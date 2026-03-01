import json
import os
import logging
import sqlite3
from collections import Counter
from contextlib import contextmanager
import importlib
from datetime import datetime
from ruamel import yaml

import spglib
import numpy as np
import pandas as pd

from ase import Atoms
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator
from ase.neighborlist import NeighborList
from ase.db.row import AtomsRow
from ase.io import jsonio

CALC_LOCK = "calc.lock"

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger()


fixsymmetry_imported = False

try:
    try:
        from ase.spacegroup.symmetrize import FixSymmetry

        fixsymmetry_imported = True
    except ImportError:
        from ase.constraints import FixSymmetry

        fixsymmetry_imported = True

except ImportError:
    fixsymmetry_imported = False
    FixSymmetry = None
    print(
        "WARNING!!!: Couldn't import ase.spacegroup.symmetrize.FixSymmetry. FixSymmetry options would be ignored. "
        "Please upgrade ase, if you want to use it"
    )


def load_yaml(fname, allow_duplicate_keys=False, typ="safe"):
    """Safely load YAML (old and new way, for compatibility reason)"""
    with open(fname, "r") as f:
        try:
            return yaml.safe_load(f)
        except AttributeError:
            yaml_loader = yaml.YAML(typ=typ)
            yaml_loader.allow_duplicate_keys = allow_duplicate_keys
            return yaml_loader.load(f)


def get_symmetry_dataset(struct, symprec=1e-5):
    """
    Wrapper for spglib.get_symmetry_dataset that handles ASE Atoms objects.
    """
    # Check if struct is an ASE Atoms object (has get_cell method)
    if hasattr(struct, "get_cell"):
        lattice = struct.get_cell()
        positions = struct.get_scaled_positions()
        numbers = struct.get_atomic_numbers()
        # Create the tuple (lattice, positions, numbers) expected by spglib
        cell = (lattice, positions, numbers)
    else:
        # Assume it is already in the (lattice, positions, numbers) format
        cell = struct

    return spglib.get_symmetry_dataset(cell, symprec=symprec)


def get_spacegroup(struct):
    """Wrapper for spglib.get_spacegroup for compatibility"""
    sym_ds = get_symmetry_dataset(struct)
    try:
        return sym_ds.number
    except (TypeError, AttributeError):
        return sym_ds["number"]


def get_wyckoffs(struct):
    """Wrapper for spglib.get_spacegroup for compatibility"""
    sym_ds = get_symmetry_dataset(struct)
    try:
        return sym_ds.wyckoffs
    except (TypeError, AttributeError):
        return sym_ds["wyckoffs"]


class JsonNumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, type):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def serialize_class(cls):
    return cls.__module__ + "." + cls.__qualname__


def deserialize_class(class_name):
    if class_name is None:
        return None
    module_name, _, class_name_short = class_name.rpartition(".")
    if module_name:
        module = importlib.import_module(module_name)
        return getattr(module, class_name_short)
    else:
        raise ValueError("Couldn't deserialize class `{}`".format(class_name))


def get_all_subclasses(cls):
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


def _compute_nn_distance(atoms, nndist_cutoff=1.5):
    """
    Compute neighbour distances for each atom
    :param atoms: ASE atoms
    :param nndist_cutoff: float, cutoff for neighbours list constructions
    :return: np.array of distances, shape (n_atoms, -1)
    """
    nl = NeighborList(cutoffs=[nndist_cutoff] * len(atoms), self_interaction=False)
    nl.update(atoms)
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    dists_list = []
    for i in range(len(atoms)):
        r0 = positions[i]
        indices, offsets = nl.get_neighbors(i)
        dists = np.array(
            [
                positions[neighbor_idx] + offset @ cell - r0
                for neighbor_idx, offset in zip(indices, offsets)
            ]
        )
        if len(dists) > 0:
            dists_list.append(np.linalg.norm(dists, axis=1))
        else:
            dists_list.append([])

    return np.hstack(dists_list)


def compute_nn_distance(atoms, nndist_cutoff=1.5, autoincrease_cutoff=True):
    if autoincrease_cutoff:
        while True:
            min_dist = _compute_nn_distance(atoms, nndist_cutoff=nndist_cutoff)
            if len(min_dist) > 0:
                return min_dist
            else:
                nndist_cutoff += 1  # increase by 1 Ang
    else:
        return _compute_nn_distance(atoms, nndist_cutoff=nndist_cutoff)


def build_job_name(*parts, separator='_', sanitize=True):
    """Build consistent job names from components.

    This utility function creates standardized job names by joining multiple
    components with a separator, optionally sanitizing special characters.

    Args:
        *parts: Variable number of components to join into a job name
        separator: String to use between components (default: '_')
        sanitize: If True, replace '.' and '/' with '_' (default: True)

    Returns:
        str: The constructed job name

    Example:
        >>> build_job_name('nndist', f'{2.5:.4f}')
        'nndist_2_5000'
        >>> build_job_name('surface', '100', 'relaxed')
        'surface_100_relaxed'
        >>> build_job_name('vacancy', 'wyck', 'Ni', 'total', sanitize=False)
        'vacancy_wyck_Ni_total'
    """
    if sanitize:
        parts = [str(p).replace('.', '_').replace('/', '_') for p in parts]
    return separator.join(str(p) for p in parts if p)


def get_nearest_neighbor_distance(atoms, nndist_cutoff=None):
    if len(atoms) == 0:
        raise ValueError("Cannot compute nearest neighbor distance for empty structure")

    if nndist_cutoff is None:
        try:
            nndist_cutoff = (atoms.get_volume() / len(atoms)) ** (1 / 3)
        except ValueError:  # non-periodic structure
            if len(atoms) == 1:
                return np.inf
            else:
                positions = atoms.get_positions()
                # Check if all positions are identical
                if len(atoms) > 1 and np.allclose(positions[0], positions[1:]):
                    raise ValueError("All atoms at same position - cannot compute nearest neighbor distance")
                nndist_cutoff = np.linalg.norm(positions[0] - positions[1])

    min_dist = compute_nn_distance(atoms, nndist_cutoff=nndist_cutoff)
    if len(min_dist) == 0:
        raise ValueError(f"No neighbors found within cutoff {nndist_cutoff}. Increase cutoff or check structure.")
    return np.min(min_dist)


def make_cell_for_non_periodic_structure(
    structure, wrapped=True, scale=None, alat=None, min_cell_len=10
):
    """
    function to generate a cell with sides s_i; s_i = diameter_i + (alat*scale) adapted
    from Minaam Quamar

    structure: ase.Atoms object
        must be an atoms object with pbc=False and cell=[0,0,0]
        with both positive and negative cartesian positions.

        In case `structure` is provided as a periodic structure,
        set argument `wrapped=False` to wrap the positions around the
        origin to resemble a non-periodic struc

    alat: float
        ground state equilibrium lattice constant for the element

    scale: float
        arbit but preferably > 5 to ensure minimal interaction
        through pbc boundary

    wrapped: bool
        if input structure is periodic then set `wrapped=False`

        This will manually wrap the atoms about (0,0,0). But may shift the
        relative atomic positions slightly. So avoid unless necessary

    """
    if not wrapped:
        structure.wrap(center=[0, 0, 0])

    # get all the x,y,z positions
    pos = structure.positions
    x, y, z = pos.T

    # get the diameter of the cluster in x,y,z
    X = max(x) - min(x)
    Y = max(y) - min(y)
    Z = max(z) - min(z)

    # to ensure minimum diameter=1 for very small clusters
    if X < 1:
        X = 1
    if Y < 1:
        Y = 1
    if Z < 1:
        Z = 1

    # generate orthogonal cell
    cell = [
        [max(X + (scale * alat), min_cell_len), 0, 0],
        [0, max(Y + (scale * alat), min_cell_len), 0],
        [0, 0, max(Z + (scale * alat), min_cell_len)],
    ]

    return cell


def make_periodic_structure(structure, wrapped=True, scale=5.0, alat=2):
    structure = structure.copy()
    if not all(structure.get_pbc()):
        orthogonal_cell = make_cell_for_non_periodic_structure(
            structure, wrapped=wrapped, scale=scale, alat=alat
        )
        structure.set_cell(orthogonal_cell)
        structure.set_pbc(True)
    return structure


def arglocalmin(a):
    """
    Return index of local minumum
    :param a: 1D np.array
    :return: np.array indices of local minima
    """
    # Find indices where value is smaller than both neighbors
    is_local_min = (a[1:-1] < a[:-2]) & (a[1:-1] < a[2:])
    # Adjust indices because we sliced off the ends
    locmin_ind = np.where(is_local_min)[0] + 1
    return locmin_ind


def numpyize_ase_atoms_dict(structure_dict):
    """
    Convert all lists into numpy arrays
    """
    for k, v in structure_dict.items():
        if isinstance(v, list):
            structure_dict[k] = np.array(v)


def attach_results(atoms: Atoms):
    new_atoms = atoms.copy()
    results = {k: v for k, v in atoms.calc.results.items() if k in all_properties}
    new_atoms.results = results
    return new_atoms


def output_structures_todict(output_structures_dict):
    serialized_output_structures_dict = {}
    for struct_name, struct in output_structures_dict.items():
        atoms_dct = atoms_todict(struct)
        serialized_output_structures_dict[struct_name] = atoms_dct
    return serialized_output_structures_dict


def output_structures_fromdict(serialzied_output_structures_dict):
    output_structures_dict = {}
    for struct_name, at_dct in serialzied_output_structures_dict.items():
        output_structures_dict[struct_name] = atoms_fromdict(at_dct)
    return output_structures_dict


def atoms_todict(atoms):
    # assert atoms is not None
    if atoms is None:
        return None
    results = atoms.results if hasattr(atoms, "results") else None
    # make copy to avoid side-effects
    atoms = atoms.copy()
    # remove constraints (they are not serializable)
    if isinstance(atoms, Atoms):
        atoms.set_constraint()
    # serialize using ASE functionality
    atoms_dict = atoms.todict()
    # attach results if available
    if results is not None:
        atoms_dict["results"] = results
    
    # support extra_info for GeneralStructure backward compatibility
    if hasattr(atoms, "info") and atoms.info:
        atoms_dict["extra_info"] = atoms.info
    return atoms_dict


def atoms_fromdict(atoms_dict):
    results = atoms_dict.pop("results") if "results" in atoms_dict else None
    extra_info = atoms_dict.pop("extra_info") if "extra_info" in atoms_dict else None
    
    # workaround: convert all lists into numpy arrays for ASE.Atoms.fromdict
    numpyize_ase_atoms_dict(atoms_dict)
    atoms = Atoms.fromdict(atoms_dict)
    
    if results:
        atoms.calc = SinglePointCalculator(atoms, **results)
    
    if extra_info:
        atoms.info.update(extra_info)
    
    return atoms


def general_atoms_copy(atoms: Atoms) -> Atoms:
    """
    Copy ASE Atoms object and restore calculator and metadata.
    """
    if atoms is None:
        return None
    new_atoms = atoms.copy()
    if atoms.calc is not None:
        new_atoms.calc = atoms.calc
    return new_atoms


def discover_dft_jobs(path, exclude=None):
    matches = []
    for root, dirnames, filenames in os.walk(path):
        if exclude is not None:
            br = False
            for ex in exclude:
                if ex in root:
                    br = True
                    break
            if br:
                continue
        for filename in filenames:
            if filename == "data.json":
                matches.append(os.path.join(root, filename))

    return matches


def collect_raw_data(output_name, path):
    logger.info("Collecting raw calculations in {}".format(path))
    raw_dft_jsons = discover_dft_jobs(path=path)  # , exclude=["gen0_AL"])
    logger.info("{} calculations found".format(len(raw_dft_jsons)))
    name_list = []
    ase_atoms = []
    results_list = []
    calc_type_list = []
    for json_file_name in raw_dft_jsons:
        try:
            dct = jsonio.read_json(json_file_name)
            # check for vasp or fhiaims in dct keys
            if "vasp_version" in dct:
                calc_type = "VASP"
            elif "fhiaims_version" in dct:
                calc_type = "FHIaims"
            else:
                calc_type = "unknown"
                # raise ValueError("DFT calculator type is not recognized for {}".format(json_file_name))
            results = dct["results"]
            atoms = AtomsRow(dct["atoms"]).toatoms()
            name_list.append(json_file_name)
            ase_atoms.append(atoms)
            results_list.append(results)
            calc_type_list.append(calc_type)
        except Exception as e:
            logger.error("{}: {}".format(json_file_name, str(e)))
    df = pd.DataFrame(
        {
            "ase_atoms": ase_atoms,
            "name": name_list,
            "results": results_list,
            "calculator": calc_type_list,
        }
    )
    df["energy"] = df["results"].map(lambda d: d.get("free_energy"))
    df["forces"] = df["results"].map(lambda d: d.get("forces"))
    df["NUMBER_OF_ATOMS"] = df["ase_atoms"].map(len)
    df["energy_per_atom"] = df["energy"] / df["NUMBER_OF_ATOMS"]
    logger.info("Calculator type(s): {}".format(list(df["calculator"].unique())))
    logger.info(
        "{} structures / {} atoms exported".format(
            len(df), int(df["NUMBER_OF_ATOMS"].sum())
        )
    )
    df.to_pickle(output_name, compression="gzip")
    logger.info("Data stored into {}".format(output_name))


def plot_df(output_name, binary_plot=False):
    """
    To quickly plot the collected dataset for first look at the dataset

    specific modules are imported inside the function to avoid circular imports
    """
    from amstools.thermodynamics import compute_compositions
    from matplotlib import cm
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    df = pd.read_pickle(output_name, compression="gzip")
    df["nnb"] = df["ase_atoms"].apply(get_nearest_neighbor_distance)

    compositions = compute_compositions(df)

    if len(compositions) != 1:
        print("More than one composition found, switching to binary plot")
        binary_plot = True
        elem1 = compositions[0]

    fig, ax = plt.subplots(constrained_layout=True)

    strucs_over_zero = len(df[df["energy_per_atom"] > 0])

    if strucs_over_zero != 0:
        df_to_plot = df[df["energy_per_atom"] < 0].copy()
        print(
            "{:d} strucs have energy above zero and are not plotted".format(
                strucs_over_zero
            )
        )

    else:
        df_to_plot = df.copy()

    e0 = df_to_plot["energy_per_atom"].min()
    e_max = df_to_plot["energy_per_atom"].max()

    if binary_plot:
        ax.scatter(
            df_to_plot["nnb"],
            df_to_plot["energy_per_atom"],
            c=df_to_plot["c_%s" % elem1],
            marker=".",
        )

        norm = mpl.colors.Normalize(
            df_to_plot["c_%s" % elem1].min(), df_to_plot["c_%s" % elem1].max()
        )
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.set_ylabel("{:s} concentration".format(elem1), fontsize=18)

        # everything below is to write an elaborate title

        comps = df_to_plot["comp_tuple"].unique()
        counts = df_to_plot["comp_tuple"].value_counts()

        formulas_and_counts = []

        for comp, count in zip(comps, counts):
            formula = []
            for com in comp:
                formula.append("".join([com[0], str(np.round(com[1], 2))]))

            formula_string = "".join(formula)
            formulas_and_counts.append(
                "{:s} = {:d} strucs".format(formula_string, count)
            )

        formulas_and_counts_string = ";\n".join(formulas_and_counts)
        title_string = "Total {:d} strucs; with \n{:s}".format(
            len(df), formulas_and_counts_string
        )

        with open("concentrations.txt", "w+") as fl:
            fl.write(title_string)

        # ax.set_title(title_string,fontsize=8)

    else:
        print("Only single composition found. Switching to unary plot")
        ax.scatter(
            df_to_plot["nnb"], df_to_plot["energy_per_atom"], color="k", marker="."
        )
        title_string = "Total %d %s strucs" % (len(df), compositions[0])
        ax.set_title(title_string, fontsize=16)

    ax.set_xlabel("nearest neighbour distance, $\mathrm{\AA}$", fontsize=18)
    ax.set_ylabel("Energy, eV/atom", fontsize=18)

    ax.tick_params(labelsize=16)

    ax.set_ylim(e0 - 0.5, e_max + 0.5)

    plt.savefig("scatterplot_{:s}.png".format(output_name.split(".")[0]))


def load_amstools_conf():
    homedir = os.path.expanduser("~")
    conf_filename = os.path.join(homedir, ".amstools")
    logger.info("Try to load config file {}".format(conf_filename))
    return load_yaml(conf_filename)


def load_amstools_queues_setup():
    conf = load_amstools_conf()
    return conf["queues"]


def load_state_dict(state_dict_json_fname="state_dict.json"):
    if os.path.isfile(state_dict_json_fname):
        with open(state_dict_json_fname, "r") as f:
            state_dict = json.load(f)
    else:
        state_dict = {}
    return state_dict


def save_state_dict(state_dict, state_dict_json_fname="state_dict.json"):
    with open(state_dict_json_fname, "w") as f:
        json.dump(state_dict, f, indent=4)


class CalculationLockException(Exception):
    pass


@contextmanager
def lock_file(path=None, lock_filename=CALC_LOCK):
    if path is not None:
        lock_filename = os.path.join(path, lock_filename)

    lock_filename = os.path.abspath(lock_filename)
    dirname = os.path.dirname(lock_filename)
    # if lock exists -raise LockException
    # create lock file
    # yield
    # finally - remove lock file
    if os.path.isfile(lock_filename):
        raise CalculationLockException(
            "Calculation in {} is running/locked".format(dirname)
        )
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    open(lock_filename, "w").close()
    try:
        yield
    finally:
        os.remove(lock_filename)


class JSONStateDict:

    def __init__(self, fname="state_dict.json"):
        self.fname = fname
        self.state_dict = {}
        self.load()

    def load(self):
        if os.path.isfile(self.fname):
            with open(self.fname, "r") as f:
                self.state_dict = json.load(f)
        else:
            self.state_dict = {}
        return self.state_dict

    def save(self):
        with open(self.fname, "w") as f:
            json.dump(self.state_dict, f, indent=4)

    def analyze_stats(self, ht_names):
        if not isinstance(ht_names, set):
            ht_names = set(ht_names)
        state_counter = dict(
            Counter(
                cur_state.get("status", "none")
                for name, cur_state in self.state_dict.items()
                if name in ht_names
            )
        )
        return state_counter

    def __contains__(self, name):
        return name in self.state_dict

    def __len__(self):
        return len(self.state_dict)

    def __getitem__(self, key):
        if key not in self.state_dict:
            self.state_dict[key] = {}
        return self.state_dict[key]

    def __setitem__(self, key, value):
        self.state_dict[key] = value

    def get(self, key, default=None):
        if key not in self.state_dict:
            return default
        return self.state_dict[key].copy()

    def save_row(self, name, state):
        self.state_dict[name] = state
        self.save()

    def has_not_finished_or_error_states(self, ht_names):
        if not isinstance(ht_names, set):
            ht_names = set(ht_names)
        has_running = False
        for name, state in self.state_dict.items():
            if (
                name in ht_names
                and state.get("status")
                and state.get("status") not in ["finished", "error"]
            ):
                has_running = True
                break
        return has_running

    def get_running_states(self, ht_names):
        if not isinstance(ht_names, set):
            ht_names = set(ht_names)

        running_states = [
            name
            for name, state in self.state_dict.items()
            if (
                name in ht_names
                and state.get("status")
                and state.get("status") not in ["finished", "error"]
            )
        ]
        return running_states


def standardize_pipe_name(pipe_name):
    if pipe_name.startswith("/"):
        pipe_name = pipe_name[1:]
    if not pipe_name.endswith("/"):
        pipe_name += "/"
    return pipe_name


class SQLiteStateDict:

    def __init__(self, fname="state_dict.db"):
        self.fname = fname
        self.load()

    def load(self):
        if not os.path.isfile(self.fname):
            with sqlite3.connect(self.fname) as conn:
                conn.execute(
                    "CREATE TABLE state_dict (name varchar(256) primary key, state json)"
                )

    def save_row(self, name, cur_state):
        ### use UPSERT
        name = standardize_pipe_name(name)
        with sqlite3.connect(self.fname) as conn:
            cur_state["timestamp"] = str(datetime.now())
            conn.execute(
                """INSERT INTO state_dict (name, state)
                        VALUES(?, ?)  ON CONFLICT(name)
                        DO UPDATE SET state=excluded.state;""",
                [name, json.dumps(cur_state)],
            )

    def __getitem__(self, name):
        name = standardize_pipe_name(name)
        with sqlite3.connect(self.fname) as conn:
            res = conn.execute("""SELECT state FROM state_dict WHERE name=?""", [name])
            row = res.fetchone()
            if row is not None:
                return json.loads(row[0])
            else:
                return {}

    def __setitem__(self, name, value):
        name = standardize_pipe_name(name)
        self.save_row(name, value)

    def __contains__(self, name):
        name = standardize_pipe_name(name)
        with sqlite3.connect(self.fname) as conn:
            res = conn.execute("""SELECT 1 FROM state_dict WHERE name=?""", [name])
            row = res.fetchone()
            return row is not None

    def __len__(self):
        with sqlite3.connect(self.fname) as conn:
            res = conn.execute("""SELECT count(1) FROM state_dict""")
            row = res.fetchone()
            return row[0]

    def analyze_stats(self, ht_names):
        ht_names = set([standardize_pipe_name(name) for name in ht_names])
        state_counter = Counter()
        with sqlite3.connect(self.fname) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT name, json_extract(state, '$.status') FROM state_dict"""
            )

            state_counter.update(
                status
                for name, status in cursor.fetchall()
                if name in ht_names and status
            )

        return dict(state_counter)

    def has_not_finished_or_error_states(self, ht_names):
        ht_names = set([standardize_pipe_name(name) for name in ht_names])
        with sqlite3.connect(self.fname) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT name, json_extract(state, '$.status') FROM state_dict"""
            )

            has_running = any(
                status not in ["finished", "error"]
                for name, status in cursor.fetchall()
                if name in ht_names and status
            )
            return has_running

    def get_running_states(self, ht_names):
        states_info = self.get_all_states(ht_names)
        return [
            name
            for name, status in states_info.items()
            if (status not in ["finished", "error", "partially_finished"])
        ]

    def get_all_states(self, ht_names):
        ht_names = set([standardize_pipe_name(name) for name in ht_names])
        with sqlite3.connect(self.fname) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT name, json_extract(state, '$.status') FROM state_dict"""
            )

            states_info = {
                name: status
                for name, status in cursor.fetchall()
                if (name in ht_names and status)
            }
            return states_info

    def get_all_info(self):
        with sqlite3.connect(self.fname) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT name, state FROM state_dict ORDER BY json_extract(state, '$.timestamp') DESC, name ASC"""
            )
            states_info = (
                {"name": name, "state": json.loads(state)}
                for name, state in cursor.fetchall()
            )
            return states_info
