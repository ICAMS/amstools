import re
from collections import Counter
from io import StringIO

import numpy as np
from ase.data import atomic_masses, chemical_symbols
from ase.io import read

from amstools.resources.scaledata import elements_alat_dict
from datetime import datetime
import string


def get_concentration_dict(atoms):
    """
    Calculate concentration of elements in ASE `atoms`

    :param atoms: ASE atoms
    :return: {"Al": 0.5, "Ni": 0.5}
    """
    atoms_chemical_symbols = atoms.get_chemical_symbols()
    natoms = len(atoms_chemical_symbols)
    cnt = Counter(atoms_chemical_symbols)
    return {el: val / natoms for el, val in cnt.items()}


class CFGIOWrapper(StringIO):

    def __init__(self, file_name: str, mapping_elements=None):
        self.metadata_dict = {}
        self.file_name = file_name

        with open(self.file_name, "rt") as f:
            self.data = f.read()  # preprocess here

        self.prototype_elements = sorted(set(re.findall("(Ele[A-Z])", self.data)))

        if isinstance(mapping_elements, str):
            mapping_elements = [mapping_elements]

        if mapping_elements is None:
            self.mapping_elements = {
                proto_el: el
                for proto_el, el in zip(self.prototype_elements, chemical_symbols[1:])
            }
        elif isinstance(mapping_elements, (list, tuple)) and len(
            mapping_elements
        ) >= len(self.prototype_elements):
            self.mapping_elements = {
                proto_el: el
                for proto_el, el in zip(self.prototype_elements, mapping_elements)
            }
        elif isinstance(mapping_elements, dict):
            self.mapping_elements = mapping_elements
        else:
            raise ValueError(
                "`mapping_elements` should be None, list of elements or dictionary {'EleA':'Al',...}"
            )
        # now self.mapping_elements =   {'EleA':'Al',...}

        # preprocess self.data
        for proto_el, el in self.mapping_elements.items():
            self.data = self.data.replace(proto_el, el)

        super().__init__(self.data)
        self.name = file_name

        self.parse_metadata()

    def parse_metadata(self):
        self.seek(0)
        for line in self:
            line = line.strip()
            if line.startswith("##"):
                try:
                    line = line.strip("##").strip()
                    splt = line.split(":")
                    k = splt[0].lower().strip()
                    v = ":".join(splt[1:])
                    v = v.strip()
                    self.metadata_dict[k] = v
                except Exception as e:
                    print("Couldn't parse metadata line: ", line)
                    print("Error: ", e)
            elif line.startswith("A") and "=" in line:
                try:
                    self.metadata_dict["A"] = float(
                        re.findall("A\s+=\s+([0-9.]+)", line)[0]
                    )
                except Exception as e:
                    print("Couldn't parse 'A = ' line: ", line)
                    print("Error: ", e)

        self.seek(0)

    def __repr__(self, *args, **kwargs):
        return "CFGIOWrapper({})".format(self.file_name)


def create_structure(
    prototype_cfg_filename: str, mapping_elements=None, chemical_symbs=None, nndist=None
):
    """

    :param prototype_cfg_filename: str - extended CFG filename
    :param mapping_elements:  None, str, list of str, dictionary {'EleA':'Al',...}
    :param nndist: float - standard nearest neighbours distance (in Angstroms)
    :return: ASE atoms
    """
    try:
        # try to load file as usual
        atoms = read(prototype_cfg_filename)
        atoms.do_permutations = False
        return atoms
    except Exception as e:
        pass
    cfgio = CFGIOWrapper(prototype_cfg_filename, mapping_elements=mapping_elements)
    metadata_dict = cfgio.metadata_dict
    atoms = read(cfgio)
    if chemical_symbs is not None:
        if len(chemical_symbs) == len(atoms):
            atoms.set_chemical_symbols(chemical_symbs)
    if (
        metadata_dict["scaling_length"] == "original"
        and metadata_dict.get("periodic") == "True"
    ):
        pass  # no scaling
    elif (
        metadata_dict["scaling_length"] == "original"
        and metadata_dict.get("periodic") == "False"
    ):
        atoms.center(vacuum=10)
        atoms.set_pbc(False)
        atoms.set_cell(None)
        pass  # no scaling
    else:
        scaling_length = eval(metadata_dict["scaling_length"])
        A = metadata_dict["A"]
        if nndist is None:
            concentraion_dict = get_concentration_dict(atoms)
            r_estimation = sum(
                elements_alat_dict.get(el, 1.0) * c
                for el, c in concentraion_dict.items()
            )
        else:
            r_estimation = nndist
        periodic = metadata_dict.get("periodic")
        if np.all(atoms.get_pbc()) and periodic != "False":
            # periodic structure
            scale_matrix = np.eye(3) * (scaling_length * A * r_estimation)
            cell = atoms.get_cell()
            new_cell = np.dot(cell, scale_matrix)
            atoms.set_cell(new_cell, scale_atoms=True)
        elif np.all(~atoms.get_pbc()) or periodic == "False":
            # non periodic
            # raise NotImplementedError("Scaling of non-periodic atoms is not yet supported")
            scale_matrix = np.eye(3) * (scaling_length * A * r_estimation)
            atoms.center(vacuum=10)
            cell = atoms.get_cell()
            new_cell = np.dot(cell, scale_matrix)
            atoms.set_cell(new_cell, scale_atoms=True)
            atoms.center(vacuum=0)
            atoms.set_pbc(False)
            atoms.set_cell(None)
        else:
            raise ValueError(
                "Either fully periodic, or fully non-periodic structures are supported"
            )

        atoms_atomic_numbers = atoms.get_atomic_numbers()
        masses = [atomic_masses[el] for el in atoms_atomic_numbers]
        atoms.set_masses(masses)

        # TODO: assign magnetic moments, charges, etc.

    atoms.info = metadata_dict
    atoms.do_permutations = True
    return atoms


def write_structure(
    fl_name,
    atoms_init,
    save_exact=False,
    scaling_length=1.0,
    metadata_dict=None,
    mapping_elements=None,
    chemical_symbs=None,
):
    """
    adapted from ase.io.cfg.write_cfg to use with internal ICAMS's structures repo

    :param fl_name: str - name of the file to save (must have .cfg extension)
    :param atoms_init: Atoms - atoms object to be written
    :param save_exact: bool - to save the structure exactly or to scale it
    :param scaling_length: float - manually provided scaling length
    :param metadata_dict: dict - metadata printed as the header of the cfg file.
                                 default values are taken from atoms.info if provided
                                 Note: 'priodic' tag will override the pbc settings of the atoms object
    """

    if save_exact == False:
        atoms = atoms_init.copy()
    else:
        atoms = atoms_init

    periodic_tag = str(np.all(atoms.get_pbc()))

    ## Inverse mapping of the elements
    prototype_elements = [
        "Ele%s" % alphabet for alphabet in list(string.ascii_uppercase)
    ]

    # inverse mapping
    if mapping_elements is None:
        elements_in_struc = sorted(set(atoms.get_chemical_symbols()))
        inv_mapping_elements = {
            proto_el: el for proto_el, el in zip(elements_in_struc, prototype_elements)
        }
    elif isinstance(mapping_elements, (list, tuple)):
        inv_mapping_elements = {
            el: proto_el for proto_el, el in zip(prototype_elements, mapping_elements)
        }
    elif isinstance(mapping_elements, dict):
        inv_mapping_elements = {v: k for k, v in mapping_elements.items()}

    else:
        raise ValueError(
            "`mapping_elements` should be None, list of elements or dictionary {'EleA':'Al',...}"
        )

    default_vals = {
        "author": atoms.info.get("author", "ICAMS"),
        "date": atoms.info.get("date", datetime.today().strftime("%Y-%m-%d")),
        "description": atoms.info.get("description", str(inv_mapping_elements)),
        "purpose": atoms.info.get("purpose", "reference structure"),
        "name": atoms.info.get("name", " "),
        "alternative names": atoms.info.get("alternative names", " "),
        "publication": atoms.info.get("publication", " "),
        "source": atoms.info.get("source", " "),
        "scaling_length": atoms.info.get("scaling_length", "%s" % scaling_length),
        "periodic": atoms.info.get("periodic", "%s" % periodic_tag),
        "A": atoms.info.get("A", 1.0),
    }

    if metadata_dict == None:
        metadata_dict = {}

    for key, val in default_vals.items():
        if key not in metadata_dict:
            metadata_dict[key] = val

    # if chem symbols are manually provided
    if chemical_symbs is not None:
        if len(chemical_symbs) == len(atoms):
            atoms.set_chemical_symbols(chemical_symbs)

    # to scale or not to scale
    if save_exact == True:
        metadata_dict["scaling_length"] = "original"

    #     print(metadata_dict)

    if (
        metadata_dict["scaling_length"] == "original"
        and metadata_dict.get("periodic") == "True"
    ):
        pass  # no scaling
    elif (
        metadata_dict["scaling_length"] == "original"
        and metadata_dict.get("periodic") == "False"
    ):
        atoms.center(vacuum=0)
        atoms.set_pbc(False)
        atoms.set_cell([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        pass  # no scaling

    else:
        scaling_length = eval(metadata_dict["scaling_length"])
        A = metadata_dict["A"]

        # get nndist==r_estimation automatically
        concentraion_dict = get_concentration_dict(atoms)
        r_estimation = sum(
            elements_alat_dict.get(el, 1.0) * c for el, c in concentraion_dict.items()
        )

        periodic = metadata_dict.get("periodic")

        if np.all(atoms.get_pbc()) and periodic != "False":
            # periodic structure
            scale_matrix = np.eye(3) * (scaling_length * A * r_estimation)
            cell = atoms.get_cell()
            new_cell = np.dot(
                cell, np.linalg.inv(scale_matrix)
            )  # inverse the scale matrix to scale down the struc
            atoms.set_cell(new_cell, scale_atoms=True)

        elif np.all(~atoms.get_pbc()) or periodic == "False":
            # non periodic
            # raise NotImplementedError("Scaling of non-periodic atoms is not yet supported")
            scale_matrix = np.eye(3) * (scaling_length * A * r_estimation)
            atoms.center(vacuum=10)
            cell = atoms.get_cell()
            new_cell = np.dot(cell, np.linalg.inv(scale_matrix))
            atoms.set_cell(new_cell, scale_atoms=True)
            atoms.center(vacuum=0)
            atoms.set_pbc(False)
            atoms.set_cell(None)

        else:
            raise ValueError(
                "Either fully periodic, or fully non-periodic structures are supported"
            )

    ## Start writing the cfg file

    cfg_default_fields = np.array(["positions", "momenta", "numbers", "magmoms"])

    fd = open(fl_name, "w+")

    fd.write("## Author: %s\n" % metadata_dict.get("author"))
    fd.write("## Date: %s\n" % metadata_dict.get("date"))
    fd.write("## Description: %s\n" % metadata_dict.get("description"))
    fd.write("## Purpose: %s\n" % metadata_dict.get("purpose"))
    fd.write("## Name: %s\n" % metadata_dict.get("name"))
    fd.write("## Alternative names: %s\n" % metadata_dict.get("alternative names"))
    fd.write("## Publication: %s\n" % metadata_dict.get("publication"))
    fd.write("## Source: %s\n" % metadata_dict.get("source"))
    fd.write("## Scaling_length: %s\n" % metadata_dict.get("scaling_length"))
    fd.write("## Periodic: %s\n" % metadata_dict.get("periodic"))

    fd.write("Number of particles = %i\n" % len(atoms))
    fd.write("A = %f Angstrom\n" % metadata_dict.get("A"))
    cell = atoms.get_cell(complete=True)
    for i in range(3):
        for j in range(3):
            fd.write("H0(%1.1i,%1.1i) = %f A\n" % (i + 1, j + 1, cell[i, j]))

    entry_count = 3
    for x in atoms.arrays.keys():
        if x not in cfg_default_fields:
            if len(atoms.get_array(x).shape) == 1:
                entry_count += 1
            else:
                entry_count += atoms.get_array(x).shape[1]

    vels = atoms.get_velocities()
    if isinstance(vels, np.ndarray):
        entry_count += 3
    else:
        fd.write(".NO_VELOCITY.\n")

    fd.write("entry_count = %i\n" % entry_count)

    i = 0
    for name, aux in atoms.arrays.items():
        if name not in cfg_default_fields:
            if len(aux.shape) == 1:
                fd.write("auxiliary[%i] = %s [a.u.]\n" % (i, name))
                i += 1
            else:
                if aux.shape[1] == 3:
                    for j in range(3):
                        fd.write(
                            "auxiliary[%i] = %s_%s [a.u.]\n"
                            % (i, name, chr(ord("x") + j))
                        )
                        i += 1

                else:
                    for j in range(aux.shape[1]):
                        fd.write("auxiliary[%i] = %s_%1.1i [a.u.]\n" % (i, name, j))
                        i += 1

    # Distinct elements
    spos = atoms.get_scaled_positions()
    for i in atoms:
        if save_exact == False:
            el = inv_mapping_elements[i.symbol]
            mass = 1.00
        else:
            el = i.symbol
            mass = atomic_masses[i.number]

        fd.write("%f\n" % mass)
        fd.write("%s\n" % el)

        x, y, z = spos[i.index, :]
        s = "%e %e %e " % (x, y, z)

        if isinstance(vels, np.ndarray):
            vx, vy, vz = vels[i.index, :]
            s = s + " %e %e %e " % (vx, vy, vz)

        for name, aux in atoms.arrays.items():
            if name not in cfg_default_fields:
                if len(aux.shape) == 1:
                    s += " %e" % aux[i.index]
                else:
                    s += (aux.shape[1] * " %e") % tuple(aux[i.index].tolist())

        fd.write("%s\n" % s)

    fd.close()


if __name__ == "__main__":
    fname = "unaries/stackingfaults/fcc/fcc_ESF_prim_12at.cfg"

    cfgio = CFGIOWrapper(fname, mapping_elements=["Al"])
    atoms = read(cfgio)
    print(atoms)
