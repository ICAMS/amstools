import os
import socket
import getpass
import logging

hostname = socket.gethostname()
username = getpass.getuser()
pid = str(os.getpid())

LOG_FMT = "%(asctime)s ({}[{}]): %(message)s".format(hostname, pid)
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
logger = logging.getLogger()

from itertools import permutations, combinations, product, chain
import yaml
import numpy as np
import glob
import spglib
import warnings

from amstools import *
from amstools.resources.cfgio import (
    create_structure,
    get_concentration_dict,
    elements_alat_dict,
)
from amstools.utils import (
    make_cell_for_non_periodic_structure,
    get_wyckoffs,
)
from amstools.highthroughput.utils import load_yaml

STATIC = "static"

RELAX = "relax"
RELAX_ATOMIC = "relax-atomic"
RELAX_FULL = "relax-full"

ENN_COARSE = "Enn-coarse"
ENN_FINE = "Enn-fine"
ENN_LOCAL = "Enn-local"
ENN_LOCAL_FEW = "Enn-local-few"

MURNAGHAN = "murnaghan"

ELASTIC = "elastic"

PHONONS = "phonons"

TP_HEXAGONAL = "tp_hexagonal"
TP_ORTHOGONAL = "tp_orthogonal"
TP_TETRAG = "tp_tetrag"
TP_TRIGONAL = "tp_trigonal"
TP_GENERAL_CUBIC_TETRAGONAL = "tp_general_cubic_tetragonal"

DEFECTFORMATION = "defectformation"

STACKING_FAULT = "stacking_fault"

RANDOMDEFORMATION = "randomdeformation"

PIPELINE_STEPS_NAMES = [
    ENN_COARSE,
    ENN_FINE,
    ENN_LOCAL,
    ENN_LOCAL_FEW,
    MURNAGHAN,
    ELASTIC,
    RELAX,
    RELAX_ATOMIC,
    RELAX_FULL,
    PHONONS,
    TP_TETRAG,
    TP_ORTHOGONAL,
    TP_TRIGONAL,
    TP_HEXAGONAL,
    TP_GENERAL_CUBIC_TETRAGONAL,
    DEFECTFORMATION,
    STATIC,
    STACKING_FAULT,
    RANDOMDEFORMATION,
]

pipeline_short_step_names = {
             "static": "stc", 
             "relax" : "rlx",
             "relax-atomic": "rla",
             "relax-full": "rlf",
             "Enn-coarse": "enc",
             "Enn-fine": "enf",
             "Enn-local": "enl",
             "murnaghan": "mrn",
             "elastic":"ela",
             "phonons": "pho",
             "tp_hexagonal": "thx",
             "tp_orthogonal":"tot",
             "tp_tetra": "tta",
             "tp_trigonal": "ttg",
             "tp_general_cubic_tetragonal": "tct",
             "defectformation":"dff",
             "stacking_fault":"sft",
             "randomdeformation": "rdf"}

pipeline_step_modifies_structure = {"static": "N", 
             "relax" : "Y",
             "relax-atomic": "Y",
             "relax-full": "Y",
             "Enn-coarse": "N",
             "Enn-fine": "N",
             "Enn-local": "N",
             "murnaghan": "Y",
             "elastic": "N",
             "phonons": "N",
             "tp_hexagonal": "N",
             "tp_orthogonal":"N",
             "tp_tetra": "N",
             "tp_trigonal": "N",
             "tp_general_cubic_tetragonal": "N",
             "defectformation": "N",
             "stacking_fault": "N",
             "randomdeformation": "N"}             


PERM_TYPE_ATOMS = "atoms"
PERM_TYPE_MIXED_WYCKOFF_ATOMS = "mixed"
PERM_TYPE_WYCKOFF = "wyckoff"
PERM_TYPE_CFG = "cfg"
PERM_TYPE_CONSTRAINED = "constrained"

STRUCTURES_PATH = "AMS_STRUCTURE_REPOSITORY_PATH"

nn_list_fine = [
    0.600,
    0.605,
    0.610,
    0.615,
    0.620,
    0.625,
    0.630,
    0.635,
    0.640,
    0.645,
    0.650,
    0.655,
    0.660,
    0.665,
    0.670,
    0.675,
    0.680,
    0.685,
    0.690,
    0.695,
    0.700,
    0.705,
    0.710,
    0.715,
    0.720,
    0.725,
    0.730,
    0.735,
    0.740,
    0.745,
    0.750,
    0.755,
    0.760,
    0.765,
    0.770,
    0.775,
    0.780,
    0.785,
    0.790,
    0.795,
    0.800,
    0.805,
    0.810,
    0.815,
    0.820,
    0.825,
    0.830,
    0.835,
    0.840,
    0.845,
    0.850,
    0.855,
    0.860,
    0.865,
    0.870,
    0.875,
    0.880,
    0.885,
    0.890,
    0.895,
    0.900,
    0.905,
    0.910,
    0.915,
    0.920,
    0.925,
    0.930,
    0.935,
    0.940,
    0.945,
    0.950,
    0.955,
    0.960,
    0.961,
    0.962,
    0.963,
    0.964,
    0.965,
    0.966,
    0.967,
    0.968,
    0.969,
    0.970,
    0.971,
    0.972,
    0.973,
    0.974,
    0.975,
    0.976,
    0.977,
    0.978,
    0.979,
    0.980,
    0.981,
    0.982,
    0.983,
    0.984,
    0.985,
    0.986,
    0.987,
    0.988,
    0.989,
    0.990,
    0.991,
    0.992,
    0.993,
    0.994,
    0.995,
    0.996,
    0.997,
    0.998,
    0.999,
    1.000,
    1.001,
    1.002,
    1.003,
    1.004,
    1.005,
    1.006,
    1.007,
    1.008,
    1.009,
    1.010,
    1.011,
    1.012,
    1.013,
    1.014,
    1.015,
    1.016,
    1.017,
    1.018,
    1.019,
    1.020,
    1.021,
    1.022,
    1.023,
    1.024,
    1.025,
    1.026,
    1.027,
    1.028,
    1.029,
    1.030,
    1.031,
    1.032,
    1.033,
    1.034,
    1.035,
    1.036,
    1.037,
    1.038,
    1.039,
    1.040,
    1.041,
    1.042,
    1.043,
    1.044,
    1.045,
    1.046,
    1.047,
    1.048,
    1.049,
    1.050,
    1.055,
    1.060,
    1.065,
    1.070,
    1.075,
    1.080,
    1.085,
    1.090,
    1.095,
    1.100,
    1.110,
    1.120,
    1.130,
    1.140,
    1.150,
    1.160,
    1.170,
    1.180,
    1.190,
    1.200,
    1.210,
    1.220,
    1.230,
    1.240,
    1.250,
    1.260,
    1.270,
    1.280,
    1.290,
    1.300,
    1.310,
    1.320,
    1.330,
    1.340,
    1.350,
    1.400,
    1.450,
    1.500,
    1.550,
    1.600,
    1.650,
    1.700,
    1.750,
    1.800,
    1.850,
    1.900,
    1.950,
    2.000,
    2.050,
    2.100,
    2.150,
    2.200,
    2.250,
    2.300,
    2.350,
    2.400,
    2.500,
    2.600,
    2.700,
    2.800,
    2.900,
    3.000,
]
nn_list_coarse = [
    0.600,
    0.650,
    0.700,
    0.800,
    0.850,
    0.900,
    0.950,
    1.000,
    1.050,
    1.100,
    1.150,
    1.200,
    1.250,
    1.300,
    1.350,
    1.400,
    1.450,
    1.500,
    1.600,
    1.700,
    1.800,
    1.900,
    2.000,
    2.200,
    2.400,
    2.600,
    2.800,
    3.000,
]
nn_list_few = [0.980, 1.000, 1.020]


def run_length_encoding(seq):
    compressed = []
    count = 1
    char = seq[0]
    for i in range(1, len(seq)):
        if seq[i] == char:
            count = count + 1
        else:
            compressed.append([char, count])
            char = seq[i]
            count = 1
    compressed.append([char, count])
    compressed_seq = ""
    for char, count in compressed:
        compressed_seq += char
        if count > 1:
            compressed_seq += str(count)
    return compressed_seq


def get_list_permutations_constrained(
    global_config_compositions, composition_dict, n_rnd_config, rnd_seed
):
    list_permutations = []
    combinations_from_composition_dict = [
        list(combinations(v_w, 1)) for k_w, v_w in composition_dict.items()
    ]
    config_compositions = generate_products_combinations(
        combinations_from_composition_dict
    )
    list_permutations.extend(
        x for x in config_compositions if x not in list_permutations
    )
    if n_rnd_config is not None:
        list_permutations = draw_random_samples(
            list_permutations, n_rnd_config, rnd_seed
        )

    return list_permutations


def get_list_permutations_wyckoff(config_compositions, cfg_file, n_sublattices):
    list_permutations = []
    config_permutations = permutations(config_compositions, n_sublattices)
    for compositions in config_permutations:
        structure = create_structure(cfg_file, list(compositions))
        chemical_symbols = list(structure.symbols)
        wyckoffs = get_wyckoffs(structure)

        counter_wyckoff_sites = {i: wyckoffs.count(i) for i in wyckoffs}
        counter_elements = {i: chemical_symbols.count(i) for i in chemical_symbols}

        dict_elements_to_wyckoff_sites = {}
        for (k_w, v_w), (k_e, v_e) in zip(
            counter_wyckoff_sites.items(), counter_elements.items()
        ):
            if (
                k_w not in dict_elements_to_wyckoff_sites
                and k_e not in dict_elements_to_wyckoff_sites.values()
            ) and v_e == v_w:
                dict_elements_to_wyckoff_sites[k_w] = k_e

        # print(dict_elements_to_wyckoff_sites)
        list_elements_to_wyckoff_sites = [
            dict_elements_to_wyckoff_sites[at] for at in wyckoffs
        ]
        list_permutations.append(list_elements_to_wyckoff_sites)
    return list_permutations


def get_list_permutations_atoms(config_compositions, cfg_file, n_sublattices):
    config_compositions_permutations = permutations(config_compositions, n_sublattices)
    list_chemical_symbols, list_structures = remove_spurious_permutations(
        config_compositions_permutations, cfg_file
    )
    list_permutations_for_given_composition = []
    for chemical_symbols, structure in zip(list_chemical_symbols, list_structures):
        list_permutations_for_given_composition += set(permutations(chemical_symbols))
    list_permutations_for_given_composition_unique = list(
        set(list_permutations_for_given_composition)
    )
    list_permutations_for_given_composition_unique.sort()
    list_permutations = [
        list(x) for x in list_permutations_for_given_composition_unique
    ]

    return list_permutations


def flattenNestedList(nested_list):
    flat_list = []
    for elem in nested_list:
        if isinstance(elem, list):
            flat_list.extend(flattenNestedList(elem))
        elif isinstance(elem, tuple):
            flat_list.append(list(elem))
        else:
            flat_list.append(elem)

    return flat_list


def adjacent_dups(lst):
    adj = [lst[x] for x in range(len(lst) - 1) if lst[x] == lst[x + 1]]
    return len(adj) > 0


def restore_order_wyckoff_sites(wyckofs, products):
    dict_wyckoff_ordered = {}
    i = 0
    for wyckoff_sites in wyckofs:
        if wyckoff_sites not in dict_wyckoff_ordered:
            dict_wyckoff_ordered[wyckoff_sites] = i
            i += 1

    wyckoff_unordered_sites = list(range(len(wyckofs)))
    wyckoff_ordered_sites = [
        v
        for wyckoff_sites in wyckofs
        for k, v in dict_wyckoff_ordered.items()
        if wyckoff_sites == k
    ]
    sorted_wyckoff_indices = [
        x for _, x in sorted(zip(wyckoff_ordered_sites, wyckoff_unordered_sites))
    ]

    products_sorted = [[y[index] for index in sorted_wyckoff_indices] for y in products]

    return products_sorted


def remove_spurious_permutations(config_permutations, cfg_file):
    """
    Remove spurious permutations of the same composition
    """
    list_chemical_symbols = []
    list_structures = []
    for config in config_permutations:
        structure = create_structure(cfg_file, list(config))
        chemical_symbols = list(structure.symbols)
        chemical_symbols.sort()
        if chemical_symbols not in list_chemical_symbols:
            list_structures.append(structure)
            list_chemical_symbols.append(chemical_symbols)

    return list_chemical_symbols, list_structures


def get_list_permutations_mixed(config_compositions, cfg_file, n_sublattices):
    list_permutations_with_duplicates = []
    config_permutations = permutations(config_compositions, n_sublattices)
    list_chemical_symbols, list_structures = remove_spurious_permutations(
        config_permutations, cfg_file
    )

    for chemical_symbols, structure in zip(list_chemical_symbols, list_structures):
        wyckoffs = get_wyckoffs(structure)
        counter_wyckoff_sites = {i: wyckoffs.count(i) for i in wyckoffs}

        combinations_wyckoff = [
            list(combinations(chemical_symbols, v_w))
            for k_w, v_w in counter_wyckoff_sites.items()
        ]
        flatten_products_combinations = generate_products_combinations(
            combinations_wyckoff
        )

        if len(wyckoffs) != len(set(wyckoffs)) and not adjacent_dups(wyckoffs):
            flatten_products_combinations = restore_order_wyckoff_sites(
                wyckoffs, flatten_products_combinations
            )

        counter_elements = {i: chemical_symbols.count(i) for i in chemical_symbols}
        for combination in flatten_products_combinations:
            counter_config = {i: combination.count(i) for i in combination}
            if counter_config == counter_elements:
                list_permutations_with_duplicates.append(combination)

    list_permutations = list(
        map(list, set(map(tuple, list_permutations_with_duplicates)))
    )
    list_permutations.sort()

    return list_permutations


def generate_products_combinations(combinations):
    products_combinations = []
    for f in list(combinations):
        if products_combinations:
            products_combinations = [list(s) for s in product(products_combinations, f)]
        else:
            products_combinations = f

    return [
        list(chain.from_iterable(flattenNestedList(f1)))
        for f1 in list(products_combinations)
    ]


def draw_random_samples(list_to_select_from, nsample=None, rnd_seed=None):
    if rnd_seed is not None:
        np.random.seed(rnd_seed)
    if nsample is None:
        return list_to_select_from
    elif nsample > len(list_to_select_from):
        return list_to_select_from
    else:
        rnd_indices = np.random.choice(len(list_to_select_from), nsample, replace=False)
        return [list_to_select_from[i] for i in rnd_indices]


def get_list_permutations(
    permutation_type, config_compositions, cfg_file, nsample=None, rnd_seed=None
):
    if config_compositions is None:
        return [None]
    structure = create_structure(cfg_file, list(config_compositions))
    if not structure.do_permutations:
        return [None]
    if not structure.get_pbc().all() and permutation_type in [
        PERM_TYPE_WYCKOFF,
        PERM_TYPE_MIXED_WYCKOFF_ATOMS,
    ]:
        raise ValueError(
            "Invalid permutation type `{}` for non-periodic structure".format(
                permutation_type
            )
        )

    n_elements = len(config_compositions)
    n_sublattices = len(list(set(structure.symbols)))
    n_atoms = len(structure)

    if structure.get_pbc().all() != False:
        # print(structure.get_pbc())
        wyckoffs = get_wyckoffs(structure)
        n_wyckofs = len(list(set(wyckoffs)))
    else:
        n_wyckofs = 0

    list_permutations = []
    if n_elements == 1:
        list_permutations.append(list(config_compositions * n_atoms))
    elif n_sublattices == 1:
        list_permutations = [
            list(x * n_atoms)
            for x in list(permutations(config_compositions, n_sublattices))
        ]
    elif n_elements < n_sublattices:
        raise ValueError(
            "more sublattices ({}) than elements ({})".format(n_sublattices, n_elements)
        )
    elif n_elements > n_wyckofs and permutation_type == PERM_TYPE_WYCKOFF:
        raise ValueError(
            "more elements ({}) than Wyckoff sites ({})".format(n_elements, n_wyckofs)
        )
    elif n_elements < n_wyckofs and permutation_type == PERM_TYPE_WYCKOFF:
        raise ValueError(
            "less elements ({}) than Wyckoff sites ({})".format(n_elements, n_wyckofs)
        )
    else:
        if permutation_type == PERM_TYPE_CFG:
            list_permutations = [
                list(x) for x in list(permutations(config_compositions, n_sublattices))
            ]
        elif permutation_type == PERM_TYPE_WYCKOFF:
            list_permutations = get_list_permutations_wyckoff(
                config_compositions, cfg_file, n_sublattices
            )
        elif permutation_type == PERM_TYPE_ATOMS:
            list_permutations = get_list_permutations_atoms(
                config_compositions, cfg_file, n_sublattices
            )
        elif permutation_type == PERM_TYPE_MIXED_WYCKOFF_ATOMS:
            if n_wyckofs == 1:
                list_permutations.append(list(structure.symbols))
            else:
                list_permutations = get_list_permutations_mixed(
                    config_compositions, cfg_file, n_sublattices
                )
        else:
            raise ValueError(
                "Invalid permutation type `{}`, should be: {},{},{} or {}".format(
                    permutation_type,
                    PERM_TYPE_ATOMS,
                    PERM_TYPE_MIXED_WYCKOFF_ATOMS,
                    PERM_TYPE_WYCKOFF,
                    PERM_TYPE_CFG,
                )
            )

    if nsample is not None and len(list_permutations) > nsample:
        warnings.warn(
            "Number of possible combinations larger than {}: a total of {} configurations will be selected randomly".format(
                nsample, nsample
            )
        )
        list_permutations = draw_random_samples(list_permutations, nsample, rnd_seed)

    return list_permutations


def generate_ht_pipelines_setup(
    filename_or_config,
    permutation_type="cfg",
    structures_repository_path=None,
    generate_pipelines=True,
    name_prefix=None,
    enforce_pbc=False,
    verbose=False,
    legacy_step_naming=False,
):
    """
    Parameters
    ----------
    filename_or_config : file or dict
        ht.yaml file or dictionary containing structure paths with corresponding pipeline steps
        and information on the seed, permutation type, nsample (maximum number of possible combinations
        to select) and composition
    permutation_type : str
        Global permutation type (default is 'cfg'). The possible values are 'atoms', 'mixed', 'wyckoff'
        'cfg' and 'constrained'.
    structures_repository_path: str, optional
        Structure folder path, if not provided the env path STRUCTURES_PATH will be used
    generate_pipelines: bool, optional
        Boolean to generate the pipeline steps found in filename_or_config (default is True)

    Returns
    -------
    dict
        {"structure": [...], "name": [...], "pipeline": [...]}
        a dict containing the generated ASE structures, the corresponding path names and
        pipelines
    """
    if structures_repository_path is None:
        structures_repository_path = os.environ.get(STRUCTURES_PATH)
    if structures_repository_path is None:
        raise ValueError(
            "No amsstructures repository is provided: use `structures_repository_path` param "
            + "or set ${} env variable".format(STRUCTURES_PATH)
        )
    if not structures_repository_path.endswith("/"):
        structures_repository_path = structures_repository_path + "/"
    if isinstance(filename_or_config, dict):
        config = filename_or_config.copy()
    else:
        config = load_yaml(filename_or_config)

    global_config_compositions = (
        config.pop("composition") if "composition" in config else None
    )
    rnd_seed = config.pop("seed") if "seed" in config else None
    if "global_permutation_type" in config:
        global_permutation_type = config.pop("global_permutation_type")
    else:
        global_permutation_type = permutation_type
    nsample = config.pop("nsample") if "nsample" in config else None

    dict_data = {"structure": [], "name": []}
    if generate_pipelines:
        dict_data["pipeline"] = []

    for cfg_name, pipeline_steps_and_options in config.items():
        filepath = structures_repository_path + cfg_name
        filenames = []
        n_rnd_config = None
        composition_dict = None
        pipeline_steps = []
        permutation_type = global_permutation_type
        if os.path.isdir(filepath):
            filenames = glob.glob(filepath + "/*.cfg")
            for step_or_option in pipeline_steps_and_options:
                if isinstance(step_or_option, dict):
                    if "random" in step_or_option and len(step_or_option.keys()) == 1:
                        n_rnd_struc = step_or_option.get("random")
                        if n_rnd_struc is not None:
                            filenames = draw_random_samples(
                                filenames, n_rnd_struc, rnd_seed
                            )
                        else:
                            if isinstance(filename_or_config, dict):
                                raise RuntimeError(
                                    "Could not get `random` config for {} block in config dict".format(
                                        cfg_name
                                    )
                                )
                            else:
                                raise RuntimeError(
                                    "Could not get `random` config for {} block in {} YAML file".format(
                                        cfg_name, filename_or_config
                                    )
                                )
                    elif "composition" in step_or_option:
                        composition_dict = step_or_option.get("composition")
                        if "random" in step_or_option:
                            n_rnd_config = step_or_option.get("random")
                    elif "permutation_type" in step_or_option:
                        permutation_type = step_or_option.get("permutation_type")
                    else:
                        pipeline_steps.append(step_or_option)
                else:
                    pipeline_steps.append(step_or_option)
        elif os.path.isfile(filepath) or "*" in filepath:
            filenames = glob.glob(filepath)
            for step_or_option in pipeline_steps_and_options:
                if isinstance(step_or_option, dict):
                    if "random" in step_or_option:
                        n_rnd_config = step_or_option.get("random")
                    elif "composition" in step_or_option:
                        composition_dict = step_or_option.get("composition")
                    elif "permutation_type" in step_or_option:
                        permutation_type = step_or_option.get("permutation_type")
                    else:
                        pipeline_steps.append(step_or_option)
                else:
                    pipeline_steps.append(step_or_option)
        else:
            if isinstance(filename_or_config, dict):
                raise RuntimeError(
                    "File or folder {} does not exist (from config dict)".format(
                        filepath
                    )
                )
            else:
                raise RuntimeError(
                    "File or folder {} does not exist (from {} YAML file)".format(
                        filepath, filename_or_config
                    )
                )
        if verbose:
            logger.info("Processing {} structure files".format(len(filenames)))
        for file in filenames:
            if verbose:
                logger.info("{}".format(file))
            if permutation_type == PERM_TYPE_CONSTRAINED:
                list_permutations = get_list_permutations_constrained(
                    global_config_compositions, composition_dict, n_rnd_config, rnd_seed
                )
            else:
                list_permutations = get_list_permutations(
                    permutation_type,
                    global_config_compositions,
                    file,
                    nsample,
                    rnd_seed,
                )

            for ordsymb in list_permutations:
                # prepare the structures
                structure = (
                    create_structure(file, ordsymb)
                    if (
                        permutation_type == PERM_TYPE_CFG
                        or permutation_type == PERM_TYPE_CONSTRAINED
                    )
                    else create_structure(file, chemical_symbs=ordsymb)
                )
                relaxation_type = "full"
                if enforce_pbc and not np.all(structure.get_pbc()):
                    r_estimation = calculate_r_estimation(structure)
                    orthogonal_cell = make_cell_for_non_periodic_structure(
                        structure, wrapped=True, scale=4.0, alat=r_estimation
                    )
                    structure.set_cell(orthogonal_cell)
                    structure.set_pbc(True)
                    structure.enforce_pbc = True
                    relaxation_type = "atomic"

                dict_data["structure"].append(structure)
                chem_symbols_rle = run_length_encoding(structure.symbols)

                composition_str = "".join(sorted(set(structure.symbols)))
                file_prefix = file.replace(structures_repository_path, "").replace(
                    ".cfg", ""
                )
                if file_prefix.startswith("/"):
                    file_prefix = file_prefix[1:]
                name = "{composition_str}/{file_prefix}/{permutation_type}/{chem_symbols_rle}/".format(
                    composition_str=composition_str,
                    file_prefix=file_prefix,
                    permutation_type=permutation_type,
                    chem_symbols_rle=chem_symbols_rle,
                )
                dict_data["name"].append(
                    name if name_prefix is None else name_prefix + name
                )

                # prepare the pipeline
                if generate_pipelines:
                    r_estimation = calculate_r_estimation(structure)
                    pipeline = generate_pipeline(
                        pipeline_steps, r_estimation, relaxation_type,
                        legacy_step_naming=legacy_step_naming,
                    )
                    dict_data["pipeline"].append(pipeline)

    return dict_data


def generate_pipeline(pipeline_steps, r_estimation, relaxation_type, legacy_step_naming=False):
    modifying_steps = []  # Track structure-modifying steps
    steps = {}

    for step in pipeline_steps:
        job_options = {}
        if isinstance(step, str):
            step_name = step
        elif isinstance(step, dict):
            step_name = next(iter(step))
            job_options = step[step_name]
        else:
            raise ValueError(
                "Unrecognized step definition (should be str or dict): {}".format(step)
            )

        if legacy_step_naming:
            dynamic_step_name = step_name
        else:
            # Determine if the step modifies the structure
            modifies_structure = pipeline_step_modifies_structure.get(step_name, "N") == "Y"
            short_name = pipeline_short_step_names.get(step_name, step_name)

            # Prepend modifying steps for subsequent steps
            dynamic_step_name = "_".join(modifying_steps + [step_name]) if modifying_steps else step_name

            if modifies_structure:
                modifying_steps.append(short_name)

        # Ensure no step is overwritten
        if dynamic_step_name in steps:
            dynamic_step_name += f"_{len(steps)}"

        # Debug: Print step details
        modifies_str = "N/A (legacy)" if legacy_step_naming else str(pipeline_step_modifies_structure.get(step_name, "N") == "Y")
        print(f"Adding step: {dynamic_step_name}, modifies_structure: {modifies_str}")

        if step_name == ENN_COARSE:
            nn_list = np.array(nn_list_coarse) * r_estimation
            job_options_default = dict(nn_distance_list=nn_list, fix_kmesh=False)
            job_options_default.update(job_options)
            if "nn_distance_range" in job_options:
                del job_options_default["nn_distance_list"]
            steps[dynamic_step_name] = NearestNeighboursExpansionCalculator(
                allow_fail=True, **job_options_default
            )
        elif step_name == ENN_FINE:
            nn_list = np.array(nn_list_fine) * r_estimation
            job_options_default = dict(nn_distance_list=nn_list, fix_kmesh=False)
            job_options_default.update(job_options)
            if "nn_distance_range" in job_options:
                del job_options_default["nn_distance_list"]
            steps[dynamic_step_name] = NearestNeighboursExpansionCalculator(
                allow_fail=True, **job_options_default
            )
        elif step_name == RELAX:
            if relaxation_type == "full":
                steps[dynamic_step_name] = StepwiseOptimizer(**job_options)
            elif relaxation_type == "atomic":
                steps[dynamic_step_name] = SpecialOptimizer(optimize_atoms_only=True, **job_options)
        elif step_name == RELAX_ATOMIC:
            steps[dynamic_step_name] = SpecialOptimizer(optimize_atoms_only=True, **job_options)
        elif step_name == RELAX_FULL:
            steps[dynamic_step_name] = StepwiseOptimizer(**job_options)
        elif step_name == ELASTIC:
            job_options_default = dict(eps_range=0.015, num_of_point=5)
            job_options_default.update(job_options)
            steps[dynamic_step_name] = ElasticMatrixCalculator(
                allow_fail=True, **job_options_default
            )
        elif step_name == PHONONS:
            steps[dynamic_step_name] = PhonopyCalculator(allow_fail=True)
        elif step_name == ENN_LOCAL:
            range = (0.88 * r_estimation, 1.120 * r_estimation)
            step = 0.02 * r_estimation
            job_options_default = dict(
                nn_distance_range=range, nn_distance_step=step, fix_kmesh=True
            )
            job_options_default.update(job_options)
            steps[dynamic_step_name] = NearestNeighboursExpansionCalculator(
                allow_fail=True, **job_options_default
            )
        elif step_name == ENN_LOCAL_FEW:
            nn_list = []
            for item in nn_list_few:
                nn_list.append(float(item) * r_estimation)
            job_options_default = dict(nn_distance_list=nn_list, fix_kmesh=True)
            job_options_default.update(job_options)
            steps[dynamic_step_name] = NearestNeighboursExpansionCalculator(
                allow_fail=True, **job_options_default
            )
        elif step_name == STATIC:
            steps[dynamic_step_name] = StaticCalculator(allow_fail=True)
        elif step_name == MURNAGHAN:
            job_options_default = dict(num_of_point=11, volume_range=0.1, fit_order=5)
            job_options_default.update(job_options)
            steps[dynamic_step_name] = MurnaghanCalculator(**job_options_default)
        elif step_name == TP_TETRAG:
            job_options_default = dict(
                num_of_point=50, transformation_type="tetragonal"
            )
            job_options_default.update(job_options)
            steps[dynamic_step_name] = TransformationPathCalculator(
                allow_fail=True, **job_options_default
            )
        elif step_name == TP_ORTHOGONAL:
            job_options_default = dict(
                num_of_point=50, transformation_type="orthogonal"
            )
            job_options_default.update(job_options)
            steps[dynamic_step_name] = TransformationPathCalculator(
                allow_fail=True, **job_options_default
            )
        elif step_name == TP_HEXAGONAL:
            job_options_default = dict(num_of_point=50, transformation_type="hexagonal")
            job_options_default.update(job_options)
            steps[dynamic_step_name] = TransformationPathCalculator(
                allow_fail=True, **job_options_default
            )
        elif step_name == TP_TRIGONAL:
            job_options_default = dict(num_of_point=50, transformation_type="trigonal")
            job_options_default.update(job_options)
            steps[dynamic_step_name] = TransformationPathCalculator(
                allow_fail=True, **job_options_default
            )
        elif step_name == TP_GENERAL_CUBIC_TETRAGONAL:
            job_options_default = dict(
                num_of_point=50, transformation_type="general_cubic_tetragonal"
            )
            job_options_default.update(job_options)
            steps[dynamic_step_name] = TransformationPathCalculator(
                allow_fail=True, **job_options_default
            )
        elif step_name == DEFECTFORMATION:
            job_options_default = dict(interaction_range=10.0, defect_type="vacancy")
            job_options_default.update(job_options)
            steps[dynamic_step_name] = DefectFormationCalculator(
                allow_fail=True, **job_options_default
            )
        elif step_name == STACKING_FAULT:
            job_options_default = dict()
            job_options_default.update(job_options)
            steps[dynamic_step_name] = StackingFaultCalculator(
                allow_fail=True, **job_options_default
            )
        elif step_name == RANDOMDEFORMATION:
            job_options_default = dict(
                nsample=3,
                random_atom_displacement=0.1,
                random_cell_strain=0.05,
                volume_range=0.05,
                num_volume_deformations=5,
                seed=42,
            )
            job_options_default.update(job_options)
            steps[dynamic_step_name] = RandomDeformationCalculator(
                allow_fail=True, **job_options_default
            )
        else:
            raise ValueError("Unrecognized step name: {}".format(step_name))

    pipeline = Pipeline(steps)
    return pipeline


def calculate_r_estimation(structure):
    concentraion_dict = get_concentration_dict(structure)
    r_estimation = sum(
        elements_alat_dict.get(el, 1.0) * c for el, c in concentraion_dict.items()
    )
    return r_estimation
