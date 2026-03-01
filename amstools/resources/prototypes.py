# Original author: Matous Mrovec (2020)
# Adapted by Yury Lysogorskiy (2020)

# Set up standard resources without relying on ASE
# first generate two lists of structure names and corresponding ASE atom objects (plus some extra info)
# and at the end combine them into a dictionary
import numpy as np
from ase.atoms import Atoms

from amstools.resources.atomicdata import atomic_volumes


class MockGenericEntry:
    pass


EleA = "Mt"  # just placeholder element Meitnerium 109
EleB = "Ds"  # just placeholder element Darmstadtium 110
EleC = "Rg"  # just placeholder element Roentgenium 111
EleD = "Cn"  # just placeholder element Copernicium 112
EleE = "Nh"  # just placeholder element Nihonium 113

sym_pro = ["Mt", "Ds", "Rg", "Cn", "Nh"]

x0 = 20.0
y0 = 20.0
z0 = 20.0

PROTOTYPES_DICT = {
    ######################################
    ### UNARY PERIODIC BULK STRUCTURES ###
    ######################################
    # fcc primitive
    "fcc": {
        "colour": "black",
        "nntolat": 2.0 / np.sqrt(2.0),
        "isbulk": True,
        "STRUKTURBERICHT": "A1",
        "atoms": Atoms(
            EleA + "1",
            scaled_positions=[(0.0, 0.0, 0.0)],
            cell=[(0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
            pbc=(1, 1, 1),
        ),
    },
    # fcc cubic
    "fcc_cubic": {
        "colour": "black",
        "nntolat": 2.0 / np.sqrt(2.0),
        "isbulk": True,
        "STRUKTURBERICHT": "A1",
        "atoms": Atoms(
            EleA + "4",
            scaled_positions=[
                (0.0, 0.0, 0.0),
                (0.0, 0.5, 0.5),
                (0.5, 0.0, 0.5),
                (0.5, 0.5, 0.0),
            ],
            cell=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            pbc=(1, 1, 1),
        ),
    },
    # bcc primitive
    "bcc": {
        "colour": "blue",
        "nntolat": 2.0 / np.sqrt(3.0),
        "isbulk": True,
        "STRUKTURBERICHT": "A2",
        "atoms": Atoms(
            EleA + "1",
            scaled_positions=[(0.0, 0.0, 0.0)],
            cell=[(-0.5, 0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5)],
            pbc=(1, 1, 1),
        ),
    },
    # bcc cubic
    "bcc_cubic": {
        "colour": "blue",
        "nntolat": 2.0 / np.sqrt(3.0),
        "isbulk": True,
        "STRUKTURBERICHT": "A2",
        "atoms": Atoms(
            EleA + "2",
            scaled_positions=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
            cell=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            pbc=(1, 1, 1),
        ),
    },
    # hcp primitive
    "hcp": {
        "colour": "red",
        "nntolat": 1.0,
        "isbulk": True,
        "STRUKTURBERICHT": "A3",
        "atoms": Atoms(
            EleA + "2",
            scaled_positions=[
                (0, 0, 0),
                (1 / 3, 2 / 3, 1 / 2),
            ],
            cell=[
                (1 / 2, -np.sqrt(3) / 2, 0),
                (1 / 2, np.sqrt(3) / 2, 0),
                (0.0, 0.0, np.sqrt(8 / 3)),
            ],
            pbc=(1, 1, 1),
        ),
    },
    # hcp orthorhombic
    "hcp_orthorhombic": {
        "colour": "red",
        "nntolat": 1.0,
        "isbulk": True,
        "STRUKTURBERICHT": "A3",
        "atoms": Atoms(
            EleA + "4",
            scaled_positions=[
                (0, 0, 0),
                (0.5, 0.5, 0.5),
                (0, 2 / 3, 0.5),
                (0.5, 1 / 6, 0.5),
            ],
            cell=[(1.0, 0.0, 0.0), (0.0, np.sqrt(3), 0.0), (0.0, 0.0, np.sqrt(8 / 3))],
            pbc=(1, 1, 1),
        ),
    },
    # diamond primitive
    "dia": {
        "colour": "orange",
        "nntolat": 4.0 / np.sqrt(3.0),
        "isbulk": True,
        "STRUKTURBERICHT": "A4",
        "atoms": Atoms(
            EleA + "2",
            scaled_positions=[(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)],
            cell=[(0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
            pbc=(1, 1, 1),
        ),
    },
    # diamond cubic
    "dia_cubic": {
        "colour": "orange",
        "nntolat": 4.0 / np.sqrt(3.0),
        "isbulk": True,
        "STRUKTURBERICHT": "A4",
        "atoms": Atoms(
            EleA + "8",
            scaled_positions=[
                (0.0, 0.0, 0.0),
                (0.5, 0.5, 0.0),
                (0.5, 0.0, 0.5),
                (0.0, 0.5, 0.5),
                (0.25, 0.25, 0.25),
                (0.75, 0.75, 0.25),
                (0.75, 0.25, 0.75),
                (0.25, 0.75, 0.75),
            ],
            cell=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            pbc=(1, 1, 1),
        ),
    },
    # sc
    "sc": dict(
        colour="green",
        nntolat=1.0,
        isbulk=True,
        STRUKTURBERICHT="A_h",
        atoms=Atoms(
            EleA + "1",
            scaled_positions=[
                (0.0, 0.0, 0.0),
            ],
            cell=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            pbc=(1, 1, 1),
        ),
    ),
    # sh
    "sh": dict(
        colour="yellow",
        nntolat=1.0,
        isbulk=True,
        STRUKTURBERICHT="A_f",
        atoms=Atoms(
            EleA + "1",
            scaled_positions=[
                (0.0, 0.0, 0.0),
            ],
            cell=[(1 / 2, -np.sqrt(3) / 2, 0), (1 / 2, np.sqrt(3) / 2, 0.0), (0, 0, 1)],
            pbc=(1, 1, 1),
        ),
    ),
    # dhcp
    "dhcp": dict(
        colour="pink",
        nntolat=1.0,
        isbulk=True,
        STRUKTURBERICHT="A3p",
        atoms=Atoms(
            EleA + "4",
            scaled_positions=[
                (0, 0, 0),
                (1 / 3, 2 / 3, 1 / 4),
                (0, 0, 1 / 2),
                (2 / 3, 1 / 3, 3 / 4),
            ],
            cell=[
                (1 / 2, -np.sqrt(3) / 2, 0),
                (1 / 2, np.sqrt(3) / 2, 0),
                (0.0, 0.0, 2 * np.sqrt(8 / 3)),
            ],
            pbc=(1, 1, 1),
        ),
    ),
    # omega
    "omega": dict(
        colour="lime",
        nntolat=3.0 / np.sqrt(3.0),
        isbulk=True,
        STRUKTURBERICHT="C32",
        atoms=Atoms(
            EleA + "6",
            scaled_positions=[
                (0.0, 0.0, 0.0),
                (0.5, 0.5, 0.0),
                (0.0, 1.0 / 3.0, 0.5),
                (0.0, 2.0 / 3.0, 0.5),
                (0.5, 1.0 / 6.0, 0.5),
                (0.5, 5.0 / 6.0, 0.5),
            ],
            cell=[(1.0, 0.0, 0.0), (0.0, np.sqrt(3.0), 0.0), (0.0, 0.0, 0.62)],
            pbc=(1, 1, 1),
        ),
    ),
    # A15
    "A15": dict(
        colour="cyan",
        nntolat=2.0,
        isbulk=True,
        STRUKTURBERICHT="A15",
        atoms=Atoms(
            EleA + "8",
            scaled_positions=[
                (0.0, 0.0, 0.0),
                (0.5, 0.5, 0.5),
                (0.5, 0.25, 0.00),
                (0.50, 0.75, 0.00),
                (0.00, 0.50, 0.25),
                (0.00, 0.50, 0.75),
                (0.25, 0.00, 0.50),
                (0.75, 0.00, 0.50),
            ],
            cell=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            pbc=(1, 1, 1),
        ),
    ),
    # C19
    "C19": dict(
        colour="green",
        nntolat=2.0,
        isbulk=True,
        STRUKTURBERICHT="C19",
        atoms=Atoms(
            EleA + "3",
            scaled_positions=[
                (0.0, 0.0, 0.0),
                (2.22222220e-01, 2.22222220e-01, 2.22222220e-01),
                (7.77777780e-01, 7.77777780e-01, 7.77777780e-01),
            ],
            cell=[
                [0.391431, 0.22599301, 1.88408005],
                [-0.391431, 0.22599301, 1.88408005],
                [0.0, -0.45198601, 1.88408005],
            ],
            pbc=(1, 1, 1),
        ),
    ),
    #######################################
    ### BINARY PERIODIC BULK STRUCTURES ###
    #######################################
    # B1
    "B1": {
        "colour": "black",
        "nntolat": 2.0 / np.sqrt(2.0),
        "isbulk": True,
        "STRUKTURBERICHT": "B1",
        "atoms": Atoms(
            EleA + "1" + EleB + "1",
            scaled_positions=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
            cell=[(0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
            pbc=(1, 1, 1),
        ),
    },
    # B2
    "B2": {
        "colour": "blue",
        "nntolat": 2.0 / np.sqrt(3.0),
        "isbulk": True,
        "STRUKTURBERICHT": "B2",
        "atoms": Atoms(
            EleA + "1" + EleB + "1",
            scaled_positions=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
            cell=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            pbc=(1, 1, 1),
        ),
    },
    # B3
    "B3": {
        "colour": "cyan",
        "nntolat": 2.0 / np.sqrt(3.0),
        "isbulk": True,
        "STRUKTURBERICHT": "B3",
        "atoms": Atoms(
            EleA + "1" + EleB + "1",
            scaled_positions=[(0.0, 0.0, 0.0), (0.25, 0.25, 0.25)],
            cell=[(0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
            pbc=(1, 1, 1),
        ),
    },
    # L1_0
    "L1_0": {
        "colour": "green",
        "nntolat": 1.0,
        "isbulk": True,
        "STRUKTURBERICHT": "L1_0",
        "atoms": Atoms(
            EleA + "1" + EleB + "1",
            scaled_positions=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
            cell=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.41421356237)],
            pbc=(1, 1, 1),
        ),
    },
    # L1_2
    "L1_2": {
        "colour": "yellow",
        "nntolat": np.sqrt(2.0),
        "isbulk": True,
        "STRUKTURBERICHT": "L1_2",
        "atoms": Atoms(
            EleA + "1" + EleB + "3",
            scaled_positions=[
                (0.0, 0.0, 0.0),
                (0.0, 0.5, 0.5),
                (0.5, 0.0, 0.5),
                (0.5, 0.5, 0.0),
            ],
            cell=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
            pbc=(1, 1, 1),
        ),
    },
    #############################
    ### NON-PERIODIC CLUSTERS ###
    #############################
    # dimer
    "dimer": dict(
        colour="brown",
        nntolat=1.0,
        isbulk=False,
        atoms=Atoms(
            EleA + "2",
            positions=[(x0, y0, z0), (x0 + 1.0, y0, z0)],
            cell=(40, 40, 40),
            pbc=(1, 1, 1),
        ),
    ),
    # trimer
    "trimer": dict(
        colour="blue",
        nntolat=1.0,
        isbulk=False,
        atoms=Atoms(
            EleA + "3",
            positions=[(x0 - 1.0, y0, z0), (x0, y0, z0), (x0 + 1.0, y0, z0)],
            cell=(40, 40, 40),
            pbc=(1, 1, 1),
        ),
    ),
    # triangle
    "triangle": dict(
        colour="cyan",
        nntolat=1.0,
        isbulk=False,
        atoms=Atoms(
            EleA + "3",
            positions=[
                (x0 - 0.5, y0 + np.sqrt(3.0) / 2.0, z0),
                (x0, y0, z0),
                (x0 + 0.5, y0 + np.sqrt(3.0) / 2.0, z0),
            ],
            cell=(40, 40, 40),
            pbc=(1, 1, 1),
        ),
    ),
    # tetramer
    "tetramer": dict(
        colour="red",
        nntolat=1.0,
        isbulk=False,
        atoms=Atoms(
            EleA + "4",
            positions=[
                (x0 - 1.5, y0, z0),
                (x0 - 0.5, y0, z0),
                (x0 + 0.5, y0, z0),
                (x0 + 1.5, y0, z0),
            ],
            cell=(40, 40, 40),
            pbc=(1, 1, 1),
        ),
    ),
    # tetrahedron
    "tetrahedron": dict(
        colour="orange",
        nntolat=1.0,
        isbulk=False,
        atoms=Atoms(
            EleA + "4",
            positions=[
                (x0, y0, z0),
                (x0 + 1.0, y0 + 1.0, z0),
                (x0 + 1.0, y0, z0 + 1.0),
                (x0, y0 + 1.0, z0 + 1.0),
            ],
            cell=(40, 40, 40),
            pbc=(1, 1, 1),
        ),
    ),
    # square
    "square": dict(
        colour="yellow",
        nntolat=1.0,
        isbulk=False,
        atoms=Atoms(
            EleA + "4",
            positions=[
                (x0, y0 + 1.0, z0),
                (x0, y0, z0),
                (x0 + 1.0, y0, z0),
                (x0 + 1.0, y0 + 1.0, z0),
            ],
            cell=(40, 40, 40),
            pbc=(1, 1, 1),
        ),
    ),
    # rhombus
    "rhombus": dict(
        colour="magenta",
        nntolat=1.0,
        isbulk=False,
        atoms=Atoms(
            EleA + "4",
            positions=[
                (x0 - 0.5, y0 + np.sqrt(3.0) / 2.0, z0),
                (x0, y0, z0),
                (x0 + 0.5, y0 + np.sqrt(3.0) / 2.0, z0),
                (x0, y0 + np.sqrt(3.0), z0),
            ],
            cell=(40, 40, 40),
            pbc=(1, 1, 1),
        ),
    ),
    # star
    "star": dict(
        colour="olive",
        nntolat=1.0,
        isbulk=False,
        atoms=Atoms(
            EleA + "4",
            positions=[
                (x0 - 2 * np.sqrt(2.0) / 3, y0 - 1 / 3, z0),
                (x0, y0, z0),
                (x0 + 2 * np.sqrt(2.0) * 1 / 3, y0 - 1 / 3, z0),
                (x0, y0 + 1.0, z0),
            ],
            cell=(40, 40, 40),
            pbc=(1, 1, 1),
        ),
    ),
}  # end of PROTOTYPES_DICT dictionary

SB_TO_COLOR_DICT = {
    v["STRUKTURBERICHT"]: v["colour"]
    for k, v in PROTOTYPES_DICT.items()
    if "STRUKTURBERICHT" in v
}
PROTOTYPE_TO_COLOR_DICT = {
    k: v["colour"] for k, v in PROTOTYPES_DICT.items() if "STRUKTURBERICHT" in v
}


def get_color_for_strukturbericht(sb: str) -> str:
    return SB_TO_COLOR_DICT.get(sb, "gray")


def get_color_for_prototype(prototype_name: str) -> str:
    return PROTOTYPE_TO_COLOR_DICT.get(prototype_name, "gray")


class MockGenericEntry:
    pass


def get_structures_dictionary(elements, include=None, exclude=None, scale_volume=True):
    """
    Return set of resources for given element
    :param elements: list of element names
    :param include: list of prototypes (str) to include
    :param exclude: list of prototypes (str) to exclude
    :param scale_volume: bool, scale volume to the atomic volume for given element
    :return: list of Atoms
    """

    # for backward compatibility check if 'elements' is string(old) or list(new)
    if isinstance(elements, str):
        elements = [elements]

    # convert both include and exclude into lists with at least one element if any
    if isinstance(exclude, str):
        exclude = [exclude]
    if isinstance(include, str):
        include = [include]

    result_dict = {}
    for prototype_name, struct_dict in PROTOTYPES_DICT.items():
        # check for exclusion
        if exclude is not None:
            if (
                prototype_name in exclude
                or struct_dict.get("STRUKTURBERICHT") in exclude
            ):
                continue
        # check for inclusion
        if include is not None:
            if (
                prototype_name not in include
                and struct_dict.get("STRUKTURBERICHT") not in include
            ):
                continue
        struct_dict = struct_dict.copy()
        atoms = struct_dict["atoms"].copy()
        symbols = atoms.get_chemical_symbols()
        i = 0
        celm = {}
        for elm in elements:
            srep = sym_pro[i]
            symbols = [s.replace(srep, elm) for s in symbols]
            celm[elm] = symbols.count(elm) / len(atoms)
            i += 1
        atoms.set_chemical_symbols(symbols)
        atoms.GENERICPARENT = MockGenericEntry()
        atoms.STRUKTURBERICHT = struct_dict.get("STRUKTURBERICHT")
        atoms.GENERICPARENT.STRUKTURBERICHT = struct_dict.get("STRUKTURBERICHT")
        if struct_dict["isbulk"] and scale_volume:
            current_volume = atoms.get_volume()
            #            guessed_volumed = atomic_volumes[element] * len(atoms)
            guessed_volumed = sum(
                atomic_volumes[elm] * celm[elm] for elm in elements
            ) * len(atoms)
            cell = atoms.get_cell()
            new_cell = (guessed_volumed / current_volume) ** (1 / 3.0) * cell
            atoms.set_cell(new_cell, scale_atoms=True)

        if "STRUKTURBERICHT" in struct_dict:
            atoms.STRUKTURBERICHT = struct_dict["STRUKTURBERICHT"]
            atoms.GENERICPARENT = MockGenericEntry()
            atoms.GENERICPARENT.STRUKTURBERICHT = struct_dict["STRUKTURBERICHT"]

        struct_dict["atoms"] = atoms
        result_dict[prototype_name] = struct_dict
    return result_dict
