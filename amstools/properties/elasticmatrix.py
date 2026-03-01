# Based on the "R. Golesorkhtabar, P. Pavone, J. Spitaler, P. Puschnig, and C. Draxl,
# ElaStic: A tool for calculating second-order elastic constants from first principles,
# Comp. Phys. Commun. 184, 1861 (2013)"
# A to C was solved using sympy

import numpy as np
from ase.optimize import BFGS

from amstools.calculators.dft.base import AMSDFTBaseCalculator, get_k_mesh_by_kspacing
from amstools.properties.generalcalculator import GeneralCalculator
from amstools.utils import get_spacegroup

eV_Ang3_to_GPA = 160.21766208  # From eV/Ang^3 to GPa

# Table 2
standard_deformations_types_dict = {
    1: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    2: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    3: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    4: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    5: [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
    6: [0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
    7: [0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
    8: [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    9: [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    10: [1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
    11: [1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
    12: [1.0, 0.0, 0.0, 0.0, 0.0, 2.0],
    13: [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    14: [0.0, 1.0, 0.0, 2.0, 0.0, 0.0],
    15: [0.0, 1.0, 0.0, 0.0, 2.0, 0.0],
    16: [0.0, 1.0, 0.0, 0.0, 0.0, 2.0],
    17: [0.0, 0.0, 1.0, 2.0, 0.0, 0.0],
    18: [0.0, 0.0, 1.0, 0.0, 2.0, 0.0],
    19: [0.0, 0.0, 1.0, 0.0, 0.0, 2.0],
    20: [0.0, 0.0, 0.0, 2.0, 2.0, 0.0],
    21: [0.0, 0.0, 0.0, 2.0, 0.0, 2.0],
    22: [0.0, 0.0, 0.0, 0.0, 2.0, 2.0],
    23: [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
    24: [-1.0, 0.5, 0.5, 0.0, 0.0, 0.0],
    25: [0.5, -1.0, 0.5, 0.0, 0.0, 0.0],
    26: [0.5, 0.5, -1.0, 0.0, 0.0, 0.0],
    27: [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    28: [1.0, -1.0, 0.0, 0.0, 0.0, 2.0],
    29: [0.0, 1.0, -1.0, 0.0, 0.0, 2.0],
}


# https://en.wikipedia.org/wiki/List_of_space_groups
# https://de.wikipedia.org/wiki/Lauegruppe
# Table 1
def space_group_to_laue_group(space_group_number):
    if 1 <= space_group_number <= 2:  # Triclinic
        laue_group = "N"
    elif 3 <= space_group_number <= 15:  # Monoclinic
        laue_group = "M"
    elif 16 <= space_group_number <= 74:  # Orthorhombic
        laue_group = "O"
    elif 75 <= space_group_number <= 88:  # Tetragonal II
        laue_group = "TII"
    elif 89 <= space_group_number <= 142:  # Tetragonal I
        laue_group = "TI"
    elif 143 <= space_group_number <= 148:  # Rhombohedral II
        laue_group = "RII"
    elif 149 <= space_group_number <= 167:  # Rhombohedral I
        laue_group = "RI"
    elif 168 <= space_group_number <= 176:  # Hexagonal II
        laue_group = "HII"
    elif 177 <= space_group_number <= 194:  # Hexagonal I
        laue_group = "HI"
    elif 195 <= space_group_number <= 206:  # Cubic II
        laue_group = "CII"
    elif 207 <= space_group_number <= 230:  # Cubic I
        laue_group = "CI"
    else:
        raise ValueError("space_group_number should be 1 <= space_group_number <= 230")
    return laue_group


# Table 4
def get_strain_list(laue_group):
    if laue_group in ["CI", "CII"]:
        strain_list = [1, 8, 23]
    elif laue_group in ["HI", "HII"]:
        strain_list = [1, 3, 4, 17, 26]
    elif laue_group in ["RI"]:
        strain_list = [1, 2, 4, 5, 8, 10]
    elif laue_group in ["RII"]:
        strain_list = [1, 2, 4, 5, 8, 10, 11]
    elif laue_group in ["TI"]:
        strain_list = [1, 4, 5, 7, 26, 27]
    elif laue_group in ["TII"]:
        strain_list = [1, 4, 5, 7, 26, 27, 28]
    elif laue_group in ["O"]:
        strain_list = [1, 3, 4, 5, 6, 7, 25, 26, 27]
    elif laue_group in ["M"]:
        strain_list = [1, 3, 4, 5, 6, 7, 12, 20, 24, 25, 27, 28, 29]
    elif laue_group in ["N"]:
        strain_list = [
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
        ]
    else:
        raise ValueError(
            "Unknown Laue group {laue_group}".format(laue_group=laue_group)
        )
    return strain_list


# List of independent C_ij: Table A.21
# A2 -> C solved with sympy
def get_C_from_A2(A, laue_group, A_std=None):
    C = np.zeros((6, 6))
    C_std = np.zeros((6, 6))

    if laue_group in ["CI", "CII"]:  # Cubic
        C[0, 0] = -2 * A[0] / 3 + 2 * A[1]
        C[0, 1] = 2 * A[0] / 3 - A[1]
        C[3, 3] = A[2] / 6.0

        C[1, 1] = C[0, 0]
        C[2, 2] = C[0, 0]
        C[4, 4] = C[3, 3]
        C[5, 5] = C[3, 3]
        C[0, 2] = C[0, 1]
        C[1, 2] = C[0, 1]

        if A_std is not None:
            C_std[0, 0] = np.linalg.norm([-2 * A_std[0] / 3, 2 * A_std[1]])
            C_std[0, 1] = np.linalg.norm([2.0 * A_std[0] / 3, -A_std[1]])
            C_std[1, 1] = C_std[0, 0]
            C_std[2, 2] = C_std[0, 0]
            C_std[3, 3] = A_std[2] / 6.0
            C_std[4, 4] = C_std[3, 3]
            C_std[5, 5] = C_std[3, 3]
            C_std[0, 2] = C_std[0, 1]
            C_std[1, 2] = C_std[0, 1]
    elif laue_group in ["HI", "HII"]:  # Hexagonal
        C[0, 0] = 2 * A[1]
        C[0, 1] = 2 * A[0] / 3 - 2 * A[1] - 2 * A[2] + 4 * A[4] / 3
        C[0, 2] = A[0] / 6 + A[2] / 2 - 2 * A[4] / 3
        C[2, 2] = 2 * A[2]
        C[3, 3] = -A[2] / 2 + A[3] / 2

        C[1, 1] = C[0, 0]
        C[1, 2] = C[0, 2]
        C[4, 4] = C[3, 3]
        C[5, 5] = (C[0, 0] - C[0, 1]) / 2

        if A_std is not None:
            C_std[0, 0] = 2 * A_std[1]
            C_std[0, 1] = np.linalg.norm(
                [2 * A_std[0] / 3, -2 * A_std[1], -2 * A_std[2], 4 * A_std[4] / 3]
            )
            C_std[0, 2] = np.linalg.norm(
                [A_std[0] / 6, A_std[2] / 2, -2 * A_std[4] / 3]
            )
            C_std[2, 2] = 2 * A_std[2]
            C_std[3, 3] = np.linalg.norm([-A_std[2] / 2, A_std[3] / 2])

            C_std[1, 1] = C_std[0, 0]
            C_std[1, 2] = C_std[0, 2]
            C_std[4, 4] = C_std[3, 3]
            C_std[5, 5] = np.linalg.norm([C_std[0, 0], C_std[0, 1]]) / 2
    elif laue_group in ["RI"]:  # Rhombohedral I
        C[0, 0] = 2 * A[1]
        C[0, 1] = -2 * A[1] + A[4]
        C[0, 2] = A[0] / 2 - A[2] / 2 - A[4] / 2
        C[0, 3] = -A[1] / 2 - A[3] / 2 + A[5] / 2
        C[2, 2] = 2 * A[2]
        C[3, 3] = A[3] / 2

        C[1, 1] = C[0, 0]
        C[1, 2] = C[0, 2]
        C[1, 3] = -C[0, 3]
        C[4, 4] = C[3, 3]
        C[4, 5] = C[0, 3]
        C[5, 5] = (C[0, 0] - C[0, 1]) / 2

        if A_std is not None:
            C_std[0, 0] = 2 * A_std[1]
            C_std[0, 1] = np.linalg.norm([-2 * A_std[1], A_std[4]])
            C_std[0, 2] = np.linalg.norm([A_std[0] / 2, -A_std[2] / 2, -A_std[4] / 2])
            C_std[0, 3] = np.linalg.norm([-A_std[1] / 2, -A_std[3] / 2, A_std[5] / 2])
            C_std[2, 2] = 2 * A_std[2]
            C_std[3, 3] = A_std[3] / 2

            C_std[1, 1] = C_std[0, 0]
            C_std[1, 2] = C_std[0, 2]
            C_std[1, 3] = C_std[0, 3]
            C_std[4, 4] = C_std[3, 3]
            C_std[4, 5] = C_std[0, 3]
            C_std[5, 5] = np.linalg.norm([C_std[0, 0], C_std[0, 1]]) / 2
    elif laue_group in ["RII"]:  # Rhombohedral II
        C[0, 0] = 2 * A[1]
        C[0, 1] = -2 * A[1] + A[4]
        C[0, 2] = A[0] / 2 - A[2] / 2 - A[4] / 2
        C[0, 3] = -A[1] / 2 - A[3] / 2 + A[5] / 2
        C[0, 4] = -A[1] / 2 - A[3] / 2 + A[6] / 2
        C[2, 2] = 2 * A[2]
        C[3, 3] = A[3] / 2

        C[1, 1] = C[0, 0]
        C[1, 2] = C[0, 2]
        C[1, 3] = -C[0, 3]
        C[1, 4] = -C[0, 4]
        C[3, 5] = -C[0, 4]
        C[4, 4] = C[3, 3]
        C[4, 5] = C[0, 3]
        C[5, 5] = (C[0, 0] - C[0, 1]) / 2

        if A_std is not None:
            C_std[0, 0] = 2 * A_std[1]
            C_std[0, 1] = np.linalg.norm([-2 * A_std[1], +A_std[4]])
            C_std[0, 2] = np.linalg.norm([A_std[0] / 2, -A_std[2] / 2, -A_std[4] / 2])
            C_std[0, 3] = np.linalg.norm([-A_std[1] / 2, -A_std[3] / 2, A_std[5] / 2])
            C_std[0, 4] = np.linalg.norm([-A_std[1] / 2, -A_std[3] / 2, A_std[6] / 2])
            C_std[2, 2] = 2 * A_std[2]
            C_std[3, 3] = A_std[3] / 2

            C_std[1, 1] = C_std[0, 0]
            C_std[1, 2] = C_std[0, 2]
            C_std[1, 3] = C_std[0, 3]
            C_std[1, 4] = C_std[0, 4]
            C_std[3, 5] = C_std[0, 4]
            C_std[4, 4] = C_std[3, 3]
            C_std[4, 5] = C_std[0, 3]
            C_std[5, 5] = 0.5 * np.linalg.norm([C_std[0, 0], C_std[0, 1]])
    elif laue_group == "TI":  # Tetragonal I
        C[0, 0] = A[0] / 3 - A[1] + 2 * A[4] / 3 + A[5] / 2
        C[0, 1] = A[0] / 3 - A[1] + 2 * A[4] / 3 - A[5] / 2
        C[0, 2] = A[0] / 6 + A[1] / 2 - 2 * A[4] / 3
        C[2, 2] = 2 * A[1]
        C[3, 3] = A[2] / 2
        C[5, 5] = A[3] / 2

        C[1, 1] = C[0, 0]
        C[1, 2] = C[0, 2]
        C[4, 4] = C[3, 3]

        if A_std is not None:
            C_std[0, 0] = np.linalg.norm(
                [A_std[0] / 3, -A_std[1], +2 * A_std[4] / 3, +A_std[5] / 2]
            )
            C_std[0, 1] = np.linalg.norm(
                [A_std[0] / 3, -A_std[1], +2 * A_std[4] / 3, -A_std[5] / 2]
            )
            C_std[0, 2] = np.linalg.norm(
                [A_std[0] / 6, +A_std[1] / 2, -2 * A_std[4] / 3]
            )
            C_std[2, 2] = 2 * A_std[1]
            C_std[3, 3] = A_std[2] / 2
            C_std[5, 5] = A_std[3] / 2

            C_std[1, 1] = C_std[0, 0]
            C_std[1, 2] = C_std[0, 2]
            C_std[4, 4] = C_std[3, 3]
    elif laue_group == "TII":  # Tetragonal II
        C[0, 0] = A[0] / 3 - A[1] + 2 * A[4] / 3 + A[5] / 2
        C[0, 1] = A[0] / 3 - A[1] + 2 * A[4] / 3 - A[5] / 2
        C[0, 2] = A[0] / 6 + A[1] / 2 - 2 * A[4] / 3
        C[0, 5] = -A[3] / 4 - A[5] / 4 + A[6] / 4
        C[2, 2] = 2 * A[1]
        C[3, 3] = A[2] / 2
        C[5, 5] = A[3] / 2

        C[1, 1] = C[0, 0]
        C[1, 2] = C[0, 2]
        C[1, 5] = -C[0, 5]
        C[4, 4] = C[3, 3]

        if A_std is not None:
            C_std[0, 0] = np.linalg.norm(
                [A_std[0] / 3, -A_std[1], +2 * A_std[4] / 3, +A_std[5] / 2]
            )
            C_std[0, 1] = np.linalg.norm(
                [A_std[0] / 3, -A_std[1], +2 * A_std[4] / 3, -A_std[5] / 2]
            )
            C_std[0, 2] = np.linalg.norm(
                [A_std[0] / 6, +A_std[1] / 2, -2 * A_std[4] / 3]
            )
            C_std[0, 5] = np.linalg.norm([-A_std[3] / 4, -A_std[5] / 4, +A_std[6] / 4])

            C_std[2, 2] = 2 * A_std[1]
            C_std[3, 3] = A_std[2] / 2
            C_std[5, 5] = A_std[3] / 2

            C_std[1, 1] = C_std[0, 0]
            C_std[1, 2] = C_std[0, 2]
            C_std[1, 5] = C_std[0, 5]
            C_std[4, 4] = C_std[3, 3]
    elif laue_group == "O":  # Orthorhombic
        C[0, 0] = 2 * A[0] / 3 - 2 * A[1] - 2 * A[2] + 4 * A[7] / 3 + A[8]
        C[0, 1] = A[0] / 3 - A[2] + 2 * A[7] / 3 - A[8] / 2
        C[0, 2] = A[0] / 3 - A[1] + 4 * A[6] / 3 - 2 * A[7] / 3 - A[8] / 2
        C[1, 1] = 2 * A[1]
        C[1, 2] = A[1] + A[2] - 4 * A[6] / 3 - 2 * A[7] / 3 + A[8] / 2
        C[2, 2] = 2 * A[2]
        C[3, 3] = A[3] / 2
        C[4, 4] = A[4] / 2
        C[5, 5] = A[5] / 2

        if A_std is not None:
            C_std[0, 0] = np.linalg.norm(
                [
                    2 * A_std[0] / 3,
                    -2 * A_std[1],
                    -2 * A_std[2],
                    +4 * A_std[7] / 3,
                    +A_std[8],
                ]
            )
            C_std[0, 1] = np.linalg.norm(
                [A_std[0] / 3, -A_std[2], +2 * A_std[7] / 3, -A_std[8] / 2]
            )
            C_std[0, 2] = np.linalg.norm(
                [
                    A_std[0] / 3,
                    -A_std[1],
                    +4 * A_std[6] / 3,
                    -2 * A_std[7] / 3,
                    -A_std[8] / 2,
                ]
            )
            C_std[1, 1] = 2 * A_std[1]

            C_std[1, 2] = np.linalg.norm(
                [
                    A_std[1],
                    +A_std[2],
                    -4 * A_std[6] / 3,
                    -2 * A_std[7] / 3,
                    +A_std[8] / 2,
                ]
            )

            C_std[2, 2] = 2 * A_std[2]
            C_std[3, 3] = A_std[3] / 2
            C_std[4, 4] = A_std[4] / 2
            C_std[5, 5] = A_std[5] / 2
    elif laue_group == "M":  # Monoclinic
        C[0, 0] = (
            2 * A[0] / 3 - 2 * A[1] - 2 * A[10] - 2 * A[2] + 8 * A[8] / 3 + 8 * A[9] / 3
        )
        C[0, 1] = A[0] / 3 - 2 * A[10] - A[2] + 4 * A[8] / 3 + 4 * A[9] / 3
        C[0, 2] = A[0] / 3 - A[1] + A[10] - 4 * A[8] / 3
        C[0, 5] = (
            -A[0] / 6
            + A[1] / 2
            + A[10] / 2
            + A[2] / 2
            - A[5] / 2
            + A[6] / 2
            - 2 * A[8] / 3
            - 2 * A[9] / 3
        )
        C[1, 1] = 2 * A[1]
        C[1, 2] = A[1] + 2 * A[10] + A[2] - 4 * A[8] / 3 - 8 * A[9] / 3
        C[1, 5] = (
            -A[0] / 6
            + A[1] / 2
            + A[10]
            - A[11] / 2
            + A[2] / 2
            + A[6] / 2
            - 2 * A[8] / 3
            - 2 * A[9] / 3
        )
        C[2, 2] = 2 * A[2]
        C[2, 5] = (
            -A[0] / 6
            + A[1] / 2
            - A[11] / 2
            - A[12] / 2
            + A[2] / 2
            + A[5] / 2
            + A[6] / 2
            + 2 * A[9] / 3
        )
        C[3, 3] = A[3] / 2
        C[3, 4] = -A[3] / 4 - A[4] / 4 + A[7] / 4
        C[4, 4] = A[4] / 2
        C[5, 5] = A[5] / 2

        if A_std is not None:
            C_std[0, 0] = np.linalg.norm(
                [
                    2 * A_std[0] / 3,
                    -2 * A_std[1],
                    -2 * A_std[10],
                    -2 * A_std[2],
                    +8 * A_std[8] / 3,
                    +8 * A_std[9] / 3,
                ]
            )
            C_std[0, 1] = np.linalg.norm(
                [
                    A_std[0] / 3,
                    -2 * A_std[10],
                    -A_std[2],
                    +4 * A_std[8] / 3,
                    +4 * A_std[9] / 3,
                ]
            )
            C_std[0, 2] = np.linalg.norm(
                [A_std[0] / 3, -A_std[1], +A_std[10], -4 * A_std[8] / 3]
            )
            C_std[0, 5] = np.linalg.norm(
                [
                    -A_std[0] / 6,
                    +A_std[1] / 2,
                    +A_std[10] / 2,
                    +A_std[2] / 2,
                    -A_std[5] / 2,
                    +A_std[6] / 2,
                    -2 * A_std[8] / 3,
                    -2 * A_std[9] / 3,
                ]
            )
            C_std[1, 1] = 2 * A_std[1]
            C_std[1, 2] = np.linalg.norm(
                [
                    A_std[1],
                    +2 * A_std[10],
                    +A_std[2],
                    -4 * A_std[8] / 3,
                    -8 * A_std[9] / 3,
                ]
            )
            C_std[1, 5] = np.linalg.norm(
                [
                    -A_std[0] / 6,
                    +A_std[1] / 2,
                    +A_std[10],
                    -A_std[11] / 2,
                    +A_std[2] / 2,
                    +A_std[6] / 2,
                    -2 * A_std[8] / 3,
                    -2 * A_std[9] / 3,
                ]
            )
            C_std[2, 2] = 2 * A_std[2]
            C_std[2, 5] = np.linalg.norm(
                [
                    -A_std[0] / 6,
                    +A_std[1] / 2,
                    -A_std[11] / 2,
                    -A_std[12] / 2,
                    +A_std[2] / 2,
                    +A_std[5] / 2,
                    +A_std[6] / 2,
                    +2 * A_std[9] / 3,
                ]
            )
            C_std[3, 3] = A_std[3] / 2
            C_std[3, 4] = np.linalg.norm([-A_std[3] / 4, -A_std[4] / 4, +A_std[7] / 4])
            C_std[4, 4] = A_std[4] / 2
            C_std[5, 5] = A_std[5] / 2
    elif laue_group == "N":  # Triclinic
        C[0, 0] = 2 * A[0]
        C[0, 1] = -A[0] - A[1] + A[6]
        C[0, 2] = -A[0] - A[2] + A[7]
        C[0, 3] = -A[0] / 2 - A[3] / 2 + A[8] / 2
        C[0, 4] = -A[0] / 2 - A[4] / 2 + A[9] / 2
        C[0, 5] = -A[0] / 2 + A[10] / 2 - A[5] / 2

        C[1, 1] = 2 * A[1]
        C[1, 2] = -A[1] + A[11] - A[2]
        C[1, 3] = -A[1] / 2 + A[12] / 2 - A[3] / 2
        C[1, 4] = -A[1] / 2 + A[13] / 2 - A[4] / 2
        C[1, 5] = -A[1] / 2 + A[14] / 2 - A[5] / 2

        C[2, 2] = 2 * A[2]
        C[2, 3] = A[15] / 2 - A[2] / 2 - A[3] / 2
        C[2, 4] = A[16] / 2 - A[2] / 2 - A[4] / 2
        C[2, 5] = A[17] / 2 - A[2] / 2 - A[5] / 2

        C[3, 3] = A[3] / 2
        C[3, 4] = A[18] / 4 - A[3] / 4 - A[4] / 4
        C[3, 5] = A[19] / 4 - A[3] / 4 - A[5] / 4

        C[4, 4] = A[4] / 2
        C[4, 5] = A[20] / 4 - A[4] / 4 - A[5] / 4

        C[5, 5] = A[5] / 2

        if A_std is not None:
            C_std[0, 0] = np.linalg.norm([2 * A_std[0]])
            C_std[0, 1] = np.linalg.norm([-A_std[0], -A_std[1], +A_std[6]])
            C_std[0, 2] = np.linalg.norm([-A_std[0], -A_std[2], +A_std[7]])
            C_std[0, 3] = np.linalg.norm([-A_std[0] / 2, -A_std[3] / 2, +A_std[8] / 2])
            C_std[0, 4] = np.linalg.norm([-A_std[0] / 2, -A_std[4] / 2, +A_std[9] / 2])
            C_std[0, 5] = np.linalg.norm([-A_std[0] / 2, +A_std[10] / 2, -A_std[5] / 2])

            C_std[1, 1] = np.linalg.norm([2 * A_std[1]])
            C_std[1, 2] = np.linalg.norm([-A_std[1], +A_std[11], -A_std[2]])
            C_std[1, 3] = np.linalg.norm([-A_std[1] / 2, +A_std[12] / 2, -A_std[3] / 2])
            C_std[1, 4] = np.linalg.norm([-A_std[1] / 2, +A_std[13] / 2, -A_std[4] / 2])
            C_std[1, 5] = np.linalg.norm([-A_std[1] / 2, +A_std[14] / 2, -A_std[5] / 2])

            C_std[2, 2] = np.linalg.norm([2 * A_std[2]])
            C_std[2, 3] = np.linalg.norm([A_std[15] / 2, -A_std[2] / 2, -A_std[3] / 2])
            C_std[2, 4] = np.linalg.norm([A_std[16] / 2, -A_std[2] / 2, -A_std[4] / 2])
            C_std[2, 5] = np.linalg.norm([A_std[17] / 2, -A_std[2] / 2, -A_std[5] / 2])

            C_std[3, 3] = np.linalg.norm([A_std[3] / 2])
            C_std[3, 4] = np.linalg.norm([A_std[18] / 4, -A_std[3] / 4, -A_std[4] / 4])
            C_std[3, 5] = np.linalg.norm([A_std[19] / 4, -A_std[3] / 4, -A_std[5] / 4])

            C_std[4, 4] = np.linalg.norm([A_std[4] / 2])
            C_std[4, 5] = np.linalg.norm([A_std[20] / 4, -A_std[4] / 4, -A_std[5] / 4])

            C_std[5, 5] = np.linalg.norm([A_std[5] / 2])
    else:
        raise ValueError(f"Unknown Laue group {laue_group}")

    if A_std is not None:
        for i in range(5):
            for j in range(i + 1, 6):
                C[j, i] = C[i, j]
                C_std[i, j] = abs(C_std[i, j])
                C_std[j, i] = C_std[i, j]
        return C, C_std
    else:
        for i in range(5):
            for j in range(i + 1, 6):
                C[j, i] = C[i, j]
    return C


class ElasticMatrixCalculator(GeneralCalculator):
    """Calculate second-order elastic constants (elastic stiffness matrix).

    Computes the elastic stiffness matrix (C_ij) by applying symmetry-adapted
    strain deformations and fitting the energy-strain relationship. Results
    include elastic constants, bulk modulus, shear modulus, Young's modulus,
    and Poisson's ratio.

    Attributes:
        num_of_point (int): Number of strain points per direction (default: 5)
        eps_range (float): Maximum strain amplitude (default: 0.005)
        sqrt_eta (bool): Use sqrt(η) for strain (default: True)
        fit_order (int): Polynomial order for fitting (default: 2)
        optimize_deformed_structure (bool): Relax atoms at each strain (default: True)
        optimizer (Type[Optimizer]): ASE optimizer (default: BFGS)
        fmax (float): Force convergence in eV/Å (default: 0.005)
        ignore_symmetry (bool): Skip symmetry detection (default: False)
        optimizer_kwargs (Dict[str, Any]): Additional optimizer arguments

    Example:
        >>> from ase.build import bulk
        >>> atoms = bulk('Al', 'fcc', a=4.05)
        >>> atoms.calc = EMT()
        >>> elastic = ElasticMatrixCalculator(atoms)
        >>> elastic.calculate()
        >>> print(f"C11 = {elastic.value['C'][0,0]:.2f} GPa")

    Reference:
        R. Golesorkhtabar et al., Comp. Phys. Commun. 184, 1861 (2013)
    """

    property_name = "elastic_matrix"

    zero_strain_job_name = "s_e_0"
    param_names = [
        "num_of_point",
        "eps_range",
        "sqrt_eta",
        "fit_order",
        "optimize_deformed_structure",
        "optimizer",
        "fmax",
        "optimizer_kwargs",
        "ignore_symmetry",
    ]

    def __init__(
        self,
        atoms=None,
        num_of_point=5,
        eps_range=0.005,
        sqrt_eta=True,
        fit_order=2,
        optimize_deformed_structure=True,
        optimizer=BFGS,
        fmax=0.005,
        ignore_symmetry=False,
        optimizer_kwargs=None,
        **kwargs,
    ):
        GeneralCalculator.__init__(self, atoms, **kwargs)

        self.num_of_point = num_of_point
        self.eps_range = eps_range
        self.sqrt_eta = sqrt_eta
        self.fit_order = fit_order
        self.optimize_deformed_structure = optimize_deformed_structure
        self.optimizer = optimizer
        self.space_group_number = 1
        self._init_kwargs(optimizer_kwargs=optimizer_kwargs)
        self.fmax = fmax
        self.ignore_symmetry = ignore_symmetry

    def symmetry_analysis(self):
        """

        Returns:

        """

        if not self.ignore_symmetry:
            self.space_group_number = get_spacegroup(self.basis_ref)
        else:
            self.space_group_number = 1
        self._value["ignore_symmetry"] = self.ignore_symmetry
        self._value["space_group_number"] = self.space_group_number
        self.v0 = self.basis_ref.get_volume()
        self._value["v0"] = self.v0
        self.laue_group = space_group_to_laue_group(self.space_group_number)
        self._value["laue_group"] = self.laue_group
        self.strain_list = get_strain_list(self.laue_group)
        self._value["strain_list"] = self.strain_list
        self.epss = np.linspace(-self.eps_range, self.eps_range, self.num_of_point)
        self._value["epss"] = self.epss

    def generate_structures(self, verbose=False):
        """

        Returns:

        """
        self.symmetry_analysis()
        basis_ref = self.basis_ref

        if 0.0 in self.epss:
            self._structure_dict[self.zero_strain_job_name] = basis_ref.copy()

        for strain_type in self.strain_list:
            deformations = np.array(standard_deformations_types_dict[strain_type])
            for eps in self.epss:
                if eps == 0.0:
                    continue
                deformation_value = eps * deformations

                eta_matrix = np.zeros((3, 3))  # Lagrangian strain tensor

                eta_matrix[0, 0] = deformation_value[0]
                eta_matrix[0, 1] = deformation_value[5] / 2.0
                eta_matrix[0, 2] = deformation_value[4] / 2.0

                eta_matrix[1, 0] = deformation_value[5] / 2.0
                eta_matrix[1, 1] = deformation_value[1]
                eta_matrix[1, 2] = deformation_value[3] / 2.0

                eta_matrix[2, 0] = deformation_value[4] / 2.0
                eta_matrix[2, 1] = deformation_value[3] / 2.0
                eta_matrix[2, 2] = deformation_value[2]

                if np.linalg.norm(eta_matrix) > 0.5:
                    raise Exception("Too large deformation {eps}".format(eps=eps))

                eps_matrix = eta_matrix.copy()  # physical strain tensor
                # Numerical solution of Eq.(1)  wrt. eps:
                # eta = eps + eps*eps / 2
                # =>
                # eps = eta - eps*eps/2

                delta = 1e9  # initial large number
                if self.sqrt_eta:
                    while delta > 1.0e-10:
                        rhs = eta_matrix - np.dot(eps_matrix, eps_matrix) / 2.0
                        delta = np.linalg.norm(rhs - eps_matrix)
                        eps_matrix = rhs

                new_cell = np.dot(basis_ref.get_cell(), np.eye(3) + eps_matrix)
                new_struct = basis_ref.copy()
                new_struct.set_cell(new_cell, scale_atoms=True)

                jobname = self.subjob_name(strain_type, eps)

                self._structure_dict[jobname] = new_struct

        return self._structure_dict

    def analyse_structures(self, output_dict):
        """

        Returns:

        """
        self.symmetry_analysis()

        ene0 = None
        if 0.0 in self.epss:
            ene0 = output_dict[self.zero_strain_job_name]
        self._value["e0"] = ene0
        strain_energy = []
        for lag_strain in self.strain_list:
            strain_energy.append([])
            for eps in self.epss:
                if not eps == 0.0:
                    jobname = self.subjob_name(lag_strain, eps)
                    ene = output_dict[jobname]
                else:
                    ene = ene0
                strain_energy[-1].append((eps, ene))
        self._value["strain_energy"] = strain_energy
        self.fit_elastic_matrix()

    def get_structure_value(self, structure, name=None):
        logfile = "-" if self.verbose else None
        if self.optimize_deformed_structure:
            if not isinstance(structure.calc, AMSDFTBaseCalculator):
                dyn = self.optimizer(
                    structure, logfile=logfile, **self.optimizer_kwargs
                )
                try:
                    dyn.set_force_consistent()
                except (TypeError, AttributeError):
                    dyn.force_consistent = False
                dyn.run(fmax=self.fmax)
                structure = dyn.atoms
            else:
                calc = structure.calc
                calc.optimize_atoms_only(ediffg=-self.fmax, max_steps=100)
                calc.auto_kmesh_spacing = False
                if calc.kmesh_spacing:
                    kmesh = get_k_mesh_by_kspacing(
                        self.basis_ref.cell, kmesh_spacing=calc.kmesh_spacing
                    )
                    calc.set_kmesh(kmesh)
                # do calculations
                structure.get_potential_energy(force_consistent=True)
                structure = structure.calc.atoms
                structure.calc = calc
        else:
            if isinstance(structure.calc, AMSDFTBaseCalculator):
                structure.calc.static_calc()
                structure.calc.auto_kmesh_spacing = False
                if structure.calc.kmesh_spacing:
                    kmesh = get_k_mesh_by_kspacing(
                        self.basis_ref.cell, kmesh_spacing=structure.calc.kmesh_spacing
                    )
                    structure.calc.set_kmesh(kmesh)

        en = structure.get_potential_energy(force_consistent=True)
        return en, structure

    @staticmethod
    def subjob_name(i, eps):
        """
        Generate subjob name
        """
        return ("s_{:02d}_e_{:.5f}".format(i, eps)).replace(".", "_").replace("-", "m")

    def calculate_modulus(self):
        """

        Returns:

        """
        C = self._value["C"]
        # Eq. (11), (12), (17), (18)
        BV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9
        GV = (
            (C[0, 0] + C[1, 1] + C[2, 2])
            - (C[0, 1] + C[0, 2] + C[1, 2])
            + 3 * (C[3, 3] + C[4, 4] + C[5, 5])
        ) / 15
        EV = (9 * BV * GV) / (3 * BV + GV)
        nuV = (1.5 * BV - GV) / (3 * BV + GV)
        self._value["BV"] = BV
        self._value["GV"] = GV
        self._value["EV"] = EV
        self._value["nuV"] = nuV

        try:
            S = np.linalg.inv(C)
            # Eq. (13), (14), (17), (18)
            BR = 1 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
            GR = 15 / (
                4 * (S[0, 0] + S[1, 1] + S[2, 2])
                - 4 * (S[0, 1] + S[0, 2] + S[1, 2])
                + 3 * (S[3, 3] + S[4, 4] + S[5, 5])
            )
            ER = (9 * BR * GR) / (3 * BR + GR)
            nuR = (1.5 * BR - GR) / (3 * BR + GR)

            # Eq. (15), (16), (17), (18)
            BH = 0.50 * (BV + BR)
            GH = 0.50 * (GV + GR)
            EH = (9.0 * BH * GH) / (3.0 * BH + GH)
            nuH = (1.5 * BH - GH) / (3.0 * BH + GH)

            # Elastic Anisotropy in polycrystalline
            AVR = 100.0 * (GV - GR) / (GV + GR)
            self._value["S"] = S

            self._value["BR"] = BR
            self._value["GR"] = GR
            self._value["ER"] = ER
            self._value["nuR"] = nuR

            self._value["BH"] = BH
            self._value["GH"] = GH
            self._value["EH"] = EH
            self._value["nuH"] = nuH

            self._value["AVR"] = AVR  # Elastic Anisotropy in polycrystalline
        except np.linalg.LinAlgError as e:
            print("LinAlgError:", e)

        eigval = np.linalg.eigh(C)
        self._value["C_eigval"] = eigval

    def fit_elastic_matrix(self):
        """

        Returns:

        """
        strain_energy = self._value["strain_energy"]

        v0 = self._value["v0"]
        laue_group = self._value["laue_group"]
        A2 = []
        A1 = []
        A2_std = []
        s_e_rms = []
        fit_order = int(self.fit_order)
        for s_e in strain_energy:
            ss = np.transpose(s_e)
            coefficients, cov = np.polyfit(ss[0], ss[1], fit_order, cov=True)
            cov = np.diag(cov)
            A2.append(coefficients[fit_order - 2])
            A1.append(coefficients[fit_order - 1])
            poly_z = np.poly1d(coefficients)
            s_e_rms.append(np.sqrt(np.mean((ss[1] - poly_z(ss[0])) ** 2)))
            A2_std.append(cov[fit_order - 2])

        A2 = np.array(A2) / v0
        A1 = np.array(A1) / v0
        A2_std = np.array(A2_std) / v0
        C, C_std = get_C_from_A2(A2, laue_group, A_std=A2_std)

        for i in range(5):
            for j in range(i + 1, 6):
                C[j, i] = C[i, j]
                C_std[j, i] = C_std[i, j]

        C *= eV_Ang3_to_GPA
        C_std *= eV_Ang3_to_GPA
        self._value["C"] = C
        self._value["A2"] = A2
        self._value["energy_rms"] = s_e_rms
        self._value["A1"] = A1
        self._value["C_std"] = C_std
        self.calculate_modulus()

    def plot(self, ax=None, **kwargs):
        from matplotlib import pylab as plt

        ax = ax or plt.gca()
        nat = len(self.basis_ref)
        ses = np.array(self.value["strain_energy"])

        strain_list = self.value.get("strain_list") or self.value.get("Lag_strain_list")
        if strain_list is None:
            raise ValueError("No strain list found")
        for strain_name, se in zip(strain_list, ses):
            ax.plot(se[:, 0], se[:, 1] / nat, label=strain_name)

        ax.set_xlabel("strain"),
        ax.set_ylabel("E, eV/at")
        ax.legend()
        return ax
