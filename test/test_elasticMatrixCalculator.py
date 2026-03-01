import json
import numpy as np
import pytest
from ase.spacegroup import crystal

from amstools import ElasticMatrixCalculator
from amstools.utils import JsonNumpyEncoder
from amstools.properties.elasticmatrix import space_group_to_laue_group
from test.utils import atoms, calculator


def create_atoms_by_spacegroup(spg, expected_vpa=20):
    a = 4.05
    if 143 <= spg < 195:
        atoms = crystal(
            "Al2",
            [(0, 0, 0), [1 / 2, 1 / 3, 1 / 4]],
            spacegroup=spg,
            cellpar=[a, a, a, 90, 90, 120],
        )
    else:
        atoms = crystal(
            "Al4",
            [
                (0, 0, 0),
                [1 / 2, 1 / 2, 1 / 2],
                [1 / 3, 1 / 4, 1 / 5],
                [1 / 4, 1 / 4, 1 / 4],
            ],
            spacegroup=spg,
            cellpar=[a, a, a, 90, 90, 90],
        )

    vpa = atoms.get_volume() / len(atoms)
    factor = expected_vpa / vpa
    new_cell = atoms.get_cell() * (factor) ** (1 / 3)
    atoms.set_cell(new_cell, scale_atoms=True)
    return atoms


@pytest.fixture
def elmat(atoms, calculator):
    atoms.calc = calculator
    return ElasticMatrixCalculator(atoms)


def test_symmetry_analysis(elmat):
    elmat.symmetry_analysis()
    assert "space_group_number" in elmat._value
    assert "v0" in elmat._value
    assert "laue_group" in elmat._value
    assert "strain_list" in elmat._value
    assert "epss" in elmat._value


def test_generate_structures(elmat):
    structures_dict = elmat.generate_structures()
    assert "s_e_0" in structures_dict
    assert "s_01_e_m0_00500" in structures_dict
    assert "s_01_e_m0_00250" in structures_dict
    assert "s_01_e_0_00250" in structures_dict
    assert "s_01_e_0_00500" in structures_dict
    assert "s_08_e_m0_00500" in structures_dict
    assert "s_08_e_m0_00250" in structures_dict
    assert "s_08_e_0_00250" in structures_dict
    assert "s_08_e_0_00500" in structures_dict
    assert "s_23_e_m0_00500" in structures_dict
    assert "s_23_e_m0_00250" in structures_dict
    assert "s_23_e_0_00250" in structures_dict
    assert "s_23_e_0_00500" in structures_dict


def test_get_structure_value(elmat, atoms):
    val, structure = elmat.get_structure_value(atoms)
    assert pytest.approx(val, abs=1e-6) == -0.001502047586230404


def test_subjob_name(elmat):
    assert elmat.subjob_name(1, -0.05) == "s_01_e_m0_05000"
    assert elmat.subjob_name(99, 0.5) == "s_99_e_0_50000"


def test_calculate_modulus(elmat):
    elmat._value["C"] = np.array(
        [
            [46.08382102, 30.74709687, 30.74709687, 0.0, 0.0, 0.0],
            [30.74709687, 46.08382102, 30.74709687, 0.0, 0.0, 0.0],
            [30.74709687, 30.74709687, 46.08382102, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 30.22242288, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 30.22242288, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 30.22242288],
        ]
    )
    elmat.calculate_modulus()

    assert pytest.approx(elmat._value["BV"], abs=1e-6) == 35.859338253333334
    assert pytest.approx(elmat._value["GV"], abs=1e-6) == 21.200798558000006
    assert pytest.approx(elmat._value["EV"], abs=1e-6) == 53.13156166218971
    assert pytest.approx(elmat._value["nuV"], abs=1e-6) == 0.2530556695030905
    assert pytest.approx(elmat._value["S"][0][0], abs=1e-6) == 0.04656717
    assert pytest.approx(elmat._value["BR"], abs=1e-6) == 35.85933825333332
    assert pytest.approx(elmat._value["GR"], abs=1e-6) == 13.885959875273606
    assert pytest.approx(elmat._value["ER"], abs=1e-6) == 36.8954828187832
    assert pytest.approx(elmat._value["nuR"], abs=1e-6) == 0.3285175511878762
    assert pytest.approx(elmat._value["BH"], abs=1e-6) == 35.85933825333333
    assert pytest.approx(elmat._value["GH"], abs=1e-6) == 17.543379216636808
    assert pytest.approx(elmat._value["EH"], abs=1e-6) == 45.25082038312408
    assert pytest.approx(elmat._value["nuH"], abs=1e-6) == 0.2896836984579242
    assert pytest.approx(elmat._value["AVR"], abs=1e-6) == 20.84786115718676


def test_fit_elastic_matrix_calculate(elmat):
    elmat.calculate()

    assert pytest.approx(elmat._value["C"][0][0], abs=1e-6) == 46.08382102
    assert pytest.approx(elmat._value["C"][0][1], abs=1e-6) == 30.74709687
    assert pytest.approx(elmat._value["C"][3][3], abs=1e-6) == 30.22242288

    assert pytest.approx(elmat._value["BV"], abs=1e-6) == 35.859338253333334
    assert pytest.approx(elmat._value["GV"], abs=1e-6) == 21.200798558000006
    assert pytest.approx(elmat._value["EV"], abs=1e-6) == 53.13156166218971
    assert pytest.approx(elmat._value["nuV"], abs=1e-6) == 0.2530556695030905
    assert pytest.approx(elmat._value["S"][0][0], abs=1e-6) == 0.04656717
    assert pytest.approx(elmat._value["BR"], abs=1e-6) == 35.85933825333332
    assert pytest.approx(elmat._value["GR"], abs=1e-6) == 13.885959875273606
    assert pytest.approx(elmat._value["ER"], abs=1e-6) == 36.8954828187832
    assert pytest.approx(elmat._value["nuR"], abs=1e-6) == 0.3285175511878762
    assert pytest.approx(elmat._value["BH"], abs=1e-6) == 35.85933825333333
    assert pytest.approx(elmat._value["GH"], abs=1e-6) == 17.543379216636808
    assert pytest.approx(elmat._value["EH"], abs=1e-6) == 45.25082038312408
    assert pytest.approx(elmat._value["nuH"], abs=1e-6) == 0.2896836984579242
    assert pytest.approx(elmat._value["AVR"], abs=1e-6) == 20.84786115718676


def test_fit_elastic_matrix_run(elmat):
    elmat.calculate()

    assert pytest.approx(elmat._value["C"][0][0], abs=1e-6) == 46.08382102
    assert pytest.approx(elmat._value["C"][0][1], abs=1e-6) == 30.74709687
    assert pytest.approx(elmat._value["C"][3][3], abs=1e-6) == 30.22242288

    assert pytest.approx(elmat._value["BV"], abs=1e-6) == 35.859338253333334
    assert pytest.approx(elmat._value["GV"], abs=1e-6) == 21.200798558000006
    assert pytest.approx(elmat._value["EV"], abs=1e-6) == 53.13156166218971
    assert pytest.approx(elmat._value["nuV"], abs=1e-6) == 0.2530556695030905
    assert pytest.approx(elmat._value["S"][0][0], abs=1e-6) == 0.04656717
    assert pytest.approx(elmat._value["BR"], abs=1e-6) == 35.85933825333332
    assert pytest.approx(elmat._value["GR"], abs=1e-6) == 13.885959875273606
    assert pytest.approx(elmat._value["ER"], abs=1e-6) == 36.8954828187832
    assert pytest.approx(elmat._value["nuR"], abs=1e-6) == 0.3285175511878762
    assert pytest.approx(elmat._value["BH"], abs=1e-6) == 35.85933825333333
    assert pytest.approx(elmat._value["GH"], abs=1e-6) == 17.543379216636808
    assert pytest.approx(elmat._value["EH"], abs=1e-6) == 45.25082038312408
    assert pytest.approx(elmat._value["nuH"], abs=1e-6) == 0.2896836984579242
    assert pytest.approx(elmat._value["AVR"], abs=1e-6) == 20.84786115718676


def test_to_from_dict(elmat):
    elmat.calculate()

    prop_dict = elmat.todict()
    prop_dict_json = json.dumps(prop_dict, cls=JsonNumpyEncoder)
    prop_dict = json.loads(prop_dict_json)
    new_prop = ElasticMatrixCalculator.fromdict(prop_dict)

    assert elmat.basis_ref == new_prop.basis_ref
    assert pytest.approx(new_prop._value["C"][0][0], abs=1e-6) == 46.08382102
    assert pytest.approx(new_prop._value["C"][0][1], abs=1e-6) == 30.74709687
    assert pytest.approx(new_prop._value["C"][3][3], abs=1e-6) == 30.22242288
    assert pytest.approx(new_prop._value["BV"], abs=1e-6) == 35.859338253333334
    assert pytest.approx(new_prop._value["GV"], abs=1e-6) == 21.200798558000006
    assert pytest.approx(new_prop._value["EV"], abs=1e-6) == 53.13156166218971
    assert pytest.approx(new_prop._value["nuV"], abs=1e-6) == 0.2530556695030905
    assert pytest.approx(new_prop._value["S"][0][0], abs=1e-6) == 0.04656717
    assert pytest.approx(new_prop._value["BR"], abs=1e-6) == 35.85933825333332
    assert pytest.approx(new_prop._value["GR"], abs=1e-6) == 13.885959875273606
    assert pytest.approx(new_prop._value["ER"], abs=1e-6) == 36.8954828187832
    assert pytest.approx(new_prop._value["nuR"], abs=1e-6) == 0.3285175511878762
    assert pytest.approx(new_prop._value["BH"], abs=1e-6) == 35.85933825333333
    assert pytest.approx(new_prop._value["GH"], abs=1e-6) == 17.543379216636808
    assert pytest.approx(new_prop._value["EH"], abs=1e-6) == 45.25082038312408
    assert pytest.approx(new_prop._value["nuH"], abs=1e-6) == 0.2896836984579242
    assert pytest.approx(new_prop._value["AVR"], abs=1e-6) == 20.84786115718676


def test_ignore_symmetry(atoms, calculator):
    elmat = ElasticMatrixCalculator(atoms, ignore_symmetry=True)
    atoms.calc = calculator

    elmat.calculate()
    assert elmat._value["space_group_number"] == 1
    assert len(elmat._value["strain_list"]) == 21


@pytest.mark.parametrize("sgn", [1, 3, 16, 75, 89, 143, 149, 168, 177, 195, 207])
def test_C_for_all_lauer_groups(sgn, calculator):
    laue_group = space_group_to_laue_group(sgn)
    atoms = create_atoms_by_spacegroup(sgn)
    atoms.calc = calculator
    elmat = ElasticMatrixCalculator(
        atoms, num_of_point=5, optimize_deformed_structure=False
    )

    elmat.calculate(verbose=True)
    value = elmat._value
    C = np.array(value["C"])
    C_std = np.array(value["C_std"])

    assert value["space_group_number"] == sgn
    assert np.allclose(C, reference_C_matrix_dict[sgn], atol=1e-4)
    assert np.allclose(C_std, reference_C_std_matrix_dict[sgn], atol=1e-4)


reference_C_matrix_dict = {
    1: np.array(
        [
            [
                191.91888721,
                24.96888332,
                92.3191319,
                36.54161569,
                -73.67591784,
                45.83464712,
            ],
            [
                24.96888332,
                -10.5967772,
                33.39534402,
                49.73635232,
                45.19366207,
                46.07950587,
            ],
            [
                92.3191319,
                33.39534402,
                15.21406293,
                52.12489666,
                4.46195672,
                40.75982811,
            ],
            [
                36.54161569,
                49.73635232,
                52.12489666,
                38.48846234,
                46.22720004,
                43.14620099,
            ],
            [
                -73.67591784,
                45.19366207,
                4.46195672,
                46.22720004,
                108.09074155,
                44.92696656,
            ],
            [
                45.83464712,
                46.07950587,
                40.75982811,
                43.14620099,
                44.92696656,
                33.36796885,
            ],
        ]
    ),
    3: np.array(
        [
            [2.47622481e02, 1.28963712e01, 8.62082207e01, 0, 0, 3.62650674e-02],
            [1.28963712e01, -1.01158835e01, 2.05906951e01, 0, 0, 4.08732596e-02],
            [8.62082207e01, 2.05906951e01, 5.37764820e00, 0, 0, 1.82990925e-02],
            [0, 0, 0, 2.80649821e01, 4.91873582e-02, 0],
            [0, 0, 0, 4.91873582e-02, 1.02208335e02, 0],
            [3.62650674e-02, 4.08732596e-02, 1.82990925e-02, 0, 0, 2.22175493e01],
        ]
    ),
    16: np.array(
        [
            [325.43832667, -15.52646244, 120.39059323, 0.0, 0.0, 0.0],
            [-15.52646244, -39.79971836, -8.26179125, 0.0, 0.0, 0.0],
            [120.39059323, -8.26179125, 19.96484275, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -4.42200578, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 131.68967539, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -9.07883903],
        ]
    ),
    75: np.array(
        [
            [98.9475712, -27.7043575, 69.73451481, 0.0, 0.0, 1.00555145],
            [-27.7043575, 98.9475712, 69.73451481, 0.0, 0.0, -1.00555145],
            [69.73451481, 69.73451481, 57.95434229, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 80.02277124, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 80.02277124, 0.0],
            [1.00555145, -1.00555145, 0.0, 0.0, 0.0, -10.45735697],
        ]
    ),
    89: np.array(
        [
            [148.5693987, -22.29544188, 70.23305912, 0.0, 0.0, 0.0],
            [-22.29544188, 148.5693987, 70.23305912, 0.0, 0.0, 0.0],
            [70.23305912, 70.23305912, -20.98094838, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 75.38013966, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 75.38013966, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -16.97748946],
        ]
    ),
    143: np.array(
        [
            [324.14081243, 69.04739145, 7.64486255, 14.79913919, -7.84074059, 0.0],
            [69.04739145, 324.14081243, 7.64486255, -14.79913919, 7.84074059, 0.0],
            [7.64486255, 7.64486255, -25.22703726, 0.0, 0.0, 0.0],
            [14.79913919, -14.79913919, 0.0, 12.64269446, 0.0, 7.84074059],
            [-7.84074059, 7.84074059, 0.0, 0.0, 12.64269446, 14.79913919],
            [0.0, 0.0, 0.0, 7.84074059, 14.79913919, 127.54671049],
        ]
    ),
    149: np.array(
        [
            [2.44178520e02, 5.74791778e01, -8.35112019e00, -6.35319828e-02, 0, 0],
            [5.74791778e01, 2.44178520e02, -8.35112019e00, 6.35319828e-02, 0, 0],
            [-8.35112019e00, -8.35112019e00, -1.91054349e01, 0, 0, 0],
            [-6.35319828e-02, 6.35319828e-02, 0, 8.36351355e00, 0, 0],
            [0, 0, 0, 0, 8.36351355e00, -6.35319828e-02],
            [0, 0, 0, 0, -6.35319828e-02, 9.33496709e01],
        ]
    ),
    168: np.array(
        [
            [300.0522306, 45.08572713, 3.24663111, 0.0, 0.0, 0.0],
            [45.08572713, 300.0522306, 3.24663111, 0.0, 0.0, 0.0],
            [3.24663111, 3.24663111, -51.89539776, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 3.99885059, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 3.99885059, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 127.48325173],
        ]
    ),
    177: np.array(
        [
            [237.73623841, 60.97627355, -12.41966065, 0.0, 0.0, 0.0],
            [60.97627355, 237.73623841, -12.41966065, 0.0, 0.0, 0.0],
            [-12.41966065, -12.41966065, -36.33017202, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -8.83390626, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, -8.83390626, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 88.37998243],
        ]
    ),
    195: np.array(
        [
            [265.8322142, 78.03755303, 78.03755303, 0.0, 0.0, 0.0],
            [78.03755303, 265.8322142, 78.03755303, 0.0, 0.0, 0.0],
            [78.03755303, 78.03755303, 265.8322142, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 105.82117704, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 105.82117704, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 105.82117704],
        ]
    ),
    207: np.array(
        [
            [225.1324571, 78.59653064, 78.59653064, 0.0, 0.0, 0.0],
            [78.59653064, 225.1324571, 78.59653064, 0.0, 0.0, 0.0],
            [78.59653064, 78.59653064, 225.1324571, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 104.85534291, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 104.85534291, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 104.85534291],
        ]
    ),
}

reference_C_std_matrix_dict = {
    1: np.array(
        [
            [0.0006866, 0.01637547, 0.04954481, 0.04992326, 0.01168052, 0.05252913],
            [0.01637547, 0.02462462, 0.10003435, 0.0263439, 0.00743988, 0.02071196],
            [0.04954481, 0.10003435, 0.09478744, 0.02554272, 0.05595476, 0.02544056],
            [0.04992326, 0.0263439, 0.02554272, 0.00952596, 0.18554903, 0.2929556],
            [0.01168052, 0.00743988, 0.05595476, 0.18554903, 0.00337596, 0.20311247],
            [0.05252913, 0.02071196, 0.02544056, 0.2929556, 0.20311247, 0.00817449],
        ]
    ),
    3: np.array(
        [
            [1.50582903e01, 7.59852476e00, 7.52883685e00, 0, 0, 6.13562874e00],
            [7.59852476e00, 1.02797303e-01, 1.28248235e00, 0, 0, 6.16462423e00],
            [7.52883685e00, 1.28248235e00, 3.67744110e-02, 0, 0, 6.21731313e00],
            [0, 0, 0, 2.13835790e-04, 1.23566517e-01, 0],
            [0, 0, 0, 1.23566517e-01, 2.34328407e-01, 0],
            [6.13562874e00, 6.16462423e00, 6.21731313e00, 0, 0, 2.11349516e-04],
        ]
    ),
    16: np.array(
        [
            [4.59000776e00, 2.29499782e00, 2.29597745e00, 0, 0, 0],
            [2.29499782e00, 1.05475696e-02, 1.85242609e00, 0, 0, 0],
            [2.29597745e00, 1.85242609e00, 1.53863107e-01, 0, 0, 0],
            [0, 0, 0, 5.78732260e-07, 0, 0],
            [0, 0, 0, 0, 3.82187362e-05, 0],
            [0, 0, 0, 0, 0, 1.06699242e-06],
        ]
    ),
    75: np.array(
        [
            [1.82550120e00, 1.82550120e00, 9.15272762e-01, 0, 0, 4.33550177e-05],
            [1.82550120e00, 1.82550120e00, 9.15272762e-01, 0, 0, 4.33550177e-05],
            [9.15272762e-01, 9.15272762e-01, 8.61793320e-02, 0, 0, 0],
            [0, 0, 0, 4.46179931e-06, 0, 0],
            [0, 0, 0, 0, 4.46179931e-06, 0],
            [4.33550177e-05, 4.33550177e-05, 0, 0, 0, 7.93593513e-07],
        ]
    ),
    89: np.array(
        [
            [7.98890017e-01, 7.98890017e-01, 4.00134424e-01, 0, 0, 0],
            [7.98890017e-01, 7.98890017e-01, 4.00134424e-01, 0, 0, 0],
            [4.00134424e-01, 4.00134424e-01, 5.57476378e-03, 0, 0, 0],
            [0, 0, 0, 2.85457596e-03, 0, 0],
            [0, 0, 0, 0, 2.85457596e-03, 0],
            [0, 0, 0, 0, 0, 5.30949507e-04],
        ]
    ),
    143: np.array(
        [
            [
                4.92525033e-01,
                2.48083783e00,
                1.84801421e00,
                1.65572958e-01,
                4.34593957e-01,
                0,
            ],
            [
                2.48083783e00,
                4.92525033e-01,
                1.84801421e00,
                1.65572958e-01,
                4.34593957e-01,
                0,
            ],
            [1.84801421e00, 1.84801421e00, 6.69183108e-03, 0, 0, 0],
            [1.65572958e-01, 1.65572958e-01, 0, 1.13897461e-03, 0, 4.34593957e-01],
            [4.34593957e-01, 4.34593957e-01, 0, 0, 1.13897461e-03, 1.65572958e-01],
            [0, 0, 0, 4.34593957e-01, 1.65572958e-01, 1.26462813e00],
        ]
    ),
    149: np.array(
        [
            [1.20152798e00, 5.14850686e00, 1.00425223e01, 7.19309861e-01, 0, 0],
            [5.14850686e00, 1.20152798e00, 1.00425223e01, 7.19309861e-01, 0, 0],
            [1.00425223e01, 1.00425223e01, 1.41013724e00, 0, 0, 0],
            [7.19309861e-01, 7.19309861e-01, 0, 2.52374016e-04, 0, 0],
            [0, 0, 0, 0, 2.52374016e-04, 7.19309861e-01],
            [0, 0, 0, 0, 7.19309861e-01, 2.64342545e00],
        ]
    ),
    168: np.array(
        [
            [0.26828372, 7.05182529, 1.78311047, 0.0, 0.0, 0.0],
            [7.05182529, 0.26828372, 1.78311047, 0.0, 0.0, 0.0],
            [1.78311047, 1.78311047, 2.04716805, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.00313073, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.00313073, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 3.52846341],
        ]
    ),
    177: np.array(
        [
            [0.58828491, 0.66958649, 0.08117607, 0.0, 0.0, 0.0],
            [0.66958649, 0.58828491, 0.08117607, 0.0, 0.0, 0.0],
            [0.08117607, 0.08117607, 0.00204924, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.04643756, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.04643756, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.44565267],
        ]
    ),
    195: np.array(
        [
            [0.60577816, 0.58255646, 0.58255646, 0.0, 0.0, 0.0],
            [0.58255646, 0.60577816, 0.58255646, 0.0, 0.0, 0.0],
            [0.58255646, 0.58255646, 0.60577816, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.37224797, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.37224797, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.37224797],
        ]
    ),
    207: np.array(
        [
            [13.39677165, 11.46751864, 11.46751864, 0.0, 0.0, 0.0],
            [11.46751864, 13.39677165, 11.46751864, 0.0, 0.0, 0.0],
            [11.46751864, 11.46751864, 13.39677165, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.08390907, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.08390907, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.08390907],
        ]
    ),
}
