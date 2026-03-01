import json
from ase import Atoms
from numpy.testing import assert_array_almost_equal

from amstools import TransformationPathCalculator
from amstools.properties.transformationpath import is_general_cubic, to_cubic_atoms
from test.utils import *
from amstools.utils import JsonNumpyEncoder


@pytest.fixture
def atoms(pipeline_calculator_zero_stress):
    atoms = atoms_factory("Al", "bcc", 3.5)
    atoms.calc = pipeline_calculator_zero_stress
    return atoms


@pytest.fixture
def tp(atoms):
    return TransformationPathCalculator(atoms, num_of_point=10)


def test_not_bcc_fcc_structure():
    fcc_atoms = atoms_factory("Al", "sc", a=4.05)
    with pytest.raises(ValueError):
        TransformationPathCalculator(fcc_atoms)


def test_not_bcc_structure():
    fcc_atoms = atoms_factory("Al", "fcc")
    TransformationPathCalculator(fcc_atoms)


def test_subjob_name(tp):
    assert tp.subjob_name(0.0) == "tp_0_00000"
    assert tp.subjob_name(4.2) == "tp_4_20000"


def test_transformation_type(atoms):
    for tt in ["tetragonal", "hexagonal", "orthogonal", "trigonal"]:
        tp = TransformationPathCalculator(atoms, transformation_type=tt)
        assert tp.transformation_type == tt


def test_deformation_path(atoms):
    # Test tetragonal path
    tp = TransformationPathCalculator(
        atoms, transformation_type="tetragonal", num_of_point=10
    )
    tet = np.array(
        [
            0.8,
            0.93333333,
            1.06666667,
            1.2,
            1.33333333,
            1.46666667,
            1.6,
            1.73333333,
            1.86666667,
            2.0,
        ]
    )
    assert_array_almost_equal(tp.deformation_path(), tet)

    # Test hexagonal path
    tp = TransformationPathCalculator(
        atoms, transformation_type="hexagonal", num_of_point=10
    )
    hex_path = np.array(
        [
            -0.5,
            -0.24444444,
            0.01111111,
            0.26666667,
            0.52222222,
            0.77777778,
            1.03333333,
            1.28888889,
            1.54444444,
            1.8,
        ]
    )
    assert_array_almost_equal(tp.deformation_path(), hex_path)

    # Test orthogonal path
    tp = TransformationPathCalculator(
        atoms, transformation_type="orthogonal", num_of_point=10
    )
    ort = np.array(
        [
            1.0,
            1.04602373,
            1.09204746,
            1.13807119,
            1.18409492,
            1.23011865,
            1.27614237,
            1.3221661,
            1.36818983,
            1.41421356,
        ]
    )
    assert_array_almost_equal(tp.deformation_path(), ort)

    # Test trigonal path
    tp = TransformationPathCalculator(
        atoms, transformation_type="trigonal", num_of_point=10
    )
    tri = np.array(
        [
            0.8,
            1.26666667,
            1.73333333,
            2.2,
            2.66666667,
            3.13333333,
            3.6,
            4.06666667,
            4.53333333,
            5.0,
        ]
    )
    assert_array_almost_equal(tp.deformation_path(), tri)


def test_generate_tetra_path(atoms):
    tp = TransformationPathCalculator(
        atoms, transformation_type="tetragonal", num_of_point=10
    )
    p, ats = tp.generate_tetra_path()
    tet = np.array(
        [
            0.8,
            0.93333333,
            1.06666667,
            1.2,
            1.33333333,
            1.46666667,
            1.6,
            1.73333333,
            1.86666667,
            2.0,
        ]
    )
    assert_array_almost_equal(p, tet)
    assert pytest.approx(ats[0].get_cell()[0][0]) == 3.770260707555796
    assert pytest.approx(ats[3].get_cell()[0][0]) == 3.2936261010835994
    assert pytest.approx(ats[6].get_cell()[0][0]) == 2.9924579066842196
    assert pytest.approx(ats[9].get_cell()[0][0]) == 2.777951840944349


def test_generate_ortho_path(atoms):
    tp = TransformationPathCalculator(
        atoms, transformation_type="orthogonal", num_of_point=10
    )
    p, ats = tp.generate_ortho_path()
    ort = np.array(
        [
            1.0,
            1.04602373,
            1.09204746,
            1.13807119,
            1.18409492,
            1.23011865,
            1.27614237,
            1.3221661,
            1.36818983,
            1.41421356,
        ]
    )
    assert_array_almost_equal(p, ort)
    assert pytest.approx(ats[0].get_cell()[0][0]) == 4.949747468305833
    assert pytest.approx(ats[3].get_cell()[0][0]) == 4.949747468305833
    assert pytest.approx(ats[6].get_cell()[0][0]) == 4.949747468305833
    assert pytest.approx(ats[9].get_cell()[0][0]) == 4.949747468305833


def test_generate_trigo_path(atoms):
    tp = TransformationPathCalculator(
        atoms, transformation_type="trigonal", num_of_point=10
    )
    p, ats = tp.generate_trigo_path()
    tri = np.array(
        [
            0.8,
            1.26666667,
            1.73333333,
            2.2,
            2.66666667,
            3.13333333,
            3.6,
            4.06666667,
            4.53333333,
            5.0,
        ]
    )
    assert_array_almost_equal(p, tri)
    assert pytest.approx(ats[0].get_cell()[0][0]) == 0.0
    assert pytest.approx(ats[3].get_cell()[0][0]) == 0.0
    assert pytest.approx(ats[6].get_cell()[0][0]) == 0.0
    assert pytest.approx(ats[9].get_cell()[0][0]) == 0.0


def test_generate_hex_path(atoms):
    tp = TransformationPathCalculator(
        atoms, transformation_type="hexagonal", num_of_point=10
    )
    p, ats = tp.generate_hex_path()
    hex_path = np.array(
        [
            -0.5,
            -0.24444444,
            0.01111111,
            0.26666667,
            0.52222222,
            0.77777778,
            1.03333333,
            1.28888889,
            1.54444444,
            1.8,
        ]
    )
    assert_array_almost_equal(p, hex_path)
    assert pytest.approx(ats[0].get_cell()[0][0]) == 4.762045546564168
    assert pytest.approx(ats[3].get_cell()[0][0]) == 5.05907362579009
    assert pytest.approx(ats[6].get_cell()[0][0]) == 5.417995065767206
    assert pytest.approx(ats[9].get_cell()[0][0]) == 5.866121358257826


def test_generate_general_tetragonal_path(atoms):
    tp = TransformationPathCalculator(
        atoms, transformation_type="general_cubic_tetragonal", num_of_point=10
    )
    p, ats = tp.generate_general_tetra_path()
    gen_tetr = np.array(
        [0.2, 0.466667, 0.733333, 1.0, 1.266667, 1.533333, 1.8, 2.066667, 2.333333, 2.6]
    )

    assert_array_almost_equal(p, gen_tetr)
    assert pytest.approx(ats[0].get_cell()[0][0]) == 5.984915813368439
    assert pytest.approx(ats[3].get_cell()[0][0]) == 3.4999999999999996
    assert pytest.approx(ats[6].get_cell()[0][0]) == 2.877247420052215
    assert pytest.approx(ats[9].get_cell()[0][0]) == 2.545327062379987


def test_generate_structures(tp):
    res = tp.generate_structures()
    for name in [
        "tp_0_80000",
        "tp_0_93333",
        "tp_1_06667",
        "tp_1_20000",
        "tp_1_33333",
        "tp_1_46667",
        "tp_1_60000",
        "tp_1_73333",
        "tp_1_86667",
        "tp_2_00000",
    ]:
        assert name in res


def test_analyse_structures(tp):
    output_dict = {
        "tp_0_80000": {"energy": 0.3524985864116612, "energy_0": 0.3524985864116612},
        "tp_0_93333": {"energy": 0.3612582046776538, "energy_0": 0.3612582046776538},
        "tp_1_06667": {"energy": 0.36057322734384023, "energy_0": 0.36057322734384023},
        "tp_1_20000": {"energy": 0.3598279134282336, "energy_0": 0.3598279134282336},
        "tp_1_33333": {"energy": 0.3591169526647384, "energy_0": 0.3591169526647384},
        "tp_1_46667": {"energy": 0.3587366702354702, "energy_0": 0.3587366702354702},
        "tp_1_60000": {"energy": 0.34779590952378303, "energy_0": 0.34779590952378303},
        "tp_1_73333": {"energy": 0.3646520878396231, "energy_0": 0.3646520878396231},
        "tp_1_86667": {"energy": 0.4066004469046245, "energy_0": 0.4066004469046245},
        "tp_2_00000": {"energy": 0.4715040630229339, "energy_0": 0.4715040630229339},
    }

    tp.analyse_structures(output_dict)
    assert "transformation_type" in tp._value
    assert "transformation_coordinates" in tp._value
    assert "energies" in tp._value
    assert "energies_0" in tp._value


def test_get_structure_value(tp, atoms):
    res, struc = tp.get_structure_value(atoms)
    assert pytest.approx(res["energy"]) == 0.18022137308618813
    assert pytest.approx(res["energy_0"]) == 0.18022137308618813


def test_calculate(tp):
    tp.calculate()
    assert "transformation_type" in tp._value
    assert "transformation_coordinates" in tp._value
    assert "energies" in tp._value
    assert "energies_0" in tp._value


def test_run(tp):
    tp.calculate()
    assert "transformation_type" in tp._value
    assert "transformation_coordinates" in tp._value
    assert "energies" in tp._value
    assert "energies_0" in tp._value


@pytest.mark.parametrize(
    "structure,expected",
    [
        (bulk("Al", "fcc", cubic=True), True),
        (bulk("Al", "fcc", cubic=False), True),
        (bulk("W", "bcc", cubic=True), True),
        (bulk("W", "bcc", cubic=False), True),
        (bulk("Si", "diamond", cubic=True), True),
        (bulk("Si", "diamond", cubic=False), True),
        (bulk("Si", "sc", a=2, cubic=True), True),
        (bulk("Si", "sc", a=2, cubic=False), True),
        (Atoms("Al", cell=[1, 1, 1]), True),
        (Atoms("Al", cell=[1, 1, 2]), False),
        (Atoms("Al", cell=[1, 2, 2]), False),
        (Atoms("Al", cell=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]), True),
        (Atoms("Al", cell=[[1, 0, 0], [0, 1, 0], [1, 2, 3]]), False),
    ],
)
def test_is_general_cubic(structure, expected):
    assert is_general_cubic(structure) == expected


def test_to_cubic_atoms():
    cell_ref = np.diag([3, 3, 3])
    test_cases = [
        (bulk("Al", "fcc", a=3, cubic=True), 4),
        (bulk("Al", "fcc", a=3, cubic=False), 4),
        (bulk("W", "bcc", a=3, cubic=True), 2),
        (bulk("W", "bcc", a=3, cubic=False), 2),
        (bulk("Si", "diamond", a=3, cubic=True), 8),
        (bulk("Si", "diamond", a=3, cubic=False), 8),
        (bulk("Si", "sc", a=3, cubic=True), 1),
        (bulk("Si", "sc", a=3, cubic=False), 1),
        (Atoms("Al", cell=[3, 3, 3]), 1),
    ]

    for atoms, expected_len in test_cases:
        at = to_cubic_atoms(atoms)
        assert len(at) == expected_len
        assert np.allclose(at.cell, cell_ref)

    # Test cases that should return None
    none_cases = [
        Atoms("Al", cell=[1, 1, 2]),
        Atoms("Al", cell=[1, 2, 2]),
        Atoms("Al", cell=[[3, 0, 0], [0, 3, 0], [1, 2, 3]]),
    ]

    for atoms in none_cases:
        assert to_cubic_atoms(atoms) is None


def test_calculate_general_tetragonal_path(atoms):
    tp = TransformationPathCalculator(
        atoms, transformation_type="general_cubic_tetragonal", num_of_point=10
    )
    tp.calculate()

    expected_energies = [
        29.54557407,
        2.25638599,
        0.39368514,
        0.36044275,
        0.35904675,
        0.35026248,
        0.38234261,
        0.50745538,
        0.6828677,
        0.9312942,
    ]
    assert np.allclose(tp._value["energies_0"], expected_energies)


def test_to_from_dict(tp):
    tp.calculate()
    prop_dict = tp.todict()
    prop_dict_json = json.dumps(prop_dict, cls=JsonNumpyEncoder)
    prop_dict = json.loads(prop_dict_json)
    new_prop = TransformationPathCalculator.fromdict(prop_dict)

    assert tp.basis_ref == new_prop.basis_ref
    assert set(tp.value) == set(new_prop.value)


def test_magnetic_moments(pipeline_calculator_zero_stress):
    atoms = atoms_factory("Al", "bcc", 3.5, cubic=True)
    n_atoms = 2
    atoms.set_initial_magnetic_moments([1.0] * len(atoms))
    init_mag_mom = atoms.get_initial_magnetic_moments()

    assert len(atoms) == n_atoms
    assert np.allclose(init_mag_mom, [1.0, 1.0])

    atoms.calc = pipeline_calculator_zero_stress
    tp = TransformationPathCalculator(atoms, num_of_point=10)
    tp.calculate()

    for at in tp._structure_dict.values():
        assert np.allclose(at.get_initial_magnetic_moments(), init_mag_mom)
