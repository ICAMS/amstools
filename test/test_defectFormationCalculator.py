import json
import numpy as np
import pytest
from amstools import DefectFormationCalculator
from amstools.utils import JsonNumpyEncoder
from test.utils import atoms, calculator


@pytest.fixture
def vac(atoms, calculator, monkeypatch):
    monkeypatch.setattr(calculator, "get_stress", lambda *args, **kwargs: np.zeros(6))
    atoms.calc = calculator
    return DefectFormationCalculator(atoms)


def test_generate_structures(vac):
    gen_structures = vac.generate_structures()
    assert "supercell_0" in gen_structures
    assert "supercell_defect_wyck_a_static" in gen_structures
    assert "supercell_defect_wyck_a_atomic" in gen_structures
    assert "supercell_defect_wyck_a_total" in gen_structures


def test_subjob_name(vac):
    for minstyle in DefectFormationCalculator.minimization_styles:
        assert f"supercell_defect_wyck_a_{minstyle}" == vac.subjob_name(minstyle, "a")


def test_analyse_structures(vac):
    output_dict = {
        "supercell_0": {
            "energy": -0.09613104551369389,
            "volume": 1062.8820000000005,
        },
        "supercell_defect_wyck_a_static": {
            "energy": 0.8788335155724916,
            "volume": 1062.8820000000005,
        },
        "supercell_defect_wyck_a_atomic": {
            "energy": 0.8670150052386862,
            "volume": 1062.8820000000005,
        },
        "supercell_defect_wyck_a_total": {
            "energy": 0.8670150052386862,
            "volume": 1062.8820000000005,
        },
    }
    vac.analyse_structures(output_dict)

    expected_keys = [
        "supercell_range",
        "n",
        "wyckoff_unique_indices",
        "energies",
        "volumes",
        "vacancy_formation_energy",
        "vacancy_formation_volume",
    ]
    for key in expected_keys:
        assert key in vac._value

    assert vac._value["supercell_range"][0] == 4
    assert len(vac._value["supercell_range"]) == 3
    assert vac._value["n"] == 64
    assert vac._value["wyckoff_unique_indices"] == {"a": 0}
    assert vac._value["energies"] == {
        "supercell_0": -0.09613104551369389,
        "supercell_defect_wyck_a_static": 0.8788335155724916,
        "supercell_defect_wyck_a_atomic": 0.8670150052386862,
        "supercell_defect_wyck_a_total": 0.8670150052386862,
    }
    assert vac._value["volumes"] == {
        "supercell_0": 1062.8820000000005,
        "supercell_defect_wyck_a_static": 1062.8820000000005,
        "supercell_defect_wyck_a_atomic": 1062.8820000000005,
        "supercell_defect_wyck_a_total": 1062.8820000000005,
    }
    assert vac._value["vacancy_formation_energy"] == {
        "a_static": 0.973462513500034,
        "a_atomic": 0.9616440031662287,
        "a_total": 0.9616440031662287,
    }
    assert vac._value["vacancy_formation_volume"] == {"a_total": 0.0}


def test_calculate_defect_formation(vac):
    vac._value = {
        "supercell_range": [4, 4, 4],
        "n": 64,
        "wyckoff_unique_indices": {"a": 0},
        "energies": {
            "supercell_0": -0.09613104551369389,
            "supercell_defect_wyck_a_static": 0.8788335155724916,
            "supercell_defect_wyck_a_atomic": 0.8670150052386862,
            "supercell_defect_wyck_a_total": 0.8670150052386862,
        },
        "volumes": {
            "supercell_0": 1062.8820000000005,
            "supercell_defect_wyck_a_static": 1062.8820000000005,
            "supercell_defect_wyck_a_atomic": 1062.8820000000005,
            "supercell_defect_wyck_a_total": 1062.8820000000005,
        },
    }

    vac.calculate_defect_formation()

    assert vac._value["vacancy_formation_energy"] == {
        "a_static": 0.973462513500034,
        "a_atomic": 0.9616440031662287,
        "a_total": 0.9616440031662287,
    }
    assert vac._value["vacancy_formation_volume"] == {"a_total": 0.0}


def test_get_structure_value(vac, atoms):
    res, structure = vac.get_structure_value(
        atoms, name=DefectFormationCalculator.SUPERCELL_0_NAME
    )
    ref = {"energy": -0.001502047586230404, "volume": 16.60753125}
    for k, v in res.items():
        if k in ref:
            assert np.allclose(v, ref[k])


def test_calculate(vac):
    vac.calculate()
    expected_keys = [
        "supercell_range",
        "n",
        "wyckoff_unique_indices",
        "energies",
        "volumes",
        "vacancy_formation_energy",
        "vacancy_formation_volume",
    ]
    for key in expected_keys:
        assert key in vac._value


def test_to_from_dict(vac):
    vac.calculate()
    prop_dict = vac.todict()
    prop_dict_json = json.dumps(prop_dict, cls=JsonNumpyEncoder)
    prop_dict = json.loads(prop_dict_json)
    new_prop = DefectFormationCalculator.fromdict(prop_dict)

    assert vac.basis_ref == new_prop.basis_ref
    assert vac.value == new_prop.value
