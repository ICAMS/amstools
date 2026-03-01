import numpy as np
import json
import pytest
from amstools.properties.randomdeform import RandomDeformationCalculator
from amstools.utils import JsonNumpyEncoder
from test.utils import atoms, calculator


@pytest.fixture
def randcalc(atoms, calculator):
    atoms.calc = calculator
    return RandomDeformationCalculator(atoms)


def test_calculate(randcalc):
    randcalc.calculate()
    assert "forces" in randcalc._value
    assert "energy" in randcalc._value


def test_generate_structures(randcalc):
    generated_structure = randcalc.generate_structures()
    expected_key_names = [
        "rnd_0__vol_0.95000",
        "rnd_0__vol_0.97500",
        "rnd_0__vol_1.00000",
        "rnd_0__vol_1.02500",
        "rnd_0__vol_1.05000",
        "rnd_1__vol_0.95000",
        "rnd_1__vol_0.97500",
        "rnd_1__vol_1.00000",
        "rnd_1__vol_1.02500",
        "rnd_1__vol_1.05000",
        "rnd_2__vol_0.95000",
        "rnd_2__vol_0.97500",
        "rnd_2__vol_1.00000",
        "rnd_2__vol_1.02500",
        "rnd_2__vol_1.05000",
    ]
    for k in expected_key_names:
        assert k in generated_structure

    assert not np.allclose(
        generated_structure["rnd_0__vol_0.95000"].cell,
        generated_structure["rnd_0__vol_1.05000"].cell,
    )

    assert not np.allclose(
        generated_structure["rnd_0__vol_0.95000"].cell,
        generated_structure["rnd_0__vol_1.00000"].cell,
    )

    assert not np.allclose(
        generated_structure["rnd_0__vol_1.00000"].cell,
        generated_structure["rnd_2__vol_1.00000"].cell,
    )


def test_generate_structures_no_vol_def(atoms, calculator):
    randcalc = RandomDeformationCalculator(atoms, num_volume_deformations=0)
    randcalc.calculator = calculator
    generated_structure = randcalc.generate_structures()

    assert "rnd_0" in generated_structure
    assert "rnd_1" in generated_structure
    assert "rnd_2" in generated_structure


def test_get_params_dict(randcalc):
    params_dict = randcalc.get_params_dict()
    expected_params = {
        "nsample": 3,
        "supercell_size": None,
        "supercell_max_num_atoms": None,
        "random_atom_displacement": 0.1,
        "random_cell_strain": 0.05,
        "volume_range": 0.05,
        "num_volume_deformations": 5,
        "seed": 42,
    }
    assert params_dict == expected_params


def test_to_from_dict(randcalc):
    randcalc.calculate()

    prop_dict = randcalc.todict()
    prop_dict_json = json.dumps(prop_dict, cls=JsonNumpyEncoder)
    prop_dict = json.loads(prop_dict_json)
    new_prop = RandomDeformationCalculator.fromdict(prop_dict)

    assert randcalc.basis_ref == new_prop.basis_ref
    assert np.allclose(new_prop._value["energy"], randcalc._value["energy"])
    assert np.allclose(new_prop._value["forces"], randcalc._value["forces"])

    for k, at in randcalc.output_structures_dict.items():
        assert at == new_prop.output_structures_dict[k]
