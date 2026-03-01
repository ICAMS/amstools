import json
import numpy as np
import pytest
from ase.optimize import BFGS

from amstools import MurnaghanCalculator
from amstools.utils import JsonNumpyEncoder
from test.utils import atoms, calculator


@pytest.fixture
def murn(atoms, calculator):
    atoms.calc = calculator
    return MurnaghanCalculator(atoms)


def test_fit_murnaghan_run(murn):
    murn.calculate()
    assert "volume" in murn._value
    assert "energy" in murn._value
    assert "equilibrium_energy" in murn._value
    assert "equilibrium_volume" in murn._value
    assert "equilibrium_bulk_modulus" in murn._value
    assert "equilibrium_b_prime" in murn._value
    assert "energy_rms" in murn._value

    assert (
        pytest.approx(murn._value["equilibrium_energy"], abs=1e-6)
        == -0.004879405557087324
    )
    assert (
        pytest.approx(murn._value["equilibrium_volume"], abs=1e-6) == 15.932462581943051
    )
    assert (
        pytest.approx(murn._value["equilibrium_bulk_modulus"], abs=1e-6)
        == 39.60166107289155
    )
    assert (
        pytest.approx(murn._value["equilibrium_b_prime"], abs=1e-6) == 2.067288162827055
    )
    assert pytest.approx(murn._value["energy_rms"], abs=1e-6) == 3.2944531712668226e-06


def test_fit_murnaghan_calculate(murn):
    murn.calculate()
    assert "volume" in murn._value
    assert "energy" in murn._value
    assert "equilibrium_energy" in murn._value
    assert "equilibrium_volume" in murn._value
    assert "equilibrium_bulk_modulus" in murn._value
    assert "equilibrium_b_prime" in murn._value
    assert "energy_rms" in murn._value

    assert (
        pytest.approx(murn._value["equilibrium_energy"], abs=1e-6)
        == -0.004879405557087324
    )
    assert (
        pytest.approx(murn._value["equilibrium_volume"], abs=1e-6) == 15.932462581943051
    )
    assert (
        pytest.approx(murn._value["equilibrium_bulk_modulus"], abs=1e-6)
        == 39.60166107289155
    )
    assert (
        pytest.approx(murn._value["equilibrium_b_prime"], abs=1e-6) == 2.067288162827055
    )
    assert pytest.approx(murn._value["energy_rms"], abs=1e-6) == 3.2944531712668226e-06


def test_get_volume_range(murn):
    expected_volumes = [
        14.946778125,
        15.278928749999995,
        15.611079375000006,
        15.943230000000005,
        16.275380625,
        16.60753125,
        16.939681875,
        17.2718325,
        17.603983125000003,
        17.936133749999996,
        18.268284375000007,
    ]
    assert pytest.approx(murn.get_volume_range(), abs=1e-6) == expected_volumes


def test_generate_structures(murn):
    generated_structure = murn.generate_structures()
    assert "strain_0_9" in generated_structure
    assert "strain_0_92" in generated_structure
    assert "strain_0_9400000000000001" in generated_structure
    assert "strain_0_9600000000000001" in generated_structure
    assert "strain_0_9800000000000001" in generated_structure
    assert "strain_1_0" in generated_structure
    assert "strain_1_02" in generated_structure
    assert "strain_1_04" in generated_structure
    assert "strain_1_06" in generated_structure
    assert "strain_1_08" in generated_structure
    assert "strain_1_1" in generated_structure

    at09 = generated_structure["strain_0_9"]
    at10 = generated_structure["strain_1_0"]

    assert (
        pytest.approx(
            np.linalg.det(at09.get_cell()) / np.linalg.det(at10.get_cell()), abs=1e-6
        )
        == 0.9
    )


def test_get_structure_value(murn, atoms):
    (en, vol, pr), structure = murn.get_structure_value(atoms, name="name")
    assert pytest.approx(en, abs=1e-6) == -0.001502047586230404
    assert pytest.approx(vol, abs=1e-6) == 16.60753125


def test_subjob_name(murn):
    assert murn.subjob_name(1.0) == "strain_1_0"
    assert murn.subjob_name(1.1) == "strain_1_1"
    assert murn.subjob_name(1.01) == "strain_1_01"


def test_get_final_structure(murn):
    murn.calculate()
    v_ret = np.linalg.det(murn.get_final_structure().get_cell())
    v_val = murn._value["equilibrium_volume"]
    assert pytest.approx(v_ret, abs=1e-6) == v_val


def test_get_params_dict(murn):
    expected_params = {
        "fit_order": 5,
        "fmax": 0.005,
        "num_of_point": 11,
        "optimize_deformed_structure": False,
        "optimizer": BFGS,
        "optimizer_kwargs": {},
        "volume_range": 0.1,
    }
    assert murn.get_params_dict() == expected_params


def test_murnaghan_volume_range_list(atoms, calculator):
    atoms.calc = calculator
    volume_range = [15, 18]
    murn = MurnaghanCalculator(atoms, volume_range=volume_range)
    murn.calculate()

    assert "volume" in murn._value
    assert "energy" in murn._value
    assert "equilibrium_energy" in murn._value
    assert "equilibrium_volume" in murn._value
    assert "equilibrium_bulk_modulus" in murn._value
    assert "equilibrium_b_prime" in murn._value
    assert "energy_rms" in murn._value

    assert pytest.approx(min(murn._value["volume"]), abs=1e-6) == volume_range[0]
    assert pytest.approx(max(murn._value["volume"]), abs=1e-6) == volume_range[1]

    assert (
        pytest.approx(murn._value["equilibrium_energy"], abs=1e-5)
        == -0.004879405557087324
    )
    assert (
        pytest.approx(murn._value["equilibrium_volume"], abs=1e-3) == 15.932462581943051
    )
    assert (
        pytest.approx(murn._value["equilibrium_bulk_modulus"], abs=1e-1)
        == 39.60166107289155
    )
    assert (
        pytest.approx(murn._value["equilibrium_b_prime"], abs=1e-1) == 2.067288162827055
    )


def test_to_from_dict(murn):
    murn.calculate()
    prop_dict = murn.todict()
    prop_dict_json = json.dumps(prop_dict, cls=JsonNumpyEncoder)
    prop_dict = json.loads(prop_dict_json)
    new_prop = MurnaghanCalculator.fromdict(prop_dict)

    assert murn.basis_ref == new_prop.basis_ref
    assert (
        pytest.approx(new_prop._value["equilibrium_energy"], abs=1e-5)
        == -0.004879405557087324
    )
    assert (
        pytest.approx(new_prop._value["equilibrium_volume"], abs=1e-3)
        == 15.932462581943051
    )
    assert (
        pytest.approx(new_prop._value["equilibrium_bulk_modulus"], abs=1e-1)
        == 39.60166107289155
    )
    assert (
        pytest.approx(new_prop._value["equilibrium_b_prime"], abs=1e-1)
        == 2.067288162827055
    )


def test_murnaghan_optimize_deformed_structure(atoms, calculator):
    atoms.calc = calculator
    murn = MurnaghanCalculator(atoms, optimize_deformed_structure=True)
    murn.calculate(verbose=True)

    assert "volume" in murn._value
    assert "energy" in murn._value
    assert "equilibrium_energy" in murn._value
    assert "equilibrium_volume" in murn._value
    assert "equilibrium_bulk_modulus" in murn._value
    assert "equilibrium_b_prime" in murn._value
    assert "energy_rms" in murn._value

    assert (
        pytest.approx(murn._value["equilibrium_energy"], abs=1e-5)
        == -0.004879405557087324
    )
    assert (
        pytest.approx(murn._value["equilibrium_volume"], abs=1e-3) == 15.932462581943051
    )
    assert (
        pytest.approx(murn._value["equilibrium_bulk_modulus"], abs=1e-1)
        == 39.60166107289155
    )
    assert (
        pytest.approx(murn._value["equilibrium_b_prime"], abs=1e-1) == 2.067288162827055
    )
