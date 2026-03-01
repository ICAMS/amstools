import json
import numpy as np
import pytest
from ase.build import bulk

from amstools.properties.interstitial import InterstitialFormationCalculator
from amstools.utils import JsonNumpyEncoder
from test.utils import atoms, calculator


@pytest.fixture
def int_calc(atoms, calculator, monkeypatch):
    monkeypatch.setattr(calculator, "get_stress", lambda: np.zeros(6))
    atoms.calc = calculator
    return InterstitialFormationCalculator(atoms, fmax=0.5)


def test_calculate(atoms, calculator, monkeypatch):
    monkeypatch.setattr(calculator, "get_stress", lambda: np.zeros(6))
    atoms.calc = calculator
    int_calc = InterstitialFormationCalculator(
        atoms=atoms,
        interstitial_type=["100_dumbbell", "octa"],
        fmax=0.5,
    )
    int_calc.calculate()

    assert "supercell_size" in int_calc._value
    assert "ideal_structure_n_at" in int_calc._value
    assert "structure_type" in int_calc._value
    assert "interstitial_type" in int_calc._value
    assert "structure_names" in int_calc._value
    assert "interstitial_formation_energy" in int_calc._value
    assert "fcc_100_dumbbell_444" in int_calc._value["interstitial_formation_energy"]
    assert "fcc_octa_444" in int_calc._value["interstitial_formation_energy"]


def test_calculate_all(int_calc):
    int_calc.calculate()

    assert "supercell_size" in int_calc._value
    assert "ideal_structure_n_at" in int_calc._value
    assert "structure_type" in int_calc._value
    assert "interstitial_type" in int_calc._value
    assert "structure_names" in int_calc._value
    assert "interstitial_formation_energy" in int_calc._value
    assert "fcc_100_dumbbell_444" in int_calc._value["interstitial_formation_energy"]
    assert "fcc_octa_444" in int_calc._value["interstitial_formation_energy"]
    assert "fcc_tetra_444" in int_calc._value["interstitial_formation_energy"]


def test_to_from_dict(int_calc):
    int_calc.calculate()

    prop_dict = int_calc.todict()
    prop_dict_json = json.dumps(prop_dict, cls=JsonNumpyEncoder)
    prop_dict = json.loads(prop_dict_json)
    new_prop = InterstitialFormationCalculator.fromdict(prop_dict)

    assert int_calc.basis_ref == new_prop.basis_ref
    assert int_calc.value == new_prop.value


def test_bcc_calculate(calculator, monkeypatch):
    atoms = bulk("Al", "bcc", a=4.0)
    monkeypatch.setattr(calculator, "get_stress", lambda: np.zeros(6))
    atoms.calc = calculator

    int_calc = InterstitialFormationCalculator(
        atoms=atoms,
        fmax=0.5,
    )
    int_calc.calculate()

    assert "supercell_size" in int_calc._value
    assert "ideal_structure_n_at" in int_calc._value
    assert "structure_type" in int_calc._value
    assert int_calc._value["structure_type"] == "bcc"
    assert "interstitial_type" in int_calc._value
    assert "structure_names" in int_calc._value
    assert "interstitial_formation_energy" in int_calc._value
    assert "bcc_100_dumbbell_444" in int_calc._value["interstitial_formation_energy"]
