import json
import numpy as np
import pytest
from ase.build import bulk

from amstools.properties.stackingfault import StackingFaultCalculator
from amstools.utils import JsonNumpyEncoder
from test.utils import atoms, calculator


@pytest.fixture
def sf_calc(atoms, calculator, monkeypatch):
    monkeypatch.setattr(calculator, "get_stress", lambda: np.zeros(6))
    atoms.calc = calculator
    return StackingFaultCalculator(atoms=atoms, fmax=0.5)


def test_calculate(sf_calc):
    sf_calc.calculate()

    assert "sf_area" in sf_calc._value
    assert "structure_type" in sf_calc._value
    assert "stacking_fault_types" in sf_calc._value
    assert "structure_names" in sf_calc._value
    assert "raw_data" in sf_calc._value
    assert "stacking_fault_energy" in sf_calc._value
    assert "stacking_fault_energy(mJ/m^2)" in sf_calc._value
    assert "fcc_ESF" in sf_calc._value["stacking_fault_energy"]
    assert "fcc_ISF" in sf_calc._value["stacking_fault_energy"]
    assert "fcc_MAX" in sf_calc._value["stacking_fault_energy"]
    assert "fcc_TWIN" in sf_calc._value["stacking_fault_energy"]
    assert "fcc_IDEAL" in sf_calc._value["stacking_fault_energy"]


def test_calculate_max(atoms, calculator, monkeypatch):
    monkeypatch.setattr(calculator, "get_stress", lambda: np.zeros(6))
    atoms.calc = calculator
    sf_calc = StackingFaultCalculator(atoms=atoms, stacking_fault_types="MAX", fmax=0.5)
    sf_calc.calculate()

    assert "sf_area" in sf_calc._value
    assert "structure_type" in sf_calc._value
    assert "stacking_fault_types" in sf_calc._value
    assert "structure_names" in sf_calc._value
    assert "raw_data" in sf_calc._value
    assert "stacking_fault_energy" in sf_calc._value
    assert "stacking_fault_energy(mJ/m^2)" in sf_calc._value
    assert "fcc_MAX" in sf_calc._value["stacking_fault_energy"]
    assert "fcc_IDEAL" in sf_calc._value["stacking_fault_energy"]


def test_calculate_wrong_symmetry():
    with pytest.raises(ValueError):
        StackingFaultCalculator(atoms=bulk("Al", "bcc", a=3), fmax=0.5)


def test_to_from_dict(atoms, calculator, monkeypatch):
    monkeypatch.setattr(calculator, "get_stress", lambda: np.zeros(6))
    atoms.calc = calculator
    sf_calc = StackingFaultCalculator(atoms=atoms, stacking_fault_types="MAX", fmax=0.5)

    sf_calc.calculate()

    prop_dict = sf_calc.todict()
    prop_dict_json = json.dumps(prop_dict, cls=JsonNumpyEncoder)
    prop_dict = json.loads(prop_dict_json)
    new_prop = StackingFaultCalculator.fromdict(prop_dict)

    assert sf_calc.basis_ref == new_prop.basis_ref
    assert sf_calc.value == new_prop.value
