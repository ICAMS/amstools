import pytest
import numpy as np
from ase.calculators.emt import EMT
from ase.build import bulk


def calculator_factory():
    return EMT()


def atoms_factory(species="Al", symmetry="fcc", a=4.05, **kwargs):
    atoms = bulk(species, symmetry, a, **kwargs)
    return atoms


@pytest.fixture
def calculator():
    calculator = calculator_factory()

    def get_stress(*args, **kwargs):
        return np.zeros(6)

    calculator.get_stress = get_stress
    return calculator


@pytest.fixture
def atoms(calculator):
    atoms = atoms_factory()
    atoms.calc = calculator
    return atoms


@pytest.fixture
def elements():
    return "Al"


@pytest.fixture
def pipeline_calculator_zero_stress(calculator, monkeypatch):
    monkeypatch.setattr(calculator, "get_stress", lambda *args: np.zeros(6))
    return calculator
