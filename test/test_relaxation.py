import json
import numpy as np
import pytest
from ase.build import bulk

from amstools.properties import MurnaghanCalculator
from amstools.properties.relaxation import (
    IsoOptimizer,
    SpecialOptimizer,
    StepwiseOptimizer,
)
from amstools.utils import JsonNumpyEncoder
from test.utils import calculator_factory

v0 = 63.723601977561984


def get_volume(at):
    return np.linalg.det(at.get_cell())


def get_stress(self, atoms=None):
    if atoms is None:
        atoms = self
    v = get_volume(atoms)
    s = 10 * 2 * (v - v0) / (v + v0)
    return np.array([s, s, s, 0, 0, 0])


@pytest.fixture
def calculator():
    calculator = calculator_factory()
    calculator.get_stress = get_stress
    return calculator


def test_has_optimized_structure(calculator):
    atoms = bulk("Al", "fcc", a=2.025 * 2, cubic=True)
    atoms.calc = calculator
    iso_optimizer = IsoOptimizer(atoms)

    iso_optimizer.run()
    assert iso_optimizer.optimized_structure is not None


def test_iso_run(calculator):
    atoms = bulk("Al", "fcc", a=2.025 * 2, cubic=True)
    atoms.calc = calculator
    iso_optimizer = IsoOptimizer(atoms)

    iso_optimizer.run()
    opt_atoms = iso_optimizer.optimized_structure
    assert opt_atoms is not None

    reference_cell = np.array(
        [
            [3.9942331717623687, 0.0, 0.0],
            [0.0, 3.9942331717623687, 0.0],
            [0.0, 0.0, 3.9942331717623687],
        ]
    )
    np.testing.assert_almost_equal(opt_atoms.get_cell(), reference_cell, decimal=3)


def test_iso_optimization_quality(calculator):
    atoms = bulk("Al", "fcc", a=2.025 * 2, cubic=True)
    atoms.calc = calculator
    iso_optimizer = IsoOptimizer(atoms)

    murn = MurnaghanCalculator(atoms)
    murn.calculate()
    murn2 = MurnaghanCalculator(murn.get_final_structure())
    murn2.calculate()

    ref_volume = np.linalg.det(murn2.get_final_structure().get_cell())
    iso_optimizer.run()
    opt_atoms = iso_optimizer.optimized_structure
    opt_volume = np.linalg.det(opt_atoms.get_cell())

    assert abs(ref_volume - opt_volume) / ref_volume <= 5e-4


def test_iso_select_optimizer(calculator):
    atoms = bulk("Al", "fcc", a=2.025 * 2, cubic=True)
    atoms.calc = calculator
    optimizer = IsoOptimizer(atoms, optimizer="FIRE")

    assert optimizer.optimizer == "FIRE"
    optimizer.run()
    opt_atoms = optimizer.optimized_structure

    assert opt_atoms is not None
    reference_cell = np.array(
        [
            [3.9942331717623687, 0.0, 0.0],
            [0.0, 3.9942331717623687, 0.0],
            [0.0, 0.0, 3.9942331717623687],
        ]
    )
    np.testing.assert_almost_equal(opt_atoms.get_cell(), reference_cell, decimal=3)


def test_iso_optimize(calculator):
    atoms = bulk("Al", "fcc", a=2.025 * 2, cubic=True)
    atoms.calc = calculator
    iso_optimizer = IsoOptimizer(atoms)

    assert not hasattr(iso_optimizer, "res")
    res = iso_optimizer.optimize(atoms)
    assert res is not None


def test_iso_to_from_dict(calculator):
    atoms = bulk("Al", "fcc", a=2.025 * 2, cubic=True)
    atoms.calc = calculator
    iso_optimizer = IsoOptimizer(atoms)

    iso_optimizer.run()
    opt_dict = iso_optimizer.todict()

    opt_dict_json = json.dumps(opt_dict, cls=JsonNumpyEncoder)
    opt_dict = json.loads(opt_dict_json)
    new_opt = IsoOptimizer.fromdict(opt_dict)

    assert iso_optimizer.structure == new_opt.structure
    assert set(iso_optimizer.value) == set(new_opt.value)
    assert iso_optimizer.value["energy"] == new_opt.value["energy"]


def test_special_atomic_optimization(calculator):
    atoms = bulk("Al", "fcc", a=2.025 * 2, cubic=True)
    atoms.calc = calculator
    special_optimizer = SpecialOptimizer(atoms)

    orig_pos = atoms.get_scaled_positions()

    deformed_atoms = atoms.copy()
    pos = deformed_atoms.get_scaled_positions()
    pos[0, 0] += 0.1

    deformed_atoms.set_scaled_positions(pos)
    special_optimizer.run(deformed_atoms, calculator)
    opt_structure = special_optimizer.optimized_structure
    opt_pos = opt_structure.get_scaled_positions()
    opt_pos = opt_pos - opt_pos[0]

    opt_vol = np.linalg.det(opt_structure.get_cell())
    dpos = np.abs(orig_pos - opt_pos)
    dpos[dpos > 0.5] -= 1
    np.testing.assert_almost_equal(dpos, np.zeros_like(dpos), decimal=3)
    np.testing.assert_almost_equal(opt_vol, v0, decimal=3)


def test_special_atomic_optimization_only(calculator):
    atoms = bulk("Al", "fcc", a=2.025 * 2, cubic=True)
    atoms.calc = calculator
    optimizer = SpecialOptimizer(atoms, optimize_atoms_only=True)

    orig_pos = atoms.get_scaled_positions()
    orig_vol = np.linalg.det(atoms.get_cell())

    deformed_atoms = atoms.copy()
    pos = deformed_atoms.get_scaled_positions()
    pos[0, 0] += 0.1

    deformed_atoms.set_scaled_positions(pos)
    optimizer.run(deformed_atoms, calculator)
    opt_structure = optimizer.optimized_structure
    opt_pos = opt_structure.get_scaled_positions()
    opt_pos = opt_pos - opt_pos[0]

    opt_vol = np.linalg.det(opt_structure.get_cell())
    dpos = np.abs(orig_pos - opt_pos)
    dpos[dpos > 0.5] -= 1
    np.testing.assert_almost_equal(dpos, np.zeros_like(dpos), decimal=3)
    np.testing.assert_almost_equal(opt_vol, orig_vol, decimal=3)


def test_special_to_from_dict(calculator):
    atoms = bulk("Al", "fcc", a=2.025 * 2, cubic=True)
    atoms.calc = calculator
    special_optimizer = SpecialOptimizer(atoms)

    special_optimizer.run()
    opt_dict = special_optimizer.todict()

    opt_dict_json = json.dumps(opt_dict, cls=JsonNumpyEncoder)
    opt_dict = json.loads(opt_dict_json)
    new_opt = SpecialOptimizer.fromdict(opt_dict)

    assert special_optimizer.structure == new_opt.structure
    assert set(special_optimizer.value) == set(new_opt.value)
    assert special_optimizer.value["energy"] == new_opt.value["energy"]


def test_stepwise_atomic_optimization(calculator):
    atoms = bulk("Al", "fcc", a=2.025 * 2, cubic=True)
    atoms.calc = calculator
    stepwise_optimizer = StepwiseOptimizer(atoms)

    orig_pos = atoms.get_scaled_positions()

    deformed_atoms = atoms.copy()
    pos = deformed_atoms.get_scaled_positions()
    pos[0, 0] += 0.1

    deformed_atoms.set_scaled_positions(pos)
    stepwise_optimizer.run(deformed_atoms, calculator)
    opt_structure = stepwise_optimizer.optimized_structure
    opt_pos = opt_structure.get_scaled_positions()
    opt_pos = opt_pos - opt_pos[0]
    dpos = np.abs(orig_pos - opt_pos)
    dpos[dpos > 0.5] -= 1
    np.testing.assert_almost_equal(dpos, np.zeros_like(dpos), decimal=3)
    opt_vol = np.linalg.det(opt_structure.get_cell())
    np.testing.assert_almost_equal(opt_vol, v0, decimal=3)


def test_stepwise_to_from_dict(calculator):
    atoms = bulk("Al", "fcc", a=2.025 * 2, cubic=True)
    atoms.calc = calculator
    stepwise_optimizer = StepwiseOptimizer(atoms)

    stepwise_optimizer.run()
    opt_dict = stepwise_optimizer.todict()

    opt_dict_json = json.dumps(opt_dict, cls=JsonNumpyEncoder)
    opt_dict = json.loads(opt_dict_json)
    new_opt = StepwiseOptimizer.fromdict(opt_dict)

    assert stepwise_optimizer.structure == new_opt.structure
    assert set(stepwise_optimizer.value) == set(new_opt.value)
    assert stepwise_optimizer.value["energy"] == new_opt.value["energy"]
