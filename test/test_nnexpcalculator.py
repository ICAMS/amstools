import json
import pytest

from amstools.properties.nnexpansion import NearestNeighboursExpansionCalculator
from amstools.utils import JsonNumpyEncoder
from test.utils import atoms, calculator


@pytest.fixture
def nnexp(atoms, calculator):
    atoms.calc = calculator
    return NearestNeighboursExpansionCalculator(atoms)


def test_NearestNeighboursExpansionCalculator_calculate(atoms, calculator):
    atoms.calc = calculator
    nnexp = NearestNeighboursExpansionCalculator(
        atoms, num_of_point=7, nn_distance_range=0.5
    )
    nnexp.calculate()

    assert "nn_distances" in nnexp._value
    assert "energy" in nnexp._value
    assert "gradient" in nnexp._value
    assert "nn_distances_step" in nnexp._value
    assert len(nnexp._value["nn_distances"]) == 7
    assert len(nnexp._value["energy"]) == 7


def test_NearestNeighboursExpansionCalculator_calculate_nn_dist_range(
    atoms, calculator
):
    atoms.calc = calculator
    nnexp = NearestNeighboursExpansionCalculator(
        atoms, num_of_point=7, nn_distance_range=[1, 5]
    )
    nnexp.calculate()

    assert "nn_distances" in nnexp._value
    assert "energy" in nnexp._value
    assert "gradient" in nnexp._value
    assert "nn_distances_step" in nnexp._value
    assert len(nnexp._value["nn_distances"]) == 7
    assert len(nnexp._value["energy"]) == 7
    assert nnexp._value["nn_distances"][0] == 1
    assert nnexp._value["nn_distances"][-1] == 5


def test_NearestNeighboursExpansionCalculator_calculate_nn_distance_list(
    atoms, calculator
):
    atoms.calc = calculator
    nnexp = NearestNeighboursExpansionCalculator(atoms, nn_distance_list=[1, 3])
    nnexp.calculate()

    assert "nn_distances" in nnexp._value
    assert "energy" in nnexp._value
    assert "gradient" in nnexp._value
    assert "nn_distances_step" in nnexp._value
    assert len(nnexp._value["nn_distances"]) == 2
    assert len(nnexp._value["energy"]) == 2
    assert nnexp._value["nn_distances"][0] == 1
    assert nnexp._value["nn_distances"][-1] == 3


def test_NearestNeighboursExpansionCalculator_return_upd_structure(atoms, calculator):
    atoms.calc = calculator
    nnexp = NearestNeighboursExpansionCalculator(atoms, return_min_structure=True)
    nnexp.calculate()

    assert "nn_distances" in nnexp._value
    assert "energy" in nnexp._value
    assert "gradient" in nnexp._value
    assert "nn_distances_step" in nnexp._value

    final_structure = nnexp.get_final_structure()
    emin = nnexp.value["energy"].min()
    assert pytest.approx(final_structure.get_potential_energy()) == emin


def test_to_from_dict(atoms, calculator):
    atoms.calc = calculator
    nnexp = NearestNeighboursExpansionCalculator(
        atoms, num_of_point=7, nn_distance_range=[1, 5]
    )
    nnexp.calculate()

    prop_dict = nnexp.todict()
    prop_dict_json = json.dumps(prop_dict, cls=JsonNumpyEncoder)
    prop_dict = json.loads(prop_dict_json)
    new_prop = NearestNeighboursExpansionCalculator.fromdict(prop_dict)

    assert nnexp.basis_ref == new_prop.basis_ref
    assert "nn_distances" in new_prop._value
    assert "energy" in new_prop._value
    assert "gradient" in new_prop._value
    assert "nn_distances_step" in new_prop._value
    assert len(new_prop._value["nn_distances"]) == 7
    assert len(new_prop._value["energy"]) == 7
    assert new_prop._value["nn_distances"][0] == 1
    assert new_prop._value["nn_distances"][-1] == 5
