import json

from amstools import ThermodynamicQHACalculator, MurnaghanCalculator
from amstools.utils import JsonNumpyEncoder
from test.utils import *


@pytest.fixture
def tqha(atoms, pipeline_calculator_zero_stress):
    atoms.calc = pipeline_calculator_zero_stress
    return ThermodynamicQHACalculator(
        atoms,
        supercell_range=2,
        num_of_point=4,
        fit_order=2,
        q_space_sample=25,
    )


def test_generate_structures(tqha):
    res = tqha.generate_structures()
    assert "murnaghan" in res


def test_generate_structures_with_murn(atoms, pipeline_calculator_zero_stress):
    atoms.calc = pipeline_calculator_zero_stress
    murn = MurnaghanCalculator(atoms, optimize_deformed_structure=True)
    murn.calculate()
    tqha = ThermodynamicQHACalculator(
        atoms,
        supercell_range=2,
        num_of_point=4,
        fit_order=2,
        q_space_sample=5,
        murnaghan=murn,
    )
    res = tqha.generate_structures()
    assert "phonopy_vol_0" in res
    assert "phonopy_vol_1" in res
    assert "phonopy_vol_2" in res
    assert "phonopy_vol_3" in res


def test_subjob_name(tqha):
    assert tqha.subjob_name(0) == "phonopy_vol_0"


def test_get_structure_value(tqha, atoms):
    murn = MurnaghanCalculator(atoms)
    res_murn, structure = tqha.get_structure_value(murn)
    assert murn is res_murn
    assert "volume" in res_murn._value
    assert "energy" in res_murn._value
    assert "equilibrium_energy" in res_murn._value
    assert "equilibrium_volume" in res_murn._value
    assert "equilibrium_bulk_modulus" in res_murn._value
    assert "equilibrium_b_prime" in res_murn._value
    assert "energy_rms" in res_murn._value


def test_calculate(tqha):
    tqha.calculate()
    assert tqha.energies == tqha._value["energy"]
    assert tqha.volumes == tqha._value["volume"]
    assert "volume" in tqha._value
    assert "energy" in tqha._value
    assert "G_QHA" in tqha._value
    assert "V_QHA" in tqha._value
    assert "B_QHA" in tqha._value
    assert "beta_QHA" in tqha._value
    assert "Cp_QHA" in tqha._value

    dec = 3
    np.testing.assert_almost_equal(
        tqha._value["G_QHA"][1][1], 2.8318304091684467, decimal=dec
    )
    np.testing.assert_almost_equal(
        tqha._value["V_QHA"][1][1], 16.22586122307998, decimal=dec
    )
    np.testing.assert_almost_equal(
        tqha._value["B_QHA"][1][1], 35.60883831715906, decimal=dec
    )
    np.testing.assert_almost_equal(
        tqha._value["beta_QHA"][1][1], 1.9606019058818492e-07, decimal=dec
    )
    np.testing.assert_almost_equal(
        tqha._value["Cp_QHA"][1][1], 0.000894179265742423, decimal=dec
    )


def test_to_from_dict(tqha):
    prop = tqha
    prop.calculate()

    prop_dict = prop.todict()
    prop_dict_json = json.dumps(prop_dict, cls=JsonNumpyEncoder)
    prop_dict = json.loads(prop_dict_json)
    new_prop = ThermodynamicQHACalculator.fromdict(prop_dict)
    print("new_prop=", new_prop)
    print("new_prop.basis_ref=", new_prop.basis_ref)

    assert prop.basis_ref == new_prop.basis_ref
    assert set(prop.value) == set(new_prop.value)


def test_calculate_with_optimize_deformed_structure(
    atoms, pipeline_calculator_zero_stress
):
    atoms.calc = pipeline_calculator_zero_stress
    tqha = ThermodynamicQHACalculator(
        atoms,
        supercell_range=2,
        num_of_point=4,
        fit_order=2,
        q_space_sample=25,
        optimize_deformed_structure=True,
        fmax=0.1,
    )
    tqha.calculate(verbose=True)

    assert tqha.energies == tqha._value["energy"]
    assert tqha.volumes == tqha._value["volume"]
    assert "volume" in tqha._value
    assert "energy" in tqha._value
    assert "G_QHA" in tqha._value
    assert "V_QHA" in tqha._value
    assert "B_QHA" in tqha._value
    assert "beta_QHA" in tqha._value
    assert "Cp_QHA" in tqha._value

    dec = 3
    np.testing.assert_almost_equal(
        tqha._value["G_QHA"][1][1], 2.8318304091684467, decimal=dec
    )
    np.testing.assert_almost_equal(
        tqha._value["V_QHA"][1][1], 16.22586122307998, decimal=dec
    )
    np.testing.assert_almost_equal(
        tqha._value["B_QHA"][1][1], 35.60883831715906, decimal=dec
    )
    np.testing.assert_almost_equal(
        tqha._value["beta_QHA"][1][1], 1.9606019058818492e-07, decimal=dec
    )
    np.testing.assert_almost_equal(
        tqha._value["Cp_QHA"][1][1], 0.000894179265742423, decimal=dec
    )
