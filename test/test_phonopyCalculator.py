import json
import numpy as np
import pytest

from amstools import PhonopyCalculator
from amstools.utils import JsonNumpyEncoder
from test.utils import atoms, calculator


@pytest.fixture
def phon(atoms, calculator):
    atoms.calc = calculator
    return PhonopyCalculator(atoms, q_mesh=5)


def test_create_phonopy(phon):
    phon.create_phonopy()
    assert list(phon._value["phonopy_data"]["supercell_range"]) == [4.0, 4.0, 4.0]
    assert list(phon._value["phonopy_data"]["supercell_matrix"][0]) == [4.0, 0.0, 0.0]
    assert pytest.approx(phon._value["phonopy_data"]["num_of_supercell"]) == 1

    displ_dataset = json.loads(phon._value["phonopy_data"]["displ_dataset"])
    assert displ_dataset["natom"] == 64
    assert displ_dataset["first_atoms"][0]["number"] == 0
    assert displ_dataset["first_atoms"][0]["displacement"] == [
        0.0,
        0.0070710678118654745,
        0.0070710678118654745,
    ]


def test_generate_structures(phon):
    gen_structures = phon.generate_structures()
    assert "supercell_phonon_0" in gen_structures


def test_subjob_name(phon):
    assert phon.subjob_name(0) == "supercell_phonon_0"


def test_analyse_structures(phon):
    phon_output_dict = {
        "supercell_phonon_0": np.array(
            [
                [-5.41233725e-16, -2.26832874e-02, -2.26832874e-02],
                [-1.80758186e-15, 6.57850515e-03, 6.57850515e-03],
                [1.94289029e-16, 9.92527774e-07, 9.92527773e-07],
                [-5.73499581e-15, 6.31573603e-03, 6.31573603e-03],
                [3.33629897e-03, -1.80747646e-04, 3.15767209e-03],
                [-9.81029924e-05, -7.76850968e-05, -1.75864417e-04],
                [9.61669480e-05, -1.72898651e-04, -7.66628632e-05],
                [-3.27651151e-03, 3.12082625e-03, -1.57950548e-04],
                [5.35133061e-07, -5.94517412e-08, 5.00285507e-07],
                [-1.52111950e-07, 2.00304619e-05, 1.98790366e-05],
                [-5.35133057e-07, 5.00285507e-07, -5.94517430e-08],
                [1.52111950e-07, 1.98790366e-05, 2.00304619e-05],
                [3.27651151e-03, -1.57950548e-04, 3.12082625e-03],
                [-3.33629897e-03, 3.15767209e-03, -1.80747646e-04],
                [9.81029924e-05, -1.75864417e-04, -7.76850968e-05],
                [-9.61669480e-05, -7.66628632e-05, -1.72898651e-04],
                [3.33629897e-03, 3.15767209e-03, -1.80747646e-04],
                [-9.81029924e-05, -1.75864417e-04, -7.76850968e-05],
                [9.61669480e-05, -7.66628632e-05, -1.72898651e-04],
                [-3.27651151e-03, -1.57950548e-04, 3.12082625e-03],
                [-1.30954956e-04, -4.41529072e-05, -4.41529072e-05],
                [-1.79457890e-09, 8.93082954e-08, 8.93082954e-08],
                [1.29569229e-04, -4.39362569e-05, -4.39362569e-05],
                [1.00608278e-06, 1.03472701e-04, 1.03472701e-04],
                [-3.31381194e-05, 5.29405058e-05, -4.42102073e-05],
                [3.31226444e-05, -4.40257169e-05, 5.32305074e-05],
                [-2.57574967e-07, 2.71754451e-07, -4.04104824e-08],
                [2.49080647e-07, -4.14359144e-08, 2.78973559e-07],
                [7.09069915e-16, -1.78552747e-04, -1.55475359e-04],
                [2.08169358e-16, -5.54866082e-04, 1.04784863e-04],
                [-7.07767178e-16, 5.15529962e-07, 5.41404230e-07],
                [3.05700120e-16, 1.02177197e-04, -5.52933266e-04],
                [5.35133060e-07, 5.00285506e-07, -5.94517385e-08],
                [-1.52111948e-07, 1.98790366e-05, 2.00304619e-05],
                [-5.35133056e-07, -5.94517422e-08, 5.00285509e-07],
                [1.52111951e-07, 2.00304619e-05, 1.98790366e-05],
                [-3.31381194e-05, -4.42102073e-05, 5.29405058e-05],
                [3.31226444e-05, 5.32305074e-05, -4.40257169e-05],
                [-2.57574965e-07, -4.04104821e-08, 2.71754453e-07],
                [2.49080647e-07, 2.78973558e-07, -4.14359141e-08],
                [1.39341070e-15, -3.51216371e-08, -3.51216381e-08],
                [4.08700851e-15, 2.13040519e-08, 2.13040505e-08],
                [-2.01574868e-15, 2.06317090e-08, 2.06317049e-08],
                [-1.18308141e-15, 2.27921147e-08, 2.27921069e-08],
                [-3.31226444e-05, -4.40257169e-05, 5.32305074e-05],
                [2.57574964e-07, 2.71754449e-07, -4.04104799e-08],
                [-2.49080638e-07, -4.14359139e-08, 2.78973560e-07],
                [3.31381194e-05, 5.29405058e-05, -4.42102073e-05],
                [3.27651151e-03, 3.12082625e-03, -1.57950548e-04],
                [-3.33629897e-03, -1.80747646e-04, 3.15767209e-03],
                [9.81029924e-05, -7.76850968e-05, -1.75864417e-04],
                [-9.61669480e-05, -1.72898651e-04, -7.66628632e-05],
                [4.67944199e-16, -1.55475359e-04, -1.78552747e-04],
                [1.42247579e-15, 1.04784863e-04, -5.54866082e-04],
                [3.11209392e-15, 5.41404229e-07, 5.15529964e-07],
                [1.29102305e-15, -5.52933266e-04, 1.02177197e-04],
                [-3.31226444e-05, 5.32305074e-05, -4.40257169e-05],
                [2.57574961e-07, -4.04104798e-08, 2.71754449e-07],
                [-2.49080652e-07, 2.78973560e-07, -4.14359167e-08],
                [3.31381194e-05, -4.42102073e-05, 5.29405058e-05],
                [-1.29569229e-04, -4.39362569e-05, -4.39362569e-05],
                [-1.00608277e-06, 1.03472701e-04, 1.03472701e-04],
                [1.30954956e-04, -4.41529072e-05, -4.41529072e-05],
                [1.79458163e-09, 8.93082984e-08, 8.93082990e-08],
            ]
        )
    }
    phon.analyse_structures(phon_output_dict)
    assert "phonopy_data" in phon._value
    assert "dos_total" in phon._value
    assert "dos_energies" in phon._value


def test__compute_dos_calculate(phon):
    phon.calculate()
    phon._compute_dos()
    assert "dos_total" in phon._value
    assert "dos_energies" in phon._value


def test__compute_dos_run(phon):
    phon.calculate()
    phon._compute_dos()
    assert "dos_total" in phon._value
    assert "dos_energies" in phon._value


def test_get_structure_value(phon, atoms):
    displ_atoms = atoms.copy()
    displ_atoms.calc = phon.calculator
    displ_atoms *= 2
    displ_atoms.positions[0, 0] += 0.01
    res, structure = phon.get_structure_value(displ_atoms)
    assert pytest.approx(res[0][0], abs=1e-6) == -3.20746068e-02


def test_phonopy(phon):
    phon.calculate()
    assert phon._phonopy is not None


def test_to_from_dict(phon):
    phon.calculate()
    prop_dict = phon.todict()
    prop_dict_json = json.dumps(prop_dict, cls=JsonNumpyEncoder)
    prop_dict = json.loads(prop_dict_json)
    new_prop = PhonopyCalculator.fromdict(prop_dict)

    assert phon.basis_ref == new_prop.basis_ref
    assert pytest.approx(list(phon.value["dos_total"])) == new_prop.value["dos_total"]
    assert (
        pytest.approx(list(phon.value["dos_energies"]))
        == new_prop.value["dos_energies"]
    )
    assert new_prop.phonopy is not None

    phonopy = phon.phonopy
    phonopy.run_mesh([10, 10, 10])
    phonopy.run_total_dos()

    new_phonopy = new_prop.phonopy
    new_phonopy.run_mesh([10, 10, 10])
    new_phonopy.run_total_dos()
    dos = phonopy.get_total_dos_dict()
    tot_dos_res = dos["frequency_points"], dos["total_dos"]

    dos = new_phonopy.get_total_dos_dict()
    new_tot_dos_res = dos["frequency_points"], dos["total_dos"]

    assert np.allclose(np.array(tot_dos_res[0]), np.array(new_tot_dos_res[0]))
    assert np.allclose(np.array(tot_dos_res[1]), np.array(new_tot_dos_res[1]))


def test_is_symmetry(atoms, calculator):
    phon = PhonopyCalculator(atoms)
    phon.calculator = calculator
    gen_structures = phon.generate_structures()
    assert len(gen_structures) == 1

    phon = PhonopyCalculator(atoms, is_symmetry=False)
    phon.calculator = calculator
    gen_structures = phon.generate_structures()
    assert len(gen_structures) == 6


def test_initial_charges(atoms, calculator):
    atoms = atoms.copy()
    atoms.set_initial_charges([+1] * len(atoms))
    phon = PhonopyCalculator(atoms)
    phon.calculator = calculator
    structures_dict = phon.generate_structures()
    for k, v in structures_dict.items():
        c = v.get_initial_charges()
        assert c is not None


def test_q_mesh_spacing(atoms, calculator):
    phon = PhonopyCalculator(atoms, q_mesh_spacing=1)
    phon.calculator = calculator
    phon.calculate()

    assert phon.q_mesh_spacing == 1
    assert phon.q_mesh == [3, 3, 3]
    assert pytest.approx(phon._value["dos_total"][14], abs=1e-6) == 0.1790342420052403


def test_plot_band_structure(phon):
    """
    Test the plot_band_structure method.
    This test verifies that the method runs without error and that plotting
    is performed on the provided axes object.
    """
    import matplotlib.pyplot as plt

    phon.calculate()

    # Define a simple path for FCC, e.g., Gamma -> X
    path = [[[0, 0, 0], [0.5, 0, 0.5]]]
    labels = [r"$\Gamma$", "X"]

    fig, ax = plt.subplots()
    assert len(ax.get_lines()) == 0  # Initially no lines on the plot

    phon.plot_band_structure(path=path, labels=labels)

    assert len(ax.get_lines()) > 0  # Check that something was plotted
    plt.close(fig)  # Close the figure to avoid displaying it during tests
