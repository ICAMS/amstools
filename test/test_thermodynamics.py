import pytest
import pandas as pd
import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT

from amstools.thermodynamics import (
    compdict_to_comptuple,
    comptuple_to_str,
    compute_compositions,
    extract_elements,
    compute_formation_energy,
    compute_convexhull_dist,
    compute_corrected_energy,
    ensure_energy_per_atom_column,
    run_convex_hull_calculation,
    plot_convex_hull,
)



@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    calc = EMT()
    structures = [
        bulk("Cu", "fcc", a=3.6),
        bulk("Ni", "fcc", a=3.52),
        bulk("Al", "fcc", a=4.05),
    ]

    at = bulk("Cu", "fcc", a=3.56, cubic=True)
    at[0].symbol = "Ni"
    structures.append(at)

    at = bulk("Cu", "fcc", a=3.56, cubic=True)
    at[0].symbol = "Al"
    structures.append(at)

    at = bulk("Ni", "fcc", a=3.56, cubic=True)
    at[0].symbol = "Al"
    structures.append(at)

    data = []
    for atoms in structures:
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        data.append(
            {
                "ase_atoms": atoms,
                "energy": energy,
            }
        )

    df = pd.DataFrame(data)
    return df


def test_compdict_to_comptuple():
    comp_dict = {"H": 2, "O": 1}
    expected = (("H", 2 / 3), ("O", 1 / 3))
    assert compdict_to_comptuple(comp_dict) == expected


def test_comptuple_to_str():
    comp_tuple = (("H", 0.6666), ("O", 0.3333))
    expected = "H_0.667 O_0.333"
    assert comptuple_to_str(comp_tuple) == expected


def test_ensure_energy_per_atom_column(sample_df):
    """Test that the energy_per_atom column is created if it doesn't exist."""
    df = sample_df.copy()
    assert "energy_per_atom" not in df.columns
    compute_compositions(df)  # To get NUMBER_OF_ATOMS
    ensure_energy_per_atom_column(df)
    assert "energy_per_atom" in df.columns
    assert np.allclose(
        df["energy_per_atom"][0], df["energy"][0] / df["NUMBER_OF_ATOMS"][0]
    )

    # Test that it does nothing if the column already exists
    df2 = df.copy()
    ensure_energy_per_atom_column(df2)
    assert df.equals(df2)


def test_extract_elements(sample_df):
    compute_compositions(sample_df)
    elements = extract_elements(sample_df)
    assert sorted(elements) == ["Al", "Cu", "Ni"]


def test_compute_compositions(sample_df):
    elements = compute_compositions(sample_df)
    assert "comp_dict" in sample_df.columns
    assert "NUMBER_OF_ATOMS" in sample_df.columns
    assert "comp_tuple" in sample_df.columns
    assert "n_Cu" in sample_df.columns
    assert "c_Cu" in sample_df.columns
    assert sample_df["n_Cu"][0] == 1
    assert sample_df["c_Ni"][1] == 1
    assert sample_df["n_Cu"][3] == 3
    assert sample_df["n_Ni"][3] == 1
    assert sample_df["c_Cu"][3] == 0.75
    assert sample_df["c_Ni"][3] == 0.25
    assert sorted(elements) == ["Al", "Cu", "Ni"]


def test_compute_formation_energy(sample_df):
    compute_compositions(sample_df)
    sample_df["energy_per_atom"] = sample_df["energy"] / sample_df["NUMBER_OF_ATOMS"]
    # get ground state energies for pure elements from the sample_df
    epa_gs_dict = {
        "Cu": sample_df["energy"][0] / len(sample_df["ase_atoms"][0]),
        "Ni": sample_df["energy"][1] / len(sample_df["ase_atoms"][1]),
        "Al": sample_df["energy"][2] / len(sample_df["ase_atoms"][2]),
    }
    compute_formation_energy(sample_df, epa_gs_dict=epa_gs_dict)
    assert "e_formation_per_atom" in sample_df.columns
    # For pure elements, formation energy should be 0
    assert np.allclose(sample_df["e_formation_per_atom"][:3], 0)

    # Check formation energy for CuNi
    e_cu_ni = sample_df["energy"][3]
    epa_cuni = sample_df["energy_per_atom"][3]
    e_form_cu_ni = epa_cuni - (0.75 * epa_gs_dict["Cu"] + 0.25 * epa_gs_dict["Ni"])
    assert np.allclose(sample_df["e_formation_per_atom"][3], e_form_cu_ni)


def test_compute_convexhull_dist(sample_df):
    compute_compositions(sample_df)
    sample_df["energy_per_atom"] = sample_df["energy"] / sample_df["NUMBER_OF_ATOMS"]
    elements = compute_convexhull_dist(sample_df)
    assert "e_chull_dist_per_atom" in sample_df.columns
    assert sorted(elements) == ["Al", "Cu", "Ni"]
    # For pure elements, convex hull distance should be 0
    assert np.allclose(sample_df["e_chull_dist_per_atom"][:3], 0)


def test_compute_corrected_energy(sample_df):
    # Mocking SINGLE_ATOM_ENERGY_DICT for this test
    from amstools.thermodynamics import SINGLE_ATOM_ENERGY_DICT

    calculator_name = "EMT"
    SINGLE_ATOM_ENERGY_DICT[calculator_name] = {
        "Cu": -0.1,
        "Ni": -0.2,
        "Al": -0.3,
    }

    compute_compositions(sample_df)
    compute_corrected_energy(sample_df, calculator_name=calculator_name)
    assert "energy_corrected" in sample_df.columns
    assert "energy_corrected_per_atom" in sample_df.columns

    expected_ec_cu = (
        sample_df["energy"][0] - SINGLE_ATOM_ENERGY_DICT[calculator_name]["Cu"]
    )
    np.testing.assert_allclose(sample_df["energy_corrected"][0], expected_ec_cu)

    # Check for Cu3Ni1
    expected_ec_cuni = sample_df["energy"][3] - (
        3 * SINGLE_ATOM_ENERGY_DICT[calculator_name]["Cu"]
        + 1 * SINGLE_ATOM_ENERGY_DICT[calculator_name]["Ni"]
    )
    np.testing.assert_allclose(sample_df["energy_corrected"][3], expected_ec_cuni)


def test_run_convex_hull_calculation():
    """Test run_convex_hull_calculation with multiple structures."""
    calc = EMT()

    # Create structure dictionary with pure elements and alloys
    cu_fcc = bulk("Cu", "fcc", a=3.6)
    ni_fcc = bulk("Ni", "fcc", a=3.52)

    # Create alloy structure
    cu_ni_alloy = bulk("Cu", "fcc", a=3.56, cubic=True)
    cu_ni_alloy[0].symbol = "Ni"

    structure_dict = {
        "Cu_fcc": cu_fcc,
        "Ni_fcc": ni_fcc,
        "CuNi_alloy": cu_ni_alloy,
    }

    # Run the calculation
    
    result_df, pipeline_dict = run_convex_hull_calculation(
        structure_dict,
        calc=calc,
        murnaghan_kwargs={"volume_range": 0.05, "num_of_point": 5},
        verbose=False,
    )

    # Verify pipeline_dict was populated in place
    assert len(pipeline_dict) == 3
    assert "Cu_fcc" in pipeline_dict
    assert "Ni_fcc" in pipeline_dict
    assert "CuNi_alloy" in pipeline_dict

    # Verify all pipelines are finished
    for name, pipeline in pipeline_dict.items():
        assert pipeline.is_finished(), f"Pipeline {name} should be finished"

    # Verify DataFrame has expected columns
    assert "name" in result_df.columns
    assert "ase_atoms" in result_df.columns
    assert "energy_per_atom" in result_df.columns
    assert "e_chull_dist_per_atom" in result_df.columns

    # Verify DataFrame has expected rows
    assert len(result_df) == 3

    # Verify convex hull distances are non-negative
    assert (result_df["e_chull_dist_per_atom"] >= -1e-10).all()



def test_run_convex_hull_calculation_with_existing_pipeline():
    """Test that existing finished pipelines are not re-run."""
    calc = EMT()

    # Start with pure Cu
    structure_dict = {
        "Cu_fcc": bulk("Cu", "fcc", a=3.6),
    }

    # First run
    result_df, pipeline_dict = run_convex_hull_calculation(
        structure_dict,
        calc=calc,
        murnaghan_kwargs={"volume_range": 0.05, "num_of_point": 5},
        verbose=False,
    )

    assert len(pipeline_dict) == 1
    original_pipeline = pipeline_dict["Cu_fcc"]
    original_energy = result_df["energy_per_atom"].iloc[0]

    # Add Ni and a CuNi alloy to have valid convex hull data
    ni_fcc = bulk("Ni", "fcc", a=3.52)
    cu_ni_alloy = bulk("Cu", "fcc", a=3.56, cubic=True)
    cu_ni_alloy[0].symbol = "Ni"

    structure_dict["Ni_fcc"] = ni_fcc
    structure_dict["CuNi_alloy"] = cu_ni_alloy

    # Second run - should not re-run Cu_fcc
    result_df_2, pipeline_dict = run_convex_hull_calculation(
        structure_dict,
        calc=calc,
        pipeline_dict=pipeline_dict,
        murnaghan_kwargs={"volume_range": 0.05, "num_of_point": 5},
        verbose=False,
    )

    # Verify Cu_fcc pipeline is the same object (not re-created)
    assert pipeline_dict["Cu_fcc"] is original_pipeline

    # Verify all structures are in the new DataFrame
    assert len(result_df_2) == 3
    assert len(pipeline_dict) == 3


def test_plot_convex_hull(sample_df):
    """Test plot_convex_hull with a sample DataFrame."""
    df = sample_df.copy()
    
    # Ensure all required columns are present
    compute_convexhull_dist(df, verbose=False)
    
    # Test single DataFrame
    axes_dict = plot_convex_hull(df)
    
    # We expect 3 binary pairs from Al, Cu, Ni: (Al, Cu), (Al, Ni), (Cu, Ni)
    assert len(axes_dict) == 3
    for pair, ax in axes_dict.items():
        assert pair in [("Al", "Cu"), ("Al", "Ni"), ("Cu", "Ni")]
        assert ax is not None

    # Test dictionary with multiple DataFrames and styles
    dfs = {"set1": df, "set2": df}
    style_dict = {
        "set1": {"color": "red", "linestyle": "--", "label": "Set 1"},
        "set2": {"color": "blue", "linestyle": "-", "label": "Set 2"}
    }
    axes_dict_2 = plot_convex_hull(dfs, style_dict=style_dict, plot_all_structures=True)
    assert len(axes_dict_2) == 3
