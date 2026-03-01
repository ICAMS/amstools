import pytest
import pandas as pd
import numpy as np
from ase import Atoms

from amstools.analyze import (
    atoms_to_dataframe,
    compute_atomic_distribution,
    compute_rdf,
)


@pytest.fixture
def sample_atoms():
    """A sample ASE Atoms object for testing."""
    # A simple cubic structure of Si and C atoms
    atoms = Atoms(
        "Si2C2",
        positions=[[0, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]],
        cell=[1, 1, 1],
        pbc=True,
    )
    return atoms


class TestAtomsToDataFrame:
    def test_conversion(self, sample_atoms):
        """Test standard conversion of an Atoms object to a DataFrame."""
        df = atoms_to_dataframe(sample_atoms)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["symbol", "x", "y", "z"]
        assert len(df) == 4
        assert df["symbol"].tolist() == ["Si", "Si", "C", "C"]
        np.testing.assert_array_almost_equal(
            df[["x", "y", "z"]].values, sample_atoms.get_positions()
        )

    # def test_empty_atoms(self):
    #     """Test conversion with an empty Atoms object."""
    #     empty_atoms = Atoms()
    #     df = atoms_to_dataframe(empty_atoms)
    #     assert isinstance(df, pd.DataFrame)
    #     assert len(df) == 0
    #     assert list(df.columns) == ['symbol', 'x', 'y', 'z']

    def test_none_input(self, capsys):
        """Test that None input is handled gracefully."""
        df = atoms_to_dataframe(None)
        assert df is None
        captured = capsys.readouterr()
        assert "Input is not a valid ASE Atoms object" in captured.err


class TestComputeAtomicDistribution:
    @pytest.fixture
    def sample_df(self, sample_atoms):
        """A sample DataFrame created from the sample_atoms fixture."""
        return atoms_to_dataframe(sample_atoms)

    def test_all_atoms(self, sample_df):
        """Test distribution computation for all atoms combined."""
        dist_data = compute_atomic_distribution(sample_df, axis="z", bins=2)
        assert isinstance(dist_data, dict)
        assert "All Atoms" in dist_data
        data = dist_data["All Atoms"]
        assert "values" in data
        assert "bin_centers" in data
        assert "bin_width" in data
        assert data["values"].sum() == 4
        np.testing.assert_array_equal(data["values"], [2, 2])

    def test_specific_elements(self, sample_df):
        """Test distribution for a specific list of elements."""
        dist_data = compute_atomic_distribution(
            sample_df, elements=["Si", "C"], axis="x", bins=2
        )
        assert "Si" in dist_data
        assert "C" in dist_data
        assert dist_data["Si"]["values"].sum() == 2
        assert dist_data["C"]["values"].sum() == 2
        np.testing.assert_array_equal(dist_data["Si"]["values"], [1, 1])
        np.testing.assert_array_equal(dist_data["C"]["values"], [1, 1])

    def test_elements_all_keyword(self, sample_df):
        """Test distribution computation with elements='all'."""
        dist_data = compute_atomic_distribution(
            sample_df, elements="all", axis="y", bins=2
        )
        assert "Si" in dist_data
        assert "C" in dist_data
        assert "All Atoms" not in dist_data
        assert dist_data["Si"]["values"].sum() == 2
        assert dist_data["C"]["values"].sum() == 2
        np.testing.assert_array_equal(dist_data["Si"]["values"], [1, 1])
        np.testing.assert_array_equal(dist_data["C"]["values"], [1, 1])

    def test_nonexistent_element(self, sample_df, capsys):
        """Test with an element not present in the DataFrame."""
        dist_data = compute_atomic_distribution(sample_df, elements=["Fe"])
        assert "Fe" not in dist_data
        assert len(dist_data) == 0
        captured = capsys.readouterr()
        assert "No atoms of type 'Fe' found" in captured.err

    def test_invalid_axis(self, sample_df, capsys):
        """Test with an invalid axis argument."""
        dist_data = compute_atomic_distribution(sample_df, axis="w")
        assert dist_data == {}
        captured = capsys.readouterr()
        assert "Invalid axis 'w'" in captured.err

    def test_empty_dataframe(self, capsys):
        """Test with an empty DataFrame."""
        empty_df = pd.DataFrame(columns=["symbol", "x", "y", "z"])
        dist_data = compute_atomic_distribution(empty_df)
        assert dist_data == {}
        captured = capsys.readouterr()
        assert "Input DataFrame is empty" in captured.err

    def test_none_dataframe(self, capsys):
        """Test with None as input."""
        dist_data = compute_atomic_distribution(None)
        assert dist_data == {}
        captured = capsys.readouterr()
        assert "Input DataFrame is empty" in captured.err

    def test_bin_centers_and_width(self, sample_df):
        """Verify the calculation of bin centers and width."""
        dist_data = compute_atomic_distribution(sample_df, axis="z", bins=2)
        data = dist_data["All Atoms"]

        # For z-axis data [0, 0, 0.5, 0.5] and bins=2,
        # bin edges should be [0, 0.25, 0.5]
        expected_bin_edges = np.array([0.0, 0.25, 0.5])
        expected_bin_centers = (expected_bin_edges[:-1] + expected_bin_edges[1:]) / 2
        expected_bin_width = expected_bin_edges[1] - expected_bin_edges[0]

        np.testing.assert_allclose(data["bin_centers"], expected_bin_centers)
        assert data["bin_width"] == pytest.approx(expected_bin_width)

    def test_concentration_option(self, sample_df):
        """Test the concentration calculation."""
        # For z-axis with 2 bins:
        # Bin 1 (0.0 to 0.25) has 2 Si atoms. Total atoms = 2. Conc Si = 2/2=1.0, Conc C = 0/2=0.0
        # Bin 2 (0.25 to 0.5) has 2 C atoms. Total atoms = 2. Conc Si = 0/2=0.0, Conc C = 2/2=1.0
        dist_data = compute_atomic_distribution(
            sample_df, axis="z", elements=["Si", "C"], bins=2, concentration=True
        )

        assert "Si" in dist_data
        assert "C" in dist_data

        si_data = dist_data["Si"]
        c_data = dist_data["C"]

        np.testing.assert_array_almost_equal(si_data["values"], [1.0, 0.0])
        np.testing.assert_array_almost_equal(c_data["values"], [0.0, 1.0])


class TestComputeRDF:
    def test_single_frame(self, sample_atoms):
        """Test RDF computation for a single frame with a specific pair."""
        rdf_data = compute_rdf(
            sample_atoms, element_pairs=[("Si", "C")], max_range=5, nbins=10
        )
        assert isinstance(rdf_data, dict)
        # Key should be a sorted tuple
        assert ("C", "Si") in rdf_data
        data = rdf_data[("C", "Si")]
        assert "r" in data
        assert "g_r" in data
        assert "g_r_std" not in data  # No std for a single frame
        assert len(data["r"]) == 10
        assert len(data["g_r"]) == 10
        # assert that r is lower than max_range
        assert all(r <= 5 for r in data["r"])
        # Check that there's a peak around the known distance
        # Si-C distance is sqrt(0.5) ~= 0.707
        peak_index = np.argmax(data["g_r"])
        assert data["r"][peak_index] == pytest.approx(1.75, abs=0.1)

    def test_trajectory(self, sample_atoms):
        """Test RDF computation for a trajectory (list of frames)."""
        # Create a small trajectory by slightly modifying the sample
        frame1 = sample_atoms.copy()
        frame2 = sample_atoms.copy()
        frame2.positions[0, 0] += 0.05  # Perturb one atom
        trajectory = [frame1, frame2]

        rdf_data = compute_rdf(
            trajectory, element_pairs=[("Si", "Si")], max_range=5, nbins=10
        )
        assert ("Si", "Si") in rdf_data
        data = rdf_data[("Si", "Si")]
        assert "r" in data
        assert "g_r" in data
        assert "g_r_std" in data  # Standard deviation should be present
        assert len(data["r"]) == 10
        assert len(data["g_r"]) == 10
        assert len(data["g_r_std"]) == 10

    def test_auto_pair_generation(self, sample_atoms):
        """Test RDF computation when pairs are generated from a list of elements."""
        rdf_data = compute_rdf(
            sample_atoms, element_pairs=["Si", "C"], max_range=5, nbins=10
        )
        assert isinstance(rdf_data, dict)
        assert ("C", "Si") in rdf_data
        assert ("Si", "Si") in rdf_data
        assert ("C", "C") in rdf_data
        assert len(rdf_data) == 3

    def test_default_element_pairs_is_none(self, sample_atoms):
        """Test RDF computation when element_pairs is None (default)."""
        rdf_data = compute_rdf(sample_atoms, max_range=5, nbins=10)
        assert isinstance(rdf_data, dict)
        assert ("C", "Si") in rdf_data
        assert ("Si", "Si") in rdf_data
        assert ("C", "C") in rdf_data
        assert len(rdf_data) == 3

    def test_empty_input(self, capsys):
        """Test RDF computation with an empty list of frames."""
        rdf_data = compute_rdf([], element_pairs=[("A", "B")])
        assert rdf_data == {}
        # The function currently doesn't print for empty list, which is fine.

    def test_no_pairs_found(self, sample_atoms, capsys):
        """Test RDF computation when the requested element pair does not exist."""
        rdf_data = compute_rdf(sample_atoms, element_pairs=[("Fe", "O")])
        assert rdf_data == {}
        # The function currently doesn't print a warning here, but returns empty dict.

    def test_empty_list_for_element_pairs(self, sample_atoms, capsys):
        """Test behavior when element_pairs is an empty list."""
        rdf_data = compute_rdf(sample_atoms, element_pairs=[])
        assert rdf_data == {}
        captured = capsys.readouterr()
        assert "No element pairs provided" in captured.err
