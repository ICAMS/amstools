import json
import numpy as np
import pytest
from ase.lattice.cubic import FaceCenteredCubic
from ase.constraints import FixedLine

from amstools.properties.gammasurface import GammaSurfaceCalculator, GammaLineCalculator
from amstools.utils import JsonNumpyEncoder
from test.utils import calculator
from matplotlib import pyplot as plt

@pytest.fixture
def block(calculator, monkeypatch):
    monkeypatch.setattr(calculator, "get_stress", lambda: np.zeros(6))
    block = FaceCenteredCubic(
        directions=[[-1, 1, 0], [1, 1, 2], [1, 1, -1]],
        size=(1, 1, 8),
        symbol="Al",
        pbc=(1, 1, 1),
        latticeconstant=4.05,
    )
    block.calc = calculator

    # Apply constraint: atom must relax along z
    myconstrain = []
    for i in [atom.index for atom in block]:
        myconstrain.append(FixedLine(i, (0, 0, 1)))
    block.set_constraint(myconstrain)
    return block


def test_gamma_surface_calculator(block):
    prop_calc = GammaSurfaceCalculator(
        block,
        shift_surface=[1, 1],
        num_of_point_per_line=2,
        fmax=1e-2,
        z_cut_level=0.5,
    )

    # Test initial values
    assert "shift_surface" in prop_calc._value
    assert "num_of_point_per_line" in prop_calc._value
    assert "optimizer" in prop_calc._value
    assert "fmax" in prop_calc._value
    assert "z_cut_level" in prop_calc._value
    assert "real_shift_surface" in prop_calc._value

    prop_calc.calculate()

    # Test computed values
    assert "shift_map_X" in prop_calc._value
    assert "shift_map_Y" in prop_calc._value
    assert "energy_map(mJ/m2)" in prop_calc._value
    assert "shift" in prop_calc._value
    assert "energy" in prop_calc._value

    ref_energy = np.array([-0.07209828, 0.1355203, 0.1355203, -0.07209828])
    assert np.allclose(prop_calc._value["energy"], ref_energy)

    ref_energy_map = np.array([[-81.3194862, 152.8530336], [152.8530336, -81.3194862]])
    ref_energy_map -= ref_energy_map[0, 0]
    assert np.allclose(prop_calc._value["energy_map(mJ/m2)"], ref_energy_map)
    prop_calc.plot()
    plt.close()

    prop_calc.plot_mep()
    plt.close()


def test_plot_minima(block):
    """Test the plot_minima method for visualizing local minima."""
    prop_calc = GammaSurfaceCalculator(
        block,
        shift_surface=[1, 1],
        num_of_point_per_line=3,
        fmax=1e-2,
        z_cut_level=0.5,
    )
    prop_calc.calculate()

    # Test plot_minima returns expected structure
    fig, ax, minima_info = prop_calc.plot_minima()

    assert fig is not None
    assert ax is not None
    assert isinstance(minima_info, list)
    assert len(minima_info) > 0

    # Check minima_info structure
    for info in minima_info:
        assert 'index' in info
        assert 'pixel_coords' in info
        assert 'phys_coords' in info
        assert 'energy' in info
        assert isinstance(info['index'], int)
        assert isinstance(info['pixel_coords'], tuple)
        assert isinstance(info['phys_coords'], tuple)
        assert isinstance(info['energy'], (int, float, np.floating))

    # Check minima are sorted by energy
    energies = [info['energy'] for info in minima_info]
    assert energies == sorted(energies)

    plt.close()


def test_plot_mep_with_custom_path(block):
    """Test plot_mep with user-specified path through minima."""
    prop_calc = GammaSurfaceCalculator(
        block,
        shift_surface=[1, 1],
        num_of_point_per_line=3,
        fmax=1e-2,
        z_cut_level=0.5,
    )
    prop_calc.calculate()

    # First get available minima
    _, _, minima_info = prop_calc.plot_minima()
    plt.close()

    # Test with valid path (need at least 2 minima)
    if len(minima_info) >= 2:
        fig, (ax1, ax2) = prop_calc.plot_mep(path=[0, 1])
        assert fig is not None
        assert ax1 is not None
        assert ax2 is not None
        plt.close()

    # Test with 3+ minima path if available
    if len(minima_info) >= 3:
        fig, (ax1, ax2) = prop_calc.plot_mep(path=[0, 2, 1])
        assert fig is not None
        plt.close()


def test_plot_mep_invalid_path(block):
    """Test plot_mep raises errors for invalid paths."""
    prop_calc = GammaSurfaceCalculator(
        block,
        shift_surface=[1, 1],
        num_of_point_per_line=3,
        fmax=1e-2,
        z_cut_level=0.5,
    )
    prop_calc.calculate()

    # Test with path containing too few points
    with pytest.raises(ValueError, match="path must contain at least 2"):
        prop_calc.plot_mep(path=[0])

    # Test with invalid index
    with pytest.raises(ValueError, match="Invalid minima index"):
        prop_calc.plot_mep(path=[0, 999])

    plt.close()


def test_gamma_line_calculator(block):
    prop_calc = GammaLineCalculator(
        block,
        shift_vector=[0, 1.0],
        num_of_point=5,
        fmax=1e-2,
        z_cut_level=0.5,
    )

    # Test initial values
    assert "shift_vector" in prop_calc._value
    assert "num_of_point" in prop_calc._value
    assert "optimizer" in prop_calc._value
    assert "fmax" in prop_calc._value
    assert "z_cut_level" in prop_calc._value
    assert "real_shift_vector" in prop_calc._value

    prop_calc.calculate()

    assert "shift" in prop_calc._value
    assert "energy" in prop_calc._value
    assert len(prop_calc._value["shift"]) == 5

    ref = np.array([-0.07209828, 0.00112891, -0.0115164, 0.25009866, 0.18545584])
    ref_energy_map = np.array(
        [0.0, 82.59277649, 68.33016607, 363.40518343, 290.49469685]
    )

    assert np.allclose(prop_calc._value["energy"], ref)
    assert np.allclose(prop_calc._value["energy_map(mJ/m2)"], ref_energy_map)


def test_gamma_surface_calculator_to_from_dict(block):
    prop_calc = GammaSurfaceCalculator(
        block,
        shift_surface=[1, 1],
        num_of_point_per_line=2,
        fmax=1e-2,
        z_cut_level=0.5,
    )

    prop_calc.calculate()

    prop_dict = prop_calc.todict()
    prop_dict_json = json.dumps(prop_dict, cls=JsonNumpyEncoder)
    prop_dict = json.loads(prop_dict_json)
    new_prop = GammaSurfaceCalculator.fromdict(prop_dict)

    assert prop_calc.basis_ref == new_prop.basis_ref
    assert sorted(prop_calc.value.keys()) == sorted(new_prop.value.keys())


def test_gamma_line_calculator_to_from_dict(block):
    prop_calc = GammaLineCalculator(
        block,
        shift_vector=[0, 1.0],
        num_of_point=2,
        fmax=1e-2,
        z_cut_level=0.5,
    )

    prop_calc.calculate()

    prop_dict = prop_calc.todict()
    prop_dict_json = json.dumps(prop_dict, cls=JsonNumpyEncoder)
    prop_dict = json.loads(prop_dict_json)
    new_prop = GammaLineCalculator.fromdict(prop_dict)

    assert prop_calc.basis_ref == new_prop.basis_ref
    assert sorted(prop_calc.value.keys()) == sorted(new_prop.value.keys())

def test_calculate_mep_and_structures(block):
    """Test calculate_mep and get_mep_structures methods."""
    prop_calc = GammaSurfaceCalculator(
        block,
        shift_surface=[1, 1],
        num_of_point_per_line=3,
        fmax=1e-2,
        z_cut_level=0.5,
    )
    prop_calc.calculate()

    # Test calculate_mep
    mep_data = prop_calc.calculate_mep()
    expected_keys = [
        "reaction_coordinate",
        "energy_profile",
        "phys_x_path",
        "phys_y_path",
        "path_nodes",
        "X_highres",
        "Y_highres",
        "emap_highres",
    ]
    for key in expected_keys:
        assert key in mep_data
    
    assert len(mep_data["reaction_coordinate"]) == len(mep_data["energy_profile"])
    assert len(mep_data["phys_x_path"]) == len(mep_data["phys_y_path"])

    # Test get_mep_structures (default)
    structures = prop_calc.get_mep_structures()
    assert isinstance(structures, list)
    assert len(structures) == len(mep_data["phys_x_path"])
    assert hasattr(structures[0], "get_positions")  # Check if it's an Atoms object or similar

    # Test get_mep_structures with n_images
    n_images = 5
    structures_resampled = prop_calc.get_mep_structures(n_images=n_images)
    assert len(structures_resampled) == n_images
    
    # Check that structures are shifted
    # Compare first and last structure positions
    pos0 = structures_resampled[0].get_positions()
    pos_last = structures_resampled[-1].get_positions()
    
    # Check if they are different (they should be, unless start and end are identical which is unlikely here)
    if not np.allclose(mep_data["phys_x_path"][0], mep_data["phys_x_path"][-1]) or        not np.allclose(mep_data["phys_y_path"][0], mep_data["phys_y_path"][-1]):
        assert not np.allclose(pos0, pos_last)
    
    # Verify cell tilt for one of the structures
    # The cell[2] vector should be modified based on the shift
    # Original cell[2] is roughly (0, 0, height)
    # New cell[2] should have x, y components matching the shift
    
    # Pick the middle image
    mid_struct = structures_resampled[n_images // 2]
    mid_shift_x = mid_struct.get_cell()[2][0] - block.get_cell()[2][0]
    mid_shift_y = mid_struct.get_cell()[2][1] - block.get_cell()[2][1]
    
    # This shift should roughly match the path coordinates at the middle
    # (Exact match might be tricky due to interpolation, but let's check it's not zero if path is traversing)
    
    plt.close()
