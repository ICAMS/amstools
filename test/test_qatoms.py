from ase.build import bulk, fcc100
from collections import Counter
import numpy as np
import pytest
from amstools import QAtoms, QAtomsCollection
from ase.calculators.emt import EMT


def test_qatoms_wrap():
    qat = QAtoms(bulk("Al", cubic=True) * (2, 2, 2))
    assert len(qat) == 32
    assert qat.num_selected == 32


def test_qatoms_bulk():
    qat = QAtoms.bulk("Al", cubic=True) * (2, 2, 2)
    assert len(qat) == 32
    assert qat.num_selected == 32


def test_qatoms_sample():
    qat = QAtoms(bulk("Al", cubic=True) * (2, 2, 2)).sample(n=1, random_state=42)
    assert len(qat) == 32
    assert qat.num_selected == 1


def test_qatoms_set():
    qat = QAtoms(bulk("Al", cubic=True) * (2, 2, 2))
    r = qat.sample(n=1, random_state=1).set(element="Li")
    assert len(r) == 32
    assert r.num_selected == 1
    comp_dict = r.comp_dict
    print("comp_dict=", comp_dict)
    assert comp_dict["Li"] == 1


def test_qatoms_select_nearby():
    qat = QAtoms(bulk("Al", cubic=True) * (2, 2, 2))
    r = (
        qat.sample(n=1, random_state=1)
        .select_nn()
        .set(element="Ni")
        .select_nearby(cutoff=3)
        .set(element="Cu")
    )
    assert len(r) == 32
    assert r.num_selected == 12
    comp_dict = r.comp_dict
    print("comp_dict=", comp_dict)
    assert comp_dict["Ni"] == 1
    assert comp_dict["Cu"] == 12


class TestSubstitute:
    @pytest.fixture
    def qat(self):
        """Fixture for a simple QAtoms object."""
        return QAtoms(bulk("Al", "fcc", a=4.0, cubic=True))

    def test_substitute_selected(self, qat):
        """Test substituting a single element on selected atoms."""
        q_selected = qat.select_explicitly([0, 1])
        q_substituted = q_selected.substitute({"Al": "Si"})

        assert q_substituted.get_chemical_symbols() == ["Si", "Si", "Al", "Al"]
        assert q_substituted.num_selected == 2
        assert q_substituted.comp_dict["Si"] == 2
        assert q_substituted.comp_dict["Al"] == 2

    def test_substitute_multiple_elements(self, qat):
        """Test substituting multiple elements at once."""
        q_multi = qat.select_explicitly([0, 1]).set(element="Ni").all()
        q_substituted = q_multi.substitute({"Al": "Si", "Ni": "Cu"})

        assert q_substituted.comp_dict["Si"] == 2
        assert q_substituted.comp_dict["Cu"] == 2
        assert "Al" not in q_substituted.comp_dict
        assert "Ni" not in q_substituted.comp_dict

    def test_substitute_respects_selection(self, qat):
        """Test that substitution only affects selected atoms."""
        # Select only atom 0, which is 'Al'
        q_selected = qat.select_explicitly(0)
        # Try to substitute 'Al' -> 'Si'. Only atom 0 should change.
        q_substituted = q_selected.substitute({"Al": "Si"})
        assert q_substituted.get_chemical_symbols() == ["Si", "Al", "Al", "Al"]


def test_qatoms_delete():
    qat = QAtoms(bulk("Al", cubic=True) * (2, 2, 2))
    r = qat.sample(n=1, random_state=42).set(element="Li").select_nn(n=2).delete()
    assert len(r) == 30
    assert r.num_selected == 0
    comp_dict = r.comp_dict
    print("comp_dict=", comp_dict)
    assert comp_dict["Li"] == 1


def test_insert_interstitial():
    qat = QAtoms(bulk("Al", cubic=True) * (2, 2, 2))
    r = qat.sample(n=1, random_state=42).insert_interstitial(
        element="H", select_new=True
    )
    assert len(r) == 33
    assert r.num_selected == 1
    comp_dict = r.comp_dict
    print("comp_dict=", comp_dict)
    assert comp_dict["H"] == 1


def test_insert_Li_interstitial_H_2():
    qat = QAtoms(bulk("Al", cubic=True) * (2, 2, 2))
    r = (
        qat.sample(n=1, random_state=42)
        .set(element="Li")
        .insert_interstitial(element="H", select_new=False)
        .insert_interstitial(element="H")
    )
    assert len(r) == 34
    assert r.num_selected == 1
    comp_dict = r.comp_dict
    print("comp_dict=", comp_dict)
    assert comp_dict["H"] == 2
    assert comp_dict["Li"] == 1


def test_tags_surf():
    surf = fcc100("Al", (2, 2, 5), periodic=True, vacuum=10)
    q = QAtoms(surf)
    q_surf = q.select(tag=5)
    assert q_surf.num_selected == 4
    q_surf_supercell = q_surf * (2, 2, 1)
    assert q_surf_supercell.num_selected == 16

    r = (
        q_surf_supercell.sample(n=1)
        .set(element="Ni")
        .select_nearby(cutoff=3)
        .select(tag=5)
        .set(element="Li")
    )
    assert len(r) == 80
    comp_dict = r.comp_dict
    print("comp_dict=", comp_dict)
    assert comp_dict["Ni"] == 1
    assert comp_dict["Li"] == 4


class TestShiftAtoms:
    @pytest.fixture
    def qat(self):
        """Fixture for a simple QAtoms object."""
        return QAtoms(bulk("Al", "fcc", a=4.0, cubic=True))

    def test_cartesian_shift_selected(self, qat):
        """Test shifting a selected atom by a Cartesian vector."""
        original_pos = qat.get_positions().copy()
        # Select atom 0
        q_selected = qat.select_explicitly(0)
        shift_vector = [0.1, 0.2, 0.3]
        q_shifted = q_selected.shift_atoms(shift=shift_vector)

        # Check that only atom 0 moved
        new_pos = q_shifted.get_positions()
        np.testing.assert_allclose(new_pos[0], original_pos[0] + shift_vector)
        np.testing.assert_allclose(new_pos[1:], original_pos[1:])

    def test_scaled_shift_selected(self, qat):
        """Test shifting selected atoms by a scaled vector."""
        original_pos = qat.get_positions().copy()
        # Select atoms 0 and 1
        q_selected = qat.select_explicitly([0, 1])
        scaled_shift_vector = [0.1, 0.0, 0.0]
        q_shifted = q_selected.shift_atoms(scaled_shift=scaled_shift_vector)

        # Calculate expected Cartesian shift
        cartesian_shift = qat.cell.T @ np.array(scaled_shift_vector)

        # Check that selected atoms moved correctly
        new_pos = q_shifted.get_positions()
        np.testing.assert_allclose(new_pos[0], original_pos[0] + cartesian_shift)
        np.testing.assert_allclose(new_pos[1], original_pos[1] + cartesian_shift)
        # Check that other atoms did not move
        np.testing.assert_allclose(new_pos[2:], original_pos[2:])

    def test_wrap_positions(self, qat):
        """Test that atoms are wrapped back into the cell."""
        # Select atom 0, which is at [0, 0, 0]
        q_selected = qat.select_explicitly(0)
        # Shift it outside the cell
        q_shifted_nowrap = q_selected.shift_atoms(shift=[-0.1, -0.1, -0.1], wrap=False)
        np.testing.assert_allclose(
            q_shifted_nowrap.get_positions()[0], [-0.1, -0.1, -0.1]
        )

        # Shift it and wrap
        q_shifted_wrap = q_selected.shift_atoms(shift=[-0.1, -0.1, -0.1], wrap=True)
        expected_pos = [-0.1 + 4.0, -0.1 + 4.0, -0.1 + 4.0]
        np.testing.assert_allclose(q_shifted_wrap.get_positions()[0], expected_pos)

    def test_shift_all_atoms_flag(self, qat):
        """Test the shift_all_atoms flag."""
        original_pos = qat.get_positions().copy()
        # Select only atom 0 initially
        q_selected = qat.select_explicitly(0)
        shift_vector = [0.1, 0.1, 0.1]

        # Shift with shift_all_atoms=True
        q_shifted_all = q_selected.shift_atoms(shift=shift_vector, shift_all_atoms=True)

        # All atoms should have moved
        expected_pos = original_pos + shift_vector
        np.testing.assert_allclose(q_shifted_all.get_positions(), expected_pos)

    def test_shift_value_error(self, qat):
        """Test that providing both shift and scaled_shift raises an error."""
        with pytest.raises(ValueError, match="Only one of 'shift' or 'scaled_shift'"):
            qat.shift_atoms(shift=[0.1, 0, 0], scaled_shift=[0.1, 0, 0])


class TestRelaxation:
    @pytest.fixture
    def qat_to_relax(self):
        """Fixture for a simple QAtoms object with an EMT calculator."""
        # Start from a slightly non-equilibrium lattice constant for Al with EMT
        atoms = QAtoms(bulk("Al", "fcc", a=3.75, cubic=True))
        atoms.calc = EMT()
        return atoms

    def test_relax_positions_only(self, qat_to_relax):
        """Test the .relax() method for atomic position relaxation."""
        initial_atoms = qat_to_relax.copy()
        # Displace an atom to create non-zero forces
        initial_atoms.positions[0] += [0.1, 0.1, 0.1]

        # Check that forces are initially high
        initial_forces = initial_atoms.get_forces()
        initial_fmax = np.sqrt((initial_forces**2).sum(axis=1).max())
        assert initial_fmax > 0.05

        # Perform relaxation
        relaxed_atoms = initial_atoms.relax(fmax=0.01)

        # Check that cell is unchanged
        np.testing.assert_allclose(initial_atoms.cell, relaxed_atoms.cell)

        # Check that positions have changed
        assert not np.allclose(initial_atoms.positions, relaxed_atoms.positions)

        # Check that forces are now low
        final_forces = relaxed_atoms.get_forces()
        final_fmax = np.sqrt((final_forces**2).sum(axis=1).max())
        assert final_fmax < 0.05

    def test_full_relax(self, qat_to_relax):
        """Test the .full_relax() method for both atomic and cell relaxation."""
        initial_volume = qat_to_relax.get_volume()

        relaxed_atoms = qat_to_relax.full_relax(fmax=0.01)
        final_volume = relaxed_atoms.get_volume()

        # For Al with EMT, the equilibrium lattice constant is ~4.04 Å, so the volume should increase from a=4.0 Å
        assert final_volume > initial_volume
        # Check that the final volume is close to the known equilibrium volume for EMT
        assert final_volume == pytest.approx(63.7257736, rel=1e-2)
