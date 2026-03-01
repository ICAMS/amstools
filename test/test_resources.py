import pytest
import numpy as np
import os
from pathlib import Path
from ase.build import bulk

from amstools.resources.data import (
    get_resources_filenames_by_glob,
    get_data_path,
    get_resource_single_filename,
)
from amstools.resources.cfgio import create_structure, write_structure

# Make test paths independent of the current working directory
TEST_DIR = Path(__file__).parent.resolve()


def test_get_data_path():
    data_path = get_data_path()
    assert data_path != ""


def test_get_resources_by_glob():
    res = get_resources_filenames_by_glob("structures/Surfaces/fcc/*/*.cfg")
    assert len(res) > 0


def test_get_resource_single_filename():
    fname = "structures/Interstitials/fcc/100_dumbbell/fcc_100_dumbbell_444.cfg"
    res = get_resource_single_filename(fname)
    assert isinstance(res, str)
    assert res.endswith(fname)


def test_create_structure_normal_cfg():
    filepath = TEST_DIR / "ht/structures/melt_Ag.1000000.cfg"
    at = create_structure(str(filepath))
    assert len(at) == 108
    assert at.get_chemical_formula() == "Ag108"


def test_create_structure_quaternary():
    filepath = (
        TEST_DIR
        / "ht/structures/quaternaries/bulk/AsNi-B8_1/reference/gen_34_B8_1_AsNi.cfg"
    )
    at = create_structure(
        str(filepath),
        mapping_elements=["Mo", "Nb", "Ta", "W"],
    )
    assert at.get_chemical_symbols() == ["Mo", "Nb", "Ta", "W"]

    at2 = create_structure(
        str(filepath),
        mapping_elements=["W", "Ta", "Nb", "Mo"],
    )
    assert at2.get_chemical_symbols() == ["W", "Ta", "Nb", "Mo"]
    assert at.get_volume() == at2.get_volume()

    at3 = create_structure(
        str(filepath),
        mapping_elements=["W", "Ta", "Nb", "Mo"],
        nndist=2.0,
    )
    assert pytest.approx(at3.get_volume()) == 31.47641734671381


def test_write_structure_exact_periodic():
    at0 = bulk("Mo", cubic=True)
    at0.set_pbc(True)
    filename = "test_bcc.cfg"
    try:
        write_structure(filename, at0, save_exact=True)
        at1 = create_structure(filename, mapping_elements={"EleA": "Mo"})

        assert np.allclose(at0.get_cell(), at1.get_cell())
        assert np.allclose(at0.get_positions(), at1.get_positions())
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_write_structure_exact_nonperiodic():
    at0 = bulk("Mo", cubic=True)
    at0.set_pbc(False)
    at0.set_cell(None)
    filename = "test_npbc.cfg"
    try:
        write_structure(filename, at0, save_exact=True)
        at1 = create_structure(filename, mapping_elements={"EleA": "Mo"})

        assert np.allclose(at0.get_cell(), at1.get_cell())
        assert np.allclose(at0.get_distance(0, 1), at1.get_distance(0, 1))
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_write_structure_non_exact():
    at0 = bulk("Mo", a=3.15, cubic=True)
    at0.set_pbc(True)
    filename = "test_scaling.cfg"
    try:
        write_structure(filename, at0, save_exact=False, scaling_length=2.0)
        at1 = create_structure(filename, mapping_elements={"EleA": "Mo"})

        assert np.allclose(at0.get_cell(), at1.get_cell())
        assert np.allclose(at0.get_positions(), at1.get_positions())
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_write_structure_with_metainfo():
    at0 = bulk("Mo", a=3.15, cubic=True)
    filename = "test_metadic.cfg"
    manual_metadict = {
        "author": "MQ",
        "periodic": "True",
        "source": "ase_atoms_bulk",
    }
    try:
        write_structure(
            filename,
            at0,
            save_exact=False,
            scaling_length=2.0,
            metadata_dict=manual_metadict,
        )
        at1 = create_structure(filename, mapping_elements={"EleA": "Mo"})

        assert np.allclose(at0.get_cell(), at1.get_cell())
        assert np.allclose(at0.get_positions(), at1.get_positions())
    finally:
        if os.path.exists(filename):
            os.remove(filename)
