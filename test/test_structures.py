import pytest
from amstools.resources import *


def test_get_structures_dictionary():
    res_dict = get_structures_dictionary(elements="Cu")
    assert "fcc" in res_dict
    assert "bcc" in res_dict
    assert "hcp" in res_dict
    assert "sc" in res_dict
    assert "dhcp" in res_dict

    struct_dict = res_dict["fcc"]
    assert "colour" in struct_dict
    assert "nntolat" in struct_dict
    assert "isbulk" in struct_dict
    assert "STRUKTURBERICHT" in struct_dict
    assert "atoms" in struct_dict

    fcc = struct_dict["atoms"]
    assert len(fcc) == 1
    assert fcc.get_chemical_symbols() == ["Cu"]


def test_get_structures_dictionary_include():
    res_dict = get_structures_dictionary(elements="Cu", include="fcc")
    assert len(res_dict) == 1
    assert "fcc" in res_dict

    struct_dict = res_dict["fcc"]
    assert "colour" in struct_dict
    assert "nntolat" in struct_dict
    assert "isbulk" in struct_dict
    assert "STRUKTURBERICHT" in struct_dict
    assert "atoms" in struct_dict

    fcc = struct_dict["atoms"]
    assert len(fcc) == 1
    assert fcc.get_chemical_symbols() == ["Cu"]


def test_get_structures_dictionary_include_strukturbericht():
    res_dict = get_structures_dictionary(elements="Cu", include="A1")
    assert "fcc" in res_dict

    struct_dict = res_dict["fcc"]
    assert "colour" in struct_dict
    assert "nntolat" in struct_dict
    assert "isbulk" in struct_dict
    assert "STRUKTURBERICHT" in struct_dict
    assert "atoms" in struct_dict

    fcc = struct_dict["atoms"]
    assert len(fcc) == 1
    assert fcc.get_chemical_symbols() == ["Cu"]


def test_get_structures_dictionary_include_multiple():
    res_dict = get_structures_dictionary(elements="Cu", include=["A1", "dimer"])
    assert "fcc" in res_dict
    assert "dimer" in res_dict

    struct_dict = res_dict["dimer"]
    assert "colour" in struct_dict
    assert "nntolat" in struct_dict
    assert "isbulk" in struct_dict
    assert "atoms" in struct_dict

    dimer = struct_dict["atoms"]
    assert len(dimer) == 2
    assert dimer.get_chemical_symbols() == ["Cu", "Cu"]


def test_get_structures_dictionary_exclude_strukturbericht():
    res_dict = get_structures_dictionary(elements="Cu", exclude="A2")
    assert "fcc" in res_dict
    assert "bcc" not in res_dict

    struct_dict = res_dict["fcc"]
    assert "colour" in struct_dict
    assert "nntolat" in struct_dict
    assert "isbulk" in struct_dict
    assert "STRUKTURBERICHT" in struct_dict
    assert "atoms" in struct_dict

    fcc = struct_dict["atoms"]
    assert len(fcc) == 1
    assert fcc.get_chemical_symbols() == ["Cu"]


def test_get_structures_dictionary_exclude():
    res_dict = get_structures_dictionary(elements="Cu", exclude="bcc")
    assert "fcc" in res_dict
    assert "bcc" not in res_dict

    struct_dict = res_dict["fcc"]
    assert "colour" in struct_dict
    assert "nntolat" in struct_dict
    assert "isbulk" in struct_dict
    assert "STRUKTURBERICHT" in struct_dict
    assert "atoms" in struct_dict

    fcc = struct_dict["atoms"]
    assert len(fcc) == 1
    assert fcc.get_chemical_symbols() == ["Cu"]


def test_get_structures_dictionary_exclude_multiple():
    res_dict = get_structures_dictionary(elements="Cu", exclude=["bcc", "dimer"])
    assert "fcc" in res_dict
    assert "bcc" not in res_dict
    assert "dimer" not in res_dict


def test_get_structures_dictionary_scale_volume_False():
    res_dict = get_structures_dictionary(
        elements="Cu", include=["fcc", "dimer"], scale_volume=False
    )
    assert "fcc" in res_dict
    assert "dimer" in res_dict
    struct_dict = res_dict["fcc"]
    vol = struct_dict["atoms"].get_volume()
    assert vol == 0.25


def test_get_structures_dictionary_scale_volume_True():
    res_dict = get_structures_dictionary(
        elements="Cu", include=["fcc", "dimer"], scale_volume=True
    )
    assert "fcc" in res_dict
    assert "dimer" in res_dict
    struct_dict = res_dict["fcc"]
    vol = struct_dict["atoms"].get_volume()
    assert vol > 1
