import os
import shutil
import copy

import numpy as np
import pytest
from ase.build import bulk
from ase.io import jsonio

from amstools.calculators.dft.base import AMSDFTBaseCalculator
from amstools.calculators.dft.vasp import ASEVaspWrapper, AMSVasp

test_vasp_calc = "test_vasp_calc_dir"
dft_test_dirname = os.path.dirname(__file__)
test_vasp_calc = os.path.join(dft_test_dirname, test_vasp_calc)

os.environ["ASE_VASP_COMMAND"] = os.path.join(
    dft_test_dirname, "mock_bin/mock_vasp_std"
)
os.environ["VASP_PP_PATH"] = os.path.join(dft_test_dirname, "mock_pp")


def clean_test_vasp_dir():
    if os.path.isdir(test_vasp_calc):
        print("Clean ", test_vasp_calc)
        shutil.rmtree(test_vasp_calc)
    else:
        print(test_vasp_calc, " does not exist")


def test_copy_method():
    clean_test_vasp_dir()
    calc = AMSVasp(
        directory=test_vasp_calc,
        xc="pbe",
        kmesh_spacing=0.125,
        setups="recommended",
        write_input_only=True,
        ispin=1,
    )
    new_calc = copy.copy(calc)

    atoms = bulk("Mo")
    atoms.calc = calc
    e = atoms.get_potential_energy()
    dir = calc.paused_calculations_dirs[0]
    with open(os.path.join(dir, "KPOINTS")) as f:
        kpts_lines = f.readlines()
    print("kpts_lines =", kpts_lines)
    assert kpts_lines[3].strip() == "23 23 23"

    # if paused file exists, then remove
    if os.path.isfile(os.path.join(dir, "paused")):
        os.remove(os.path.join(dir, "paused"))

    # new_calc = copy.copy(calc)
    new_calc = calc.copy()
    new_calc.kmesh_spacing = None
    new_calc.set_kmesh([1, 1, 1])
    print(new_calc.get_kmesh())
    atoms = bulk("Mo")
    atoms.calc = new_calc
    e = atoms.get_potential_energy()
    dir = new_calc.paused_calculations_dirs[0]
    with open(os.path.join(dir, "KPOINTS")) as f:
        kpts_lines = f.readlines()
    print("kpts_lines=", kpts_lines)
    assert kpts_lines[3].strip() == "1 1 1"

    # if paused file exists, then remove
    if os.path.isfile(os.path.join(dir, "paused")):
        os.remove(os.path.join(dir, "paused"))

    atoms = bulk("Mo")
    atoms.calc = calc
    e = atoms.get_potential_energy()
    dir = calc.paused_calculations_dirs[0]
    with open(os.path.join(dir, "KPOINTS")) as f:
        kpts_lines = f.readlines()
    print("kpts_lines=", kpts_lines)
    assert kpts_lines[3].strip() == "23 23 23"

    clean_test_vasp_dir()


def test_kmesh_max_and_to_dict_from_dict():
    clean_test_vasp_dir()
    if os.path.isfile("tmp.json"):
        os.remove("tmp.json")
    calc = AMSVasp(
        directory=test_vasp_calc,
        xc="pbe",
        kmesh_spacing=0.04,
        kmesh_max=24,
        setups="recommended",
        write_input_only=True,
        ispin=1,
    )

    atoms = bulk("Mo")
    atoms.calc = calc
    atoms.get_potential_energy()
    dir = calc.paused_calculations_dirs[0]
    with open(os.path.join(dir, "KPOINTS")) as f:
        kpts_lines = f.readlines()
    assert kpts_lines[3].strip() == "24 24 24"
    # clean_test_vasp_dir()

    dct = calc.todict()
    assert "kmesh_max" in dct and dct["kmesh_max"] == 24
    jsonio.write_json("tmp.json", dct)
    dct_loaded = jsonio.read_json("tmp.json")
    assert "kmesh_max" in dct_loaded and dct_loaded["kmesh_max"] == 24
    vasp2 = AMSVasp.fromdict(dct_loaded)
    print(vasp2.directory)
    assert vasp2.kmesh_max == 24
    clean_test_vasp_dir()
    if os.path.isfile("tmp.json"):
        os.remove("tmp.json")
