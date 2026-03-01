import pytest
import os
import re
import tempfile
from pathlib import Path
from ase.build import bulk
from ase.io import write

from amstools.lammps import write_in_lammps


def _check_line_in_script(line, script, present=True):
    """
    Checks if a whitespace-normalized line is in the script.
    `present=True` asserts the line is present.
    `present=False` asserts the line is NOT present.
    """
    normalized_line = " ".join(line.strip().split())

    found = any(
        normalized_line in " ".join(script_line.strip().split())
        for script_line in script.split("\n")
    )
    if present:
        assert found, f"Line '{normalized_line}' not found in script."
    else:
        assert (
            not found
        ), f"Line '{normalized_line}' was found in script, but should not be."


@pytest.fixture
def lammps_env():
    """Set up a temporary directory and dummy files for tests."""
    with tempfile.TemporaryDirectory() as test_dir:
        working_dir = test_dir
        structure = bulk("Al", "fcc", a=4.05, cubic=True)
        structure_multi = bulk("NaCl", "rocksalt", a=5.6)
        potname = "dummy.yace"
        Path(working_dir, potname).touch()

        def _read_script():
            """Helper to read the generated LAMMPS script."""
            with open(os.path.join(working_dir, "in.lammps"), "r") as f:
                return f.read()

        yield {
            "working_dir": working_dir,
            "structure": structure,
            "structure_multi": structure_multi,
            "potname": potname,
            "read_script": _read_script,
        }


def test_min_full_step(lammps_env):
    """Test generation of a full minimization step."""
    steps = [{"type": "min_full"}]
    write_in_lammps(
        lammps_env["structure"],
        lammps_env["potname"],
        lammps_env["working_dir"],
        steps=steps,
    )

    script = lammps_env["read_script"]()
    print("SCRIPT:")
    print(script)
    _check_line_in_script(
        'print "-------------- MINIMIZE min_full ---------------------"', script
    )
    _check_line_in_script("fix box_relax all box/relax aniso 0.0 vmax 0.001", script)
    _check_line_in_script("min_style cg", script)
    _check_line_in_script("minimize 0 1.0e-3 500 500", script)
    _check_line_in_script("write_data min_full.lammps-data", script)
    assert os.path.exists(
        os.path.join(lammps_env["working_dir"], "structure.lammps-data")
    )

    # Check relative order
    assert script.find("fix box_relax") < script.find("min_style cg")
    assert script.find("min_style cg") < script.find("minimize")
    assert script.find("minimize") < script.find("unfix box_relax")


def test_min_atomic_step(lammps_env):
    """Test generation of an atomic-only minimization step."""
    steps = [{"type": "min_atomic"}]
    write_in_lammps(
        lammps_env["structure"],
        lammps_env["potname"],
        lammps_env["working_dir"],
        steps=steps,
    )

    script = lammps_env["read_script"]()
    _check_line_in_script(
        'print "-------------- MINIMIZE min_atomic ---------------------"', script
    )
    _check_line_in_script("fix box_relax", script, present=False)
    _check_line_in_script("minimize 0 1.0e-3 500 500", script)


def test_eq_npt_step(lammps_env):
    """Test generation of an NPT equilibration step."""
    steps = [
        {"type": "eq", "eq_type": "npt", "T1": 500, "T2": 500, "P1": 1.0, "P2": 1.0}
    ]
    write_in_lammps(
        lammps_env["structure"],
        lammps_env["potname"],
        lammps_env["working_dir"],
        steps=steps,
    )

    script = lammps_env["read_script"]()
    print("SCRIPT:")
    print(script)
    _check_line_in_script(
        'print "-------------- EQUILIBRATION eq ---------------------"', script
    )
    _check_line_in_script("velocity all create", script)
    _check_line_in_script("fix eq all npt temp 500 500 0.1 aniso 1.0 1.0 0.1", script)
    assert re.search(r"run\s+10000", script)

    # Check relative order
    assert script.find("velocity all create") < script.find("fix eq all npt")
    assert script.find("fix eq all npt") < script.find("run")
    assert script.find("run") < script.find("unfix 	eq")


def test_mdmc_step(lammps_env):
    """Test generation of an MDMC (atom swap) step."""
    steps = [{"type": "mdmc", "elements": ["Na", "Cl"], "T_MDMC": 1200}]
    write_in_lammps(
        lammps_env["structure_multi"],
        lammps_env["potname"],
        lammps_env["working_dir"],
        steps=steps,
        specorder=["Na", "Cl"],
    )

    script = lammps_env["read_script"]()
    print("SCRIPT:")
    print(script)
    _check_line_in_script(
        'print "-------------- MDMC mdmc ---------------------"', script
    )
    _check_line_in_script(
        "fix NaCl all atom/swap 100 100 1021 1200 ke yes types 1 2", script
    )
    _check_line_in_script("fix mdmc_npt all npt", script)
    _check_line_in_script(
        "thermo_style custom step cpuremain temp pe fmax c_min_dist f_NaCl[2]", script
    )

    # Check relative order
    assert script.find("fix NaCl all atom/swap") < script.find("fix mdmc_npt all npt")
    assert script.find("fix mdmc_npt all npt") < script.find("run")
    assert script.find("run") < script.find("unfix NaCl")


def test_gcmc_single_element(lammps_env):
    """Test generation of a GCMC step with a single element."""
    steps = [{"type": "gcmc", "elem": "H", "mu": -1.5, "X": 50}]
    # Add H to specorder even if not in structure, as it will be inserted
    write_in_lammps(
        lammps_env["structure"],
        lammps_env["potname"],
        lammps_env["working_dir"],
        steps=steps,
        specorder=["Al", "H"],
    )

    script = lammps_env["read_script"]()
    _check_line_in_script(
        'print "-------------- GCMC gcmc ---------------------"', script
    )
    _check_line_in_script("group H type 2", script)
    _check_line_in_script(
        "fix gcmc_H H gcmc 100 50 0 2 12345 300 -1.5 0.0 overlap_cutoff 1.0", script
    )
    _check_line_in_script("variable H_conc equal count(H)/count(all)", script)
    _check_line_in_script("variable H_ins equal f_gcmc_H[4]", script)
    _check_line_in_script("variable H_del equal f_gcmc_H[6]", script)
    _check_line_in_script(
        "thermo_style custom step cpuremain temp pe fmax c_min_dist v_H_conc v_H_ins v_H_del",
        script,
    )

    # Check relative order: variables must be defined after the fix they reference
    assert script.find("fix gcmc_H") < script.find("variable H_conc")
    assert script.find("variable H_conc") < script.find("variable H_ins")
    assert script.find("variable H_ins") < script.find("variable H_del")
    assert script.find("variable H_del") < script.find("thermo_style")


def test_gcmc_multi_element(lammps_env):
    """Test generation of a GCMC step with multiple elements."""
    steps = [
        {
            "type": "gcmc",
            "elem": ["H", "He"],
            "mu": [-1.5, -2.0],
            "X": [50, 60],
            "M": [10, 0],
            "seed": 54321,
        }
    ]
    write_in_lammps(
        lammps_env["structure"],
        lammps_env["potname"],
        lammps_env["working_dir"],
        steps=steps,
        specorder=["Al", "H", "He"],
    )

    script = lammps_env["read_script"]()
    _check_line_in_script("group H type 2", script)
    _check_line_in_script("group He type 3", script)
    # Check first fix for H
    _check_line_in_script(
        "fix gcmc_H H gcmc 100 50 10 2 54321 300 -1.5 0.0 overlap_cutoff 1.0", script
    )
    # Check second fix for He, note incremented seed
    _check_line_in_script(
        "fix gcmc_He He gcmc 100 60 0 3 54322 300 -2.0 0.0 overlap_cutoff 1.0", script
    )
    # Check thermo columns
    _check_line_in_script("v_H_conc v_H_ins v_H_del v_H_trans", script)
    _check_line_in_script("v_He_conc v_He_ins v_He_del", script)
    _check_line_in_script("v_He_trans", script, present=False)

    # Check variables are defined after the fixes they reference
    assert script.find("fix gcmc_H") < script.find("variable H_conc")
    assert script.find("fix gcmc_He") < script.find("variable He_conc")
    # Check unfix commands are present
    assert script.find("run") < script.find("unfix gcmc_H")
    assert script.find("run") < script.find("unfix gcmc_He")


def test_gcmc_param_mismatch_raises_error(lammps_env):
    """Test GCMC raises ValueError for mismatched parameter list lengths."""
    steps = [
        {"type": "gcmc", "elem": ["H", "He"], "mu": [-1.5]}
    ]  # mu list is too short
    with pytest.raises(ValueError, match="Length of 'mu' list"):
        write_in_lammps(
            lammps_env["structure"],
            lammps_env["potname"],
            lammps_env["working_dir"],
            steps=steps,
            specorder=["Al", "H", "He"],
        )


def test_gcmc_no_exchange(lammps_env):
    """Test GCMC with X=0 doesn't output ins/del columns."""
    steps = [{"type": "gcmc", "elem": "H", "mu": -1.5, "X": 0, "M": 5}]
    write_in_lammps(
        lammps_env["structure"],
        lammps_env["potname"],
        lammps_env["working_dir"],
        steps=steps,
        specorder=["Al", "H"],
    )

    script = lammps_env["read_script"]()
    # With X=0, ins, del, and conc variables should not be present
    _check_line_in_script("variable H_conc", script, present=False)
    _check_line_in_script("variable H_ins", script, present=False)
    _check_line_in_script("variable H_del", script, present=False)
    # With M>0, trans variable should be present
    _check_line_in_script("variable H_trans", script)
    _check_line_in_script(
        "thermo_style custom step cpuremain temp pe fmax c_min_dist v_H_trans", script
    )


def test_gcmc_mdmc_combined_step(lammps_env):
    """Test generation of a combined GCMC and atom swap (MDMC) step."""
    steps = [
        {
            "type": "gcmc_mdmc",
            "T": 800,
            # GCMC part
            "gcmc_elements": ["H"],
            "mu": [-1.5],
            "X": 50,
            # MDMC part
            "atom_swap_elements": ["Na", "Cl"],
            "swap_T": 800,
            "swap_N": 50,
            "swap_M": 50,
        }
    ]
    write_in_lammps(
        lammps_env["structure_multi"],
        lammps_env["potname"],
        lammps_env["working_dir"],
        steps=steps,
        specorder=["Na", "Cl", "H"],
    )

    script = lammps_env["read_script"]()
    print("SCRIPT:")
    print(script)

    # Check for the correct step title
    _check_line_in_script(
        'print "-------------- GCMC+MDMC gcmc_mdmc ---------------------"', script
    )

    # Check GCMC parts
    _check_line_in_script("group H type 3", script)
    _check_line_in_script(
        "fix gcmc_H H gcmc 100 50 0 3 12345 800 -1.5 0.0 overlap_cutoff 1.0", script
    )
    _check_line_in_script("variable H_conc equal count(H)/count(all)", script)
    _check_line_in_script("variable H_ins equal f_gcmc_H[4]", script)
    _check_line_in_script("variable H_del equal f_gcmc_H[6]", script)

    # Check MDMC (atom swap) part
    _check_line_in_script(
        "fix NaCl all atom/swap 50 50 13346 800 ke yes types 1 2", script
    )

    # Check thermo style for combined outputs
    _check_line_in_script(
        "thermo_style custom step cpuremain temp pe fmax c_min_dist v_H_conc v_H_ins v_H_del f_NaCl[2]",
        script,
    )

    # Check unfix commands are present and in order
    _check_line_in_script("unfix gcmc_H", script)
    _check_line_in_script("unfix NaCl", script)
    assert script.find("run") < script.find("unfix gcmc_H")
    assert script.find("run") < script.find("unfix NaCl")


def test_with_extrapolation_grade(lammps_env):
    """Test generation with extrapolation grade."""
    potname_yaml = "dummy.yaml"
    potname_asi = "dummy.asi"
    Path(lammps_env["working_dir"], potname_yaml).touch()
    Path(lammps_env["working_dir"], potname_asi).touch()

    steps = [{"type": "min_atomic"}]
    write_in_lammps(
        lammps_env["structure"],
        potname_yaml,
        lammps_env["working_dir"],
        steps=steps,
        with_extrapolation_grade=True,
    )

    script = lammps_env["read_script"]()
    print("SCRIPT:")
    print(script)
    _check_line_in_script("pair_style pace/extrapolation", script)
    _check_line_in_script(f"pair_coeff * * {potname_yaml} {potname_asi} Al", script)
    _check_line_in_script(
        "fix gamma all pair 50 pace/extrapolation gamma 1", script
    )
    _check_line_in_script("compute max_gamma all reduce max f_gamma", script)
    _check_line_in_script(
        "thermo_style custom step cpuremain pe  fmax c_min_dist c_max_gamma  press vol density pxx pyy pzz pxy pxz pyz xy xz yz",
        script,
    )

    # Check relative order
    assert script.find("pair_style") < script.find("pair_coeff")
    assert script.find("pair_coeff") < script.find("fix gamma")
    assert script.find("fix gamma") < script.find("compute max_gamma")


def test_dump_extrapolative(lammps_env):
    """Test generation with extrapolative structure dumping."""
    potname_yaml = "dummy.yaml"
    potname_asi = "dummy.asi"
    Path(lammps_env["working_dir"], potname_yaml).touch()
    Path(lammps_env["working_dir"], potname_asi).touch()

    steps = [{"type": "min_atomic"}]
    write_in_lammps(
        lammps_env["structure"],
        potname_yaml,
        lammps_env["working_dir"],
        steps=steps,
        with_extrapolation_grade=True,
        dump_extrapolative=True,
        gamma_lo=5,
        gamma_hi=20,
    )

    script = lammps_env["read_script"]()
    print("SCRIPT:")
    print(script)
    _check_line_in_script('variable dump_skip equal "c_max_gamma < 5"', script)
    _check_line_in_script(
        "dump dump_extrapolative all custom 50 extrapolative_structures.min_atomic.dump",
        script,
    )
    _check_line_in_script("dump_modify dump_extrapolative skip v_dump_skip", script)
    _check_line_in_script(
        "fix extreme_extrapolation all halt 50 v_max_gamma > 20", script
    )

    # Check relative order
    assert script.find("variable dump_skip") < script.find("dump dump_extrapolative")
    assert script.find("dump dump_extrapolative") < script.find(
        "dump_modify dump_extrapolative skip"
    )
    assert script.find("dump_modify dump_extrapolative skip") < script.find("minimize")
    assert script.find("minimize") < script.find("undump dump_extrapolative")


def test_gpu_option(lammps_env):
    """Test GPU option adds 'chunksize' to pair_style."""
    steps = [{"type": "min_atomic"}]
    write_in_lammps(
        lammps_env["structure"],
        lammps_env["potname"],
        lammps_env["working_dir"],
        steps=steps,
        is_gpu=True,
    )

    script = lammps_env["read_script"]()
    _check_line_in_script("pair_style pace product chunksize 1024", script)


def test_multiple_steps(lammps_env):
    """Test generation of a script with multiple sequential steps."""
    steps = [
        {"type": "min_atomic", "name": "relax_atoms"},
        {"type": "eq", "name": "heat_up", "T_init": 300, "T1": 600, "T2": 600},
    ]
    write_in_lammps(
        lammps_env["structure"],
        lammps_env["potname"],
        lammps_env["working_dir"],
        steps=steps,
    )

    script = lammps_env["read_script"]()
    _check_line_in_script(
        'print "-------------- MINIMIZE relax_atoms ---------------------"', script
    )
    _check_line_in_script("write_data relax_atoms.lammps-data", script)
    _check_line_in_script(
        'print "-------------- EQUILIBRATION heat_up ---------------------"', script
    )
    _check_line_in_script("write_data heat_up.lammps-data", script)

    relax_pos = script.find(
        'print "-------------- MINIMIZE  relax_atoms ---------------------"'
    )
    eq_pos = script.find(
        'print "-------------- EQUILIBRATION heat_up ---------------------"'
    )
    assert relax_pos < eq_pos


def test_invalid_step_type_raises_error(lammps_env):
    """Test that an unknown step type raises a ValueError."""
    steps = [{"type": "this_is_not_a_real_step"}]
    with pytest.raises(ValueError, match="Unknown step: this_is_not_a_real_step"):
        write_in_lammps(
            lammps_env["structure"],
            lammps_env["potname"],
            lammps_env["working_dir"],
            steps=steps,
        )


def test_some_lammps_setup(lammps_env):
    """Test generation of a script with some LAMMPS setup."""
    potname = lammps_env["potname"].replace(".yace", ".yaml")
    asi_filename = potname.replace(".yaml", ".asi")
    Path(lammps_env["working_dir"], asi_filename).touch()

    write_in_lammps(
            lammps_env["structure"],
            potname,
            lammps_env["working_dir"],
            pairstyle='grace/fs',
            with_extrapolation_grade=True,
            steps=[
                    {
                    "type": "min_full",
                    "thermo_freq": 100,
                    "steps": 1000,
                    "aniso_style": "aniso",
                },                
            ],
    )

    script = lammps_env["read_script"]()
    print("SCRIPT:")
    print(script)
    
    # check that grace/fs extrapolation is used
    _check_line_in_script("pair_style grace/fs extrapolation", script)
    _check_line_in_script(f"pair_coeff * * {potname} {asi_filename} Al", script)
    _check_line_in_script("fix gamma all pair 50 grace/fs gamma 1", script)
    _check_line_in_script("compute max_gamma all reduce max f_gamma", script)
    _check_line_in_script(
        "thermo_style custom step cpuremain pe  fmax c_min_dist c_max_gamma  press vol density pxx pyy pzz pxy pxz pyz xy xz yz",
        script,
    )
