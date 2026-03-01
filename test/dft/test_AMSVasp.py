import os
import shutil

from contextlib import contextmanager
import numpy as np
import pytest
from ase.build import bulk
from ase.io import jsonio

from amstools.calculators.dft.base import CalculationNotConverged
from amstools.utils import CalculationLockException, lock_file
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


@contextmanager
def restore_files_after_test(directory):
    """A context manager to clean up new files created in a directory."""
    if not os.path.isdir(directory):
        os.makedirs(directory)
        initial_contents = set()
    else:
        initial_contents = set(os.listdir(directory))

    error_raised = False
    try:
        yield
    except Exception as e:
        error_raised = True
        raise e
    finally:
        if os.path.isdir(directory) and not error_raised:
            final_contents = set(os.listdir(directory))
            new_contents = final_contents - initial_contents
            for item in new_contents:
                path = os.path.join(directory, item)
                if os.path.isfile(path) or os.path.islink(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)


def test_ASEVaspWrapper():
    with restore_files_after_test(test_vasp_calc):

        calc = ASEVaspWrapper(directory=test_vasp_calc, xc="pbe", setups="recommended")
        atoms = bulk("Mo")
        atoms.calc = calc

        e = atoms.get_potential_energy()
        e_ref = -10.12441798  # total energy (not free energy)

        fe = atoms.get_potential_energy(force_consistent=True)
        fe_ref = -10.17419569  # total energy (not free energy)

        forces = atoms.get_forces()
        forces_ref = [[0, 0, 0]]
        print("forces=", forces)

        stresses = atoms.get_stress()
        print("stresses=", stresses)
        stresses_ref = [-0.90440714, -0.90440714, -0.90440714, -0.0, -0.0, 0.0]

        assert abs(e - e_ref) < 1e-7
        assert abs(fe - fe_ref) < 1e-7
        assert np.max(abs(forces - forces_ref)) < 1e-7
        assert np.max(abs(stresses - stresses_ref)) < 1e-7


def test_AMSVasp():
    with restore_files_after_test(test_vasp_calc):
        calc = AMSVasp(
            directory=test_vasp_calc, xc="pbe", setups="recommended", ispin=1
        )
        assert not calc.converged
        atoms = bulk("Mo")
        atoms.calc = calc

        e = atoms.get_potential_energy(force_consistent=False)
        fe = atoms.get_potential_energy(force_consistent=True)
        assert calc.is_new_calc
        assert calc.converged

        e_ref = -10.12441798  # total energy (not free energy)
        fe_ref = -10.17419569  # total energy (not free energy)

        assert abs(e - e_ref) < 1e-7
        assert abs(fe - fe_ref) < 1e-7
        assert os.path.isfile(os.path.join(test_vasp_calc, "data.json"))
        assert os.path.isfile(os.path.join(test_vasp_calc, "vasp.tar.gz"))

        calc2 = AMSVasp(directory=test_vasp_calc)
        print("calc2.atoms=", calc2.atoms)
        print("calc2.results=", calc2.results)

        assert calc2.atoms is not None
        assert calc2.results


def test_AMSVasp_auto_kspacing():
    with restore_files_after_test(test_vasp_calc):
        calc = AMSVasp(
            directory=test_vasp_calc,
            xc="pbe",
            gamma=True,
            setups="recommended",
            kmesh_spacing=0.15,
            write_input_only=True,
        )

        atoms = bulk("Mo")
        atoms.calc = calc
        e = atoms.get_potential_energy()
        assert calc.is_new_calc
        assert not calc.converged
        assert e is None
        assert calc.paused
        print("calc.paused_calculations_dirs=", calc.paused_calculations_dirs)
        assert len(calc.paused_calculations_dirs) == 1
        dir = calc.paused_calculations_dirs[0]
        with open(os.path.join(dir, "KPOINTS")) as f:
            kpts_lines = f.readlines()
        print("kpts_lines=", kpts_lines)
        assert kpts_lines[3].strip() == "19 19 19"


def test_AMSVasp_reload_from_data_json():
    directory = os.path.join(dft_test_dirname, "restore_vasp_json")
    with restore_files_after_test(directory):

        calc = AMSVasp(directory=directory)
        print("calc.atoms=", calc.atoms)
        print("calc.results=", calc.results)
        assert calc.converged
        assert calc.atoms is not None
        assert calc.results

        assert not calc.is_new_calc
        e = calc.get_potential_energy()
        fe = calc.get_potential_energy(force_consistent=True)

        e_ref = -10.12441798  # total energy (not free energy)
        fe_ref = -10.17419569  # total free energy

        assert abs(e - e_ref) < 1e-7
        assert abs(fe - fe_ref) < 1e-7


def test_AMSVasp_reload_from_raw_calc():
    directory = os.path.join(dft_test_dirname, "restore_vasp_raw")
    with restore_files_after_test(directory):

        calc = AMSVasp(directory=directory)
        calc.decompress()
        print("calc.atoms=", calc.atoms)
        print("calc.results=", calc.results)
        assert calc.converged
        assert calc.atoms is not None
        assert calc.results
        e = calc.get_potential_energy()
        fe = calc.get_potential_energy(force_consistent=True)
        assert not calc.is_new_calc
        e_ref = -10.12441798  # total energy (not free energy)
        fe_ref = -10.17419569  # total energy (not free energy)

        assert abs(e - e_ref) < 1e-7
        assert abs(fe - fe_ref) < 1e-7

        #  test_AMSVasp_lock_file():
        with lock_file(directory):
            calc = AMSVasp(directory=directory)
            assert calc.results

            calc.inner_calculator.results.clear()
            assert len(calc.results) == 0
            with pytest.raises(CalculationLockException):
                calc.get_potential_energy()


def test_AMSVasp_reload_from_broken_raw_calc():
    # Expected to be not restored (is_new_calc) and non converged
    directory = os.path.join(dft_test_dirname, "broken_vasp")
    with restore_files_after_test(directory):

        calc = AMSVasp(directory=directory)
        atoms = calc.atoms
        assert not calc.converged
        assert calc.is_new_calc
        assert atoms is None
        with pytest.raises(Exception):
            calc.get_potential_energy()


def test_AMSVasp_reload_from_broken_contcar_raw_calc():
    # Expected not to be restored
    directory = os.path.join(dft_test_dirname, "broken_vasp_contcar")
    with restore_files_after_test(directory):
        calc = AMSVasp(directory=directory)
        with pytest.raises(CalculationNotConverged):
            assert calc.get_potential_energy()
        # assert not calc.is_new_calc
        # atoms = calc.atoms
        # assert atoms is not None


def test_AMSVasp_to_from_dict():
    directory = os.path.join(dft_test_dirname, "restore_vasp_json")
    with restore_files_after_test(directory):
        calc = AMSVasp(directory=directory)
        # atoms = calc.atoms
        assert not calc.is_new_calc
        assert calc.converged
        vasp_dct = calc.todict()
        print("vasp_dct=", vasp_dct)

        calc = AMSVasp.fromdict(vasp_dct)

        print("calc.atoms=", calc.atoms)
        print("calc.results=", calc.results)
        assert calc.atoms is not None
        assert calc.results
        assert not calc.is_new_calc
        assert calc.converged
        e = calc.get_potential_energy()
        fe = calc.get_potential_energy(force_consistent=True)

        e_ref = -10.12441798  # total energy (not free energy)
        fe_ref = -10.17419569  # total energy (not free energy)

        assert abs(e - e_ref) < 1e-7
        assert abs(fe - fe_ref) < 1e-7


def test_AMSVasp_to_from_dict_initialized():
    with restore_files_after_test(test_vasp_calc):
        # check/remove tmp.json
        if os.path.isfile("tmp.json"):
            os.remove("tmp.json")

        atoms = bulk("Mo")
        vasp = AMSVasp(
            kpts=[1, 1, 1], kmesh_spacing=None, xc="pbe", setups="recommended", ispin=1
        )
        vasp.static_calc()
        vasp.atoms = atoms
        dct = vasp.todict()
        jsonio.write_json("tmp.json", dct)
        dct_loaded = jsonio.read_json("tmp.json")
        vasp2 = AMSVasp.fromdict(dct_loaded)
        vasp2.directory = test_vasp_calc
        assert vasp2.atoms is not None
        vasp2.get_potential_energy()
        os.remove("tmp.json")


def test_AMSVasp_magnetic_collinear():
    with restore_files_after_test(test_vasp_calc):

        calc = AMSVasp(
            directory=test_vasp_calc,
            xc="pbe",
            setups="recommended",
            write_input_only=True,
        )
        atoms = bulk("Mo", cubic=True, a=3.6)
        atoms.set_initial_magnetic_moments([1, 2])
        atoms.calc = calc

        e = atoms.get_potential_energy()

        with open(os.path.join(test_vasp_calc, "INCAR")) as f:
            lines = [l.strip() for l in f]
        # assert that there is a line with ISPIN
        print(
            "\n".join(lines),
        )
        assert any("ISPIN" in line for line in lines)
        assert any("MAGMOM" in line for line in lines)
        assert any("MAGMOM = 1*1.0000 1*2.0000" in line for line in lines)


def test_AMSVasp_magnetic_non_collinear():
    with restore_files_after_test(test_vasp_calc):

        calc = AMSVasp(
            directory=test_vasp_calc,
            xc="pbe",
            setups="recommended",
            lnoncollinear=True,
            write_input_only=True,
        )
        atoms = bulk("Mo", cubic=True, a=3.6)
        atoms.set_initial_magnetic_moments([[1, 0.1, 0.2], [2, 0.3, 0.4]])
        atoms.calc = calc

        e = atoms.get_potential_energy()

        with open(os.path.join(test_vasp_calc, "INCAR")) as f:
            lines = [l.strip() for l in f]
        print(
            "\n".join(lines),
        )
        # assert that there is a line with ISPIN

        assert any("ISPIN" in line for line in lines)
        assert any("LNONCOLLINEAR" in line for line in lines)
        assert any("MAGMOM" in line for line in lines)
        assert any(
            "MAGMOM = 1*1.0000 1*0.1000 1*0.2000 1*2.0000 1*0.3000 1*0.4000" in line
            for line in lines
        )


def test_AMSVasp_magnetic_and_ldau():
    with restore_files_after_test(test_vasp_calc):

        calc = AMSVasp(
            directory=test_vasp_calc,
            xc="pbe",
            setups="recommended",
            write_input_only=True,
            ldau_luj={"Si": {"L": 1, "U": 3, "J": 0}, "Mo": {"L": 2, "U": 4, "J": 1}},
        )
        atoms = bulk("Mo", cubic=True, a=3.6)
        atoms.set_initial_magnetic_moments([1, 2])
        atoms.calc = calc

        e = atoms.get_potential_energy()

        with open(os.path.join(test_vasp_calc, "INCAR")) as f:
            lines = [l.strip() for l in f]
        # assert that there is a line with ISPIN
        print(
            "\n".join(lines),
        )
        assert any("ISPIN" in line for line in lines)
        assert any("MAGMOM" in line for line in lines)
        assert any("MAGMOM = 1*1.0000 1*2.0000" in line for line in lines)
        assert any("LDAU = .TRUE." in line for line in lines)
        assert any("LDAUL = 2" in line for line in lines)
        assert any("LDAUU = 4.000" in line for line in lines)
        assert any("LDAUJ = 1.000" in line for line in lines)
