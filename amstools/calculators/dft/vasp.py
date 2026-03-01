import logging
import os

import numpy as np
from ase.calculators import calculator
from ase.calculators.calculator import ReadError, PropertyNotImplementedError
from ase.calculators.vasp.vasp import Vasp, check_atoms
from ase.io import read

from amstools.calculators.dft.base import (
    AMSDFTBaseCalculator,
    WriteInputOnlyException,
    CalculationNotConverged,
)
from amstools.utils import lock_file


class ASEVaspWrapper(Vasp):
    _error_code_filename = "vasp_error_code.dat"

    def __init__(self, *args, **kwargs):
        self.is_new_calc = True
        self.write_input_only = False
        Vasp.__init__(self, *args, **kwargs)

    def calculate(
        self,
        atoms=None,
        properties=("energy",),
        system_changes=tuple(calculator.all_changes),
    ):
        """Do a VASP calculation in the specified directory.

        This will generate the necessary VASP input files, and then
        execute VASP. After execution, the energy, forces. etc. are read
        from the VASP output files.
        """
        # Check for zero-length lattice vectors and PBC
        # and that we actually have an Atoms object.

        errorcode = self.check_error_code_file()
        if errorcode and errorcode != "0":
            msg = f"Previous run has error code {errorcode}, stopping this VASP calculation"
            logging.warning(msg)
            raise RuntimeError(msg)

        check_atoms(atoms)

        self.clear_results()

        if atoms is not None:
            self.atoms = atoms.copy()
        # check if self.directory contains 'paused' file
        if os.path.exists(os.path.join(self.directory, "paused")):
            raise WriteInputOnlyException()
        self.write_input(self.atoms, properties, system_changes)
        if self.write_input_only:
            with open(os.path.join(self.directory, "paused"), "w") as f:
                f.write("paused")
            logging.info(
                f"Write input only for VASP calculation in folder: {self.directory}"
            )
            raise WriteInputOnlyException()

        command = self.make_command(self.command)
        logging.info(
            f"Run VASP calculation (command={command}) in folder: {self.directory}"
        )
        with self._txt_outstream() as out:
            errorcode = self._run(command=command, out=out, directory=self.directory)
        # bugfix for change errorcode format:  (0, '')
        if isinstance(errorcode, tuple):
            errorcode = errorcode[0]
        self.write_error_code_file(errorcode)
        if errorcode:
            raise calculator.CalculationFailed(
                f"{self.name} in {self.directory} returned an error: {errorcode}"
            )

        # Read results from calculation
        self.update_atoms(atoms)
        self.read_results()

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError(f"{name} property not implemented")

        if atoms is None:
            atoms = self.atoms
            system_changes = []
        else:
            system_changes = self.check_state(atoms)
            if system_changes:
                self.reset()
        if name not in self.results:
            if not allow_calculation:
                return None
            # lock the calculation
            with lock_file(self.directory):
                self.calculate(atoms, [name], system_changes)

        if name not in self.results and not self.write_input_only:
            # For some reason the calculator was not able to do what we want,
            # and that is OK.
            raise PropertyNotImplementedError(f"{name} not present in this calculation")

        elif name not in self.results and self.write_input_only:
            return None
        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        if not self.converged:
            raise CalculationNotConverged(
                f"Calculation in {self.directory} is not converged"
            )
        return result

    def write_error_code_file(self, error_code):
        filename = os.path.join(self.directory, self._error_code_filename)
        with open(filename, "w") as f:
            print(error_code, file=f)

    def check_error_code_file(self):
        """check if error code file exists and return its value
        otherwise - 0"""
        filename = os.path.join(self.directory, self._error_code_filename)
        if os.path.isfile(filename):
            with open(filename, "r") as f:
                line = f.readline().strip()
                return line
        else:
            return 0

    def reset(self):
        if self.is_new_calc:
            super().reset()

    def clear_results(self):
        if self.is_new_calc:
            super().clear_results()

    def set(self, **kwargs):
        old_results = self.results.copy()
        super().set(**kwargs)
        # protecting old results
        if not self.is_new_calc:
            self.results = old_results


class AMSVasp(AMSDFTBaseCalculator):
    _compressed_tar_gz_filename = "vasp.tar.gz"
    _files_to_not_compress = [
        AMSDFTBaseCalculator._data_json_filename,
        _compressed_tar_gz_filename,
    ]
    _files_to_remove = ["EIGENVAL", "DOSCAR", "POTCAR", "vaspout.h5", "PROCAR"]
    _default_kwargs = {
        "xc": "pbe",
        "gamma": True,
        "setups": "recommended",
        "ismear": 0,
        "sigma": 0.1,
        "prec": "Accurate",
        "encut": 500,
        "ispin": 2,
        "ediff": 1e-6,
        "nelm": 120,
        "lreal": False,
        "lcharg": False,
        "lwave": False,
        "addgrid": True,
    }

    def __init__(
        self,
        atoms=None,
        restart=True,
        directory=".",
        label="vasp",
        command=None,
        txt="vasp.out",
        write_input_only=False,
        kmesh_spacing=0.125,
        kmesh_max=None,
        **kwargs,
    ):

        AMSDFTBaseCalculator.__init__(
            self, write_input_only=write_input_only, kmesh_max=kmesh_max
        )

        self.init_params = dict(
            atoms=atoms,
            restart=restart,
            directory=directory,
            label=label,
            command=command,
            txt=txt,
        )
        self.init_params.update(AMSVasp._default_kwargs)
        for k, v in kwargs.items():
            self.init_params[k] = v

        if kmesh_spacing:
            self.set_kmesh_spacing(kmesh_spacing)

    def _init_calculator(self):
        """
        Initiate calculator with restart or new params
        """
        full_data_json_filename = os.path.join(self.directory, self._data_json_filename)

        if os.path.isfile(full_data_json_filename):
            # try to load from json file, set _is_new_calc to False
            self._calculator = ASEVaspWrapper()
            self.deserialize()
            self._is_new_calc = False
            self._calculator.is_new_calc = self._is_new_calc
        elif not self.init_params["restart"]:
            # if no json file, then initiate with new params, set _is_new_calc to True
            self._calculator = ASEVaspWrapper(**self.init_params)
            self._calculator.set(**self.set_kwargs)
            logging.debug("Initiate with new params")
            self._is_new_calc = True
            self._calculator.is_new_calc = self._is_new_calc
        else:
            # if no json file and restart=True, then try to initiate with restart=True
            try:
                # try to initiate with restart=True, set _is_new_calc to False
                self._calculator = ASEVaspWrapper(
                    directory=self.directory,
                    label=self.init_params["label"],
                    restart=True,
                )
                logging.debug("Initiate with restart=True only")
                self._is_new_calc = False
                self._calculator.is_new_calc = self._is_new_calc
            except (ReadError, IndexError):
                # if failed, then initiate with new params, set _is_new_calc to True
                init_params = self.init_params.copy()
                init_params.pop("restart")
                self._calculator = ASEVaspWrapper(**init_params)
                self._calculator.set(**self.set_kwargs)
                logging.debug("Fallback initiate with new params")
                self._is_new_calc = True
                self._calculator.is_new_calc = self._is_new_calc

        self._calculator.set(directory=self.directory)
        self._calculator.write_input_only = self.write_input_only

    def read_init_atoms(self):
        # try to read data.json
        # if not succeed, then try to read POSCAR
        return super().read_init_atoms() or self.read_poscar_file()

    def read_poscar_file(self):
        filename = os.path.join(self.directory, "POSCAR")
        if os.path.isfile(filename):
            return read(filename=filename)

    def optimize_atoms_only(self, max_steps=200, ediffg=-1e-3, ibrion=2):
        self.set(ibrion=ibrion, nsw=max_steps, isif=2, ediffg=ediffg)

    def optimize_full(self, max_steps=200, ediffg=-1e-3, ibrion=2):
        self.set(ibrion=ibrion, nsw=max_steps, isif=3, ediffg=ediffg)

    def optimize_cell_only(self, max_steps=200, ediffg=-1e-3, ibrion=2):
        self.set(ibrion=ibrion, nsw=max_steps, isif=6, ediffg=ediffg)

    def optimize_volume_only(self, max_steps=200, ediffg=-1e-3, ibrion=2):
        self.set(ibrion=ibrion, nsw=max_steps, isif=7, ediffg=ediffg)

    def static_calc(self):
        self.set(ibrion=-1, nsw=0, isif=2)
