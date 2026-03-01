import logging
import os
import ase
import numpy as np

from ase.calculators import calculator
from ase.calculators.calculator import (
    ReadError,
    PropertyNotImplementedError,
    FileIOCalculator,
    Parameters,
)
from ase.calculators.aims import Aims
from ase.io import read
from ase.io.aims import read_aims

from amstools.calculators.dft.base import AMSDFTBaseCalculator, CalculationNotConverged
from amstools.utils import lock_file


class ASEFHIaimsWrapper(Aims):
    _error_code_filename = "aims_error_code.dat"

    def __init__(self, *args, **kwargs):
        self.is_new_calc = True
        self.write_input_only = False
        if "restart" in kwargs and kwargs["restart"]:
            self.directory = kwargs.get("directory", ".")
            kwargs["restart"] = self.directory
        Aims.__init__(self, *args, **kwargs)
        self.version = None

    def read_version(self):
        """Get the version number"""
        # The version number is the first occurrence, so we can just
        # load the outfilename, as we will return soon anyway
        outfilename = os.path.join(self.directory, self.outfilename)
        if not os.path.isfile(outfilename):
            return None
        with open(outfilename) as f:
            for line in f:
                if "Version " in line:
                    return " ".join(line.strip().split(" ")[1:])
        # We didn't find the version
        return None

    def get_version(self):
        if self.version is None:
            # Try if we can read the version number
            self.version = self.read_version()
        return self.version

    @property
    def out(self):
        return os.path.join(self.directory, self.outfilename)

    def set_label(self, label, update_outfilename=False):
        # msg = "Aims.set_label is not supported anymore, please use `directory`"
        # raise RuntimeError(msg)
        self.label = label
        self.directory = label

    def read(self, label=None):
        if label is None:
            label = self.label
        FileIOCalculator.read(self, label)
        geometry = os.path.join(self.directory, "geometry.in")
        control = os.path.join(self.directory, "control.in")

        for filename in [geometry, control, self.out]:
            if not os.path.isfile(filename):
                raise ReadError

        # FIX THIS LINE:
        # self.atoms, symmetry_block = read_aims(geometry, True)
        self.atoms = read_aims(geometry, True)
        self.parameters = Parameters.read(
            os.path.join(self.directory, "parameters.ase")
        )
        # AND REMOVE THIS LINE
        # if symmetry_block:
        #     self.parameters["symmetry_block"] = symmetry_block
        try:
            self.read_results()
        except RuntimeError as e:  # catch non-converged error
            raise CalculationNotConverged(e)

    def read_energy(self):
        for line in open(self.out, "r"):
            if line.rfind("Total energy corrected") > -1:
                E0 = float(line.split()[5])
            elif line.rfind("Total energy uncorrected") > -1:
                Euncorr = float(line.split()[5])
            elif line.rfind("Electronic free energy        :") > -1:
                F = float(line.split()[5])
        self.results["free_energy"] = F
        self.results["energy"] = E0
        self.results["energy_uncorr"] = Euncorr

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name not in self.implemented_properties:
            raise PropertyNotImplementedError(
                "{} property not implemented".format(name)
            )

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
            raise PropertyNotImplementedError(
                "{} not present in this " "calculation".format(name)
            )

        elif name not in self.results and self.write_input_only:
            return None
        result = self.results[name]
        if isinstance(result, np.ndarray):
            result = result.copy()
        # if not self.converged:
        #     raise CalculationNotConverged("Calculation in {} is not converged".format(self.directory))
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

    def asdict(self):
        """Return a dictionary representation of the calculator state.
        Does NOT contain information on the ``command``, ``txt`` or
        ``directory`` keywords.
        Contains the following keys:

            - ``ase_version``
            - ``vasp_version``
            - ``inputs``
            - ``results``
            - ``atoms`` (Only if the calculator has an ``Atoms`` object)
        """
        # Get versions
        asevers = ase.__version__
        version = self.get_version()

        # Store input parameters which have been set
        inputs = {
            key: value for key, value in self.parameters.items() if value is not None
        }

        dct = {
            "ase_version": asevers,
            "fhiaims_version": version,
            # '__ase_objtype__': self.ase_objtype,
            "inputs": inputs,
            "results": self.results.copy(),
        }

        if self.atoms:
            # Encode atoms as dict
            from ase.db.row import atoms2dict

            dct["atoms"] = atoms2dict(self.atoms)

        return dct

    def fromdict(self, dct):
        """Restore calculator from a :func:`~ase.calculators.vasp.Vasp.asdict`
        dictionary.

        Parameters:

        dct: Dictionary
            The dictionary which is used to restore the calculator state.
        """
        if "fhiaims_version" in dct:
            self.version = dct["fhiaims_version"]
        if "inputs" in dct:
            self.set(**dct["inputs"])
            # self._store_param_state()
        if "atoms" in dct:
            from ase.db.row import AtomsRow

            atoms = AtomsRow(dct["atoms"]).toatoms()
            self.atoms = atoms
        if "results" in dct:
            self.results.update(dct["results"])


class AMSFHIaims(AMSDFTBaseCalculator):
    _compressed_tar_gz_filename = "fhiaims.tar.gz"
    _files_to_not_compress = [
        AMSDFTBaseCalculator._data_json_filename,
        _compressed_tar_gz_filename,
    ]

    _default_kwargs = {
        "xc": "pbe",
        "charge": 0.0,
        "spin": "none",
        "occupation_type": "gaussian 0.10",
        "mixer": "pulay",
        "n_max_pulay": 10,
        "charge_mix_param": 0.05,
        "sc_iter_limit": 500,
        "sc_accuracy_rho": 1e-5,
        "sc_accuracy_eev": 1e-3,
        "sc_accuracy_etot": 1e-7,
        "relativistic": "atomic_zora scalar",
        "compute_forces": True,
        "sc_accuracy_forces": 1e-4,
    }

    def __init__(
        self,
        atoms=None,
        restart=True,
        directory=".",
        label=".",
        command=None,
        # txt='vasp.out',
        write_input_only=False,
        kmesh_spacing=0.125,
        **kwargs
    ):

        AMSDFTBaseCalculator.__init__(self, write_input_only=write_input_only)

        self.init_params = dict(
            atoms=atoms,
            restart=restart,
            directory=directory,
            label=label,
            command=command,
            # txt=txt
        )
        self.init_params.update(AMSFHIaims._default_kwargs)
        for k, v in kwargs.items():
            self.init_params[k] = v

        if kmesh_spacing:
            self.set_kmesh_spacing(kmesh_spacing)

    def _init_calculator(self):
        full_data_json_filename = os.path.join(self.directory, self._data_json_filename)

        if os.path.isfile(full_data_json_filename):
            # create calc
            self._calculator = ASEFHIaimsWrapper()
            self.deserialize()
            self._is_new_calc = False
            self._calculator.is_new_calc = self._is_new_calc
        elif not self.init_params["restart"]:
            self._calculator = ASEFHIaimsWrapper(**self.init_params)
            self._calculator.set(**self.set_kwargs)
            logging.debug("Initiate with new params")
            self._is_new_calc = True
            self._calculator.is_new_calc = self._is_new_calc
        else:
            try:
                self._calculator = ASEFHIaimsWrapper(
                    directory=self.directory,
                    # label=self.init_params["label"],
                    restart=True,
                )
                logging.debug("Initiate with restart=True only")
                self._is_new_calc = False
                self._calculator.is_new_calc = self._is_new_calc
            except (ReadError, IndexError, CalculationNotConverged):
                init_params = self.init_params.copy()
                init_params.pop("restart")
                self._calculator = ASEFHIaimsWrapper(**init_params)
                self._calculator.set(**self.set_kwargs)
                logging.debug("Fallback initiate with new params")
                self._is_new_calc = True
                self._calculator.is_new_calc = self._is_new_calc

        self._calculator.directory = self.directory
        self._calculator.write_input_only = self.write_input_only

    def read_init_atoms(self):
        # try to read data.json
        # if not succeed, then try to read POSCAR

        return super().read_init_atoms() or self.read_geometry_in_file()

    def read_geometry_in_file(self):
        geometry = os.path.join(self.directory, "geometry.in")
        if os.path.isfile(geometry):
            return read_aims(geometry)

    def optimize_atoms_only(self, max_steps=200):
        raise NotImplementedError()

    def optimize_full(self, max_steps=200):
        raise NotImplementedError()

    def optimize_cell_only(self, max_steps=200):
        raise NotImplementedError()

    def optimize_volume_only(self, max_steps=200):
        raise NotImplementedError()

    def static_calc(self):
        pass
        # raise NotImplementedError()
