import copy
import logging
import os
import tarfile

import numpy as np
from ase import Atoms
from ase.calculators.calculator import compare_atoms
from ase.io import jsonio
from amstools.pipeline import scheduler as pq

from amstools.utils import serialize_class, deserialize_class

MAX_KPOINT_PER_DIRECTION = 25


class WriteInputOnlyException(Exception):
    pass


class CalculationNotConverged(Exception):
    pass


def get_k_mesh_by_kspacing(cell, kmesh_spacing=1.0):
    """
    Args:
        cell:
        kmesh_spacing:
    Returns:
    """
    omega = np.linalg.det(cell)
    l1, l2, l3 = cell
    g1 = 2 * np.pi / omega * np.cross(l2, l3)
    g2 = 2 * np.pi / omega * np.cross(l3, l1)
    g3 = 2 * np.pi / omega * np.cross(l1, l2)

    kmesh = np.rint(np.array([np.linalg.norm(g) for g in [g1, g2, g3]]) / kmesh_spacing)
    kmesh[kmesh < 1] = 1
    kmesh[kmesh > MAX_KPOINT_PER_DIRECTION] = MAX_KPOINT_PER_DIRECTION
    return [int(k) for k in kmesh]


class AMSDFTBaseCalculator:
    _data_json_filename = "data.json"
    _compressed_tar_gz_filename = "calculation.tar.gz"
    _files_to_not_compress = []
    _files_to_remove = []

    def __init__(
        self, write_input_only=False, auto_kmesh_spacing=False, kmesh_max=None
    ):
        self.kmesh_spacing = None
        self.auto_kmesh_spacing = auto_kmesh_spacing
        self.kmesh_max = kmesh_max
        self.init_params = {}
        self.set_kwargs = {}
        self._calculator = None
        self._is_new_calc = True
        self.write_input_only = write_input_only
        self.paused = False
        self.paused_calculations_dirs = []

    def static_calc(self):
        raise NotImplementedError()

    def optimize_atoms_only(self, *args, **kwargs):
        raise NotImplementedError()

    def optimize_full(self, *args, **kwargs):
        raise NotImplementedError()

    def optimize_cell_only(self, *args, **kwargs):
        raise NotImplementedError()

    def optimize_volume_only(self, *args, **kwargs):
        raise NotImplementedError()

    def set_kmesh_spacing(self, kmesh_spacing=1.0):
        self.kmesh_spacing = kmesh_spacing
        self.auto_kmesh_spacing = True

    def set_kmesh(self, kmesh=None, kmesh_spacing=None):
        if self.set_kwargs.get("kpts") != kmesh:
            logging.debug("AMSDFTCalculator set kmesh = {}".format(kmesh))
            self.set(kpts=kmesh)

    def get_kmesh(self):
        if self._calculator is not None:
            return self._calculator.input_params.get("kpts")

    def update_kmesh_from_spacing(self, cell_or_structure, kmesh_spacing=None):

        kmesh_spacing = self.kmesh_spacing if kmesh_spacing is None else kmesh_spacing

        if isinstance(cell_or_structure, Atoms):
            cell = cell_or_structure.cell
        else:
            cell = cell_or_structure
        if kmesh_spacing is not None:
            kmesh = get_k_mesh_by_kspacing(cell, kmesh_spacing=kmesh_spacing)
            if self.kmesh_max is not None:
                for index, k in enumerate(kmesh):
                    if k > self.kmesh_max:
                        kmesh[index] = self.kmesh_max
            self.set_kmesh(kmesh)

    def _init_calculator(self):
        raise NotImplementedError()

    @property
    def is_new_calc(self):
        if self._calculator is None:
            self._init_calculator()
        return self._is_new_calc

    def set(self, **kwargs):
        self.set_kwargs.update(kwargs)

        if self._calculator is not None:
            # to prevent the reset of ASE calculator state,
            # we do update only if parameters is different!
            for k, v in kwargs.items():
                eq = self._calculator.parameters.get(k) == v
                if not isinstance(eq, bool):
                    eq = np.all(eq)
                if not eq:
                    self._calculator.set(**{k: v})

    @property
    def inner_calculator(self):
        if self._calculator is None:
            self._init_calculator()
        return self._calculator

    @property
    def directory(self):
        return self.init_params["directory"]

    @directory.setter
    def directory(self, val):
        # TODO: update/reload DFT calc from new directory if possible
        logging.debug("AMSDFTBaseCalculator.directory = {}".format(val))
        self.init_params["directory"] = val
        if self._calculator is not None:
            self._calculator.set(directory=val)
            # TODO: check if we really need it?
            old_atoms = self.atoms
            self._init_calculator()
            if self.atoms is None:
                if old_atoms is not None:
                    self.atoms = old_atoms
            logging.debug("AMSDFTBaseCalculator._calculator.directory = {}".format(val))

    def calculate(self, *args, **kwargs):
        return self.inner_calculator.calculate(*args, **kwargs)

    @property
    def results(self):
        return self.inner_calculator.results

    @property
    def atoms(self):
        return self.inner_calculator.atoms

    @atoms.setter
    def atoms(self, val):
        self.inner_calculator.atoms = val

    def deserialize(self):
        full_data_json_filename = os.path.join(self.directory, self._data_json_filename)
        logging.debug("Deserialize from {}".format(full_data_json_filename))
        dct = jsonio.read_json(full_data_json_filename)
        self._calculator.fromdict(dct)
        self._calculator.converged = dct.get("converged")

    def serialize(self, json_filename=None):
        if json_filename is None:
            json_filename = os.path.join(self.directory, self._data_json_filename)
        if not os.path.isfile(json_filename):
            dct = self._calculator.asdict()
            init_atoms = self.read_init_atoms()
            init_atoms.calc = None
            dct["init_atoms"] = init_atoms.todict()
            dct["converged"] = self.converged
            jsonio.write_json(json_filename, dct)
            logging.debug("Serialization to {}".format(json_filename))

    def read_json_init_atoms(self):
        full_data_json_filename = os.path.join(self.directory, self._data_json_filename)
        if os.path.isfile(full_data_json_filename):
            dct = jsonio.read_json(full_data_json_filename)
            if "init_atoms" in dct:
                from ase.db.row import AtomsRow

                atoms = AtomsRow(dct["init_atoms"]).toatoms()
                return atoms

    def _prepare_atoms_and_kmesh(self, atoms):
        inner_calculator = self.inner_calculator
        # reset atoms to None if data.json exists and identical
        atoms = atoms or self.atoms
        if not self.is_new_calc:  # or self._compare_to_init_atoms(atoms):
            logging.debug(
                "DFT calculations are loaded from {}, no calculations performed".format(
                    self.directory
                )
            )
            atoms = None
        elif (
            self.auto_kmesh_spacing
            and atoms is not None
            and self.kmesh_spacing is not None
        ):
            if all(atoms.pbc):
                kmesh = get_k_mesh_by_kspacing(atoms.cell, self.kmesh_spacing)
                logging.debug("Adjusting kmesh to {}".format(kmesh))
                if self.kmesh_max is not None:
                    for index, k in enumerate(kmesh):
                        if k > self.kmesh_max:
                            kmesh[index] = self.kmesh_max
                self.set_kmesh(kmesh)
        return atoms, inner_calculator

    def is_compressed(self):
        tar_filename = os.path.join(self.directory, self._compressed_tar_gz_filename)
        return os.path.isfile(tar_filename)

    def compress(self):
        # pass
        abs_directory = os.path.abspath(self.directory)
        tar_filename = os.path.join(abs_directory, self._compressed_tar_gz_filename)
        tmp_tar_filename = tar_filename + ".tmp"
        logging.debug(
            "{} job compression at {} to {}".format(
                self.__class__.__name__, abs_directory, tar_filename
            )
        )
        old_cwd = os.getcwd()
        try:
            os.chdir(abs_directory)
            # remove unnecessary files
            for filename in self._files_to_remove:
                if os.path.isfile(filename):
                    os.remove(filename)
            # compress necessary files
            compressed_files = []
            with tarfile.open(tmp_tar_filename, "w:gz") as tar:
                for filename in os.listdir():
                    if filename not in self._files_to_not_compress:
                        tar.add(filename)
                        compressed_files.append(filename)
            os.rename(tmp_tar_filename, tar_filename)
            logging.debug("Removing DFT job files")

            # remove comressed files
            for filename in compressed_files:
                if os.path.isfile(filename):
                    os.remove(filename)
            logging.debug("DFT job compression done")
        finally:
            os.chdir(old_cwd)

    def decompress(self):
        abs_directory = os.path.abspath(self.directory)
        tar_filename = os.path.join(abs_directory, self._compressed_tar_gz_filename)
        logging.debug(
            "DFT job compression at {} to {}".format(abs_directory, tar_filename)
        )
        old_cwd = os.getcwd()
        try:
            os.chdir(abs_directory)
            with tarfile.open(tar_filename, "r:gz") as tar:
                tar.extractall()
        finally:
            os.chdir(old_cwd)

    def get_potential_energy(self, atoms=None, force_consistent=False, **kwargs):
        logging.debug(
            "{}.get_potential_energy(atoms={})".format(self.__class__.__name__, atoms)
        )
        atoms, calc = self._prepare_atoms_and_kmesh(atoms)

        try:
            self.paused = False
            res = calc.get_potential_energy(
                atoms=atoms, force_consistent=force_consistent, **kwargs
            )
            self.serialize()
            if not self.is_compressed():
                self.compress()
        except WriteInputOnlyException:
            res = None
            self.paused = True
            if self.directory not in self.paused_calculations_dirs:
                self.paused_calculations_dirs.append(self.directory)
        return res

    def get_property(self, atoms=None, **kwargs):
        atoms, calc = self._prepare_atoms_and_kmesh(atoms)
        try:
            self.paused = False
            res = calc.get_property(atoms=atoms, **kwargs)
            self.serialize()
            if not self.is_compressed():
                self.compress()
        except WriteInputOnlyException:
            res = None
            self.paused = True
            if self.directory not in self.paused_calculations_dirs:
                self.paused_calculations_dirs.append(self.directory)

        return res

    def get_forces(self, atoms=None, **kwargs):
        atoms, calc = self._prepare_atoms_and_kmesh(atoms)
        try:
            self.paused = False
            res = calc.get_forces(atoms=atoms, **kwargs)
            self.serialize()
            if not self.is_compressed():
                self.compress()
        except WriteInputOnlyException:
            res = None
            self.paused = True
            if self.directory not in self.paused_calculations_dirs:
                self.paused_calculations_dirs.append(self.directory)
        return res

    def get_stresses(self, atoms=None, **kwargs):
        atoms, calc = self._prepare_atoms_and_kmesh(atoms)
        try:
            self.paused = False
            res = calc.get_stresses(atoms=atoms, **kwargs)
            self.serialize()
            if not self.is_compressed():
                self.compress()
        except WriteInputOnlyException:
            res = None
            self.paused = True
            if self.directory not in self.paused_calculations_dirs:
                self.paused_calculations_dirs.append(self.directory)

        return res

    def get_stress(self, atoms=None, **kwargs):
        atoms, calc = self._prepare_atoms_and_kmesh(atoms)
        try:
            self.paused = False
            res = calc.get_stress(atoms=atoms, **kwargs)
            self.serialize()
            if not self.is_compressed():
                self.compress()
        except WriteInputOnlyException:
            res = None
            self.paused = True
            if self.directory not in self.paused_calculations_dirs:
                self.paused_calculations_dirs.append(self.directory)
        return res

    @property
    def name(self):
        return self.inner_calculator.name

    @property
    def converged(self):
        try:
            return bool(self.inner_calculator.converged)
        except AttributeError:
            return False

    def todict(self):
        calc_dct = self.inner_calculator.asdict()
        calc_dct["converged"] = self.converged
        calc_dct["__cls__"] = serialize_class(self.inner_calculator.__class__)
        dct = {
            "__cls__": serialize_class(self.__class__),
            "_calculator": calc_dct,
            "write_input_only": self.write_input_only,
            "init_params": self.init_params,
            "kmesh_spacing": self.kmesh_spacing,
            "auto_kmesh_spacing": self.auto_kmesh_spacing,
            "kmesh_max": self.kmesh_max,
            "set_kwargs": self.set_kwargs,
        }
        return dct

    @classmethod
    def fromdict(cls, dct):
        _ams_calculator_class = dct.get("__cls__") or dct.get("_ams_calculator_class")

        # for backward compatibility
        if _ams_calculator_class == "AMSVasp":
            from amstools.calculators.dft.vasp import AMSVasp

            des_cls = AMSVasp
        else:
            des_cls = deserialize_class(_ams_calculator_class)
        new_calculator = des_cls()

        inner_calc_dct = dct.get("_calculator") or dct.get(
            "_vasp_calculator"
        )  # for backward compat
        write_input_only = dct["write_input_only"]
        init_params = dct["init_params"]
        set_kwargs = dct["set_kwargs"]

        new_calculator.init_params = init_params
        new_calculator.set_kwargs = set_kwargs
        new_calculator.write_input_only = write_input_only

        calc_cls = deserialize_class(inner_calc_dct.get("__cls__"))
        # for backward compatibility
        if calc_cls is None:
            from amstools.calculators.dft.vasp import ASEVaspWrapper

            calc_cls = ASEVaspWrapper

        _calculator = calc_cls()
        _calculator.fromdict(inner_calc_dct)
        _calculator.converged = inner_calc_dct.get("converged")
        new_calculator._calculator = _calculator
        new_calculator.kmesh_spacing = dct["kmesh_spacing"]
        new_calculator.kmesh_max = dct.get("kmesh_max")
        new_calculator.auto_kmesh_spacing = dct.get(
            "auto_kmesh_spacing", True if new_calculator.kmesh_spacing else False
        )
        new_calculator._is_new_calc = False
        return new_calculator

    def _compare_to_init_atoms(self, atoms):
        """
        Compare `atoms` to init atoms from data.json or POSCAR (if any)
        """
        if atoms is not None:
            init_atoms = self.read_init_atoms()
            system_changes = compare_atoms(init_atoms, atoms, tol=1e-9)
            if not system_changes:
                return True
        return False

    def read_init_atoms(self):
        # try to read data.json
        return self.read_json_init_atoms()

    def save(self, filename="calculation.json", **kwargs):
        dct = self.todict()
        dct.update(kwargs)
        working_dir = os.path.dirname(filename)
        if working_dir and not os.path.isdir(working_dir):
            os.makedirs(working_dir)
        jsonio.write_json(filename, dct)

    @classmethod
    def load(cls, filename="calculation.json"):
        dct = jsonio.read_json(filename)
        _cls = deserialize_class(dct.get("__cls__"))
        if _cls is not None:
            calc = _cls.fromdict(dct)
            return calc
        else:
            raise ValueError(
                "Could not deserialize class from file {}".format(filename)
            )

    def submit_to_scheduler(self, options: dict, working_dir=None):
        # raise NotImplementedError()
        if working_dir is None:
            if self.directory is not None:
                working_dir = self.directory
            else:
                raise ValueError(
                    "Neither AMSDFTCalculator.directory nor working_dir are provided"
                )

        working_dir = os.path.abspath(working_dir)
        if not os.path.isdir(working_dir):
            os.makedirs(working_dir)

        if options["scheduler"] == "slurm":
            scheduler = pq.SLURM(options, cores=options["cores"], directory=working_dir)
        elif options["scheduler"] == "sge":
            scheduler = pq.SGE(options, cores=options["cores"], directory=working_dir)
        else:
            raise ValueError("Unknown scheduler")

        fname = os.path.join(working_dir, "calculation.json")
        self.save(fname)
        scheduler.maincommand = "ams_calculation {} -d {}".format(fname, working_dir)
        submission_script = os.path.join(working_dir, "ams_pipeline_job.sh")
        scheduler.write_script(submission_script)

        res = scheduler.submit()
        lines = res.stdout.readlines()
        # print("Submission result: {}".format(lines[0].decode()))
        return scheduler.get_job_id(lines)

    def submit_remote_calculation(
        self, remote_connection_options=None, connection=None, sftp=None
    ):
        # 1. save calculation to file
        # 2. connect to remote login node
        # 3. open sftp,
        # 4. navigate to specific location (root project path) remotely
        # 5. create new calc folder, put calculator.json there
        # 6. store locally the remote path
        # 7. submit it to the queue (options submission stored locally/remotely)
        from fabric.connection import Connection

        calc_json = os.path.join(self.directory, "calculation.json")
        self.save(calc_json)

        if connection is not None:
            c = connection
        elif "connection" in remote_connection_options:
            c = Connection(**remote_connection_options.get("connection"))
        else:
            raise ValueError(
                "Provide either connect:fabric.Connection or remote_connection_options:dict"
            )

        if sftp is not None:
            s = sftp
        else:
            s = c.sftp()

        remote_directory = self.get_remote_path(c, s)

        # dirlist = s.listdir()
        # if self.directory not in dirlist:
        if c.run("test -d {}".format(remote_directory), warn=True).failed:
            # Folder doesn't exist
            s.mkdir(remote_directory)

        s.chdir(remote_directory)
        dirlist = s.listdir()
        if "calculation.json" in dirlist or self._data_json_filename in dirlist:
            # do not send remote job if calculation.json or data.json are already there
            self.save(calc_json, host=c.host, remote_path=remote_directory)
            return False

        remote_calculation_json = os.path.join(remote_directory, "calculation.json")

        s.put(calc_json, remote_calculation_json)

        res = c.run(
            "conda activate ace && cd {} && ams_calculation -s".format(remote_directory)
        )

        if res.return_code == 0:
            lines = [l for l in res.stderr.split("\n") if l]
            submission_result = lines[-1].split("Submission result:")[-1].strip()
            self.save(
                calc_json,
                host=c.host,
                remote_path=remote_directory,
                submission_result=submission_result,
            )
        return True

    def get_remote_path(self, c, s):
        """
        Get remote path = remote root path + current calculations's working directory
        :param c: fabric.Connection
        :param s: fabric.sftp
        :return: remote_directory:str
        """
        if hasattr(s, "remote_project_root_path"):
            remote_project_root_path = s.remote_project_root_path
        else:
            remote_project_root_path = c.run(
                "echo $AMS_REMOTE_CALC_PATH", hide=True
            ).stdout.strip()
            if len(remote_project_root_path) == 0:
                remote_project_root_path = "${HOME}/ams_remote_calc"
            s.remote_project_root_path = remote_project_root_path
        logging.debug("remote_project_root_path = {}".format(remote_project_root_path))
        s.chdir(remote_project_root_path)
        remote_directory = os.path.join(remote_project_root_path, self.directory)
        return remote_directory

    def check_calculation(
        self, remote_connection_options=None, connection=None, sftp=None
    ):
        # Check status
        # 8. connect, go to specific location, check data.json, get it back, store to local mirror folder
        from fabric.connection import Connection

        calc_json = os.path.join(self.directory, "calculation.json")
        dct = jsonio.read_json(calc_json)
        host = dct.get("host")
        remote_path = dct.get("remote_path")

        if connection is not None:
            c = connection
        elif "connection" in remote_connection_options:
            c = Connection(**remote_connection_options.get("connection"))
        else:
            raise ValueError(
                "Provide either connect:fabric.Connection or remote_connection_options:dict"
            )

        if sftp is not None:
            s = sftp
        else:
            s = c.sftp()

        logging.debug("remote_path = {}".format(remote_path))

        s.chdir(remote_path)

        dirlist = s.listdir()
        if self._data_json_filename in dirlist:
            _data_json_filename = os.path.join(self.directory, self._data_json_filename)
            # data_json
            if not os.path.isfile(_data_json_filename):
                remote_calculation_json = os.path.join(
                    remote_path, self._data_json_filename
                )
                s.get(remote_calculation_json, _data_json_filename)
            # _compressed_tar_gz_filename

            _compressed_tar_gz_filename = os.path.join(
                self.directory, self._compressed_tar_gz_filename
            )
            if not os.path.isfile(_compressed_tar_gz_filename):
                remote_compressed_tar_gz_filename = os.path.join(
                    remote_path, self._compressed_tar_gz_filename
                )
                s.get(remote_compressed_tar_gz_filename, _compressed_tar_gz_filename)
            self._init_calculator()
            return True
        else:
            return False

    def is_finished(self):
        """
        Check if resulting data json file is presented
        """
        _data_json_filename = os.path.join(self.directory, self._data_json_filename)
        if os.path.isfile(_data_json_filename):
            return True
        else:
            return False

    def copy(self):
        return copy.copy(self)
