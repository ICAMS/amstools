import json
import copy
import logging
import os
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from ase import Atoms
try:
    from ase.calculators.calculator import BaseCalculator
except ImportError:
    from ase.calculators.calculator import Calculator as BaseCalculator
from ase.io import jsonio

from amstools.utils import general_atoms_copy

from amstools.pipeline.exc import PausedException
from amstools.pipeline.pipelinestep import (
    serialize_class,
    deserialize_class,
    PipelineStep,
    CALCULATING,
    FINISHED,
)
from amstools.utils import (
    attach_results,
    output_structures_todict,
    output_structures_fromdict,
    atoms_todict,
    atoms_fromdict,
)

PROPERTY_JSON = "property.json"


class GeneralCalculator(PipelineStep):
    property_name = "property"
    param_names: List[str] = []

    @property
    def JOB_NAME(self):
        """Use property_name as the job name for pipeline steps."""
        return self.property_name

    @JOB_NAME.setter
    def JOB_NAME(self, value):
        """Allow pipeline to override job name."""
        self.property_name = value

    def __init__(
        self, atoms: Atoms = None, calculator: Optional[BaseCalculator] = None, **kwargs
    ):
        # Support both property-style (atoms, calculator) and pipeline-style (structure, engine)
        # Extract structure/engine from kwargs to avoid conflict when passed to parent
        structure_from_kwargs = kwargs.pop("structure", None)
        engine_from_kwargs = kwargs.pop("engine", None)

        # Use pipeline-style if provided, otherwise use property-style
        atoms = atoms or structure_from_kwargs
        calculator = calculator or engine_from_kwargs

        # Extract calculator from engine if it's a PipelineEngine
        if hasattr(calculator, "calculator"):
            calculator = calculator.calculator

        super().__init__(structure=atoms, engine=calculator, **kwargs)

        self.calculator = (
            calculator
            or getattr(atoms, "calculator", None)
            or getattr(atoms, "calc", None)
        )

        self.verbose = False
        self._value = OrderedDict()
        self._structure_dict = OrderedDict()
        self.output_dict = {}
        self.output_structures_dict = {}
        # collection of errors {structure_name: error}
        self.collected_errors = {}
        self.flag_job_is_done = False

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, value):
        if value is not None:
            if isinstance(value, Atoms):
                self._structure = general_atoms_copy(value)
            elif hasattr(value, "atoms"):
                self._structure = general_atoms_copy(value.atoms)
            else:
                self._structure = value
            self._basis_ref = self._structure
        else:
            self._structure = None
            self._basis_ref = None

    @property
    def basis_ref(self):
        return self._basis_ref

    @basis_ref.setter
    def basis_ref(self, value):
        if value is not None:
            if isinstance(value, Atoms):
                self._structure = general_atoms_copy(value)
                self._basis_ref = self._structure
            elif hasattr(value, "atoms"):
                self._structure = general_atoms_copy(value.atoms)
                self._basis_ref = self._structure
            else:
                self._structure = value
                self._basis_ref = value
        else:
            self._structure = None
            self._basis_ref = None

    def get_params_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.param_names}

    def _init_kwargs(self, **kwargs_dict):
        """Initialize optional kwargs, converting None to empty dicts.

        This helper reduces code duplication across calculator classes that
        accept optional kwargs parameters (e.g., optimizer_kwargs, phonopy_kwargs).

        Args:
            **kwargs_dict: Keyword arguments to initialize, where None values
                          will be converted to empty dicts

        Example:
            def __init__(self, atoms=None, optimizer_kwargs=None, **kwargs):
                GeneralCalculator.__init__(self, atoms, **kwargs)
                self._init_kwargs(optimizer_kwargs=optimizer_kwargs)
        """
        for name, value in kwargs_dict.items():
            setattr(self, name, value if value is not None else {})

    def calculate(
        self,
        calculator: Optional[BaseCalculator] = None,
        verbose: Optional[bool] = None,
        raise_errors: bool = False,
        **kwargs,
    ) -> None:
        """Run calculations on all generated structures.

        This method orchestrates the calculation process by:
        1. Preparing calculator and structures
        2. Running calculation loop for each structure
        3. Analyzing and finalizing results

        Args:
            calculator: Optional ASE calculator to use
            verbose: Whether to print progress messages
            raise_errors: Whether to raise errors immediately or collect them

        Raises:
            PausedException: If calculator is paused during execution
            RuntimeError: If errors were collected and raise_errors is False
        """
        self._prepare_calculation(calculator, verbose)
        self._structure_dict = self.generate_structures(verbose=self.verbose)
        self.output_dict = {}

        self._run_calculation_loop(raise_errors)
        self._finalize_calculation()

    def _prepare_calculation(self, calculator: Optional[BaseCalculator], verbose: Optional[bool]) -> None:
        """Set up calculator and verbose flag before calculation.

        Args:
            calculator: Optional calculator to use
            verbose: Whether to enable verbose output
        """
        self.verbose = verbose or getattr(self, "verbose", False)
        # Use provided calculator, or the one from the engine, or the one already set
        if calculator is not None:
            self.calculator = calculator
        elif self.engine is not None and self.engine.calculator is not None:
            self.calculator = self.engine.calculator

    def _run_calculation_loop(self, raise_errors: bool) -> None:
        """Main calculation loop with error handling.

        Args:
            raise_errors: Whether to raise errors immediately or collect them

        Raises:
            PausedException: If calculator is paused
            RuntimeError: If errors were collected
        """
        finished = False
        paused = False
        self.collected_errors = {}
        ase_calculator = self.calculator

        while not finished:
            # Freeze current structure_dict
            current_name_structures_pairs = list(self._structure_dict.items())
            total_items = len(current_name_structures_pairs)

            pbar = tqdm(
                current_name_structures_pairs,
                total=total_items,
                disable=not self.verbose,
            )

            for name, structure in pbar:
                if name in self.output_dict:  # Skip already processed
                    continue

                self._update_progress_bar(pbar, name, structure)

                with self._calculator_directory_context(ase_calculator, name):
                    if self._process_single_structure(name, structure, ase_calculator, raise_errors):
                        paused = True

            if paused:
                raise PausedException()

            if self.collected_errors:
                self._raise_collected_errors()

            # Check if all structures are processed
            finished = all(name in self.output_dict for name in self._structure_dict)

    @contextmanager
    def _calculator_directory_context(self, calculator: BaseCalculator, name: str):
        """Context manager for calculator directory switching.

        Temporarily changes the calculator's working directory for a specific
        structure calculation, then restores the original directory.

        Args:
            calculator: The ASE calculator
            name: Name of the structure (used as subdirectory name)

        Yields:
            None
        """
        old_directory = calculator.directory or "."
        directory_changed = old_directory != "."

        if directory_changed:
            calculator.directory = os.path.join(old_directory, name)
            logging.debug(
                f"GeneralCalculator::calculate: ase_calculator.directory = {calculator.directory}"
            )

        try:
            yield
        finally:
            if directory_changed:
                calculator.directory = old_directory

    def _process_single_structure(
        self,
        name: str,
        structure: Atoms,
        calculator: BaseCalculator,
        raise_errors: bool
    ) -> bool:
        """Process a single structure with error handling.

        Args:
            name: Name/identifier of the structure
            structure: ASE Atoms object to calculate
            calculator: Calculator to use
            raise_errors: Whether to raise errors immediately

        Returns:
            bool: True if calculator was paused, False otherwise

        Raises:
            Exception: If raise_errors is True and calculation fails
        """
        structure.calc = calculator
        try:
            val, out_struct = self.get_structure_value(structure, name)
            self.output_dict[name] = val
            if isinstance(out_struct, Atoms):
                out_struct = attach_results(out_struct)
            self.output_structures_dict[name] = out_struct
            return False
        except Exception as e:
            is_paused = hasattr(calculator, "paused") and calculator.paused
            if is_paused:
                return True
            if raise_errors:
                raise e
            # Collect errors to throw later
            self.collected_errors[name] = e
            return False

    def _update_progress_bar(self, pbar, name: str, structure) -> None:
        """Update tqdm progress bar with current structure info.

        Args:
            pbar: tqdm progress bar object
            name: Name of the structure
            structure: Structure being processed
        """
        msg = f"Processing: {name}"
        if isinstance(structure, Atoms):
            msg += f" ({len(structure)} atom(s))"
        pbar.set_description(msg)

    def _finalize_calculation(self) -> None:
        """Finalize calculation and set completion status."""
        self.analyse_structures(self.output_dict)
        self.flag_job_is_done = True
        self.status = FINISHED

    def _raise_collected_errors(self):
        msg = "\n".join([f"{struc}:{e}" for struc, e in self.collected_errors.items()])
        raise RuntimeError(
            f"During calculation following errors were collected:\n{msg}"
        )

    def get_structure_value(
        self,
        structure: Atoms,
        name: Optional[str] = None
    ) -> Tuple[Dict[str, Any], Atoms]:
        """Calculate properties for a single structure.

        This method must be implemented by subclasses to define what
        property/properties to calculate for each structure.

        Args:
            structure: ASE Atoms object to calculate properties for
            name: Optional identifier for this structure (used for logging/debugging)

        Returns:
            A tuple containing:
                - Dict[str, Any]: Dictionary of calculated properties (e.g., {'energy': -3.5, 'volume': 12.3})
                - Atoms: The output structure (may be relaxed/modified during calculation)

        Raises:
            NotImplementedError: Must be implemented by subclass

        Example:
            >>> def get_structure_value(self, structure, name=None):
            ...     energy = structure.get_potential_energy()
            ...     volume = structure.get_volume()
            ...     return {'energy': energy, 'volume': volume}, structure
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_structure_value()"
        )

    def analyse_structures(self, output_dict: Dict[str, Any]) -> None:
        """Analyze all calculated structures and populate self._value.

        This method is called after all structures have been calculated.
        It should process the results from get_structure_value() and store
        the final analysis in self._value dictionary.

        Args:
            output_dict: Dictionary mapping structure names to their calculation
                        results (output from get_structure_value())

        Returns:
            None: Results are stored in self._value

        Raises:
            NotImplementedError: Must be implemented by subclass

        Example:
            >>> def analyse_structures(self, output_dict):
            ...     energies = [v['energy'] for v in output_dict.values()]
            ...     self._value['min_energy'] = min(energies)
            ...     self._value['max_energy'] = max(energies)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement analyse_structures()"
        )

    def generate_structures(self, verbose: bool = False) -> OrderedDict:
        """Generate structures for calculation.

        This method must be implemented by subclasses to define what
        structures to generate based on the calculator's parameters.

        Args:
            verbose: Whether to print progress/debug information

        Returns:
            OrderedDict: Mapping of structure names (str) to ASE Atoms objects.
                        The order matters as it determines calculation sequence.

        Raises:
            NotImplementedError: Must be implemented by subclass

        Example:
            >>> def generate_structures(self, verbose=False):
            ...     structures = OrderedDict()
            ...     for strain in [0.95, 1.0, 1.05]:
            ...         atoms = self.basis_ref.copy()
            ...         atoms.set_cell(atoms.cell * strain, scale_atoms=True)
            ...         structures[f'strain_{strain}'] = atoms
            ...     return structures
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement generate_structures()"
        )

    @property
    def calculator(self):
        if self.basis_ref:
            return self.basis_ref.calc
        else:
            return self._calculator

    @calculator.setter
    def calculator(self, value):
        self._calculator = value
        if self.basis_ref:
            self.basis_ref.calc = value

    def clean(self):
        """Call the associated calculator clean method"""
        calculator = self.calculator
        if calculator and hasattr(calculator, "clean"):
            calculator.clean()

    @property
    def VALUE(self):
        # 2. Issue the warning when accessed
        warnings.warn(
            "Accessing .VALUE is deprecated. Please use .value instead.",
            DeprecationWarning,
            stacklevel=2,  # Points the warning to the line calling the code, not this line
        )
        # 3. Redirect to the new attribute
        return self.value

    @VALUE.setter
    def VALUE(self, new_val):
        # 4. Handle setting the old attribute
        warnings.warn(
            "Setting .VALUE is deprecated. Please use .value instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.value = new_val

    @property
    def value(self):
        if self.job is not None and self.job is not self:
            return self.job.value
        return self._value

    @value.setter
    def value(self, x):
        self._value = x

    def __repr__(self):
        """Friendly representation of the property calculator"""
        params = self.get_params_dict()
        params_str = ", ".join([f"{k}={repr(v)}" for k, v in params.items()])
        return f"{self.__class__.__name__}(name={repr(self.name)}, status={repr(self.status)}, {params_str})"

    def todict(self) -> Dict[str, Any]:
        calc_dct = {
            "__cls__": serialize_class(self.__class__),
            "property_name": self.property_name,
            "status": self.status,
            "finished": self.finished,
        }

        # serialize calculator's parameters
        params = self.get_params_dict()
        for k, v in params.items():
            if isinstance(v, type):
                params[k] = {"__cls__": serialize_class(v)}
        calc_dct["params"] = params

        # serialize initial structure (basis_ref)
        # TODO: preserve the constraints ?
        # if self.basis_ref:
        calc_dct["basis_ref"] = atoms_todict(self.basis_ref)

        # serialize values
        calc_dct["_VALUE"] = self.value

        # serialize self.output_structures_dict
        calc_dct["output_structures"] = output_structures_todict(
            self.output_structures_dict
        )
        return calc_dct

    @classmethod
    def fromdict(cls, calc_dct: Dict[str, Any]):
        calc_dct = copy.deepcopy(calc_dct)
        _value = calc_dct.get("_value") or calc_dct.get(
            "_VALUE"
        )  # "_VALUE" for backward compat
        basis_ref_dict = calc_dct.get("basis_ref")
        basis_ref = atoms_fromdict(basis_ref_dict) if basis_ref_dict else None

        params = calc_dct["params"]
        # deserialize classes in params:
        for k, v in params.items():
            if isinstance(v, dict) and "__cls__" in v:
                params[k] = deserialize_class(v["__cls__"])

        cls_resolved = deserialize_class(calc_dct["__cls__"])
        new_calc = cls_resolved(basis_ref, **params)
        new_calc._value = _value

        serialized_output_structures = calc_dct.get("output_structures", {})
        new_calc.output_structures_dict = output_structures_fromdict(
            serialized_output_structures
        )

        # Restore status
        if "status" in calc_dct:
            new_calc.status = calc_dct["status"]
        if "finished" in calc_dct:
            new_calc.finished = calc_dct["finished"]

        return new_calc

    @classmethod
    def from_dict(cls, calc_dct: Dict[str, Any]):
        """
        Alias for fromdict
        """
        return cls.fromdict(calc_dct)

    def to_json(self, filename="property_calculator.json"):
        dct = self.todict()
        jsonio.write_json(filename, dct)

    @classmethod
    def read_json(cls, filename="property_calculator.json"):
        dct = jsonio.read_json(filename)
        return cls.fromdict(dct)

    # # Implement abstract methods from PipelineStep
    def job_is_done(self):
        if self.finished:
            self.flag_job_is_done = True
            return True

        if bool(self.output_dict):  # Legacy check
            self.flag_job_is_done = True
            return True

        if not self.flag_job_is_done and self.working_dir is not None:
            # try to find property.json
            property_json_filename = os.path.join(self.working_dir, PROPERTY_JSON)
            if os.path.isfile(property_json_filename):
                try:
                    self.load_job(property_json_filename)
                    self.status = FINISHED
                    self.flag_job_is_done = True
                except Exception as e:
                    logging.warning(
                        f"Failed to load job from {property_json_filename}: {e}"
                    )

        return self.flag_job_is_done

    def load_job(self, property_json_filename):
        with open(property_json_filename) as f:
            calc_dct = json.load(f, object_hook=jsonio.object_hook)
        logging.debug(f"Trying to load job from {property_json_filename}")

        # Manually update self from dict
        # Since fromdict creates a new object, we need to extract the relevant parts
        calc_object = deserialize_class(calc_dct["__cls__"]).fromdict(calc_dct)

        self._value = calc_object._value
        self.output_structures_dict = calc_object.output_structures_dict
        # We might also want to update params or basis_ref if they differ,
        # but usually we trust the loaded job matches the current intentions if we are loading it.
        # Actually, if we are loading, we might want to respect the parameters on disk?
        # But 'self' is already initialized with parameters.
        # Ideally they should match.

    def job_has_error(self):
        return bool(self.collected_errors)

    def job_repair(self):
        self.status = CALCULATING

    def job_is_running(self):
        return False

    def load_final_structure(self):
        return general_atoms_copy(self.basis_ref)

    def job_cleanup(self):
        self.clean()

    def job_delete(self):
        # Implementation required by PipelineStep abstract base class
        # Currently no cleanup logic beyond job_cleanup
        pass

    def save_results_locally(self, path=None):
        if self.job_is_done() and path is not None:
            dct = self.todict()
            os.makedirs(path, exist_ok=True)
            property_json_filename = os.path.join(path, PROPERTY_JSON)
            jsonio.write_json(property_json_filename, dct)
            logging.debug(f"Property {self} saved into {property_json_filename}")

    def copy_reset(self, copy_engine=False):
        params = self.get_params_dict()
        # Merge params with options from PipelineStep - access via job_options!
        kwargs = self.job_options.options.copy()
        kwargs.update(params)

        return self.__class__(
            structure=self.structure,
            engine=None if not copy_engine else self.engine,
            allow_fail=self.allow_fail,
            name_tag=self.name_tag,
            cleanup=self.cleanup,
            force_restart=self.force_restart,
            **kwargs,
        )
