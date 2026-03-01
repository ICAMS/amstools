import logging
import os
from typing import Optional, Any, Dict, Type

from amstools.pipeline.generalstructure import GeneralStructure

try:
    from ase.filters import UnitCellFilter, StrainFilter
except ImportError:
    from ase.constraints import UnitCellFilter, StrainFilter

from ase.optimize import BFGS, FIRE, MDMin
from ase.optimize.optimize import Optimizer
from ase import Atoms

from amstools.calculators.dft.base import AMSDFTBaseCalculator
from amstools.utils import atoms_todict, atoms_fromdict, general_atoms_copy
from amstools.properties.generalcalculator import GeneralCalculator


def get_force_consistent_energy(atoms: Atoms) -> float:
    """
    Try to get force consistent energy, otherwise return potential energy.
    """
    try:
        return atoms.get_potential_energy(force_consistent=True)
    except Exception:
        return atoms.get_potential_energy()


def get_optimizer_class(optimizer_name: str) -> Type[Optimizer]:
    optimizers = {
        "BFGS": BFGS,
        "FIRE": FIRE,
        "MDMin": MDMin,
    }
    if optimizer_name not in optimizers:
        raise ValueError(
            f"Unknown optimizer class: {optimizer_name}. "
            f"Accepted: {', '.join(optimizers.keys())}"
        )
    return optimizers[optimizer_name]


class GenericOptimizer(GeneralCalculator):
    property_name = "optimization"
    param_names = []

    def __init__(
        self,
        atoms: Optional[Atoms] = None,
        calculator: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__(atoms=atoms, calculator=calculator, **kwargs)
        self.optimized_structure: Optional[Atoms] = None
        if calculator is not None:
            self.calculator = calculator


    def set_structure_calculator(
        self, structure: Optional[Atoms], calculator: Optional[Any]
    ):
        if structure is not None:
            self.structure = structure
            self.calculator = structure.calc
            self.optimized_structure = self.structure.copy()

        if calculator is not None:
            self.calculator = calculator
            self.structure.calc = self.calculator
        elif structure is not None and hasattr(structure, "calc") and structure.calc is not None:
            self.calculator = structure.calc

    def generate_structures(self, verbose: bool = False) -> Dict[str, Atoms]:
        self._structure_dict = {"optimized_structure": self.basis_ref}
        return self._structure_dict

    def get_structure_value(self, structure: Atoms, name: Optional[str] = None):
        # structure has the calculator attached
        optimized_structure = self.optimize(structure)

        # Collect results
        results = {
            "energy": get_force_consistent_energy(optimized_structure),
            "forces": optimized_structure.get_forces(),
            "n_atom": len(optimized_structure),
        }

        for prop in ["volume", "stress", "stresses"]:
            try:
                method = getattr(optimized_structure, f"get_{prop}")
                results[prop] = method()
            except Exception:
                pass

        return results, optimized_structure

    def analyse_structures(self, output_dict: Dict[str, Any]):
        self._value.update(output_dict["optimized_structure"])
        self.optimized_structure = self.output_structures_dict["optimized_structure"]
        self.optimized_structure.calc = self.calculator

    def optimize(self, init_atoms: Atoms) -> Atoms:
        raise NotImplementedError("Method optimize should be implemented in subclasses")

    def get_final_structure(self) -> Optional[Atoms]:
        return general_atoms_copy(self.optimized_structure)

    def run(
        self,
        structure: Optional[Atoms] = None,
        calculator: Optional[Any] = None,
        verbose: bool = False,
        raise_errors: bool = False,
    ) -> Atoms:
        # Support legacy signature run(structure, calculator, verbose)
        # and GeneralCalculator signature run(calculator, verbose)

        if structure is not None:
            if isinstance(structure, Atoms) or hasattr(
                structure, "get_potential_energy"
            ):
                self.structure = structure
            else:
                raise NotImplementedError(
                    f"Structure {type(structure)} is not supported"
                )

        # Set working_dir from parent_path and JOB_NAME, mirroring run_submit_check
        self.working_dir = (
            os.path.join(self.parent_path, self.JOB_NAME) if self.parent_path else None
        )

        calculator = calculator or self.calculator or self.engine.calculator

        if self.working_dir:
            self.engine.calculator.directory = self.working_dir

        self.status = "calculating"
        super().calculate(
            calculator=calculator, verbose=verbose, raise_errors=raise_errors
        )
        self.optimized_structure.calc = calculator
        if self.job_is_done():
            self.status = "finished"
            self.flag_job_is_done = True

        # Save results to disk, mirroring what run_submit_check does for other steps
        if self.working_dir and not self.engine.paused:
            self.save_results_locally(path=self.working_dir)

        return self.optimized_structure

    def todict(self) -> Dict[str, Any]:
        # Use GeneralCalculator todict but ensure structure is also saved if needed for back compat
        calc_dct = super().todict()

        # Add 'structure' as alias to 'basis_ref' for backward compatibility
        calc_dct["structure"] = calc_dct["basis_ref"]

        # Add 'optimized_structure' for backward compatibility
        if self.optimized_structure:
            calc_dct["optimized_structure"] = atoms_todict(self.optimized_structure)

        return calc_dct

    @classmethod
    def fromdict(cls, calc_dct: Dict[str, Any]):
        # Handle both 'structure' and 'basis_ref'
        if "basis_ref" not in calc_dct and "structure" in calc_dct:
            calc_dct["basis_ref"] = calc_dct["structure"]

        new_optimizer = super().fromdict(calc_dct)

        # Restore optimized_structure if present
        if "optimized_structure" in calc_dct:
            new_optimizer.optimized_structure = atoms_fromdict(
                calc_dct["optimized_structure"]
            )
        elif "optimized_structure" in new_optimizer.output_structures_dict:
            new_optimizer.optimized_structure = new_optimizer.output_structures_dict[
                "optimized_structure"
            ]

        return new_optimizer


class IsoOptimizer(GenericOptimizer):
    param_names = ["fmax", "max_steps", "optimizer"]

    def __init__(
        self,
        atoms: Optional[Atoms] = None,
        calculator: Optional[Any] = None,
        fmax: float = 0.05,
        max_steps: int = 500,
        optimizer: str = "BFGS",
        **kwargs,
    ):
        super().__init__(atoms=atoms, calculator=calculator, **kwargs)
        self.fmax = fmax
        self.max_steps = max_steps
        self.optimizer = optimizer

    def optimize(
        self, init_atoms: Optional[Atoms] = None, verbose: bool = False
    ) -> Atoms:
        init_atoms = init_atoms or self.structure
        # if PBC, run relaxation, else - skip
        if all(init_atoms.get_pbc()):
            # cell only optimization
            cell_opt = StrainFilter(init_atoms)
            opt_class = get_optimizer_class(self.optimizer)
            optimizer = opt_class(
                cell_opt, master=True, logfile="-" if verbose else None
            )
            optimizer.run(fmax=self.fmax, steps=self.max_steps)
        init_atoms.calc = self.calculator
        return init_atoms


class SpecialOptimizer(GenericOptimizer):
    param_names = ["fmax", "max_steps", "optimize_atoms_only", "optimizer"]

    def __init__(
        self,
        atoms: Optional[Atoms] = None,
        calculator: Optional[Any] = None,
        fmax: float = 0.005,
        max_steps: int = 500,
        optimize_atoms_only: bool = False,
        optimizer: str = "BFGS",
        **kwargs,
    ):
        super().__init__(atoms, calculator, **kwargs)
        self.fmax = fmax
        self.max_steps = max_steps
        self.optimize_atoms_only = optimize_atoms_only
        self.optimizer = optimizer

    @staticmethod
    def _next_optimizer(opt_class: Type[Optimizer]) -> Optional[Type[Optimizer]]:
        transitions = {BFGS: FIRE, FIRE: MDMin}
        return transitions.get(opt_class)

    def optimize(
        self,
        init_atoms: Optional[Atoms] = None,
        opt_class: Optional[Type[Optimizer]] = None,
        verbose: bool = False,
    ) -> Atoms:
        if isinstance(self.calculator, AMSDFTBaseCalculator):
            return self.optimize_dft(init_atoms=init_atoms)

        optimizer_class = opt_class or get_optimizer_class(self.optimizer)
        init_atoms = init_atoms or self.structure
        atoms = init_atoms.copy()
        atoms.calc = self.calculator
        e_init = get_force_consistent_energy(atoms)

        # State for this optimization run
        best_atoms = [atoms.copy()]
        min_energy = [e_init]

        if not self.optimize_atoms_only:
            sf = UnitCellFilter(atoms)
            dyn = optimizer_class(sf, logfile="-" if verbose else None)
        else:
            dyn = optimizer_class(atoms, logfile="-" if verbose else None)

        def observer():
            # dyn.atoms is the object being optimized (Atoms or Filter)
            # We need the underlying Atoms object for energy and storage
            obj = dyn.atoms
            en = get_force_consistent_energy(obj)
            if en < min_energy[0]:
                min_energy[0] = en
                if self.optimize_atoms_only:
                    best_atoms[0] = obj.copy()
                else:
                    # obj is UnitCellFilter, obj.atoms is Atoms
                    best_atoms[0] = obj.atoms.copy()

        dyn.attach(observer)
        try:
            dyn.run(fmax=self.fmax, steps=self.max_steps)
        except Exception as e:
            print(f"OPTIMIZATION error: {e}")
            atoms = init_atoms

        e_final = get_force_consistent_energy(atoms)
        energy_eps = 1e-3

        if e_init < e_final and abs(e_init - e_final) > energy_eps:
            logging.warning(
                f"! BAD OPTIMIZATION, energy increases by {e_final - e_init} "
                f"from {e_init} to {e_final}"
            )
            # Fallback to best found atoms
            atoms = best_atoms[0]
            atoms.calc = self.calculator

            next_opt = self._next_optimizer(optimizer_class)
            if next_opt is not None:
                try:
                    atoms = self.optimize(atoms, next_opt)
                    atoms.calc = self.calculator
                    if get_force_consistent_energy(atoms) > e_init:
                        atoms = init_atoms
                except Exception as e:
                    logging.error(f"OPTIMIZATION error: {e}")
                    raise e

        atoms.calc = self.calculator
        return atoms

    def optimize_dft(self, init_atoms: Optional[Atoms] = None) -> Atoms:
        init_atoms = init_atoms or self.structure
        atoms = init_atoms.copy()

        if self.optimize_atoms_only:
            self.calculator.optimize_atoms_only(
                ediffg=-self.fmax, max_steps=self.max_steps
            )
        else:
            self.calculator.optimize_full(ediffg=-self.fmax, max_steps=self.max_steps)

        atoms.calc = self.calculator
        self.calculator.auto_kmesh_spacing = True
        # DFT must have force consistent energy
        atoms.get_potential_energy(force_consistent=True)
        self.calculator.auto_kmesh_spacing = False
        atoms = self.calculator.atoms
        atoms.calc = self.calculator
        return atoms


class StepwiseOptimizer(GenericOptimizer):
    param_names = [
        "fmax",
        "max_steps",
        "dEmax",
        "relative_volume_change_tolerance",
        "max_restart",
        "optimizer",
    ]

    def __init__(
        self,
        atoms: Optional[Atoms] = None,
        calculator: Optional[Any] = None,
        fmax: float = 0.005,
        dEmax: float = 1e-3,
        max_steps: int = 500,
        relative_volume_change_tolerance: float = 0.01,
        max_restart: int = 5,
        optimizer: str = "BFGS",
        **kwargs,
    ):
        super().__init__(atoms=atoms, calculator=calculator, **kwargs)
        self.fmax = float(fmax)
        self.max_steps = int(max_steps)
        self.dEmax = float(dEmax)
        self.relative_volume_change_tolerance = float(relative_volume_change_tolerance)
        self.max_restart = int(max_restart)
        self.optimizer = optimizer

    def stepwise_optimization(
        self,
        init_atoms: Atoms,
        calc: Any,
        max_steps: int = 500,
        dEmax: float = 1e-3,
        fmax: float = 1e-6,
        verbose: bool = False,
    ) -> Atoms:
        init_atoms.calc = calc
        converged = False
        iteration = 0
        current_atoms = init_atoms.copy()
        prev_en = 0.0

        opt_class = get_optimizer_class(self.optimizer)

        while not converged and iteration < self.max_restart:
            # atomic only optimization
            current_atoms.calc = calc
            atomic_opt = opt_class(current_atoms, logfile="-" if verbose else None)
            atomic_opt.run(fmax=fmax, steps=max_steps)

            # cell only optimization
            cell_opt = StrainFilter(current_atoms)
            opt = opt_class(cell_opt, master=True, logfile="-" if verbose else None)
            opt.run(fmax=fmax, steps=self.max_steps)

            dE = get_force_consistent_energy(current_atoms) - prev_en
            prev_en = current_atoms.get_potential_energy(force_consistent=True)
            if abs(dE) < dEmax:
                converged = True

            iteration += 1
        return current_atoms

    def optimize(
        self,
        init_atoms: Optional[Atoms] = None,
        max_steps: Optional[int] = None,
        dEmax: Optional[float] = None,
        fmax: Optional[float] = None,
        verbose: bool = False,
    ) -> Atoms:
        if isinstance(self.calculator, AMSDFTBaseCalculator):
            opt_struct = self.optimize_dft(init_atoms=init_atoms)
        else:
            init_atoms = init_atoms or self.structure
            max_steps = max_steps or self.max_steps
            dEmax = dEmax or self.dEmax
            fmax = fmax or self.fmax
            opt_struct = self.stepwise_optimization(
                init_atoms,
                self.calculator,
                max_steps=max_steps,
                dEmax=dEmax,
                fmax=fmax,
                verbose=verbose,
            )
        opt_struct.calc = self.calculator
        return opt_struct

    def optimize_dft(self, init_atoms: Optional[Atoms] = None) -> Atoms:
        init_atoms = init_atoms or self.structure
        atoms = init_atoms.copy()
        self.calculator.optimize_full(ediffg=-self.fmax, max_steps=self.max_steps)
        atoms.calc = self.calculator
        self.calculator.auto_kmesh_spacing = True
        path = self.calculator.directory

        iteration = 0
        while iteration < self.max_restart:
            volume = atoms.get_volume()
            # DFT must have force-consistent energy
            atoms.get_potential_energy(force_consistent=True)
            # optimized structures
            atoms = self.calculator.atoms
            atoms.calc = self.calculator

            vol_change = abs(atoms.get_volume() - volume)
            if (
                volume == 0
                or vol_change <= self.relative_volume_change_tolerance * volume
            ):
                break

            iteration += 1
            self.calculator.directory = os.path.join(path, f"stage_{iteration}")

        self.calculator.auto_kmesh_spacing = False
        return atoms
