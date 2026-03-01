import numpy as np

from ase.optimize import BFGS
from amstools.calculators.dft.base import AMSDFTBaseCalculator

from amstools.utils import (
    fixsymmetry_imported,
    FixSymmetry,
    get_wyckoffs,
    build_job_name,
)


from amstools.properties.generalcalculator import GeneralCalculator
from amstools.properties.relaxation import StepwiseOptimizer


class DefectFormationCalculator(GeneralCalculator):
    """Calculation of vacancy formation energy in different Wyckoff positions

    Args:
         :param atoms: original ASE Atoms object
         :param interaction_range: supercell size, in Angstroms (default = 10.)
         :param defect_type: only "vacancy" is supported
         :param optimizer: optimizer class from ase.optimize, default = BFGS,
            could be also LBFGS, BFGSLineSearch, LBFGSLineSearch, MDMin, FIRE, QuasiNewton, GoodOldQuasiNewton
         :param fmax: fmax option for optimizer
         :param optimizer_kwargs: additional keyword arguments for optimizer class,
            f.e. optimizer_kwargs={"maxstep":0.05}

    Usage:
    >>  atoms.calc=calculator
    >>  defect_formation = DefectFormationCalculator(atoms)
    >>  defect_formation.calculate()
    >>  print(defect_formation.value["vacancy_formation_energy"]) # will print vacancy formation energy
    """

    property_name = "defect"

    param_names = [
        "interaction_range",
        "defect_type",
        "optimizer",
        "fmax",
        "dEmax",
        "optimizer_kwargs",
        "fix_symmetry",
    ]
    SUPERCELL_DEFECT_NAME = "supercell_defect"
    SUPERCELL_0_NAME = "supercell_0"
    STATIC = "static"
    ATOMIC = "atomic"
    TOTAL = "total"
    minimization_styles = [STATIC, ATOMIC, TOTAL]

    def __init__(
        self,
        atoms=None,
        interaction_range=10.0,
        defect_type="vacancy",
        optimizer=BFGS,
        fmax=0.005,
        dEmax=1e-3,
        optimizer_kwargs=None,
        fix_symmetry=True,
        **kwargs,
    ):
        GeneralCalculator.__init__(self, atoms, **kwargs)

        self.fix_symmetry = fix_symmetry
        self.interaction_range = interaction_range
        self.defect_type = defect_type
        self.optimizer = optimizer
        self._init_kwargs(optimizer_kwargs=optimizer_kwargs)
        self.fmax = fmax
        self.dEmax = dEmax

    def generate_structures(self, verbose=False):
        INTERACTION_DIST = self.interaction_range
        unitcell = self.basis_ref.get_cell()
        supercell_range = np.round(
            np.ceil(
                INTERACTION_DIST / np.array([np.linalg.norm(vec) for vec in unitcell])
            )
        )
        supercell_range = [int(val) for val in supercell_range]
        self._value["supercell_range"] = supercell_range

        struct = self.basis_ref.copy()
        supercell_0 = struct.repeat(supercell_range)
        n = len(supercell_0)
        self._value["n"] = n

        wyckofs = get_wyckoffs(supercell_0)
        wyckofs_set = set(wyckofs)
        wyck_dict = {}
        for wyck in wyckofs_set:
            w = wyckofs[0]
            i = 0
            for i, w in enumerate(wyckofs):
                if w == wyck:
                    break
            wyck_dict[w] = i
        self._value["wyckoff_unique_indices"] = wyck_dict

        # Stage 1. Supercell
        self._structure_dict[self.SUPERCELL_0_NAME] = supercell_0

        # Stage 2. Supercell with vacancy
        for wyck_name, wyck_ind in wyck_dict.items():
            supercell_defect = supercell_0.copy()
            del supercell_defect[wyck_ind]
            for minstyle in self.minimization_styles:
                job_name = self.subjob_name(minstyle, wyck_name)
                self._structure_dict[job_name] = supercell_defect

        return self._structure_dict

    @staticmethod
    def subjob_name(minstyle, wyck_name):
        return build_job_name(
            DefectFormationCalculator.SUPERCELL_DEFECT_NAME,
            'wyck',
            wyck_name,
            minstyle
        )

    def analyse_structures(self, output_dict):
        self.generate_structures()
        energies = {}
        volumes = {}
        for job_name in self._structure_dict.keys():
            energies[job_name] = output_dict[job_name]["energy"]
            volumes[job_name] = output_dict[job_name]["volume"]

        self._value["energies"] = energies
        self._value["volumes"] = volumes
        self.calculate_defect_formation()

    def calculate_defect_formation(self):
        self.generate_structures()
        # check number of species types
        symb = sorted(set(self.basis_ref.get_chemical_symbols()))

        if len(symb) > 1:
            raise NotImplementedError(
                f"Vacancy formation energy calculation is only valid for unary systems. "
                f"Found {len(symb)} species: {symb}. "
                f"Please compute manually with proper chemical potentials."
            )

        wyck_dict = self._value["wyckoff_unique_indices"]
        n = self._value["n"]
        e0 = self._value["energies"][self.SUPERCELL_0_NAME]
        v0 = self._value["volumes"][self.SUPERCELL_0_NAME]
        defect_formation_energy_value = {}
        defect_formation_volume_value = {}
        for wyck_name in wyck_dict.keys():
            for minstyle in self.minimization_styles:
                e1 = self._value["energies"][
                    self.SUPERCELL_DEFECT_NAME + "_wyck_" + wyck_name + "_" + minstyle
                ]
                v1 = self._value["volumes"][
                    self.SUPERCELL_DEFECT_NAME + "_wyck_" + wyck_name + "_" + minstyle
                ]
                def_fe = e1 - (n - 1.0) / n * e0  # TODO: valid only for unaries!
                def_v = v1 - v0
                defect_formation_energy_value[wyck_name + "_" + minstyle] = def_fe
                if minstyle == "total":
                    defect_formation_volume_value[wyck_name + "_" + minstyle] = def_v
        self._value["vacancy_formation_energy"] = defect_formation_energy_value
        self._value["vacancy_formation_volume"] = defect_formation_volume_value

    def get_structure_value(self, structure, name=None):
        logfile = "-" if self.verbose else None
        if isinstance(structure.calc, AMSDFTBaseCalculator):
            calc = structure.calc
            structure.calc.auto_kmesh_spacing = True
            if name == DefectFormationCalculator.SUPERCELL_0_NAME:
                structure.calc.static_calc()
            elif DefectFormationCalculator.SUPERCELL_DEFECT_NAME in name:
                if DefectFormationCalculator.STATIC in name:
                    structure.calc.static_calc()
                elif DefectFormationCalculator.ATOMIC in name:
                    calc.optimize_atoms_only(ediffg=-self.fmax, max_steps=100)
                elif DefectFormationCalculator.TOTAL in name:
                    calc.optimize_full(ediffg=-self.fmax, max_steps=100)

            # do calculations
            structure.get_potential_energy(force_consistent=True)
            structure = structure.calc.atoms
            structure.calc = calc
        else:
            if name == DefectFormationCalculator.SUPERCELL_0_NAME:
                pass
            elif DefectFormationCalculator.SUPERCELL_DEFECT_NAME in name:
                if DefectFormationCalculator.STATIC in name:
                    # dyn = structure
                    pass
                elif DefectFormationCalculator.ATOMIC in name:
                    if fixsymmetry_imported and self.fix_symmetry:
                        fix_atoms = FixSymmetry(structure, verbose=True)
                        structure.set_constraint(fix_atoms)
                    dyn = self.optimizer(
                        structure, logfile=logfile, **self.optimizer_kwargs
                    )
                    try:
                        dyn.set_force_consistent()
                    except (TypeError, AttributeError):
                        dyn.force_consistent = False
                    dyn.run(fmax=self.fmax)
                    structure = dyn.atoms
                elif DefectFormationCalculator.TOTAL in name:
                    if fixsymmetry_imported and self.fix_symmetry:
                        fix_atoms = FixSymmetry(structure, verbose=True)
                        structure.set_constraint(fix_atoms)
                    full_opt = StepwiseOptimizer(
                        atoms=structure, fmax=self.fmax, dEmax=self.dEmax
                    )
                    structure = full_opt.run()

        en = structure.get_potential_energy(force_consistent=True)
        vol = structure.get_volume()
        return {"energy": en, "volume": vol}, structure

    def plot(self, ax=None, **kwargs):
        """
        Plot vacancy formation energy across different Wyckoff positions
        """
        import matplotlib.pyplot as plt

        if "vacancy_formation_energy" not in self.value:
            print("No vacancy formation energy found in results")
            return

        engs = self.value["vacancy_formation_energy"]
        
        # Filter for 'total' minimization style if available, otherwise 'atomic', then 'static'
        plot_data = {}
        for style in ["total", "atomic", "static"]:
            data = {k.replace("_" + style, ""): v for k, v in engs.items() if k.endswith("_" + style)}
            if data:
                plot_data = data
                label = style
                break
        
        if not plot_data:
            return

        if ax is None:
            fig, ax = plt.subplots()
        
        names = list(plot_data.keys())
        values = list(plot_data.values())
        
        ax.bar(names, values, **kwargs)
        ax.set_ylabel("Formation Energy (eV)")
        ax.set_xlabel("Wyckoff Position")
        ax.set_title(f"Vacancy Formation Energy ({label} relaxation)")
        plt.xticks(rotation=45)
        
        return ax
