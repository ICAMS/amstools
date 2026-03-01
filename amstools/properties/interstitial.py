import os
from collections import defaultdict

import numpy as np
from ase.io import read
from ase.optimize import FIRE

from amstools.calculators.dft.base import AMSDFTBaseCalculator
from amstools.utils import get_spacegroup, build_job_name
from amstools.utils import fixsymmetry_imported, FixSymmetry


from amstools.properties.generalcalculator import GeneralCalculator
from amstools.resources.data import get_resource_single_filename

SEPARATOR = "___"


class InterstitialFormationCalculator(GeneralCalculator):
    """Calculation of interstitial formation energy in FCC and BCC

    Args:
         :param atoms: original ASE Atoms object
         :param interstitial_type: str "all" or  str or list of str
                        for FCC - "100_dumbbell", "octa", "tetra"
                        for BCC - "100_dumbbell", "110_dumbbell", "111_dumbbell", "crowdion", "octahedral"
         :param supercell_size: int, [3..8]
         :param optimizer: optimizer class from ase.optimize, default = FIRE,
            could be also BFGS, LBFGS, BFGSLineSearch, LBFGSLineSearch, MDMin, QuasiNewton, GoodOldQuasiNewton
         :param fmax: fmax option for optimizer
         :param optimizer_kwargs: additional keyword arguments for optimizer class,
            f.e. optimizer_kwargs={"maxstep":0.05}
         :param fix_symmetry: bool, flag if use FixSymmetry constraint for optimization

    Usage:
    >>  atoms.calc = calculator
    >>  interstitial_formation = InterstitialFormationCalculator(atoms)
    >>  interstitial_formation.calculate()
    >>  print(interstitial_formation.value["interstitial_formation_energy"]) # will print interstitial formation energy
    """

    property_name = "interstitial"

    param_names = [
        "interstitial_type",
        "supercell_size",
        "optimizer",
        "fmax",
        "optimizer_kwargs",
        "fix_symmetry",
    ]

    SUPERCELL_DEFECT = "defect"
    STATIC = "static"
    ATOMIC = "atomic"
    SUPERCELL_0_NAME = SUPERCELL_DEFECT + SEPARATOR + "IDEAL" + SEPARATOR + STATIC
    minimization_styles = [STATIC, ATOMIC]
    ALL_INTERSTITIALS_DICT = {
        "fcc": ["100_dumbbell", "octa", "tetra"],
        "bcc": ["100_dumbbell", "110_dumbbell", "111_dumbbell", "111_crowdion", "octa"],
    }

    def __init__(
        self,
        atoms=None,
        interstitial_type="all",
        supercell_size=4,
        optimizer=FIRE,
        fmax=0.01,
        optimizer_kwargs=None,
        fix_symmetry=True,
        **kwargs,
    ):
        GeneralCalculator.__init__(self, atoms, **kwargs)

        self._value["spgn"] = get_spacegroup(atoms)
        if self._value["spgn"] not in [225, 229]:  # 225 - FCC, 229- BCC
            raise ValueError(
                "Only FCC(sg #225) or BCC (sg #229) structures is acceptable, but you provide "
                "structure with space group #{}".format(self._value["spgn"])
            )
        self.volume = atoms.get_volume() / len(atoms)
        if self._value["spgn"] == 225:  # FCC
            self._value["structure_type"] = "fcc"
            self._value["lattice_param"] = (self.volume * 4.0) ** (
                1.0 / 3.0
            )  # lattice constant for FCC
        elif self._value["spgn"] == 229:  # BCC
            self._value["structure_type"] = "bcc"
            self._value["lattice_param"] = (self.volume * 2.0) ** (
                1.0 / 3.0
            )  # lattice constant for BCC

        if isinstance(interstitial_type, str):
            if interstitial_type == "all":
                self.interstitial_type = self.ALL_INTERSTITIALS_DICT[
                    self._value["structure_type"]
                ]
            else:
                self.interstitial_type = [interstitial_type]
        else:
            self.interstitial_type = interstitial_type
        self.supercell_size = supercell_size
        self.optimizer = optimizer
        self._init_kwargs(optimizer_kwargs=optimizer_kwargs)
        self.fmax = fmax
        self.fix_symmetry = fix_symmetry

    def generate_structures(self, verbose=False):
        # unrelaxed and atomic only-relaxed
        structure_names_filenames_dict = {}
        structure_names = []
        for int_type in self.interstitial_type:
            resources_path = os.path.join(
                "structures", "Interstitials", self._value["structure_type"], int_type
            )
            supercell_size_suffix = "".join([str(self.supercell_size)] * 3)
            struct_name = "{structure_type}_{interstitial_type}_{size_suffix}".format(
                resources_path=resources_path,
                structure_type=self._value["structure_type"],
                interstitial_type=int_type,
                size_suffix=supercell_size_suffix,
            )
            structure_names_filenames_dict[struct_name] = os.path.join(
                resources_path, struct_name + ".cfg"
            )

        ideal_structure = self.basis_ref.repeat([self.supercell_size] * 3)
        self._value["supercell_size"] = self.supercell_size
        self._value["ideal_structure_n_at"] = len(ideal_structure)
        self._value["interstitial_type"] = self.interstitial_type
        self._value["structure_names"] = list(structure_names_filenames_dict.keys())

        elm = self.basis_ref.get_chemical_symbols()[0]

        # Stage 1. Supercell
        self._structure_dict[self.SUPERCELL_0_NAME] = ideal_structure

        # Stage 2. Supercell with vacancy
        for struct_name, file_name in structure_names_filenames_dict.items():
            structure_filename = get_resource_single_filename(file_name)
            if verbose:
                print("Reading {}".format(structure_filename))
            defect_structure = read(structure_filename)
            defect_structure.set_chemical_symbols(elm * len(defect_structure))
            tmp_cell = defect_structure.get_cell()
            defect_structure.set_cell(
                tmp_cell * self._value["lattice_param"], scale_atoms=True
            )
            for minstyle in self.minimization_styles:
                job_name = self.subjob_name(minstyle, struct_name)
                self._structure_dict[job_name] = defect_structure

        return self._structure_dict

    def get_structure_value(self, structure, name=None):
        logfile = "-" if self.verbose else None
        if isinstance(structure.calc, AMSDFTBaseCalculator):
            raise NotImplementedError()
        initial_structure = structure.copy()
        if name == InterstitialFormationCalculator.SUPERCELL_0_NAME:
            pass
        elif name.startswith(InterstitialFormationCalculator.SUPERCELL_DEFECT):
            if name.endswith(InterstitialFormationCalculator.STATIC):
                pass
            elif InterstitialFormationCalculator.ATOMIC in name:
                # TODO: use special optimizer?
                if fixsymmetry_imported and self.fix_symmetry:
                    fix_atoms = FixSymmetry(structure, verbose=True)
                    structure.set_constraint(fix_atoms)
                dyn = self.optimizer(
                    structure, logfile=logfile, **self.optimizer_kwargs
                )
                dyn.run(fmax=self.fmax)
                structure = dyn.atoms

        en = structure.get_potential_energy(force_consistent=True)
        forces = structure.get_forces()
        return {
            "energy": en,
            "forces": forces,
            "n_at": len(structure),
            "volume": structure.get_volume(),
            "atoms": structure,
            "initial_atoms": initial_structure,
        }, structure

    @staticmethod
    def subjob_name(minstyle, struct_name):
        return build_job_name(
            InterstitialFormationCalculator.SUPERCELL_DEFECT,
            struct_name,
            minstyle,
            separator=SEPARATOR,
            sanitize=False
        )

    @staticmethod
    def parse_job_name(jobname: str):
        """
        Split joname into structure name and minimization type
        :param jobname: str, job name
        :return: (structure_name, minstyle)
        """
        splits = jobname.split(SEPARATOR)
        return splits[1], splits[2]

    def analyse_structures(self, output_dict):
        # self.generate_structures()
        raw_data_dict = {}
        for job_name in self._structure_dict.keys():
            struct_name, min_style = self.parse_job_name(job_name)
            raw_data_dict[struct_name + SEPARATOR + min_style] = {
                "energy": output_dict[job_name]["energy"],
                "max_force": np.max(
                    np.linalg.norm(output_dict[job_name]["forces"], axis=1)
                ),
                "volume": output_dict[job_name]["volume"],
                "n_at": output_dict[job_name]["n_at"],
            }

        self._value["raw_data"] = raw_data_dict
        self.calculate_interstitial_formation()

    def calculate_interstitial_formation(self):
        self.generate_structures()
        structure_names = self._value["structure_names"]
        n_at_ideal = self._value["ideal_structure_n_at"]

        e0 = (
            self._value["raw_data"][
                "IDEAL" + SEPARATOR + InterstitialFormationCalculator.STATIC
            ]["energy"]
            / n_at_ideal
        )

        interstitial_formation_energy_value = defaultdict(dict)
        for struct_name in structure_names:
            for minstyle in self.minimization_styles:
                e_int = self._value["raw_data"][struct_name + SEPARATOR + minstyle][
                    "energy"
                ]
                n_int = self._value["raw_data"][struct_name + SEPARATOR + minstyle][
                    "n_at"
                ]
                e_form = e_int - n_int * e0
                interstitial_formation_energy_value[struct_name][minstyle] = e_form

        self._value["interstitial_formation_energy"] = dict(
            interstitial_formation_energy_value
        )
    def plot(self, ax=None, **kwargs):
        """
        Plot interstitial formation energy for different types
        """
        import matplotlib.pyplot as plt

        if "interstitial_formation_energy" not in self.value:
            print("No interstitial formation energy found in results")
            return

        engs = self.value["interstitial_formation_energy"]
        
        # Filter for 'atomic' minimization style if available, otherwise 'static'
        plot_data = {}
        for style in ["atomic", "static"]:
            data = {k: v[style] for k, v in engs.items() if style in v}
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
        ax.set_xlabel("Interstitial Type")
        ax.set_title(f"Interstitial Formation Energy ({label} relaxation)")
        plt.xticks(rotation=45)
        
        return ax
