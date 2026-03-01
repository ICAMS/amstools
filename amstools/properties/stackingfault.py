import os
from collections import defaultdict

import numpy as np
from amstools.calculators.dft.base import AMSDFTBaseCalculator
from ase.constraints import FixedLine
from ase.io import read
from ase.optimize import FIRE

from amstools.utils import compute_nn_distance, get_spacegroup


from amstools.properties.generalcalculator import GeneralCalculator
from amstools.resources.data import get_resource_single_filename

SEPARATOR = "___"


class StackingFaultCalculator(GeneralCalculator):
    """Calculation stacking fault energies in FCC

    Args:
         :param atoms: original ASE Atoms object
         :param stacking_fault_type: str: "all", or any combination of "ESF", "ISF", "MAX", "MIDDLE", "TWIN"
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
    >>  print(interstitial_formation._value["stacking_fault_energy"]) # will print stacking fault energies
    """

    property_name = "stacking_fault"

    param_names = [
        "stacking_fault_types",
        "optimizer",
        "fmax",
        "optimizer_kwargs",
        "fix_symmetry",
    ]
    STATIC = "static"
    ATOMIC = "atomic"
    minimization_styles = [STATIC, ATOMIC]

    eV_A2_to_mJ_m2 = 1.60218e-19 * 1e20 * 1e3

    def __init__(
        self,
        atoms=None,
        stacking_fault_types="all",
        optimizer=FIRE,
        fmax=0.01,
        optimizer_kwargs=None,
        fix_symmetry=True,
        **kwargs,
    ):
        GeneralCalculator.__init__(self, atoms, **kwargs)
        if isinstance(stacking_fault_types, str):
            if stacking_fault_types == "all":
                self.stacking_fault_types = ["ESF", "ISF", "MAX", "MIDDLE", "TWIN"]
            else:
                self.stacking_fault_types = [stacking_fault_types]
        else:
            self.stacking_fault_types = stacking_fault_types
        self.optimizer = optimizer
        self._init_kwargs(optimizer_kwargs=optimizer_kwargs)
        self.fmax = fmax
        self.fix_symmetry = fix_symmetry

        if atoms is not None:
            self.initialize_structure_properties(atoms)

    def initialize_structure_properties(self, atoms):
        self._value["spgn"] = get_spacegroup(atoms)
        if self._value["spgn"] not in [225]:  # 225 - FCC, 229- BCC
            raise ValueError(
                "Only FCC(sg #225) structures is acceptable, but you provide "
                "structure with space group #{}".format(self._value["spgn"])
            )
        self.volume = atoms.get_volume() / len(atoms)
        if self._value["spgn"] == 225:  # FCC
            self._value["structure_type"] = "fcc"
            self._value["lattice_param"] = (self.volume * 4.0) ** (
                1.0 / 3.0
            )  # lattice constant for FCC
            self._value["d0"] = np.mean(compute_nn_distance(atoms))

    def generate_structures(self, verbose=True):
        if not hasattr(self, "volume"):
            self.initialize_structure_properties(self.basis_ref)

        # unrelaxed and atomic only-relaxed
        structure_names_filenames_dict = {}

        if "IDEAL" not in self.stacking_fault_types:
            self.stacking_fault_types.append("IDEAL")

        for stacking_fault_type in self.stacking_fault_types:
            resources_path = os.path.join(
                "structures", "Stacking_faults", self._value["structure_type"]
            )
            struct_name = "{structure_type}_{stacking_fault_type}".format(
                structure_type=self._value["structure_type"],
                stacking_fault_type=stacking_fault_type,
            )
            structure_names_filenames_dict[struct_name] = os.path.join(
                resources_path, struct_name + "_prim_12at.cfg"
            )

        self._value["stacking_fault_types"] = self.stacking_fault_types
        self._value["structure_names"] = list(structure_names_filenames_dict.keys())

        elm = self.basis_ref.get_chemical_symbols()[0]

        # Stage 1. Supercell with vacancy
        for struct_name, file_name in structure_names_filenames_dict.items():
            structure_filename = get_resource_single_filename(file_name)
            if verbose:
                print("Reading {}".format(structure_filename))
            defect_structure = read(structure_filename)
            defect_structure.set_chemical_symbols(elm * len(defect_structure))
            d = 1  # reference NN distance
            cell = defect_structure.cell
            scale_factor = self._value["d0"] / d
            transform_matrix = np.eye(3) * scale_factor
            new_cell = np.dot(cell, transform_matrix)
            defect_structure.set_cell(new_cell, scale_atoms=True)
            for minstyle in self.minimization_styles:
                job_name = self.subjob_name(minstyle, struct_name)
                self._structure_dict[job_name] = defect_structure

        self._value["sf_area"] = np.linalg.det(new_cell[:2, :2])

        return self._structure_dict

    def get_structure_value(self, structure, name=None):
        logfile = "-" if self.verbose else None

        if isinstance(structure.calc, AMSDFTBaseCalculator):
            # raise NotImplementedError()
            if name.endswith(self.STATIC):
                structure.calc.static_calc()
                structure.calc.auto_kmesh_spacing = True
            elif self.ATOMIC in name:
                const_list = [FixedLine(i, [0, 0, 1]) for i in range(len(structure))]
                structure.set_constraint(const_list)
                structure.calc.optimize_atoms_only(ediffg=-self.fmax, max_steps=100)
                # do calculations
                structure.get_potential_energy()
                calc = structure.calc
                structure = structure.calc.atoms
                structure.calc = calc
        else:
            if name.endswith(self.STATIC):
                pass
            elif self.ATOMIC in name:
                # if fixsymmetry_imported and self.fix_symmetry:
                const_list = [FixedLine(i, [0, 0, 1]) for i in range(len(structure))]
                structure.set_constraint(const_list)
                dyn = self.optimizer(
                    structure, logfile=logfile, **self.optimizer_kwargs
                )
                dyn.run(fmax=self.fmax)
                structure = dyn.atoms
        en = structure.get_potential_energy(force_consistent=True)
        forces = structure.get_forces()
        return {"energy": en, "forces": forces}, structure

    @staticmethod
    def subjob_name(minstyle, struct_name):
        job_name = struct_name + SEPARATOR + minstyle
        return job_name

    @staticmethod
    def parse_job_name(jobname: str):
        """
        Split joname into structure name and minimization type
        :param jobname: str, job name
        :return: (structure_name, minstyle)
        """
        splits = jobname.split(SEPARATOR)
        return splits[0], splits[1]

    def analyse_structures(self, output_dict):
        self.generate_structures()
        raw_data_dict = {}
        for job_name in self._structure_dict.keys():
            struct_name, min_style = self.parse_job_name(job_name)
            raw_data_dict[struct_name + SEPARATOR + min_style] = {
                "energy": output_dict[job_name]["energy"],
                "max_force": np.max(
                    np.linalg.norm(output_dict[job_name]["forces"], axis=1)
                ),
            }

        self._value["raw_data"] = raw_data_dict
        self.calculate_stacking_fault_energies()

    def calculate_stacking_fault_energies(self):
        self.generate_structures()
        structure_names = self._value["structure_names"]

        e_ideal_atomic = self._value["raw_data"]["fcc_IDEAL" + SEPARATOR + self.ATOMIC][
            "energy"
        ]
        e_ideal_static = self._value["raw_data"]["fcc_IDEAL" + SEPARATOR + self.STATIC][
            "energy"
        ]
        e_ideal_dict = {self.ATOMIC: e_ideal_atomic, self.STATIC: e_ideal_static}
        sf_area = self._value["sf_area"]

        stacking_fault_energy_value = defaultdict(dict)
        stacking_fault_energy_mJ_m2_value = defaultdict(dict)
        for struct_name in structure_names:
            for minstyle in self.minimization_styles:
                e_int = self._value["raw_data"][struct_name + SEPARATOR + minstyle][
                    "energy"
                ]
                e_form = (e_int - e_ideal_dict[minstyle]) / sf_area
                stacking_fault_energy_value[struct_name][minstyle] = e_form
                stacking_fault_energy_mJ_m2_value[struct_name][minstyle] = (
                    e_form * self.eV_A2_to_mJ_m2
                )

        self._value["stacking_fault_energy"] = dict(stacking_fault_energy_value)
        self._value["stacking_fault_energy(mJ/m^2)"] = dict(
            stacking_fault_energy_mJ_m2_value
        )
    def plot(self, ax=None, **kwargs):
        """
        Plot stacking fault energy for different types
        """
        import matplotlib.pyplot as plt

        if "stacking_fault_energy(mJ/m^2)" not in self.value:
            print("No stacking fault energy found in results")
            return

        engs = self.value["stacking_fault_energy(mJ/m^2)"]
        
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
        ax.set_ylabel("SFE (mJ/m^2)")
        ax.set_xlabel("Stacking Fault Type")
        ax.set_title(f"Stacking Fault Energy ({label} relaxation)")
        plt.xticks(rotation=45)
        
        return ax
