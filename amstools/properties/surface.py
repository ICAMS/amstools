import os

import numpy as np
import pandas as pd
from ase import Atom
from ase.io import read
from ase.optimize import FIRE

from amstools.calculators.dft.base import AMSDFTBaseCalculator
from amstools.properties.generalcalculator import GeneralCalculator
from amstools.properties.relaxation import SpecialOptimizer, StepwiseOptimizer
from amstools.resources.data import get_resource_single_filename
from amstools.utils import compute_nn_distance, get_spacegroup, build_job_name

SEPARATOR = "___"
eV_A2_to_mJ_m2 = 1.60218e-19 * 1e20 * 1e3


class SurfaceEnergyCalculator(GeneralCalculator):
    """Calculation of surface energies of bcc(100,110,111,112), fcc(100,110,111,112) and dia(100) structures

    Args:
         :param atoms: original ASE Atoms object
         :param surface_orientation: str, "100", "110", "111", "112"
         :param surface_name: str: i.e. "X100_Y010_Z001_6at" or "default"
         :param optimizer: optimizer class from ase.optimize, default = FIRE,
         :param fmax: fmax option for optimizer
            could be also BFGS, LBFGS, BFGSLineSearch, LBFGSLineSearch, MDMin, QuasiNewton, GoodOldQuasiNewton
         :param optimizer_kwargs: additional keyword arguments for optimizer class,
            f.e. optimizer_kwargs={"maxstep":0.05}
         :param fix_symmetry: bool, flag if use FixSymmetry constraint for optimization

    Usage:
    >>  atoms.calc = calculator
    >>  interstitial_formation = InterstitialFormationCalculator(atoms)
    >>  interstitial_formation.calculate()
    >>  print(interstitial_formation.value["stacking_fault_energy"]) # will print stacking fault energies
    """

    property_name = "surface_energy"

    param_names = [
        "surface_orientation",
        "surface_name",
        "optimizer",
        "fmax",
        "optimizer_kwargs",
        "fix_symmetry",
    ]
    ATOMIC = "atomic"
    minimization_styles = [ATOMIC]

    def __init__(
        self,
        atoms=None,
        surface_orientation: str = "100",
        surface_name: str = "default",
        optimizer=FIRE,
        fmax=0.01,
        optimizer_kwargs=None,
        fix_symmetry=True,
        **kwargs,
    ):
        GeneralCalculator.__init__(self, atoms, **kwargs)
        self.surface_orientation = surface_orientation
        if isinstance(surface_name, str):
            if surface_name == "default":
                raise NotImplementedError(
                    "'default' surface_name is not yet implemented. Use SurfaceEnergyCalculator.available_surfaces for suggestions"
                )
            else:
                self.surface_name = surface_name
        else:
            raise ValueError(
                "surface_name could be only string, but got {} (type {})".format(
                    surface_name, type(surface_name)
                )
            )

        self.optimizer = optimizer
        self._init_kwargs(optimizer_kwargs=optimizer_kwargs)
        self.fmax = fmax
        self.fix_symmetry = fix_symmetry

        if atoms is not None:
            self.initialize_structure_properties(atoms)
            
    def initialize_structure_properties(self, atoms):
        self._value["spgn"] = get_spacegroup(atoms)
        self._value["volume"] = atoms.get_volume() / len(atoms)

        self.structure_type, self.lattice_param, self.d0 = get_structure_parameters(
            self.basis_ref, self._value["spgn"], self._value["volume"]
        )

    def generate_structures(self, verbose=True):
        if not hasattr(self, "structure_type"):
            self.initialize_structure_properties(self.basis_ref)

        # unrelaxed and atomic only-relaxed
        self._value["structure_type"] = self.structure_type
        self._value["surface_name"] = self.surface_name
        self._value["surface_orientation"] = self.surface_orientation

        elm = self.basis_ref.get_chemical_symbols()[0]

        # Stage 1. Ideal structure energy
        self._structure_dict["IDEAL"] = self.basis_ref.copy()

        # Stage 2. Surface
        surface_structure, struct_name = get_surface_structure_and_name(
            self.structure_type,
            self.surface_orientation,
            self.surface_name,
            element=elm,
            verbose=verbose,
        )
        self._value["structure_name"] = struct_name
        rescale_structure_to_nn_distance(surface_structure, self.d0)
        job_name = self.subjob_name(self.ATOMIC, "SURFACE")
        self._structure_dict[job_name] = surface_structure

        return self._structure_dict

    def get_structure_value(self, structure, name=None):
        if isinstance(structure.calc, AMSDFTBaseCalculator):
            raise NotImplementedError()
        logfile = "-" if self.verbose else None
        result_dict = {"n_at": len(structure)}
        if name == "IDEAL":
            pass
        elif name.endswith(self.ATOMIC):
            dyn = self.optimizer(structure, logfile=logfile, **self.optimizer_kwargs)
            dyn.run(fmax=self.fmax)
            structure = dyn.atoms
            result_dict["area"] = np.linalg.det(structure.get_cell()[:2, :2])
        result_dict["energy"] = structure.get_potential_energy(force_consistent=True)
        result_dict["forces"] = structure.get_forces()
        result_dict["atoms"] = structure
        return result_dict, structure

    @staticmethod
    def subjob_name(minstyle, struct_name):
        return build_job_name(struct_name, minstyle, separator=SEPARATOR, sanitize=False)

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
        # self.generate_structures(verbose=self.verbose)

        e0 = output_dict["IDEAL"]["energy"] / output_dict["IDEAL"]["n_at"]
        job_name = self.subjob_name(self.ATOMIC, "SURFACE")

        self._value["surface_structure_energy"] = output_dict[job_name]["energy"]
        self._value["number_of_atoms"] = output_dict[job_name]["n_at"]
        self._value["surface_area"] = output_dict[job_name]["area"]
        self._value["max_force"] = np.max(
            np.linalg.norm(output_dict[job_name]["forces"], axis=1)
        )
        self._value["ref_energy_per_atom"] = e0

        surface_energy_value = (
            self._value["surface_structure_energy"]
            - self._value["ref_energy_per_atom"] * self._value["number_of_atoms"]
        ) / (2 * self._value["surface_area"])
        surface_energy_mJ_m2_value = surface_energy_value * eV_A2_to_mJ_m2

        self._value["surface_energy"] = surface_energy_value
        self._value["surface_energy(mJ/m^2)"] = surface_energy_mJ_m2_value

    def plot(self, ax=None, **kwargs):
        """
        Plot surface energy (bar plot)
        """
        import matplotlib.pyplot as plt

        if "surface_energy(mJ/m^2)" not in self.value:
            print("No surface energy found in results")
            return

        if ax is None:
            fig, ax = plt.subplots()

        name = self.value.get("surface_name", "unknown")
        orient = self.value.get("surface_orientation", "")
        ax.bar([f"{orient} {name}"], [self.value["surface_energy(mJ/m^2)"]], **kwargs)
        ax.set_ylabel("Surface Energy (mJ/m^2)")
        return ax

    @staticmethod
    def available_surfaces(atoms, surface_orientation=None):
        spgn = get_spacegroup(atoms)
        if spgn not in [225, 229, 227]:  # 225 - FCC, 229- BCC, 227-diam
            raise ValueError(
                "Only FCC(sg #225), BCC(sg #229) or DIAMOND(sg #227) structures are acceptable, but you provide "
                "structure with space group #{}".format(spgn)
            )
        # TODO: refactor
        spgn_to_structure_type_dict = {225: "fcc", 229: "bcc", 227: "dia"}

        structure_type = spgn_to_structure_type_dict[spgn]

        if surface_orientation is None:
            surface_orientation = "*"

        from amstools.resources.data import get_resources_filenames_by_glob

        surfaces_file_names = get_resources_filenames_by_glob(
            "structures/Surfaces/{structure_type}/{orientation}/*.cfg".format(
                structure_type=structure_type, orientation=surface_orientation
            )
        )

        structure_types = []
        orientations = []
        surface_types = []
        for s in surfaces_file_names:
            struct_type, orientation, fname = s.split("/")[-3:]
            surface_type = fname.split("surf_")[-1].split(".cfg")[0]
            structure_types.append(struct_type)
            orientations.append(orientation)
            surface_types.append(surface_type)
        return pd.DataFrame(
            {
                "structure_type": structure_types,
                "surface_orientation": orientations,
                "surface_name": surface_types,
            }
        )


class SurfaceAtomAdsorptionCurveCalculator(GeneralCalculator):
    """
    Compute adsorption curve of the atom to the surface.

    :param surface: ASE atoms surface structure along XY direction
    :param adsorption_position_atom_indices: default=(0,),  tuple of indices of atoms on the top layer.
                 adsorption site would be AVERAGE position of those atoms
    :param z_min: default=1, min z-distance from the surface
    :param z_max: default=9.1, max z-distance from the surface
    :param dz: default=0.1, step in z
    :param surface_supercell_size: default=(2, 3), supercell size along X,Y of original *atoms* surface
    :param adsorption_energy_alignment: default="last", adsorption energy alignment:
                "last" - adsorption atom at z_max has zero energy
    :param elem=chemical symbol of adsorbate (default read from surface block)
    :param verbose=False

    """

    property_name = "surface_atom_adsorption"

    param_names = [
        "z_min",
        "z_max",
        "dz",
        "adsorption_position_atom_indices",
        "surface_supercell_size",
        "adsorption_energy_alignment",
        "elem",
    ]

    def __init__(
        self,
        surface,
        adsorption_position_atom_indices=(0,),
        z_min=1,
        z_max=9.1,
        dz=0.1,
        surface_supercell_size=(2, 3),
        adsorption_energy_alignment="last",
        elem=None,
        verbose=False,
        **kwargs,
    ):
        GeneralCalculator.__init__(self, surface, **kwargs)
        self.surface_structure = surface
        self.adsorption_position_atom_indices = adsorption_position_atom_indices
        self.surface_supercell_size = surface_supercell_size
        self.adsorption_energy_alignment = adsorption_energy_alignment
        self.z_min = z_min
        self.z_max = z_max
        self.dz = dz
        self.z_shift = np.arange(z_min, z_max, dz)
        self.verbose = verbose
        self.elem = elem

        self.value["surface_supercell_size"] = surface_supercell_size
        self.value["z_shift"] = list(self.z_shift)
        self.value["adsorption_position_atom_indices"] = (
            adsorption_position_atom_indices
        )
        self.value["adsorption_energy_alignment"] = adsorption_energy_alignment

        if self.adsorption_energy_alignment != "last":
            raise NotImplementedError(
                "'{}' adsorption energy alignment is invalid or not implemented".format(
                    self.adsorption_energy_alignment
                )
            )

    @staticmethod
    def subjob_name(z):
        return build_job_name('z_shift', f'{z:.5f}')

    def generate_structures(self, verbose=False):

        surface_supercell_size = list(self.surface_supercell_size) + [1]

        self.big_surface_structure = self.basis_ref.repeat(surface_supercell_size)

        # added flexibility to choose adsorbtion element
        if self.elem == None:
            elem = self.basis_ref[0].symbol

        else:
            elem = self.elem

        positions = self.big_surface_structure.get_positions()

        max_z = max(positions[:, 2])
        upper_layer_mask = positions[:, 2] > max_z - 0.5

        pm = positions[upper_layer_mask]

        pos_x, pos_y = np.mean(pm[self.adsorption_position_atom_indices, :2], axis=0)

        self._value["top_atoms_positions"] = pm.tolist()
        self._value["adsorption_position"] = [pos_x, pos_y]

        for cur_z_shift in self.z_shift:
            if self.verbose:
                print(".", end="")
            z = max_z + cur_z_shift
            current_structure = self.big_surface_structure.copy()
            current_structure.append(Atom(elem, position=[pos_x, pos_y, z]))
            job_name = self.subjob_name(cur_z_shift)
            self._structure_dict[job_name] = current_structure

        return self._structure_dict

    def get_structure_value(self, structure, name=None):
        if isinstance(structure.calc, AMSDFTBaseCalculator):
            raise NotImplementedError()
        return structure.get_potential_energy(force_consistent=True), structure

    def analyse_structures(self, output_dict):
        adsorption_data = []
        for cur_z_shift in self.z_shift:
            job_name = self.subjob_name(cur_z_shift)
            adsorption_data.append(output_dict[job_name])
        self._value["raw_adsorption_data"] = adsorption_data

        if self.adsorption_energy_alignment == "last":
            e_align = adsorption_data[-1]
        else:
            raise NotImplementedError(
                "'{}' adsorption energy alignment is invalid or not implemented".format(
                    self.adsorption_energy_alignment
                )
            )

        adsorption_energies = np.array(adsorption_data) - e_align
        self._value["adsorption_energies"] = adsorption_energies.tolist()

    def plot(self, ax=None, **kwargs):
        """
        Plot adsorption energy curve
        """
        import matplotlib.pyplot as plt

        if "adsorption_energies" not in self.value:
            print("No adsorption energies found in results")
            return

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.value["z_shift"], self.value["adsorption_energies"], **kwargs)
        ax.set_xlabel("z-shift (A)")
        ax.set_ylabel("Adsorption Energy (eV)")
        ax.set_title("Surface Adsorption Curve")
        return ax


class SurfaceDecohesionCalculator(SurfaceEnergyCalculator):
    """
    Compute surface decohesion energy curve
    :param atoms: original ASE Atoms object
    :param surface_orientation: str, "100", "110", "111", "112"
    :param surface_name: str: i.e. "X100_Y010_Z001_6at" or "default"
    :param optimizer: optimizer class from ase.optimize, default = FIRE,
    :param fmax: fmax option for optimizer
    could be also BFGS, LBFGS, BFGSLineSearch, LBFGSLineSearch, MDMin, QuasiNewton, GoodOldQuasiNewton
    :param optimizer_kwargs: additional keyword arguments for optimizer class,
    f.e. optimizer_kwargs={"maxstep":0.05}
    :param fix_symmetry: bool, flag if use FixSymmetry constraint for optimization
    :param z_min: default=-0.1, min z-distance  between surfaces, negative value corresponds to compression
    :param z_max: default=7.3, max z-distance between the surfaces
    :param dz: default=0.1, step in z
    """

    property_name = "surface_decohesion"

    param_names = [
        "zmin",
        "zmax",
        "dz",
        "surface_orientation",
        "surface_name",
        "fix_symmetry",
        "optimizer",
        "fmax",
        "optimizer_kwargs",
    ]

    def __init__(
        self,
        atoms,
        surface_orientation,
        surface_name="default",
        optimizer=FIRE,
        fmax=0.01,
        optimizer_kwargs=None,
        fix_symmetry=True,
        zmin=-0.1,
        zmax=7.3,
        dz=0.1,
        **kwargs,
    ):
        super().__init__(
            atoms,
            surface_orientation,
            surface_name,
            optimizer,
            fmax,
            optimizer_kwargs,
            fix_symmetry,
            **kwargs,
        )
        self.zmin = zmin
        self.zmax = zmax
        self.dz = dz
        self._value["zmin"] = zmin
        self._value["zmax"] = zmax
        self._value["dz"] = dz

    def generate_bulk_surface_structure(self):
        surface_structure = self._structure_dict["SURFACE___atomic"]
        surf_block = surface_structure.copy()
        pos = surf_block.get_positions()
        pos[:, 2] -= pos[:, 2].min()
        z_pos = pos[:, 2]
        dzpos = abs(z_pos[1:] - z_pos[:-1])
        dzpos = dzpos[dzpos > 0]

        dzpos_mean = dzpos.mean()

        new_cell = surf_block.cell.copy()
        new_cell[2, 2] = z_pos.max() + dzpos_mean

        surf_block.set_positions(pos)
        surf_block.set_cell(new_cell, scale_atoms=False)

        self._structure_dict["BULK_SURFACE_BLOCK"] = surf_block

    def generate_decohesion_surface_structures(self, opt_surf_block):
        z_lat_eq = opt_surf_block.cell[2, 2]
        zs = np.arange(self.zmin, self.zmax + self.dz / 2, self.dz)
        for z in zs:
            z_lat = z + z_lat_eq
            new_fcc_block = opt_surf_block.copy()
            new_cell = new_fcc_block.cell.copy()
            new_cell[2, 2] = z_lat
            new_fcc_block.set_cell(new_cell, scale_atoms=False)
            job_name = "SURF_DECOHESION_Z__{:.3f}".format(z)
            job_name = job_name.replace(".", "_").replace("-", "m")
            self._structure_dict[job_name] = new_fcc_block

    def get_structure_value(self, structure, name=None):
        if isinstance(structure.calc, AMSDFTBaseCalculator):
            raise NotImplementedError()
        if name == "BULK_SURFACE_BLOCK":
            surf_opt = StepwiseOptimizer(structure)
            opt_surf_block = surf_opt.optimize()
            block_e_pot = opt_surf_block.get_potential_energy(force_consistent=True)
            # add series of SURF_DECOHESION_Z__ steps
            self.generate_decohesion_surface_structures(opt_surf_block)
            return {"energy": block_e_pot}, structure
        elif name.startswith("SURF_DECOHESION_Z"):
            new_block_e_pot = structure.get_potential_energy(force_consistent=True)
            opt = SpecialOptimizer(atoms=structure, optimize_atoms_only=True)
            opt_struct = opt.run()
            opt_new_block_e_pot = opt_struct.get_potential_energy(force_consistent=True)
            z = name.split("__")[-1]
            z = float(z.replace("m", "-").replace("_", "."))
            res_dict = {"z": z, "e0": new_block_e_pot, "e_relax": opt_new_block_e_pot}
            return res_dict, structure
        else:
            result_dict, structure = super().get_structure_value(structure, name)
            if name == "SURFACE___atomic":
                # add BULK_SURFACE_BLOCK step
                self.generate_bulk_surface_structure()
            return result_dict, structure

    def analyse_structures(self, output_dict):
        super().analyse_structures(output_dict)
        block_en_list = []
        for name, res_dict in output_dict.items():
            if name.startswith("SURF_DECOHESION_Z"):
                block_en_list.append(
                    [res_dict["z"], res_dict["e0"], res_dict["e_relax"]]
                )
        surf_decoh_df = pd.DataFrame(
            block_en_list, columns=["z", "e_pot", "e_pot_relax"]
        )
        self._value["surface_bulk_energy"] = output_dict["BULK_SURFACE_BLOCK"]["energy"]
        e0 = self._value["ref_energy_per_atom"]
        surf_decoh_area = self._value["surface_area"]
        n_at = self._value["number_of_atoms"]
        surf_decoh_df["e_surf"] = (surf_decoh_df["e_pot"] - e0 * n_at) / (
            2 * surf_decoh_area
        )
        surf_decoh_df["e_surf_relax"] = (surf_decoh_df["e_pot_relax"] - e0 * n_at) / (
            2 * surf_decoh_area
        )
        surf_decoh_df["e_surf(mJ/m^2)"] = surf_decoh_df["e_surf"] * eV_A2_to_mJ_m2
        surf_decoh_df["e_surf_relax(mJ/m^2)"] = (
            surf_decoh_df["e_surf_relax"] * eV_A2_to_mJ_m2
        )
        self._value["z"] = surf_decoh_df["z"].to_list()
        self._value["e_surf"] = surf_decoh_df["e_surf"].to_list()
        self._value["e_surf_relax"] = surf_decoh_df["e_surf_relax"].to_list()

        self._value["e_surf(mJ/m^2)"] = surf_decoh_df["e_surf(mJ/m^2)"].to_list()
        self._value["e_surf_relax(mJ/m^2)"] = surf_decoh_df[
            "e_surf_relax(mJ/m^2)"
        ].to_list()

    def plot(self, ax=None, **kwargs):
        """
        Plot decohesion energy curve
        """
        import matplotlib.pyplot as plt

        if "e_surf(mJ/m^2)" not in self.value:
            print("No decohesion data found in results")
            return

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.value["z"], self.value["e_surf(mJ/m^2)"], label="static", **kwargs)
        if "e_surf_relax(mJ/m^2)" in self.value:
            ax.plot(self.value["z"], self.value["e_surf_relax(mJ/m^2)"], label="relaxed", **kwargs)
        
        ax.set_xlabel("Separation (A)")
        ax.set_ylabel("Energy (mJ/m^2)")
        ax.set_title("Surface Decohesion Curve")
        ax.legend()
        return ax


def get_structure_parameters(atoms, spgn, volume):
    if spgn == 225:  # FCC
        structure_type = "fcc"
        lattice_param = (volume * 4.0) ** (1.0 / 3.0)  # lattice constant for FCC
        d0 = np.mean(compute_nn_distance(atoms))
        return structure_type, lattice_param, d0
    elif spgn == 229:  # BCC
        structure_type = "bcc"
        lattice_param = (volume * 2.0) ** (1.0 / 3.0)  # lattice constant for FCC
        d0 = np.mean(compute_nn_distance(atoms))
        return structure_type, lattice_param, d0
    elif spgn == 227:  # DIAM
        structure_type = "dia"
        lattice_param = (volume * 8.0) ** (1.0 / 3.0)  # lattice constant for FCC
        d0 = np.mean(compute_nn_distance(atoms))
        return structure_type, lattice_param, d0

    raise ValueError(
        "Only FCC(sg #225), BCC(sg #229) or DIAMOND(sg #227) structures are acceptable, but you provide "
        "structure with space group #{}".format(spgn)
    )


def get_surface_structure_and_name(
    structure_type, surface_orientation, surface_type, element=None, verbose=True
):
    resources_path = os.path.join(
        "structures", "Surfaces", structure_type, surface_orientation
    )
    struct_name = "{structure_type}_{orientation}surf_{surface_type}".format(
        structure_type=structure_type,
        orientation=surface_orientation,
        surface_type=surface_type,
    )
    structure_names_filename = os.path.join(resources_path, struct_name + ".cfg")
    structure_filename = get_resource_single_filename(structure_names_filename)
    if verbose:
        print("Reading {}".format(structure_filename))
    surface_structure = read(structure_filename)
    if element:
        surface_structure.set_chemical_symbols(element * len(surface_structure))
    return surface_structure, struct_name


def rescale_structure_to_nn_distance(surface_structure, d0):
    d = compute_nn_distance(surface_structure).min()  # reference NN distance
    cell = surface_structure.cell
    scale_factor = d0 / d
    transform_matrix = np.eye(3) * scale_factor
    new_cell = np.dot(cell, transform_matrix)
    surface_structure.set_cell(new_cell, scale_atoms=True)
