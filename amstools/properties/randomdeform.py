import numpy as np
from ase.optimize import BFGS

from amstools.calculators.dft.base import AMSDFTBaseCalculator
from amstools.properties.generalcalculator import GeneralCalculator


def shake_atoms(atoms, pos_shake=0.05, cell_shake=0.05, seed=None):
    if seed is not None:
        np.random.seed(seed)

    orig_pos = atoms.get_positions()
    orig_cell = atoms.get_cell()
    dpos = np.random.randn(*orig_pos.shape) * pos_shake

    dcell_matrix = np.eye(3) + np.random.randn(3, 3) * cell_shake

    new_cell = np.dot(orig_cell, dcell_matrix)

    new_pos = orig_pos + dpos

    new_atoms = atoms.copy()

    new_atoms.set_positions(new_pos)

    new_atoms.set_cell(new_cell, scale_atoms=True)

    return new_atoms


class RandomDeformationCalculator(GeneralCalculator):
    """Randomly sample structure by randomly deforming cell, shifting atoms and performing uniform E-V strain
    :param atoms: original ASE Atoms object
    :param nsample: number of random deformation (incl. simultaneous cell deform and atomic displacement)
    :param supercell_size: default = (1,1,1)
    :param supercell_max_num_atoms: default = None

    :param random_atom_displacement: default = 0.1 (in Angstrom)
    :param random_cell_strain: default = 0.05 (=5%, dimensionless)

    :param volume_range: volume deformation range, default = 0.1 (+/- 10%)
    :param num_volume_deformations: default = 5
    :param seed: default = 42
    Usage:
    >>  atoms.calc = calculator
    >>  murn = RandomSamplingCalculator(atoms)
    >>  murn.calculate()
    >>  print(murn.value)
    """

    property_name = "randomdeformation"

    # TODO: update
    param_names = [
        "nsample",
        "supercell_size",
        "supercell_max_num_atoms",
        "random_atom_displacement",
        "random_cell_strain",
        "volume_range",
        "num_volume_deformations",
        "seed",
    ]

    def __init__(
        self,
        atoms=None,
        nsample=3,
        supercell_size=None,
        supercell_max_num_atoms=None,
        random_atom_displacement=0.1,
        random_cell_strain=0.05,
        volume_range=0.05,
        num_volume_deformations=5,
        seed=42,
        **kwargs,
    ):
        GeneralCalculator.__init__(self, atoms, **kwargs)
        self.nsample = nsample
        self.supercell_size = supercell_size
        self.supercell_max_num_atoms = supercell_max_num_atoms
        self.random_atom_displacement = random_atom_displacement
        self.random_cell_strain = random_cell_strain

        self.volume_range = volume_range
        self.num_volume_deformations = num_volume_deformations
        self.seed = seed

    def generate_structures(self, verbose=False):
        self.actual_structure = self.basis_ref.copy()

        # construct supercell
        if self.supercell_size:
            self.actual_structure = self.actual_structure * self.supercell_size
        elif self.supercell_max_num_atoms:
            raise NotImplementedError()
            # TODO implement
            # self.actual_structure = self.actual_structure
            # self._VALUE["supercell_size"]=...

        for sample_ind in range(self.nsample):
            current_structure = self.actual_structure.copy()
            random_sample_atoms = shake_atoms(
                current_structure,
                pos_shake=self.random_atom_displacement,
                cell_shake=self.random_cell_strain,
                seed=self.seed + sample_ind,
            )

            if self.num_volume_deformations and self.num_volume_deformations > 1:
                cell = random_sample_atoms.get_cell().copy()
                for strain in np.linspace(
                    1 - self.volume_range,
                    1 + self.volume_range,
                    self.num_volume_deformations,
                ):
                    def_cell = cell * (strain ** (1.0 / 3.0))
                    random_sample_atoms.set_cell(def_cell, scale_atoms=True)
                    jobname = "rnd_{}__vol_{:.5f}".format(sample_ind, strain)
                    self._structure_dict[jobname] = random_sample_atoms.copy()
            else:
                jobname = "rnd_{}".format(sample_ind)
                self._structure_dict[jobname] = random_sample_atoms

        return self._structure_dict

    def get_structure_value(self, structure, name=None):
        if isinstance(structure.calc, AMSDFTBaseCalculator):
            calc = structure.calc
            calc.static_calc()
            calc.auto_kmesh_spacing = False
            calc.update_kmesh_from_spacing(self.actual_structure)
            # do actual calculations
            structure.get_potential_energy(force_consistent=True)
            # update structure to optimized structure
            structure = calc.atoms
            structure.calc = calc
        en = structure.get_potential_energy(force_consistent=True)
        f = structure.get_forces()
        return (en, f), structure

    def analyse_structures(self, output_dict):
        energy_list = []
        forces_list = []

        for name, (e, f, *atoms) in output_dict.items():
            energy_list.append(e)
            forces_list.append(f)

        energy_list = np.array(energy_list)
        forces_list = np.array(forces_list)

        self._value["energy"] = energy_list
        self._value["forces"] = forces_list

    def plot(self, ax=None, **kwargs):
        """
        Plot histograms of energies and flattened forces
        """
        import matplotlib.pyplot as plt

        if "energy" not in self.value:
            print("No energy found in results")
            return

        engs = self.value["energy"]
        forces = self.value.get("forces")

        if ax is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        elif isinstance(ax, (list, np.ndarray)):
            ax1 = ax[0]
            ax2 = ax[1] if len(ax) > 1 else None
        else:
            ax1 = ax
            ax2 = None

        # Energy histogram
        ax1.hist(engs, bins=kwargs.get("bins", 20), **kwargs)
        ax1.set_xlabel("Energy (eV)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Energy Distribution")

        # Force histogram
        if forces is not None and ax2 is not None:
            forces_flat = np.array(forces).flatten()
            ax2.hist(forces_flat, bins=kwargs.get("bins", 20), color="orange", **kwargs)
            ax2.set_xlabel("Force Components (eV/A)")
            ax2.set_ylabel("Frequency")
            ax2.set_title("Force Components Distribution")
            
        if ax is None:
            plt.tight_layout()
            return ax1, ax2
        
        return ax1 if ax2 is None else (ax1, ax2)
