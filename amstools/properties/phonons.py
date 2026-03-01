import copy
import json

import numpy as np
from ase.atoms import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

try:
    from phonopy.physical_units import get_physical_units

    VaspToTHz = get_physical_units().DefaultToTHz
except ImportError:
    # for backward compatibility with older phonopy versions
    from phonopy.units import VaspToTHz

from amstools.calculators.dft.base import AMSDFTBaseCalculator, get_k_mesh_by_kspacing
from amstools.properties.generalcalculator import GeneralCalculator
from amstools.utils import JsonNumpyEncoder


def phonopy_to_ase_atoms(ph_atoms):
    """
    Convert Phonopy Atoms to ASE-like Atoms
    Args:
        ph_atoms: Phonopy Atoms object

    Returns: ASE-like Atoms object

    """
    ase_atoms = Atoms(
        symbols=list(ph_atoms.symbols),
        positions=list(ph_atoms.positions),
        cell=list(ph_atoms.cell),
        pbc=True,
    )
    ase_atoms.set_initial_magnetic_moments(ph_atoms.magnetic_moments)
    return ase_atoms


def ase_to_phonopy_atoms(ase_atoms):
    """
    Convert ASE-like Atoms to Phonopy Atoms
    Args:
        ase_atoms: ASE-like Atoms

    Returns:
        Phonopy Atoms

    """
    ph_atoms = PhonopyAtoms(
        symbols=list(ase_atoms.get_chemical_symbols()),
        scaled_positions=list(ase_atoms.get_scaled_positions()),
        cell=list(ase_atoms.get_cell()),
        # pbc=True, # Deprecated, always True
    )
    ph_atoms.magnetic_moments = ase_atoms.get_initial_magnetic_moments()
    return ph_atoms


band_pathes = {
    "bcc": np.array(
        [
            [0, 0, 0],  # Gamma
            [0.5, -0.5, 0.5],  # H
            [0.25, 0.25, 0.25],  # P
            [0, 0, 0],  # Gamma
            [0, 0, 0.5],
        ]
    ),  # N
    "diam": np.array(
        [
            [0, 0, 0],  # Gamma
            [1.0, 0, 0],  # X
            [3 / 4, -3 / 4, 0],  # K
            [0, 0, 0],  # Gamma
            [1, 1, 1],  # L
        ]
    )
    / 2,
}


def generate_band_path(path):
    bands = []
    for i in range(len(path) - 1):
        q_start = np.array(path[i])
        q_end = np.array(path[i + 1])
        band = []
        for i in range(51):
            band.append(q_start + (q_end - q_start) / 50 * i)
        bands.append(band)
    return bands


def attach_charges(atoms, charge_dict):
    symb = atoms.get_chemical_symbols()
    charges = [charge_dict[s] for s in symb]
    atoms.set_initial_charges(charges)


class PhonopyCalculator(GeneralCalculator):
    """Calculation of phonopy object (see https://atztogo.github.io/phonopy/phonopy-module.html#phonopy-module for details)

    :param atoms: original ASE Atoms object with calculator
    :param interaction_range: supercell size (default is 10 A)
    :param supercell_range:  supercell range (optional). If specified, interaction_range is ignored
    :param is_symmetry: use symmetry of the cell (default - True)
    :param displacement: atom displacement from original position, in angstroms (default is 0.01)
    :param factor: conversion factor from energy units to THz (default is from eV to THz)

     Usage:
     >>  atoms.calc = calculator
     >> phon = PhonopyProperty(atoms)
     >> phon.calculate()
     >> phonopy =  phon.phonopy #generated phonopy object
     >> phonopy.set_mesh([50,50,50])
     >> phonopy.run_total_dos()

     >> phonopy.plot_total_DOS()
    """

    property_name = "phonons"

    param_names = [
        "interaction_range",
        "displacement",
        "supercell_range",
        "factor",
        "force_cutoff",
        "q_mesh",
        "q_mesh_spacing",
    ]

    def __init__(
        self,
        atoms=None,
        interaction_range=10.0,
        supercell_range=None,
        displacement=0.01,
        factor=None,
        force_cutoff=None,
        q_mesh=75,
        q_mesh_spacing=None,
        is_symmetry=True,
        **kwargs,
    ):

        GeneralCalculator.__init__(self, atoms, **kwargs)
        self._postpone_load_phonopy_fromdict = False
        self.interaction_range = interaction_range
        self.displacement = displacement
        self.supercell_range = supercell_range
        self.factor = factor
        self.force_cutoff = force_cutoff
        self.is_symmetry = is_symmetry
        self.q_mesh = q_mesh
        self.q_mesh_spacing = q_mesh_spacing
        self._phonopy = None
        self._dos_total = None
        self._dos_energies = None

    def create_phonopy(self):
        basis_ref = self.basis_ref
        unitcell = ase_to_phonopy_atoms(basis_ref)

        if "phonopy_data" in self._value:
            phonopy_data = self._value["phonopy_data"]
        else:
            phonopy_data = {}
            # basis_ref = self.basis_ref
            # unitcell = ase_to_phonopy_atoms(basis_ref)
            PHONON_INTERACTION_DIST = self.interaction_range
            if self.supercell_range is None:
                supercell_range = np.ceil(
                    PHONON_INTERACTION_DIST
                    / np.array([np.linalg.norm(vec) for vec in unitcell.cell])
                )
            else:
                supercell_range = self.supercell_range

            phonopy_data["supercell_range"] = supercell_range
            supercell_matrix = np.eye(3) * supercell_range
            phonopy_data["supercell_matrix"] = supercell_matrix
        self._phonopy = Phonopy(
            unitcell=unitcell,
            supercell_matrix=phonopy_data["supercell_matrix"],
            factor=self.factor,
            is_symmetry=self.is_symmetry,
        )
        self._phonopy.generate_displacements(distance=self.displacement)

        if "displ_dataset" not in phonopy_data:
            phonopy_displ_dataset = self._phonopy.dataset
            phonopy_data["displ_dataset"] = json.dumps(
                phonopy_displ_dataset, cls=JsonNumpyEncoder
            )
            if "num_of_supercell" not in phonopy_data:
                phonopy_data["num_of_supercell"] = len(
                    phonopy_displ_dataset["first_atoms"]
                )

        self._value["phonopy_data"] = phonopy_data

    def generate_structures(self, verbose=False):
        self.create_phonopy()
        supercells = self.phonopy.supercells_with_displacements
        # extract atoms charges mapping
        symb_charged_dict = None
        charges = self.basis_ref.get_initial_charges()
        if np.any(charges != 0.0):
            symb = self.basis_ref.get_chemical_symbols()
            symb_charged_dict = {s: c for s, c in zip(symb, charges)}
            # check that symbol-charge correspondance is consistent
            for s, c in zip(symb, charges):
                if symb_charged_dict[s] != c:
                    raise ValueError(
                        "Chemical specie {} has inconsistent charges: {} and {}".format(
                            s, c, symb_charged_dict[s]
                        )
                    )

        for ind, sc in enumerate(supercells):
            jobname = self.subjob_name(ind)
            atoms = phonopy_to_ase_atoms(sc)
            if symb_charged_dict:
                attach_charges(atoms, symb_charged_dict)
            self._structure_dict[jobname] = atoms
        return self._structure_dict

    @staticmethod
    def subjob_name(ind):
        return "supercell_phonon_%d" % ind

    def analyse_structures(self, output_dict):
        self.create_phonopy()
        supercells = self.phonopy.supercells_with_displacements
        set_of_forces = []
        supercell_forces_dict = {}
        for ind in range(len(supercells)):
            jobname = self.subjob_name(ind)
            set_of_forces.append(output_dict[jobname])
            supercell_forces_dict[jobname] = output_dict[jobname]

        phonopy_data = self._value["phonopy_data"]
        phonopy_data["supercell_forces_dict"] = supercell_forces_dict

        self._value["phonopy_data"] = phonopy_data

        self.phonopy.forces = set_of_forces
        self.phonopy.produce_force_constants()
        if self.force_cutoff:
            self.phonopy.set_force_constants_zero_with_radius(self.force_cutoff)
        self._compute_dos()

        self._value["dos_total"] = self._dos_total
        self._value["dos_energies"] = self._dos_energies

    def _compute_dos(self):
        if self.q_mesh_spacing is not None:
            self.q_mesh = get_k_mesh_by_kspacing(
                self.basis_ref.cell, kmesh_spacing=self.q_mesh_spacing
            )
        if self.q_mesh is not None:
            if isinstance(self.q_mesh, list):
                self.phonopy.run_mesh(mesh=self.q_mesh)
            else:
                self.phonopy.run_mesh(mesh=[self.q_mesh] * 3)
            self.phonopy.run_total_dos(
                use_tetrahedron_method=False
            )  # "False" for backward compatibility
            dos = self.phonopy.get_total_dos_dict()
            erg, dos = dos["frequency_points"], dos["total_dos"]
            self._dos_total = dos
            self._dos_energies = erg

    def get_structure_value(self, structure, name=None):
        if isinstance(structure.calc, AMSDFTBaseCalculator):
            structure.calc.static_calc()
            structure.calc.auto_kmesh_spacing = False
            if structure.calc.kmesh_spacing:
                kmesh = get_k_mesh_by_kspacing(
                    structure.cell, kmesh_spacing=structure.calc.kmesh_spacing
                )
                structure.calc.set_kmesh(kmesh)

        return structure.get_forces(), structure

    @property
    def phonopy(self):
        if self._phonopy is None:
            self.create_phonopy()

        if self._postpone_load_phonopy_fromdict:
            self._postpone_load_phonopy_fromdict = False
            self.init_phonopy_fromdict()
        return self._phonopy

    @phonopy.setter
    def phonopy(self, value):
        self._phonopy = value

    def plot_band_structure(
        self,
        path,
        labels=None,
        npoints=51,
        ylabel="Frequency, THz",
        ax=None,
        plot_kwargs=None,
    ):
        """
        Plot phonon dispersion

        :param path: i.e. [[[0, 0, 0], [0.5, 0, 0.5]]]
        :param labels:, i.e. ["G","X"]
        :param npoints: number of point in each section
        :param ax: matplotlib ax
        :param plot_kwargs: kwargs for plotting band structure lines
        """

        from matplotlib import pylab as plt

        from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

        qpoints, connections = get_band_qpoints_and_path_connections(
            path, npoints=npoints
        )
        self.phonopy.run_band_structure(
            qpoints, path_connections=connections, labels=labels
        )

        try:
            _, distances, frequencies, _ = self.phonopy.get_band_structure()
        except AttributeError:
            band_structure = self.phonopy.get_band_structure_dict()
            distances = band_structure["distances"]
            frequencies = band_structure["frequencies"]

        max_dist = max([max(dist) for dist in distances])
        if ax is None:
            ax = plt.gca()
        if not plot_kwargs:
            plot_kwargs = {"color": "red"}
        x_ticks_positions = []
        for dist, freq, lab in zip(distances, frequencies, labels):
            n_freq = freq.shape[1]
            for n in range(n_freq):
                ax.plot(dist / max_dist, freq[:, n], **plot_kwargs)
            ax.axvline(dist[0] / max_dist, color="lightgray")
            x_ticks_positions.append(
                dist[0] / max_dist
            )  # modified by MQ to align x_tick positions

        ax.axvline(dist[-1] / max_dist, color="lightgray")
        x_ticks_positions.append(dist[-1] / max_dist)
        ax.set_xticks(x_ticks_positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)

    def plot_total_DOS(
        self,
        ax=None,
        xlabel="Frequency, THz",
        ylabel="Phonon DOS, THz$^{-1}$",
        flip_xy=False,
        plot_kwargs=None,
    ):
        """
        Plot total DOS
        :param xlabel: label for X axis
        :param ylabel: label for Y axis
        :param flip_xy: flip X and Y axis
        :param ax: matplotlib ax
        :param plot_kwargs: kwargs for plotting band structure lines
        :return:
        """
        from matplotlib import pylab as plt

        if self._dos_total is None:
            self._compute_dos()

        dos = self.phonopy.get_total_dos_dict()
        frequency_points, total_dos = dos["frequency_points"], dos["total_dos"]
        if ax is None:
            ax = plt.gca()

        if not plot_kwargs:
            plot_kwargs = {"color": "red"}

        if not flip_xy:
            ax.plot(frequency_points, total_dos, **plot_kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else:
            ax.plot(total_dos, frequency_points, **plot_kwargs)
            ax.set_xlabel(ylabel)
            ax.set_ylabel(xlabel)

    def plot_band_structure_and_dos(
        self, path, labels=None, npoints=51, plot_kwargs=None
    ):
        from matplotlib import pylab as plt
        import matplotlib.gridspec as gridspec

        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax2 = plt.subplot(gs[0, 1])
        self.plot_total_DOS(ax=ax2, xlabel="", flip_xy=True, plot_kwargs=plot_kwargs)
        ax2.set_xlim((0, None))
        plt.setp(ax2.get_yticklabels(), visible=False)

        ax1 = plt.subplot(gs[0, 0], sharey=ax2)
        self.plot_band_structure(path, labels, npoints, ax=ax1, plot_kwargs=plot_kwargs)
        plt.subplots_adjust(wspace=0.03)
        plt.tight_layout()

    @classmethod
    def fromdict(cls, calc_dct):
        calc_dct = copy.deepcopy(calc_dct)
        phonon_calc = super().fromdict(calc_dct)
        # phonon_calc.init_phonopy_fromdict()
        phonon_calc._postpone_load_phonopy_fromdict = True
        return phonon_calc

    def init_phonopy_fromdict(self):
        phonopy_data = self._value["phonopy_data"]
        displ_dataset_s = phonopy_data["displ_dataset"]
        displ_dataset = json.loads(displ_dataset_s).copy()
        for i in range(len(displ_dataset["first_atoms"])):
            if "forces" in displ_dataset["first_atoms"][i]:
                displ_dataset["first_atoms"][i]["forces"] = np.array(
                    displ_dataset["first_atoms"][i]["forces"]
                )
        self.phonopy.dataset = displ_dataset
        # old format
        if (
            "num_of_supercell" in phonopy_data
            and "supercell_forces_dict" in phonopy_data
        ):
            supercell_forces_dict = phonopy_data["supercell_forces_dict"].copy()
            set_of_forces = []

            for ind in range(len(supercell_forces_dict)):
                jobname = self.subjob_name(ind)
                set_of_forces.append(np.array(supercell_forces_dict[jobname]))
            self.phonopy.forces = set_of_forces
        try:
            self.phonopy.produce_force_constants()
            if self.force_cutoff:
                self.phonopy.set_force_constants_zero_with_radius(self.force_cutoff)
        except RuntimeError as e:
            print("Error: ", e)

    def todict(self):
        prop_dict = super().todict()
        value_dict = prop_dict.get("_VALUE")

        if value_dict and "phonopy_data" in value_dict:
            phonopy_data = value_dict["phonopy_data"]
            if not "supercell_forces_dict" in phonopy_data:
                paused = (
                    self.basis_ref.calc.paused
                    if hasattr(self.basis_ref.calc, "paused")
                    else False
                )
                if not paused:
                    logging.warning(
                        "'supercell_forces_dict' is not in _VALUE/phonopy_data. 'phonopy_data' contains only {}".format(
                            phonopy_data.keys()
                        )
                    )

        return prop_dict

    def plot(self, ax=None, **kwargs):
        from matplotlib import pyplot as plt

        if ax is None:
            ax = plt.gca()

        ax.plot(self.value["dos_energies"], self.value["dos_total"], **kwargs)
        ax.set_xlabel("f, THz")
        ax.set_ylabel("Ph.DOS")
