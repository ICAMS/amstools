import numpy as np
from amstools.calculators.dft.base import AMSDFTBaseCalculator
from ase.atoms import Atoms
from ase.build import bulk

from amstools.properties.generalcalculator import GeneralCalculator
from amstools.utils import get_spacegroup

spgn_dict = {"fcc": 225, "bcc": 229, "diam": 227, "sc": 221}


def is_general_cubic(atoms):
    # either primitive/cubic fcc, bcc, diamond
    # or just cubic cell
    cell = np.array(atoms.get_cell())
    spgn = get_spacegroup(atoms)
    if spgn in [spgn_dict[sg_name] for sg_name in ["fcc", "bcc", "diam", "sc"]]:
        return True
    else:
        if cell[0, 0] == cell[1, 1] == cell[2, 2] and np.sum(
            np.abs(np.diag(cell))
        ) == np.sum(np.abs(cell)):
            return True
        else:
            return False


def to_cubic_atoms(atoms):
    # either primitive/cubic fcc, bcc, diamond
    # or just cubic cell
    cell = np.array(atoms.get_cell())
    volume_per_atom = np.linalg.det(cell) / len(atoms)
    elem = atoms.get_chemical_symbols()[0]
    spgn = get_spacegroup(atoms)
    if spgn in [spgn_dict[sg_name] for sg_name in ["fcc", "bcc", "diam", "sc"]]:
        if spgn == spgn_dict["fcc"]:
            a0 = (volume_per_atom * 4.0) ** (1.0 / 3.0)
            new_atoms = bulk(elem, "fcc", a=a0, cubic=True)
        elif spgn == spgn_dict["bcc"]:
            a0 = (volume_per_atom * 2.0) ** (1.0 / 3.0)
            new_atoms = bulk(elem, "bcc", a=a0, cubic=True)
        elif spgn == spgn_dict["diam"]:
            a0 = (volume_per_atom * 8.0) ** (1.0 / 3.0)
            new_atoms = bulk(elem, "diamond", a=a0, cubic=True)
        elif spgn == spgn_dict["sc"]:
            a0 = (volume_per_atom * 1.0) ** (1.0 / 3.0)
            new_atoms = bulk(elem, "sc", a=a0, cubic=True)
        else:
            return None
        return new_atoms
    else:
        if cell[0, 0] == cell[1, 1] == cell[2, 2] and np.sum(
            np.abs(np.diag(cell))
        ) == np.sum(np.abs(cell)):
            return atoms
        else:
            return None


def set_magnetic_moments(src_atoms, dest_atoms):
    """
    Extract initial magnetic moments from src_atoms and apply it to dest_atoms
    :param src_atoms:
    :param dest_atoms:
    """
    magmoms = src_atoms.get_initial_magnetic_moments()
    if magmoms is not None:
        magmoms = list(set(magmoms))
        if len(magmoms) > 1:
            raise ValueError("Only NM or FM magnetic order are possible")
        if len(magmoms) == 1:
            dest_atoms.set_initial_magnetic_moments(magmoms * len(dest_atoms))


class TransformationPathCalculator(GeneralCalculator):
    """
    Calculation of bcc,fcc, cubic transformation paths

    Args:
        :param atoms - ASE Atoms with calculator. Only  BCC/FCC/cubic input structures are valid
        :param transformation_type - str, transformation transformation_type: "tetragonal", "trigonal", "hexagonal", "orthogonal", "general_cubic_tetragonal"
        :param num_of_point - int, number of points along transformation path

    Usage:
    >>  atoms.calc = calculator
    >>  trans_path = TransformationPathCalculator(atoms, type="hexagonal")
    >>  trans_path .calculate()
    >>  print(trans_path .value["energies_0"]) # will print energies along transformation path
    """

    property_name = "transformation_path"

    HEXAGONAL = "hexagonal"
    TRIGONAL = "trigonal"
    ORTHOGONAL = "orthogonal"
    TETRAGONAL = "tetragonal"
    GENERAL_CUBIC_TETRAGONAL = "general_cubic_tetragonal"

    param_names = ["transformation_type", "num_of_point"]

    @staticmethod
    def subjob_name(p):
        return ("tp_%.5f" % p).replace(".", "_").replace("-", "m")

    def __init__(self, atoms=None, transformation_type="tetragonal", num_of_point=50, **kwargs):
        GeneralCalculator.__init__(self, atoms, **kwargs)
        self.num_of_point = num_of_point
        self.transformation_type = transformation_type
        self._initialized = False
        if self.basis_ref:
            self.initialize()

    def initialize(self):
        if self._initialized:
            return
        elements = set(self.basis_ref.get_chemical_symbols())
        if len(elements) != 1:
            raise ValueError(
                "Only single-species structures are acceptable, but you got following elements: {}".format(
                    elements
                )
            )
        spgn = get_spacegroup(self.basis_ref)
        if self.transformation_type in [
            self.HEXAGONAL,
            self.TRIGONAL,
            self.ORTHOGONAL,
            self.TETRAGONAL,
        ] and spgn not in [
            225,
            229,
        ]:  # 225 - FCC, 229- BCC
            raise ValueError(
                "Only FCC(sg #225) or BCC (sg #229) structures are acceptable for {} transformation type, but you provide "
                "structure with space group #{}".format(self.transformation_type, spgn)
            )
        elif self.transformation_type == self.GENERAL_CUBIC_TETRAGONAL:
            if not is_general_cubic(self.basis_ref):
                raise ValueError(
                    "Only FCC(sg #225), BCC (sg #229), SC (#221), DIAM(#227) or cubic-cell structures are acceptable for {} transformation type, but you provide "
                    "structure with space group #{} and cell {}".format(
                        self.transformation_type, spgn, self.basis_ref.get_cell()
                    )
                )
        elem = self.basis_ref.get_chemical_symbols()[0]
        self.transformation_type = self.transformation_type
        self.element = elem
        # self.a0 = a0
        self.num_of_point = self.num_of_point

        self._initialized = True

    @property
    def transformation_type(self):
        return self._value["transformation_type"]

    @transformation_type.setter
    def transformation_type(self, value):
        self._value["transformation_type"] = value

    def deformation_path(self):
        if self.transformation_type == TransformationPathCalculator.TETRAGONAL:
            path_indices = np.linspace(0.8, 2, self.num_of_point)
        elif self.transformation_type == TransformationPathCalculator.ORTHOGONAL:
            path_indices = np.linspace(1.0, np.sqrt(2.0), self.num_of_point)
        elif self.transformation_type == TransformationPathCalculator.TRIGONAL:
            path_indices = np.linspace(0.8, 5.0, self.num_of_point)
        elif self.transformation_type == TransformationPathCalculator.HEXAGONAL:
            path_indices = np.linspace(-0.5, 1.8, self.num_of_point)
        elif (
            self.transformation_type
            == TransformationPathCalculator.GENERAL_CUBIC_TETRAGONAL
        ):
            path_indices = np.linspace(0.2, 2.6, self.num_of_point)
        else:
            raise NotImplementedError(
                "Transformation path <"
                + str(self.transformation_type)
                + "> is not implemented"
            )
        return path_indices

    def generate_tetra_path(self, indices_only=False):
        def gen_tetr(a0, base_atoms, p):
            a = (a0**3 / p) ** (1.0 / 3.0)
            c = a0**3 / a**2.0
            atoms = base_atoms.copy()
            atoms.set_cell([(a, 0, 0), (0, a, 0), (0, 0, c)], scale_atoms=True)
            return atoms

        path_indices = self.deformation_path()
        if indices_only:
            return path_indices
        else:
            volume = self.basis_ref.get_volume() / len(
                self.basis_ref
            )  # compute volume_per_atoms
            a0 = (volume * 2.0) ** (
                1.0 / 3.0
            )  # lattice constant for BCC, even if original structure was FCC
            base_atoms = Atoms(
                [self.element] * 2,
                scaled_positions=[(0, 0, 0), (1 / 2.0, 1.0 / 2, 1.0 / 2)],
                cell=[(a0, 0, 0), (0, a0, 0), (0, 0, a0)],
                pbc=True,
            )
            structures = []
            for p in path_indices:
                atoms = gen_tetr(a0, base_atoms, p)
                structures.append(atoms)

            atoms = gen_tetr(a0, base_atoms, p=1.0)
            self.base_structure = atoms

            return path_indices, structures

    def generate_ortho_path(self, indices_only=False):
        def gen_orth(a0, p):
            a1 = a0 * np.array([np.sqrt(2.0), 0.0, 0.0])
            a2 = a0 * np.array([0.0, p, 0.0])
            a3 = a0 * np.array([0.0, 0.0, np.sqrt(2.0) / p])
            cell = np.array([a1, a2, a3])
            atoms = Atoms(
                [self.element] * 4,
                scaled_positions=[
                    (0, 0, 0),
                    (0.5, 0.5, 0.0),
                    (0.5, 0.0, 0.5),
                    (0.0, 0.5, 0.5),
                ],
                cell=cell,
                pbc=True,
            )
            return atoms

        path_indices = self.deformation_path()
        if indices_only:
            return path_indices
        else:
            volume = self.basis_ref.get_volume() / len(
                self.basis_ref
            )  # compute volume_per_atoms
            a0 = (volume * 2.0) ** (
                1.0 / 3.0
            )  # lattice constant for BCC, even if original structure was FCC
            structures = []
            for p in path_indices:
                atoms = gen_orth(a0, p)
                structures.append(atoms)
            self.base_structure = gen_orth(a0, p=1.0)
            return path_indices, structures

    def generate_trigo_path(self, indices_only=False):
        def gen_tri(base_atoms, cell, p):
            trigo = np.array(
                [
                    [
                        0.0,
                        (-2.0 / np.sqrt(6.0))
                        / ((np.power(p, 2.0 / 3.0)) ** (1.0 / 2.0)),
                        np.power(p, 2.0 / 3.0) / np.sqrt(3.0),
                    ],
                    [
                        (np.sqrt(2.0) / 2.0)
                        / ((np.power(p, 2.0 / 3.0)) ** (1.0 / 2.0)),
                        1.0 / np.sqrt(6.0) / ((np.power(p, 2.0 / 3.0)) ** (1.0 / 2.0)),
                        np.power(p, 2.0 / 3.0) / np.sqrt(3.0),
                    ],
                    [
                        (-np.sqrt(2.0) / 2.0)
                        / ((np.power(p, 2.0 / 3.0)) ** (1.0 / 2.0)),
                        1.0 / np.sqrt(6.0) / ((np.power(p, 2.0 / 3.0)) ** (1.0 / 2.0)),
                        np.power(p, 2.0 / 3.0) / np.sqrt(3.0),
                    ],
                ]
            )

            atoms = base_atoms.copy()
            atoms.set_cell(np.array(np.dot(cell, trigo)), scale_atoms=True)
            return atoms

        path_indices = self.deformation_path()
        if indices_only:
            return path_indices
        else:
            volume = self.basis_ref.get_volume() / len(
                self.basis_ref
            )  # compute volume_per_atoms
            a0 = (volume * 2.0) ** (
                1.0 / 3.0
            )  # lattice constant for BCC, even if original structure was FCC
            base_atoms = Atoms(
                [self.element] * 2,
                scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)],
                cell=[(a0, 0, 0), (0, a0, 0), (0, 0, a0)],
                pbc=True,
            )
            cell = np.array(base_atoms.get_cell())
            structures = []
            for p in path_indices:
                atoms = gen_tri(base_atoms, cell, p)
                structures.append(atoms)
            self.base_structure = gen_tri(base_atoms, cell, p=1.0)
            return path_indices, structures

    def generate_hex_path(self, indices_only=False):
        def gen_hex(a0, p):
            brac1 = p * (2 * np.sqrt(3) - 3 * np.sqrt(2)) / 6.0 + np.sqrt(2) / 2.0
            brac2 = p * (2 * np.sqrt(2) - 3.0) / 3.0 + 1
            vv0 = np.sqrt(2) * brac1 * brac2
            a = a0 * np.sqrt(2) / (vv0 ** (1.0 / 3.0))
            b = a * brac1
            c = a * brac2
            pos1 = (0.5 - p / 6.0, 0, 0.5)
            pos2 = (0, 0, 0)
            pos3 = (0.5, 0.5, 0)
            pos4 = (-p / 6.0, 0.5, 0.5)
            a1 = a * np.array([1, 0, 0])
            a2 = b * np.array([0, 1, 0])
            a3 = c * np.array([0, 0, 1])
            cell = np.array([a1, a2, a3])
            atoms = Atoms(
                [self.element] * 4,
                scaled_positions=[pos1, pos2, pos3, pos4],
                cell=cell,
                pbc=True,
            )
            cell *= (a * b * c / atoms.get_volume()) ** (1 / 3.0)
            atoms.set_cell(cell, scale_atoms=True)
            return atoms

        path_indices = self.deformation_path()
        if indices_only:
            return path_indices
        else:
            volume = self.basis_ref.get_volume() / len(
                self.basis_ref
            )  # compute volume_per_atoms
            a0 = (volume * 2.0) ** (
                1.0 / 3.0
            )  # lattice constant for BCC, even if original structure was FCC

            structures = []
            for p in path_indices:
                atoms = gen_hex(a0, p)
                structures.append(atoms)
            self.base_structure = gen_hex(a0, p=0.0)
            return path_indices, structures

    def generate_general_tetra_path(self, indices_only=False):
        def gen_tetr(a0, base_atoms, p):

            a = (a0**3 / p) ** (1.0 / 3.0)
            c = a0**3 / a**2.0
            atoms = base_atoms.copy()
            atoms.set_cell([(a, 0, 0), (0, a, 0), (0, 0, c)], scale_atoms=True)
            return atoms

        path_indices = self.deformation_path()
        if indices_only:
            return path_indices
        else:
            base_atoms = to_cubic_atoms(self.basis_ref)
            a0 = base_atoms.cell[0, 0]

            structures = []
            for p in path_indices:
                atoms = gen_tetr(a0, base_atoms, p)
                structures.append(atoms)

            atoms = gen_tetr(a0, base_atoms, p=1.0)
            self.base_structure = atoms

            return path_indices, structures

    def generate_path(self, indices_only=False):
        if self.transformation_type == TransformationPathCalculator.TETRAGONAL:
            return self.generate_tetra_path(indices_only)
        elif self.transformation_type == TransformationPathCalculator.ORTHOGONAL:
            return self.generate_ortho_path(indices_only)
        elif self.transformation_type == TransformationPathCalculator.TRIGONAL:
            return self.generate_trigo_path(indices_only)
        elif self.transformation_type == TransformationPathCalculator.HEXAGONAL:
            return self.generate_hex_path(indices_only)
        elif (
            self.transformation_type
            == TransformationPathCalculator.GENERAL_CUBIC_TETRAGONAL
        ):
            return self.generate_general_tetra_path(indices_only)
        else:
            raise NotImplementedError(
                "Transformation path <"
                + str(self.transformation_type)
                + "> is not implemented"
            )

    def generate_structures(self, verbose=False):
        if not self._initialized:
            self.initialize()
        path_ind, structures = self.generate_path()
        self.indices = path_ind
        self._value["transformation_coordinates"] = self.indices
        for p, sc in zip(path_ind, structures):
            job_name = self.subjob_name(p)
            set_magnetic_moments(self.basis_ref, sc)
            self._structure_dict[job_name] = sc
        return self._structure_dict

    def analyse_structures(self, output_dict):
        self.generate_structures()
        energies = []
        energies_0 = []
        n_at = None
        for p in self.indices:
            job_name = self.subjob_name(p)
            dct = output_dict[job_name]
            energies.append(dct["energy"])
            if "energy_0" in dct:
                energies_0.append(dct["energy_0"])
            n_at = dct.get("n_at")

        self._value["energies"] = np.array(energies)
        self._value["n_at"] = n_at
        if len(energies_0) > 0:
            self._value["energies_0"] = np.array(energies_0)
        self._value["transformation_coordinates"] = self.indices

    def get_structure_value(self, structure, name=None):
        if isinstance(structure.calc, AMSDFTBaseCalculator):
            calc = structure.calc
            calc.static_calc()
            calc.auto_kmesh_spacing = False
            calc.update_kmesh_from_spacing(self.basis_ref)
            # do actual calculations
            structure.get_potential_energy(force_consistent=True)
            # update structure to optimized structure
            structure = calc.atoms
            structure.calc = calc
        en0 = structure.get_potential_energy(force_consistent=True)
        en = en0
        return {"energy": en, "energy_0": en0, "n_at": len(structure)}, structure

    def plot(self, ax=None, **kwargs):
        from matplotlib import pylab as plt

        ax = ax or plt.gca()
        if "label" not in kwargs:
            kwargs["label"] = self.value["transformation_type"]
        ax.plot(
            self.value["transformation_coordinates"], self.value["energies_0"], **kwargs
        )
        ax.set_xlabel("Transformation coordinate, p")
        ax.set_ylabel("Energy, eV")
