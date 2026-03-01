from collections import Counter, OrderedDict
from itertools import groupby

import numpy as np

from amstools.utils import atoms_todict, atoms_fromdict, get_spacegroup


def get_composition(occupation):
    cnt = Counter(occupation)
    od = OrderedDict(sorted(cnt.items()))
    composition = " ".join(["%s-%d" % (k, v) for (k, v) in od.items()])
    return composition


def RLE_encoding(iterable):
    grouped = [list(g) for k, g in groupby(iterable)]
    return "".join([gr[0] + str(len(gr)) for gr in grouped])


class GeneralStructure:
    def __init__(self, structure):
        if isinstance(structure, GeneralStructure):
            structure = structure.structure
        # original structure
        self.structure = structure
        if structure is None:
            self.atoms = None
            return

        self.atoms = structure.copy()
        if structure.calc is not None:
            self.atoms.calc = structure.calc

    def todict(self):
        # TODO: Test!
        structure_dict = atoms_todict(self.atoms)  # .todict()
        # structure_dict["extra_info"] = extra_info_dict
        return structure_dict

    @classmethod
    def fromdict(cls, structure_dict):
        # TODO: Test!
        extra_info_dict = (
            structure_dict.pop("extra_info") if "extra_info" in structure_dict else {}
        )
        atoms = atoms_fromdict(structure_dict)
        general_structure = cls(atoms)
        for key, val in extra_info_dict.items():
            setattr(general_structure, key, val)

        return general_structure

    @property
    def calc(self):
        return self.atoms.calc

    @calc.setter
    def calc(self, value):
        self.atoms.calc = value

    def __getattr__(self, name):
        return getattr(self.atoms, name)

    def __eq__(self, other):
        if not isinstance(other, GeneralStructure):
            return False
        a1 = self.atoms
        a2 = other.atoms
        if a1 is None or a2 is None:
            return a1 is a2
        return (
            np.allclose(a1.get_positions(), a2.get_positions())
            and np.allclose(a1.get_cell(), a2.get_cell())
            and np.array_equal(a1.get_pbc(), a2.get_pbc())
            and a1.get_chemical_symbols() == a2.get_chemical_symbols()
        )

    def __repr__(self):
        if self.atoms is None:
            return "GeneralStructure(None)"
        return f"GeneralStructure({self.atoms.get_chemical_formula()}, pbc={self.atoms.get_pbc()})"
