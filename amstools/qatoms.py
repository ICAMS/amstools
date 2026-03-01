from __future__ import annotations

import pathlib
from collections import Counter
from collections.abc import Callable, Iterable
from typing import IO, Optional, Union

import numpy as np
import pandas as pd
from ase import Atoms, Atom
from ase.build import bulk, make_supercell
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.neighborlist import neighbor_list
from ase.optimize import BFGS
from ase.visualize import view

from amstools.analyze import compute_and_plot_rdf, compute_rdf, plot_rdf


class QAtoms(Atoms):

    @staticmethod
    def bulk(*args, **kwargs):
        return QAtoms(bulk(*args, **kwargs))

    def _init_default_selection(self):
        self.arrays["_selected_mask"] = np.ones(len(self)).astype(bool)

    @property
    def selected_mask(self):
        if "_selected_mask" not in self.arrays:
            self._init_default_selection()
        return self.arrays["_selected_mask"]

    @selected_mask.setter
    def selected_mask(self, val):
        if len(val) != len(self):
            raise ValueError("length of mask does not equal to number of atoms")
        if not hasattr(self, "_selected_mask"):
            self._init_default_selection()
        self.arrays["_selected_mask"] = val

    @property
    def selected_inds(self):
        return np.where(self.selected_mask)[0]

    @selected_inds.setter
    def selected_inds(self, val):
        new_mask = np.zeros(len(self)).astype(bool)
        new_mask[val] = True
        self.selected_mask = new_mask

    def __repr__(self):
        old_repr = super().__repr__()
        return old_repr[:-1] + f", {len(self.selected_inds)} selected atoms)"

    def __mul__(self, rep):
        return super().repeat(rep)

    def copy(self) -> QAtoms:
        new_at = super().copy()
        new_at.selected_mask = self.selected_mask
        new_at.calc = self.calc
        return new_at

    def select(
        self, tag=None, tag_min=None, tag_max=None, element=None, from_all=False
    ) -> QAtoms:
        """
        Select atoms by tag, range of tag (tag_min or tag_max), element.
        tag: tag of atoms to select
        tag_min, tag_max: inclusive range of tag [tag_min, tag_max] to select. Can be used separately
        all: (default=False) select from all atoms or from previous selection
        """
        if from_all:
            sel_mask = np.ones_like(self).astype(bool)
        else:
            sel_mask = self.selected_mask

        tags = np.array(self.get_tags())
        symbols = np.array(self.get_chemical_symbols())

        if tag is not None:
            sel_mask = sel_mask & (tags == tag)

        if tag_min is not None:
            sel_mask = sel_mask & (tags >= tag_min)

        if tag_max is not None:
            sel_mask = sel_mask & (tags <= tag_max)

        if element is not None:
            sel_mask = sel_mask & (symbols == element)

        new_at = self.copy()
        new_at.selected_mask = sel_mask
        return new_at

    def all(self) -> QAtoms:
        """
        Select all atoms
        """
        new_at = self.copy()
        new_at.selected_mask = np.ones_like(self).astype(bool)
        return new_at

    def clear(self) -> QAtoms:
        """
        Clear selection, i.e. select None atoms
        """
        new_at = self.copy()
        new_at.selected_mask = np.zeros_like(self).astype(bool)
        return new_at

    def invert_selection(self) -> QAtoms:
        """
        Invert the current selection. All previously selected atoms will be
        deselected, and all deselected atoms will be selected.
        """
        new_at = self.copy()
        new_at.selected_mask = ~self.selected_mask
        return new_at

    def sample(
        self, n=None, frac=None, random_state=None, ignore_errors=False
    ) -> QAtoms:
        """
        Select `n` atom(s) randomly from previously selection
        """
        sel_inds = self.selected_inds
        if len(sel_inds) < 1 and not ignore_errors:
            raise ValueError(
                "No selected atoms. Use .all() to selecte all atoms before or any other filters"
            )
        if frac is not None:
            n = int(np.ceil(frac * len(sel_inds)))
        if n > len(sel_inds) and not ignore_errors:
            raise ValueError(
                f"Cannot filter more atoms (n={n}) than already selected ({len(sel_inds)})"
            )
        if random_state is not None:
            np.random.seed(random_state)
        inds = np.random.choice(sel_inds, size=min(n, len(sel_inds)), replace=False)
        return self.copy().select_explicitly(inds)

    def set(self, element=None, tag=None) -> QAtoms:
        """
        For  SELECTED atoms, set element or tag. Selection remains the same.
        """
        new_at = self.copy()

        if element is not None:
            symbs = np.array(self.get_chemical_symbols(), dtype=object)
            symbs[self.selected_mask] = element
            new_at.set_chemical_symbols(symbs)
        if tag is not None:
            tags = np.array(self.get_tags())
            tags[self.selected_mask] = tag
            new_at.set_tags(tags)
        return new_at

    def substitute(self, substitutions: dict) -> QAtoms:
        """
        Substitute elements for selected atoms based on a mapping.

        This operation applies only to the currently selected atoms.

        Args:
            substitutions (dict): A dictionary mapping element symbols to be
                                  replaced to their new symbols.
                                  e.g., {'Al': 'Si', 'O': 'N'}

        Returns:
            QAtoms: A new QAtoms object with substituted atoms.
        """
        new_at = self.copy()
        symbols = np.array(new_at.get_chemical_symbols(), dtype=object)

        for old_elem, new_elem in substitutions.items():
            symbols[(symbols == old_elem) & self.selected_mask] = new_elem

        new_at.set_chemical_symbols(symbols)
        return new_at

    def select_nearby(self, cutoff=2.5, self_exclude=True) -> QAtoms:
        """
        Select atoms within cutoff nearby  every already selected atoms.
        cutoff: (default=2.5) cutoff radius for selection of new atoms
        self_exclude: (default=True) remove originally selected atoms from new selection
        """
        selected_inds = self.selected_inds
        ii, jj, DD = neighbor_list("ijd", self, cutoff=cutoff)

        new_selected_inds = set()

        for si in selected_inds:
            mask = ii == si
            mask2 = DD[mask] <= cutoff
            jj_sel = jj[mask][mask2]
            new_selected_inds.update(jj_sel)

        if self_exclude:
            new_selected_inds = [i for i in new_selected_inds if i not in selected_inds]

        new_selected_inds = sorted(new_selected_inds)

        return self.copy().select_explicitly(new_selected_inds)

    def select_nn(self, n=1, max_cutoff=5, self_exclude=True) -> QAtoms:
        """
        Select first `n` nearest neighbours of each currently selected atom.
        Consider only nearest atoms within `max_cutoff` radius.

        n: (default=1) number of nearest atoms to select
        max_cutoff: (default=5) maximal cutoff for consideration
        self_exclude: (default=True) remove originally selected atoms from new selection
        """

        ii, jj, DD = neighbor_list("ijd", self, cutoff=max_cutoff)

        new_selected_inds = set()

        for i in self.selected_inds:
            mask = ii == i  # bool mask, nat
            jj_mask = jj[mask]  # n_neigh of i
            dd_mask = DD[mask]  # n_neigh of i
            sort_inds = np.argsort(dd_mask)
            nearest_neigh_ind = sort_inds[:n]
            jj_sel = jj_mask[nearest_neigh_ind]
            new_selected_inds.update(jj_sel)

        if self_exclude:
            new_selected_inds = [
                i for i in new_selected_inds if i not in self.selected_inds
            ]

        new_selected_inds = sorted(new_selected_inds)

        return self.copy().select_explicitly(new_selected_inds)

    def shift_atoms(
        self, shift=None, scaled_shift=None, wrap=True, shift_all_atoms=False
    ) -> QAtoms:
        """
        Shift SELECTED atoms by a given vector.

        Either `shift` (Cartesian) or `scaled_shift` (fractional) can be provided,
        but not both. This operation applies only to the selected atoms.

        Args:
            shift (list or np.ndarray, optional): Cartesian vector [dx, dy, dz]
                to shift atoms by (in Angstroms).
            scaled_shift (list or np.ndarray, optional): Fractional (scaled)
                vector [dsx, dsy, dsz] to shift atoms by.
            wrap (bool): If True (default), wrap atoms back into the unit cell
                after shifting.

        Returns:
            QAtoms: A new QAtoms object with shifted atoms.
        """
        if shift is not None and scaled_shift is not None:
            raise ValueError("Only one of 'shift' or 'scaled_shift' can be provided.")

        new_at = self.copy()
        positions = new_at.get_positions()
        if shift_all_atoms:
            new_at = new_at.all()

        if shift is not None:
            positions[new_at.selected_inds] += np.array(shift)
        if scaled_shift is not None:
            positions[new_at.selected_inds] += new_at.cell.T @ np.array(scaled_shift)

        new_at.set_positions(positions)
        if wrap:
            new_at.wrap()
        return new_at

    def select_explicitly(self, selection_inds=None) -> QAtoms:
        """
        Set selection explicitly
        """

        if isinstance(selection_inds, (int, np.integer)):
            selection_inds = [selection_inds]
        if selection_inds is None:
            selection_inds = []

        new_at = self.copy()
        new_at.selected_inds = selection_inds
        return new_at

    def insert_interstitial(
        self,
        element,
        cutoff=2.5,
        min_dist=1,
        max_attempts=100,
        select_new=True,
        random_state=None,
    ):
        """
        Randomly insert interstitial of type `element` within `cutoff` distance from currently SINGLE selected atoms.
        New interstitials should be at minimal distance `min_dist` from any atoms.

        element: element of interstitial
        cutoff: (default=2.5) maximal distance between currently selected atom and new interstitial
        min_dist: (default=1) minimal distance between new interstitial and any atom
        max_attempts: (default=100) maximal number of trials to insert new interstitial
        select_new: (default=True) flag, whether to select new interstitial or leave old selection
        """
        if random_state is not None:
            np.random.seed(random_state)
        ii, DD = neighbor_list("iD", self, cutoff=cutoff + min_dist)

        # selected_inds=self.selected_inds
        if len(self.selected_inds) > 1:
            raise ValueError("Only one pre-selected atom is supported")

        i = self.selected_inds[0]
        i_pos = self.get_positions()[i]
        cur_mask = ii == i
        cur_D = DD[cur_mask]

        found = False
        new_pos = [0, 0, 0]
        for _ in range(max_attempts):
            # 1. suggest new pos within cutoff around atom "i"
            while True:
                # pos wrt. atom "i"
                new_pos = (np.random.rand(3) - 0.5) * 2 * cutoff
                d = np.linalg.norm(new_pos)
                if d >= min_dist and d <= cutoff:
                    break
            # print("new_pos=",new_pos)
            if (
                len(cur_D) == 0
                or np.min(np.linalg.norm(new_pos - cur_D, axis=1)) >= min_dist
            ):
                found = True
                break

        if not found:
            raise ValueError(
                f"Can't find interstitial position with min_dist={min_dist}"
            )
        new_at = QAtoms(self.to_atoms() + Atom(element, position=i_pos + new_pos))
        if select_new:
            new_at.selected_inds = [len(new_at) - 1]
        else:
            new_at.selected_inds = self.selected_inds
        new_at.wrap()
        return new_at

    def delete(self) -> QAtoms:
        """
        Delete all currently selected atoms.
        """
        new_at = self.copy()
        for i in reversed(new_at.selected_inds):
            del new_at[i]
        new_at.selected_inds = []
        return new_at

    def filter(self, pred: Callable[[Atom], bool]) -> QAtoms:
        """
        Select atoms within structure by explicitly function (predicate)
        pred: function, that take sinlge ase.Atom and return True/False

        Example:
        q.filter(lambda a: (a.symbol=='Al' and a.tag>0))
        """
        return self.select_explicitly([a.index for a in self if pred(a)])

    def to_atoms(self) -> Atoms:
        """
        Convert QAtoms to ase.Atoms
        """
        return Atoms(self)

    def flat(self) -> QAtomsCollection:
        """
        Convert to a collection of QAtoms (QAtomsCollection),
        where each new QAtoms  structure has only one selected atom from
        original structure
        """
        return QAtomsCollection(
            [self.select_explicitly(si) for si in self.selected_inds]
        )

    def map(self, func) -> QAtomsCollection:
        """
        For each selected atom, create a separate QAtoms,
        then apply function `func`, than returns SINGLE QAtoms.
        Finally,  join all results in one list.

        Return QAtomsCollection
        """
        return self.flat().map(func)

    def flatmap(self, func) -> QAtomsCollection:
        """
        For each selected atom, create a separate QAtoms,
        then apply function `func`, that returns LIST of QAtoms.
        Finally,  join/flatten all lists.

        Return QAtomsCollection
        """
        return self.flat().flatmap(func)

    def view(self, *args, **kwargs):
        """
        Call ase.visualize.view function for given QAtoms
        """
        view(self, *args, **kwargs)

    @property
    def num_selected(self):
        """
        Return number of selected atoms
        """
        return len(self.selected_inds)

    @property
    def comp_dict(self):
        return Counter(self.get_chemical_symbols())

    def name(self, str_or_func, append=True, sep="/"):
        current_name = self.info.get("name")
        if isinstance(str_or_func, str):
            name = str_or_func
        elif isinstance(str_or_func, Callable):
            name = str_or_func(self)
        else:
            raise RuntimeError(
                f"Unsupported type of str_or_func: got {type(str_or_func)}, expect: str or Callable"
            )

        if append and current_name:
            name = current_name + sep + name

        new_at = self.copy()
        new_at.info["name"] = name
        return new_at

    def supercell(self, supercell_size_or_P, align_cell=True, **kwargs):
        if isinstance(supercell_size_or_P, int):
            supercell_size_or_P = (
                supercell_size_or_P,
                supercell_size_or_P,
                supercell_size_or_P,
            )

        assert (
            len(supercell_size_or_P) == 3
        ), "len(supercell_size_or_P) should be equal to  3"

        if isinstance(supercell_size_or_P[0], int):
            return self * supercell_size_or_P
        else:
            for v in supercell_size_or_P:
                assert (
                    len(v) == 3
                ), "supercell_size_or_P should be vector of len=3 or 3x3 matrix of int"
            new_at = make_supercell(self, supercell_size_or_P, **kwargs)
            if align_cell:
                new_at.rotate(new_at.cell[0], [1, 0, 0], rotate_cell=True)
                new_at.rotate(new_at.cell[1], [0, 1, 0], rotate_cell=True)
                new_at.rotate(new_at.cell[2], [0, 0, 1], rotate_cell=True)

            return QAtoms(new_at)

    def relax(self, fmax=0.05, calc=None):
        new_atoms = self.copy()
        new_atoms.calc = calc or self.calc

        BFGS(new_atoms).run(fmax=fmax)
        return new_atoms

    def full_relax(self, fmax=0.05, calc=None):
        new_atoms = self.copy()
        new_atoms.calc = calc or self.calc

        BFGS(FrechetCellFilter(new_atoms)).run(fmax=fmax)
        return new_atoms

    @classmethod
    def read(
        cls,
        filename: Union[str, pathlib.PurePath, IO],
        format: Optional[str] = None,
        **kwargs,
    ) -> QAtoms:
        return QAtoms(read(filename, format=format, **kwargs))

    def compute_rdf(
        self, element_pairs=None, max_range=10, nbins=50, for_selected_only=True
    ):
        """
        Compute Radial Distribution Function (RDF).

        This is a wrapper around `amstools.analyze.compute_rdf`.

        Args:
            element_pairs (list, optional): List of element symbol tuples (e.g., [('Si', 'O')])
                                          or a list of symbols (e.g., ['Si', 'O']) from which
                                          pairs are generated. If None, all pairs are used.
            max_range (float): The maximum distance (in Angstroms) for the RDF. Defaults to 10.
            nbins (int): The number of bins for the distance histogram. Defaults to 50.
            for_selected_only (bool): If True (default), compute RDF only for selected atoms.
                                      If False, use all atoms in the structure.

        Returns:
            dict: A dictionary containing the RDF data.
        """
        atoms_to_use = self[self.selected_inds] if for_selected_only else self
        if len(atoms_to_use) == 0:
            print(
                "Warning: No atoms to compute RDF for (selection is empty or structure is empty)."
            )
            return {}
        return compute_rdf(
            atoms_to_use, element_pairs=element_pairs, max_range=max_range, nbins=nbins
        )

    def plot_rdf(
        self,
        element_pairs=None,
        max_range=10,
        nbins=50,
        std_factor=1.0,
        for_selected_only=True,
    ):
        """
        Compute and plot Radial Distribution Function (RDF).

        If RDF data is already computed and stored in `self.info['rdf_data']`,
        it will be plotted directly using `amstools.analyze.plot_rdf`.
        Otherwise, it computes and plots the RDF using `amstools.analyze.compute_and_plot_rdf`.


        Args:
            element_pairs (list, optional): List of element symbol tuples (e.g., [('Si', 'O')])
                                          or a list of symbols (e.g., ['Si', 'O']) from which
                                          pairs are generated. If None, all pairs are used.
            max_range (float): The maximum distance (in Angstroms) for the RDF. Defaults to 10.
            nbins (int): The number of bins for the distance histogram. Defaults to 50.
            std_factor (float): Factor to scale the standard deviation for the shaded uncertainty
                                region in the plot. Defaults to 1.0.
            for_selected_only (bool): If True (default), compute and plot RDF only
                                      for selected atoms. If False, use all atoms.
        """
        if "rdf_data" in self.info:
            plot_rdf(self.info["rdf_data"], std_factor=std_factor)
        else:
            atoms_to_use = self[self.selected_inds] if for_selected_only else self
            if len(atoms_to_use) > 0:
                compute_and_plot_rdf(
                    atoms_to_use,
                    element_pairs=element_pairs,
                    max_range=max_range,
                    nbins=nbins,
                    std_factor=std_factor,
                )
            else:
                print(
                    "Warning: No atoms to plot RDF for (selection is empty or structure is empty)."
                )


def to_QAtoms(atoms) -> QAtoms:
    if isinstance(atoms, QAtoms):
        return atoms
    elif isinstance(atoms, Atoms):
        return QAtoms(atoms)
    else:
        raise ValueError(
            f"Only ASE.Atoms or Qatoms are supported, but got {type(atoms)}"
        )


class QAtomsCollection:

    def __init__(self, initial_atoms_or_list=None):
        self._qatoms_list = []
        if isinstance(initial_atoms_or_list, Atoms):
            self._qatoms_list = [to_QAtoms(initial_atoms_or_list)]
        elif isinstance(initial_atoms_or_list, Iterable):
            self._qatoms_list = [to_QAtoms(a) for a in initial_atoms_or_list]
        elif initial_atoms_or_list is not None:
            raise ValueError(
                f"Unsupported type {type(initial_atoms_or_list)}. Only ASE.Atoms, QAtoms or list of those are supported"
            )

    def __len__(self):
        return len(self._qatoms_list)

    def __iter__(self):
        return iter(self._qatoms_list)

    def __getitem__(self, ind):
        return self._qatoms_list[ind]

    def __repr__(self):
        return f"QAtomsCollection({len(self)} structures)"

    def __add__(self, other: QAtomsCollection) -> QAtomsCollection:
        if isinstance(other, (Atoms, QAtoms)):
            return QAtomsCollection(self._qatoms_list + to_QAtoms(other))
        elif isinstance(other, QAtomsCollection):
            return QAtomsCollection(self._qatoms_list + other._qatoms_list)
        else:
            raise RuntimeError(
                f"Can't add QAtomsCollection and {type(other)}. Only Atoms/QAtoms or QAtomsCollection are possible"
            )

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert collection of QAtoms to pandas.DataFrame with
        `ase_atoms` and `name` columns. Name is extracted from Atoms.info["name"] (if available)
        """
        df = pd.DataFrame({"ase_atoms": [a.copy() for a in self._qatoms_list]})
        df["name"] = df["ase_atoms"].map(lambda at: at.info.get("name"))
        return df

    def drop_duplicates(self) -> QAtomsCollection:
        """
        Drop duplicates in QAtomsCollection
        """
        df = self.to_pandas()
        df = df.drop_duplicates(subset=["ase_atoms"])
        return QAtomsCollection(df["ase_atoms"])

    # general operations
    def map(self, func) -> QAtomsCollection:
        """
        Apply function `func` to each atom in collection
        func: Function(QAtoms)->QAtoms
        """
        return QAtomsCollection([func(qat) for qat in self])

    def flat(self) -> QAtomsCollection:
        """
        Build a new QAtomsCollection from current collection, where each QAtoms has exactly one selected atom.
        Example: QAtomsCollection([QAtoms with 2 selected atom and QAtoms with 3 selected atoms]) -> QAtomsCollection([5 QAtoms with 1 selected atom each])
        """
        return QAtomsCollection([new_qat for qat in self for new_qat in qat.flat()])

    def flatmap(self, func) -> QAtomsCollection:
        """
        Combination of .map(func).flat().
        Apply function `func` to each QAtoms in collection  and then flat the resulted collection,
        i.e. create new collection of QAtoms with single selected atom each.
        """
        return self.map(func).flat()

    # "syntactic sugar" methods (map+QAtoms.method)
    def set(self, element=None, tag=None) -> QAtomsCollection:
        """
        Apply QAtoms.set function to each QAtoms in collection. Number of QAtoms in collection is preserved.
        See QAtoms.set for more details.
        """
        return self.map(lambda q: q.set(element=element, tag=tag))

    def select_nearby(self, cutoff=2.5, self_exclude=True) -> QAtomsCollection:
        """
        Apply QAtoms.select_nearby function to each QAtoms in collection. Number of QAtoms in collection is preserved.
        See QAtoms.select_nearby for more details.
        """
        return self.map(
            lambda q: q.select_nearby(cutoff=cutoff, self_exclude=self_exclude)
        )

    def select_nn(self, n=1, max_cutoff=5, self_exclude=True) -> QAtomsCollection:
        """
        Apply QAtoms.select_nn function to each QAtoms in collection. Number of QAtoms in collection is preserved.
        See QAtoms.select_nn for more details.
        """
        return self.map(
            lambda q: q.select_nn(n=n, max_cutoff=max_cutoff, self_exclude=self_exclude)
        )

    def insert_interstitial(
        self, element, cutoff=2.5, min_dist=1, max_attempts=100, select_new=True
    ) -> QAtomsCollection:
        """
        Apply QAtoms.insert_interstitial function to each QAtoms in collection. Number of QAtoms in collection is preserved.
        See QAtoms.insert_interstitial for more details.
        """
        return self.map(
            lambda q: q.insert_interstitial(
                element=element,
                cutoff=cutoff,
                min_dist=min_dist,
                max_attempts=max_attempts,
                select_new=select_new,
            )
        )

    def select(
        self, tag=None, tag_min=None, tag_max=None, element=None, all=False
    ) -> QAtomsCollection:
        """
        Apply QAtoms.select function to each QAtoms in collection. Number of QAtoms in collection is preserved.
        See QAtoms.select for more details.
        """
        return self.map(
            lambda q: q.select(
                tag=tag, tag_min=tag_min, tag_max=tag_max, element=element, all=all
            )
        )

    def sample(self, n=1, random_state=None, ignore_errors=False) -> QAtomsCollection:
        """
        Apply QAtoms.sample function to each QAtoms in collection. Number of QAtoms in collection is preserved.
        See QAtoms.select for more details.
        """
        return self.map(
            lambda q: q.sample(
                n=n, random_state=random_state, ignore_errors=ignore_errors
            )
        )

    def delete(self):
        """
        Apply QAtoms.delete function to each QAtoms in collection. Number of QAtoms in collection is preserved.
        See QAtoms.delete for more details.
        """
        return self.map(lambda q: q.delete())

    # Visualization
    def view(self, *args, **kwargs) -> None:
        """
        Call ase.visualize.view on whole collection
        """
        view(self, *args, **kwargs)
