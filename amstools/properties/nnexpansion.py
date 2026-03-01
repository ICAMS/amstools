import logging

import numpy as np

from amstools.calculators.dft.base import AMSDFTBaseCalculator
from amstools.pipeline.generalstructure import GeneralStructure
from amstools.properties.generalcalculator import GeneralCalculator
from amstools.utils import compute_nn_distance, arglocalmin, make_periodic_structure, general_atoms_copy, build_job_name


class NearestNeighboursExpansionCalculator(GeneralCalculator):
    """Calcualtion of E-NN distance curve and related quantities
    :param atoms: original ASE Atoms object
    :param num_of_point: number of points
    :param nn_distance_range: nearest neighbours distance deformation range, default = 0.1 (+/- 10%) or [nn_min, nn_max]
    :param nn_distance_step: NN distance step
    :param return_min_structure: (default=False) whether return structure's volume according to minimum energy

    Usage:
    >>  atoms.calc = calculator
    >>  nnexpansion = NearestNeighboursExpansionCalculator(atoms)
    >>  nnexpansion.calculate()
    >>  print(nnexpansion.value)
    """

    property_name = "energy_nn_distance"

    param_names = [
        "num_of_point",
        "nn_distance_range",
        "nn_distance_step",
        "nn_distance_list",
        "fix_kmesh",
        "return_min_structure",
    ]

    def __init__(
        self,
        atoms=None,
        num_of_point=None,
        nn_distance_range=(2, 5),
        nn_distance_step=0.05,
        nn_distance_list=None,
        fix_kmesh=False,
        return_min_structure=False,
        **kwargs,
    ):
        GeneralCalculator.__init__(self, atoms, **kwargs)
        self.num_of_point = num_of_point

        if self.num_of_point is not None:
            self.nn_distance_step = None  # ignoring
        else:
            self.nn_distance_step = nn_distance_step

        self.nn_distance_range = nn_distance_range
        self.fix_kmesh = fix_kmesh
        self.nn_distance_list = nn_distance_list
        self.return_min_structure = return_min_structure

    def get_nn_min_max(self):
        if isinstance(self.nn_distance_range, float):
            min_dist = min(compute_nn_distance(self.basis_ref))
            return min_dist * (1 - self.nn_distance_range), min_dist * (
                1 + self.nn_distance_range
            )
        else:
            return self.nn_distance_range

    def generate_structures(self, verbose=False):
        basis_ref, enforced_pbc = self.enforce_pbc_structure(self.basis_ref)

        nn_min, nn_max = self.get_nn_min_max()
        if self.nn_distance_list is not None:
            self._value["nn_distances"] = self.nn_distance_list
            self._value["nn_distances_step"] = None
            self._value["nn_distance_range"] = [
                min(self.nn_distance_list),
                max(self.nn_distance_list),
            ]
        elif self.nn_distance_step is None:
            self._value["nn_distances"] = np.linspace(nn_min, nn_max, self.num_of_point)
            self._value["nn_distances_step"] = (
                self._value["nn_distances"][1] - self._value["nn_distances"][0]
            )
        else:
            self._value["nn_distances"] = np.arange(
                nn_min, nn_max, self.nn_distance_step
            )
            self._value["nn_distances_step"] = self.nn_distance_step

        volume_list = []
        min_nn_dist = compute_nn_distance(basis_ref).min()
        for nn_dist in self._value["nn_distances"]:
            basis = basis_ref.copy()
            cell = basis.get_cell()
            cell *= nn_dist / min_nn_dist
            basis.set_cell(cell, scale_atoms=True)
            if enforced_pbc:
                self.remove_pbc(basis)
            else:
                try:
                    volume_list.append(basis.get_volume() / len(basis))
                except (ValueError, RuntimeError):
                    # Non-periodic structure has no volume
                    pass
            jobname = self.subjob_name(nn_dist)
            self._structure_dict[jobname] = basis

        if not enforced_pbc:
            self._value["volumes"] = volume_list

        return self._structure_dict

    def enforce_pbc_structure(self, basis_ref):
        if not all(basis_ref.get_pbc()):
            basis_ref = make_periodic_structure(basis_ref)
            enforced_pbc = True
        else:
            enforced_pbc = False
        return basis_ref, enforced_pbc

    def remove_pbc(self, atoms):
        atoms.set_cell(None, scale_atoms=False)
        atoms.set_pbc(False)

    def analyse_structures(self, output_dict):

        energy_list = []
        stresses_list = []
        volume_list = []
        # extrapolation grades for PACE
        gamma_list = []
        gamma_max_list = []
        for nn_dist in self._value["nn_distances"]:
            jobname = self.subjob_name(nn_dist)
            structure_value_dict = output_dict[jobname]
            energy_list.append(structure_value_dict["energy"])
            if "stresses" in structure_value_dict:
                stresses_list.append(structure_value_dict["stresses"])
            if "volume" in structure_value_dict:
                volume_list.append(structure_value_dict["volume"])
            if "gamma" in structure_value_dict:
                gamma_list.append(structure_value_dict["gamma"])
                gamma_max_list.append(structure_value_dict["gamma_max"])

        energy_list = np.array(energy_list)
        self._value["energy"] = energy_list
        if len(stresses_list) == len(energy_list):
            self._value["stresses"] = np.array(stresses_list)
        if len(volume_list) == len(energy_list):
            self._value["volume"] = volume_list
        if len(gamma_list) == len(energy_list):
            self._value["gamma"] = np.array(gamma_list)
            self._value["gamma_max"] = np.array(gamma_max_list)

        try:
            if self._value["nn_distances_step"] is not None:
                self._value["gradient"] = np.gradient(
                    self._value["energy"], self._value["nn_distances_step"]
                )
            else:
                self._value["gradient"] = None
        except (ValueError, TypeError) as e:
            # Gradient computation failed (wrong shape/type)
            logging.debug(f"Failed to compute gradient: {e}")
            self._value["gradient"] = None

        try:
            self._value["energy_min"] = self.get_local_energy_minimum()
            self._value["energy_min_nn_dist"] = self.get_local_energy_minimum_nndist()
        except (ValueError, IndexError, TypeError) as e:
            # No local minimum found or computation failed
            logging.debug(f"Could not determine local energy minimum: {e}")
            pass

    def get_structure_value(self, structure, name=None):
        if isinstance(structure.calc, AMSDFTBaseCalculator):
            structure.calc.static_calc()
            if all(structure.pbc) and self.fix_kmesh:
                # fixed kmesh for small ranges at kmesh for basis_ref
                structure.calc.update_kmesh_from_spacing(self.basis_ref)
                logging.debug("NNexpansion:update_kmesh_from_spacing")
                structure.calc.auto_kmesh_spacing = False
            else:
                # automatic kmesh for larger ranges if fix_kmesh=False (default)
                structure.calc.auto_kmesh_spacing = True
        en = structure.get_potential_energy(force_consistent=True)
        res_dict = {"energy": en}
        try:
            stresses = structure.get_stress()
            volume = structure.get_volume()
            res_dict["stresses"] = stresses
            res_dict["volume"] = volume
        except (ValueError, RuntimeError, NotImplementedError):
            # Calculator doesn't provide stress or structure has no volume
            pass
        if (
            "gamma" in structure.calc.results
            and len(structure.calc.results["gamma"]) > 0
        ):
            res_dict["gamma"] = structure.calc.results["gamma"]
            res_dict["gamma_max"] = np.max(res_dict["gamma"])
        return res_dict, structure

    @staticmethod
    def subjob_name(nn_dist):
        return build_job_name('nndist', f'{nn_dist:.4f}')

    # def get_final_structure(self):
    #     return self.basis_ref

    def get_local_energy_minimum(self):
        mininds = arglocalmin(self._value["energy"])
        if mininds.size > 0:
            minvals = self._value["energy"][mininds]
            return np.min(minvals)

    def get_local_energy_minimum_nndist(self):
        mininds0 = arglocalmin(self._value["energy"])
        if mininds0 is not None:
            mininds = mininds0[np.argmin(self._value["energy"][mininds0])]
            minvals = self._value["nn_distances"][mininds]
            return minvals

    def load_final_structure(self):
        """
        Returns: Structure with equilibrium volume
        """
        assert self.basis_ref is not None
        final_structure, enforced_pbc = self.enforce_pbc_structure(self.basis_ref)

        calc = self.basis_ref.calc
        final_structure.calc = calc

        if not self.return_min_structure:
            return general_atoms_copy(final_structure)

        min_nn_dist = compute_nn_distance(final_structure).min()

        nndist = self.get_local_energy_minimum_nndist()
        cell = final_structure.get_cell()
        cell *= nndist / min_nn_dist
        final_structure.set_cell(cell, scale_atoms=True)
        if enforced_pbc:
            self.remove_pbc(final_structure)
        
        return general_atoms_copy(final_structure)

    def plot(
        self, ax=None, key="energy", per_atom=True, with_interpolation=False, **kwargs
    ):
        from matplotlib import pyplot as plt

        if ax is None:
            ax = plt.gca()

        y_val = self._value[key]
        if per_atom:
            y_val = y_val / len(self.basis_ref)
        p = ax.plot(self._value["nn_distances"], y_val, **kwargs)
        ax.set_xlabel("z, A")
        if key == "energy" and per_atom:
            ax.set_ylabel("E, eV/at")
        if with_interpolation:
            mask = self._value["gamma_max"] <= 1
            kwargs = kwargs.copy()
            kwargs["color"] = p[0].get_color()
            ax.scatter(self._value["nn_distances"][mask], y_val[mask], **kwargs)
        
        return ax
