import warnings

import numpy as np

from ase.optimize import BFGS

from amstools.calculators.dft.base import AMSDFTBaseCalculator
from amstools.properties.generalcalculator import GeneralCalculator
from amstools.utils import general_atoms_copy


class MurnaghanCalculator(GeneralCalculator):
    """Calculate energy-volume (Murnaghan) equation of state and bulk modulus.

    Computes the equation of state by calculating energies at multiple volumes,
    then fitting to extract equilibrium properties including bulk modulus,
    equilibrium volume, and energy.

    Attributes:
        num_of_point (int): Number of volume points to calculate (default: 11)
        volume_range (float | Tuple[float, float]): Volume deformation range
            - If float: fractional range, e.g., 0.1 = ±10%
            - If tuple: absolute [V_min, V_max] in Å³ (default: 0.1)
        fit_order (int): Polynomial order for E-V curve fitting (default: 5)
        optimize_deformed_structure (bool): Whether to relax atoms at each volume
            - True: Full relaxation at each volume
            - False: Only rescale cell (faster, default)
        optimizer (Type[Optimizer]): ASE optimizer class (default: BFGS)
        fmax (float): Force convergence in eV/Å (default: 0.005)
        optimizer_kwargs (Dict[str, Any]): Additional optimizer arguments

    Example:
        >>> from ase.build import bulk
        >>> from ase.calculators.emt import EMT
        >>> atoms = bulk('Cu', 'fcc', a=3.6)
        >>> atoms.calc = EMT()
        >>> murn = MurnaghanCalculator(atoms, num_of_point=11)
        >>> murn.calculate()
        >>> print(f"B0 = {murn.value['equilibrium_bulk_modulus']:.2f} GPa")
    """

    property_name = "murnaghan"

    param_names = [
        "num_of_point",
        "volume_range",
        "fit_order",
        "optimize_deformed_structure",
        "optimizer",
        "fmax",
        "optimizer_kwargs",
    ]

    def __init__(
        self,
        atoms=None,
        num_of_point=11,
        volume_range=0.1,
        fit_order=5,
        optimize_deformed_structure=False,
        optimizer=BFGS,
        fmax=0.005,
        optimizer_kwargs=None,
        **kwargs,
    ):
        GeneralCalculator.__init__(self, atoms, **kwargs)
        self.num_of_point = num_of_point
        self.volume_range = volume_range
        self.fit_order = fit_order

        self.optimize_deformed_structure = optimize_deformed_structure
        self.optimizer = optimizer
        self._init_kwargs(optimizer_kwargs=optimizer_kwargs)
        self.fmax = fmax

    def get_volume_range(self):
        vol_max, vol_min = self.get_volume_min_max()

        volumes = []
        for strain in np.linspace(vol_min, vol_max, self.num_of_point):
            basis = self.basis_ref.copy()
            cell = basis.get_cell()
            cell *= strain ** (1.0 / 3.0)
            basis.set_cell(cell, scale_atoms=True)
            volumes.append(basis.get_volume())
        return volumes

    def get_volume_min_max(self):
        if isinstance(self.volume_range, (list, tuple, np.ndarray)):
            if len(self.volume_range) != 2:
                raise ValueError(
                    "Unsupported shape of volume_range(list): expect 2 elements, but got {}".format(
                        len(self.volume_range)
                    )
                )
            basis = self.basis_ref.copy()
            cell = basis.get_cell()
            V0 = np.linalg.det(cell)
            vol_min = self.volume_range[0] / V0
            vol_max = self.volume_range[1] / V0
        elif isinstance(self.volume_range, float):
            vol_min = 1 - self.volume_range
            vol_max = 1 + self.volume_range
        else:
            raise ValueError(
                "Unsupported type ({}) of volume_range. Support only [vmin, vmax] (list, tuple, np.array) or v_fraction (float)".format(
                    type(self.volume_range)
                )
            )
        return vol_max, vol_min

    def generate_structures(self, verbose=False):
        basis_ref = self.basis_ref

        vol_max, vol_min = self.get_volume_min_max()

        for strain in np.linspace(vol_min, vol_max, self.num_of_point):
            basis = basis_ref.copy()

            cell = basis.get_cell()
            cell *= strain ** (1.0 / 3.0)

            basis.set_cell(cell, scale_atoms=True)

            jobname = self.subjob_name(strain)
            self._structure_dict[jobname] = basis

        return self._structure_dict

    def analyse_structures(self, output_dict):

        energy_list = []
        volume_list = []
        pressure_list = []
        for name, (e, v, p, *atoms) in output_dict.items():
            energy_list.append(e)
            volume_list.append(v)
            pressure_list.append(p)
        volume_list = np.array(volume_list)
        energy_list = np.array(energy_list)
        pressure_list = np.array(pressure_list)

        ind_order = np.argsort(volume_list)
        volume_list = volume_list[ind_order]
        energy_list = energy_list[ind_order]
        pressure_list = pressure_list[ind_order]

        self._value["volume"] = volume_list
        self._value["energy"] = energy_list
        self._value["pressure"] = pressure_list
        self.fit_murnaghan(self.fit_order)

    def get_structure_value(self, structure, name=None):
        logfile = "-" if self.verbose else None
        if isinstance(structure.calc, AMSDFTBaseCalculator):
            calc = structure.calc
            if self.optimize_deformed_structure:
                calc.optimize_atoms_only()
            else:
                calc.static_calc()
            calc.auto_kmesh_spacing = False
            calc.update_kmesh_from_spacing(self.basis_ref)
            # do actual calculations
            structure.get_potential_energy(force_consistent=True)
            # update structure to optimized structure
            structure = calc.atoms
            structure.calc = calc
        else:
            if self.optimize_deformed_structure:
                dyn = self.optimizer(
                    structure, logfile=logfile, **self.optimizer_kwargs
                )
                try:
                    dyn.set_force_consistent()
                except (TypeError, AttributeError):
                    dyn.force_consistent = False
                dyn.run(fmax=self.fmax)
                structure = dyn.atoms
        en = structure.get_potential_energy(force_consistent=True)
        vol = structure.get_volume()
        stress = structure.get_stress()
        pr = -1.0 / 3.0 * sum(stress[0:3])
        return (en, vol, pr), structure

    @staticmethod
    def subjob_name(strain):
        return "strain_" + str(strain).replace(".", "_")

    def fit_murnaghan(self, fit_order=5):

        fit_dict = {}
        df = self._value
        # compute a polynomial fit
        with warnings.catch_warnings():
            z = np.polyfit(df["volume"], df["energy"], fit_order)
        p_fit = np.poly1d(z)
        fit_dict["poly_fit"] = z

        # get equilibrium lattice constant
        # search for the local minimum with the lowest energy
        p_deriv_1 = np.polyder(p_fit, 1)
        roots = np.roots(p_deriv_1)
        # volume_eq_lst = np.array([np.real(r) for r in roots if np.abs(np.imag(r)) < 1e-10])
        volume_eq_lst = np.array(
            [
                np.real(r)
                for r in roots
                if (
                    abs(np.imag(r)) < 1e-10
                    and min(df["volume"]) <= r <= max(df["volume"])
                )
            ]
        )

        e_eq_lst = p_fit(volume_eq_lst)
        arg = np.argsort(e_eq_lst)
        # print ("v_eq:", arg, e_eq_lst)
        if len(e_eq_lst) == 0:
            print("Minimum could not be found!")
            return None
        e_eq = e_eq_lst[arg][0]
        volume_eq = volume_eq_lst[arg][0]

        eV_div_A3_to_GPa = 160.21766208

        # get bulk modulus at equ. lattice const.
        p_2deriv = np.polyder(p_fit, 2)
        p_3deriv = np.polyder(p_fit, 3)
        a2 = p_2deriv(volume_eq)
        a3 = p_3deriv(volume_eq)

        b_prime = -(volume_eq * a3 / a2 + 1)

        fit_dict["fit_order"] = fit_order
        fit_dict["volume_eq"] = volume_eq
        fit_dict["energy_eq"] = e_eq
        fit_dict["bulkmodul_eq"] = eV_div_A3_to_GPa * volume_eq * a2
        fit_dict["b_prime_eq"] = b_prime
        fit_dict["least_square_error"] = self.get_error(
            df["volume"], df["energy"], p_fit
        )
        fit_dict["energy_rms"] = np.sqrt(
            self.get_error(df["volume"], df["energy"], p_fit)
        )

        df["equilibrium_energy"] = e_eq
        df["equilibrium_volume"] = volume_eq
        df["equilibrium_bulk_modulus"] = fit_dict["bulkmodul_eq"]
        df["equilibrium_b_prime"] = b_prime
        df["energy_rms"] = fit_dict["energy_rms"]

    @staticmethod
    def get_error(x_lst, y_lst, p_fit):
        y_fit_lst = np.array(p_fit(x_lst))
        error_lst = (y_lst - y_fit_lst) ** 2
        return np.mean(error_lst)

    def load_final_structure(self):
        """
        Returns: Structure with equilibrium volume
        """
        assert self.basis_ref is not None
        snapshot = self.basis_ref.copy()

        if hasattr(self.basis_ref, "calculator_id"):
            self.basis_ref.calculator_id = self.basis_ref.calculator_id
        if hasattr(self.basis_ref, "calculator"):
            self.basis_ref.calculator = self.basis_ref.calculator

        old_vol = snapshot.get_volume()
        new_vol = self._value["equilibrium_volume"]
        k = (new_vol / old_vol) ** (1.0 / 3.0)
        new_cell = snapshot.cell * k
        snapshot.set_cell(new_cell, scale_atoms=True)
        snapshot.calc = self.basis_ref.calc
        return general_atoms_copy(snapshot)

    def plot(self, ax=None, **kwargs):
        from matplotlib import pylab as plt

        ax = ax or plt.gca()
        nat = len(self.basis_ref)
        vs = np.array(self.value["volume"]) / nat
        es = np.array(self.value["energy"]) / nat
        ax.plot(vs, es, **kwargs)

        ax.set_xlabel("V, A^3/at")
        ax.set_ylabel("E, eV/at")

        return ax
