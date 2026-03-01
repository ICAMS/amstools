import numpy as np
from ase.optimize import BFGS
from scipy.optimize import curve_fit

from amstools.properties.generalcalculator import GeneralCalculator
from amstools.utils import output_structures_todict, output_structures_fromdict
from amstools.properties.murnaghan import MurnaghanCalculator
from amstools.properties.phonons import PhonopyCalculator


def birch_murn(v, e0, v0, b0, bp):
    return e0 + 9 * v0 * b0 / 16 * (
        ((v0 / v) ** (2.0 / 3.0) - 1.0) ** 3.0 * bp
        + ((v0 / v) ** (2.0 / 3.0) - 1.0) ** 2.0 * (6 - 4 * (v0 / v) ** (2.0 / 3.0))
    )


eV_to_kJ_per_mol = 96.485307
kJ_per_mol_to_eV = 1.0 / eV_to_kJ_per_mol
eV_div_A3_to_GPa = 160.21766208


class ThermodynamicQHACalculator(GeneralCalculator):
    """Calculation of thermodynamic quantities in quasiharmonic approximation

    :param atoms: original ASE Atoms object with calculator
    :param interaction_range: supercell size (default is 10 A)
    :param supercell_range:  supercell range (optional). If specified, interaction_range is ignored
    :param displacement: atom displacement from original position, in angstroms (default is 0.01)
    :param imaginary_phonons_fraction_threshold: maximum fraction of DOS with imaginary frequencies (default = 0.01)
    :param murnaghan: MurnaghanCalculator, default - None, E-V curve would be computed
    :param volume_range: default=0.1, volume range for E-V curve
    :param fit_order: default=5, polynomial fit order for E-V curve
    :param num_of_point: default=11, number of points in E-V curve
    :param optimize_deformed_structure: (for E-V) whether to optimize (atomic only relaxation) of deformed structure (default: False)
    :param optimizer: (for E-V) which optimizer to use (default: BFGS)
    :param fmax: (for E-V) maximum force tolerance for structure optimization
    :param optimizer_kwargs: (for E-V) extra keyword argumetns for `optimizer`
    :param q_space_sample: default=75, q-mesh size for computing/integrating phonon DOS
    :param tmin, tmax: default=(1,3000) temperature range to compute thermodynamic properties
    """

    property_name = "qha"

    param_names = [
        "interaction_range",
        "displacement",
        "supercell_range",
        "force_cutoff",
        "fit_order",
        "num_of_point",
        "q_space_sample",
        "volume_range",
        "optimize_deformed_structure",
        "optimizer",
        "fmax",
        "optimizer_kwargs",
        "imaginary_phonons_fraction_threshold",
        "tmin",
        "tmax",
    ]

    def __init__(
        self,
        atoms=None,
        interaction_range=10.0,
        supercell_range=None,
        displacement=0.01,
        force_cutoff=None,
        murnaghan=None,
        volume_range=0.1,
        fit_order=5,
        num_of_point=11,
        optimize_deformed_structure=False,
        optimizer=BFGS,
        fmax=0.005,
        optimizer_kwargs=None,
        q_space_sample=75,
        imaginary_phonons_fraction_threshold=0.01,
        tmin=1,
        tmax=3000,
        **kwargs,
    ):
        if atoms is None and murnaghan is not None:
            if hasattr(murnaghan, "original_structure"):
                atoms = murnaghan.original_structure.get_atoms()
                atoms.calc = murnaghan.CALCULATOR.create_ase_calculator()
            elif hasattr(murnaghan, "basis_ref"):
                atoms = murnaghan.basis_ref
                atoms.calc = murnaghan.basis_ref.calc
            else:
                raise ValueError(
                    "Neither atoms nor correct murnaghan have been provided"
                )
        GeneralCalculator.__init__(self, atoms, **kwargs)

        self.imaginary_phonons_fraction_threshold = imaginary_phonons_fraction_threshold
        self.interaction_range = interaction_range
        self.displacement = displacement
        self.supercell_range = supercell_range
        self.force_cutoff = force_cutoff
        self.q_space_sample = q_space_sample
        self.tmin = tmin
        self.tmax = tmax
        self._structure_dict = {}

        self.optimize_deformed_structure = optimize_deformed_structure
        self.optimizer = optimizer
        self._init_kwargs(optimizer_kwargs=optimizer_kwargs)
        self.fmax = fmax

        self.murnaghan = None
        if murnaghan is not None and hasattr(murnaghan, "_value"):
            self.murnaghan = murnaghan
            self.num_of_point = len(self.murnaghan.value["energy"])
            self.fit_order = "BM"
        else:
            self.volume_range = volume_range
            self.num_of_point = num_of_point
            self.fit_order = fit_order

    def generate_structures(self, verbose=False):
        # if no murnaghan data were provided, calculate them
        if not self.murnaghan:
            self.murnaghan = MurnaghanCalculator(
                self.basis_ref,
                num_of_point=self.num_of_point,
                volume_range=self.volume_range,
                optimize_deformed_structure=self.optimize_deformed_structure,
                optimizer=self.optimizer,
                fmax=self.fmax,
                optimizer_kwargs=self.optimizer_kwargs,
            )
            self._structure_dict["murnaghan"] = self.murnaghan
            return self._structure_dict

        self._structure_dict = self.qha_generate_phonons_jobs()
        return self._structure_dict

    def qha_generate_phonons_jobs(self):
        volumes = []
        energies = []
        pressures = []
        murn_atoms_list = []
        for name, (e, v, p) in self.murnaghan.output_dict.items():
            volumes.append(v)
            energies.append(e)
            pressures.append(p)
            murn_atoms_list.append(self.murnaghan.output_structures_dict[name])
        inds = np.argsort(volumes)
        volumes = np.array(volumes)[inds]
        energies = np.array(energies)[inds]
        murn_atoms_list = [murn_atoms_list[i] for i in inds]
        self.volumes = volumes
        self.energies = energies
        atoms = self.basis_ref.copy()
        unitcell = atoms.get_cell()
        PHONON_INTERACTION_DIST = self.interaction_range
        if self.supercell_range is None:
            supercell_range = np.ceil(
                PHONON_INTERACTION_DIST
                / np.array([np.linalg.norm(vec) for vec in unitcell])
            )
        else:
            supercell_range = self.supercell_range
        for ind, new_atoms in enumerate(murn_atoms_list):
            new_atoms.calc = self.basis_ref.calc
            phonopy_calc = PhonopyCalculator(
                new_atoms,
                q_mesh=1,
                displacement=self.displacement,
                force_cutoff=self.force_cutoff,
                supercell_range=supercell_range,
            )
            jobname = self.subjob_name(ind)
            self._structure_dict[jobname] = phonopy_calc
        return self._structure_dict

    @staticmethod
    def subjob_name(ind):
        return "phonopy_vol_%d" % ind

    def analyse_structures(self, output_dict):

        if "murnaghan" in output_dict:
            murn = output_dict["murnaghan"]
            self.volumes = murn._value["volume"]
            self.energies = murn._value["energy"]

        F_V_T = []
        new_en = []
        new_vol = []
        for ind, (elec_en, volume) in enumerate(zip(self.energies, self.volumes)):
            jobname = self.subjob_name(ind)
            pj = output_dict[jobname]
            pj.phonopy.run_mesh([self.q_space_sample] * 3)
            pj.phonopy.run_total_dos(
                use_tetrahedron_method=False
            )  # "False" for backward compatibility
            dos_dict = pj.phonopy.get_total_dos_dict()
            te, td = dos_dict["frequency_points"], dos_dict["total_dos"]
            im = td[te < 0].sum() / td.sum()
            if self.verbose:
                print("Volume=", volume, " imag.frequencies fraction = ", im)
            if im < self.imaginary_phonons_fraction_threshold:
                pj.phonopy.run_thermal_properties(t_step=self.tmin, t_max=self.tmax)
                tp = pj.phonopy.get_thermal_properties_dict()
                F_V_T.append([volume, tp["free_energy"] + elec_en * eV_to_kJ_per_mol])
                new_en.append(elec_en)
                new_vol.append(volume)

        self.energies = new_en
        self.volumes = new_vol

        if len(F_V_T) > 0:
            temperatures = tp["temperatures"]
            F_V_T = sorted(F_V_T)

            fvt = []
            for v, fv in F_V_T:
                fvt.append(fv)
            fvt = np.array(fvt)

            Fmin = []
            Vmin = []
            Bmin = []
            for tind, t in enumerate(temperatures):
                es = fvt[:, tind]
                emin = np.min(es)
                vmin = self.volumes[np.argmin(es)]
                try:
                    # noinspection PyTypeChecker
                    if self.fit_order == "BM":
                        popt, pcov = curve_fit(
                            birch_murn, self.volumes, es, p0=(emin, vmin, 0, 0)
                        )
                        e_eq = popt[0]
                        volume_eq = popt[1]
                        B_eq = popt[2] * 10 / 6.0221409
                        if volume_eq < min(self.volumes) and volume_eq > max(
                            self.volumes
                        ):
                            volume_eq = None
                    else:
                        z = np.polyfit(self.volumes, es, self.fit_order)
                        p_fit = np.poly1d(z)
                        p_deriv_1 = np.polyder(p_fit, 1)
                        roots = np.roots(p_deriv_1)
                        volume_eq_lst = np.array(
                            [
                                np.real(r)
                                for r in roots
                                if (
                                    abs(np.imag(r)) < 1e-10
                                    and r >= min(self.volumes)
                                    and r <= max(self.volumes)
                                )
                            ]
                        )
                        e_eq_lst = p_fit(volume_eq_lst)
                        arg = np.argsort(e_eq_lst)
                        e_eq = e_eq_lst[arg][0]
                        if len(e_eq_lst) == 0:
                            volume_eq = None
                        else:
                            volume_eq = volume_eq_lst[arg][0]
                            # get bulk modulus at equ. lattice const.
                            p_2deriv = np.polyder(p_fit, 2)
                            a2 = p_2deriv(volume_eq)
                            B_eq = volume_eq * a2 * kJ_per_mol_to_eV * eV_div_A3_to_GPa

                    if volume_eq is None:
                        print(
                            "Minimum could not be found at T={} K! Stopping at this temperature".format(
                                t
                            )
                        )
                        break
                    Fmin.append(e_eq)
                    Vmin.append(volume_eq)
                    Bmin.append(B_eq)
                except Exception as e:
                    # raise e
                    break

            temperatures = temperatures[: len(Fmin)]
            Fmin = np.array(Fmin)
            Vmin = np.array(Vmin)
            Bmin = np.array(Bmin)

            self.Fmin = Fmin
            self.temperatures = temperatures
            self.F_V_T = fvt[:, : len(temperatures)]

            self._value["G_QHA"] = (temperatures, Fmin)
            self._value["V_QHA"] = (temperatures, Vmin)
            self._value["B_QHA"] = (temperatures, Bmin)

            betas = (
                2
                * (Vmin[1:] - Vmin[:-1])
                / (Vmin[1:] + Vmin[:-1])
                / (temperatures[1:] - temperatures[:-1])
            )
            T1 = (temperatures[1:] + temperatures[:-1]) / 2
            self._value["beta_QHA"] = (T1, betas)

            DFmin = (Fmin[1:] - Fmin[:-1]) / (temperatures[1] - temperatures[0])
            DDFmin = (DFmin[1:] - DFmin[:-1]) / (T1[1] - T1[0])
            T2 = (T1[1:] + T1[:-1]) / 2
            Cp = -T2 * DDFmin * 1000

            self._value["Cp_QHA"] = (T2, Cp)
        else:
            raise ValueError("All phonon calculations have imaginary frequencies")

    def get_structure_value(self, structure, name=None):
        structure.calculate(verbose=self.verbose)
        if name == "murnaghan":
            self.qha_generate_phonons_jobs()
            # self.energies = structure._value["energy"]
        return structure, structure

    @property
    def energies(self):
        return self._value.get("energy")

    @energies.setter
    def energies(self, value):
        self._value["energy"] = value

    @property
    def volumes(self):
        return self._value.get("volume")

    @volumes.setter
    def volumes(self, value):
        self._value["volume"] = value

    def todict(self):
        serialized_output_structures_dict = {}
        for sname, prop in self.output_structures_dict.items():
            serialized_output_structures_dict[sname] = output_structures_todict(
                prop.output_structures_dict
            )

        # temporary copy
        output_structures_dict = self.output_structures_dict
        # clear
        self.output_structures_dict = {}
        calc_dct = super().todict()
        # restore
        self.output_structures_dict = output_structures_dict

        calc_dct["output_structures"] = serialized_output_structures_dict
        # TODO: dump phonopy subjobs
        return calc_dct

    @classmethod
    def fromdict(cls, calc_dct):
        # pop serialized output_structures
        output_structures_dict = calc_dct.pop("output_structures")
        # deserializie it manually
        deserialized_output_structures_dict = {}
        for sname, sprop in output_structures_dict.items():
            deserialized_output_structures_dict[sname] = output_structures_fromdict(
                sprop
            )
        # call parent deserialization
        new_calc = super().fromdict(calc_dct)
        # assign deserialized_output_structures_dict
        new_calc.output_structures_dict = deserialized_output_structures_dict

        # TODO: load phonopy subjobs
        return new_calc
    def plot(self, ax=None, key="G_QHA", **kwargs):
        """
        Plot thermodynamic quantities in QHA
        :param key: "G_QHA", "V_QHA", "B_QHA", "beta_QHA", "Cp_QHA"
        """
        import matplotlib.pyplot as plt

        if key not in self.value:
            print(f"Key {key} not found in results")
            return

        t, val = self.value[key]

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(t, val, **kwargs)
        ax.set_xlabel("Temperature (K)")
        
        labels = {
            "G_QHA": "Free Energy (kJ/mol)",
            "V_QHA": "Equilibrium Volume (A$^3$)",
            "B_QHA": "Bulk Modulus (GPa)",
            "beta_QHA": "Thermal Expansion (K$^{-1}$)",
            "Cp_QHA": "Heat Capacity (J/mol/K)"
        }
        ax.set_ylabel(labels.get(key, key))
        ax.set_title(f"QHA: {key}")
        
        return ax
