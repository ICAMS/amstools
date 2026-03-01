import pandas as pd
import json
import numpy as np

from typing import List
import json

from collections import Counter
from scipy.spatial import ConvexHull

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(seq, *args, **kwargs):
        return seq


from amstools import *
from amstools.utils import JsonNumpyEncoder
from amstools.pipeline.pipeline import Pipeline

try:
    from structdborm import *

    structdb_imported = True
except ImportError:
    structdb_imported = False
    print("No structdborm support")


def save_to_json(data, filename="data.json"):
    with open(filename, "w") as f:
        json.dump(
            data,
            f,
            separators=(",", ":"),
            sort_keys=True,
            indent=4,
            cls=JsonNumpyEncoder,
        )


def get_dataframe_from_props_list(prop_list):
    prop_df = pd.DataFrame(prop_list, columns=["property"])
    prop_df["SB"] = prop_df["property"].map(
        lambda p: p.original_structure.GENERICPARENT.STRUKTURBERICHT
    )
    prop_df["VALUE"] = prop_df["property"].map(lambda p: p._value)
    prop_df["n_at"] = prop_df["property"].map(lambda p: len(p.original_structure))
    prop_df["CALCULATOR"] = prop_df["property"].map(lambda p: p.CALCULATOR.SHORT_NAME)
    return prop_df


def preprocess_murnaghan_df(tot_murn_df):
    tot_murn_df["E0"] = (
        tot_murn_df["VALUE"].map(lambda d: d.get("equilibrium_energy"))
        / tot_murn_df["n_at"]
    )
    tot_murn_df["V0"] = (
        tot_murn_df["VALUE"].map(lambda d: d.get("equilibrium_volume"))
        / tot_murn_df["n_at"]
    )
    tot_murn_df["B0"] = tot_murn_df["VALUE"].map(
        lambda d: d.get("equilibrium_bulk_modulus")
    )
    tot_murn_df["Bp"] = tot_murn_df["VALUE"].map(lambda d: d.get("equilibrium_b_prime"))
    tot_murn_df["energy"] = (
        tot_murn_df["VALUE"].map(lambda d: np.array(d.get("energy")))
        / tot_murn_df["n_at"]
    )
    tot_murn_df["volume"] = (
        tot_murn_df["VALUE"].map(lambda d: np.array(d.get("volume")))
        / tot_murn_df["n_at"]
    )


def preprocess_elast_properties_df(elast_df):
    elast_df["C"] = elast_df["VALUE"].map(lambda v: v["C"])
    elast_df["C11"] = elast_df["C"].map(lambda C: C[0][0])
    elast_df["C12"] = elast_df["C"].map(lambda C: C[0][1])
    elast_df["C13"] = elast_df["C"].map(lambda C: C[0][2])
    elast_df["C33"] = elast_df["C"].map(lambda C: C[2][2])
    elast_df["C44"] = elast_df["C"].map(lambda C: C[3][3])


def preprocess_phonon_df(tot_phon_df):
    tot_phon_df["dos_energies"] = tot_phon_df["VALUE"].map(lambda d: d["dos_energies"])
    tot_phon_df["dos_total"] = tot_phon_df["VALUE"].map(lambda d: d["dos_total"])


from collections import defaultdict


def extract_vac_energies(vacancy_formation_energy):
    vac_dict = defaultdict(list)
    for k, v in vacancy_formation_energy.items():
        symb, typ = k.split("_")
        vac_dict[typ].append(v)
    return dict(vac_dict)


def preprocess_defects_df(tot_defects_df):
    tot_defects_df["vacancy_formation_energy"] = tot_defects_df["VALUE"].map(
        lambda d: d.get("vacancy_formation_energy")
    )
    tot_defects_df["vac_energies"] = tot_defects_df["vacancy_formation_energy"].map(
        extract_vac_energies
    )
    for k in ["static", "atomic", "total"]:
        tot_defects_df[k] = tot_defects_df["vac_energies"].map(lambda d: d.get(k))


def preprocess_transformation_path_df(dft_trans_df):
    for k in ["energies_0", "transformation_type", "transformation_coordinates"]:
        dft_trans_df[k] = dft_trans_df["VALUE"].map(lambda d: d.get(k))
    return dft_trans_df


class PotentialValidation:
    def __init__(self, reference_calc_name: List[str] = None):

        self.reference_calc_name = reference_calc_name
        self.reference_calc_list = None
        self._storage = None
        self.comp_prop_df = None
        self.smoothness_structures_dict = {}
        self.pipeline_structures_dict = {}

    @property
    def storage(self):
        if structdb_imported:
            if self._storage is None:
                self._storage = StructSQLStorage()
            return self._storage

    def get_reference_calculators(self):
        if structdb_imported and self.reference_calc_name is not None:
            if self.reference_calc_list is None:
                self.reference_calc_list = []
                if isinstance(self.reference_calc_name, (list, tuple)):
                    for calc_name in self.reference_calc_list:
                        self.reference_calc_list.append(
                            self.storage.query(CalculatorType)
                            .filter(CalculatorType.NAME == calc_name)
                            .one()
                        )
                elif isinstance(self.reference_calc_name, str):
                    self.reference_calc_list.append(
                        self.storage.query(CalculatorType)
                        .filter(CalculatorType.NAME == self.reference_calc_name)
                        .one()
                    )

            return self.reference_calc_list
        else:
            return []

    def plot_radial_functions(self, ase_calc):
        pass

    def validate_smoothness(
        self,
        ase_calculator,
        elements,
        include_prototypes=None,
        exclude_prototypes=("dimer"),
        max_cutoff=8.5,
        dx=0.05,
        plot=True,
        json_filename="smoothness.json",
        plot_filename="smoothness.pdf",
    ):
        """
        Perform a validation of the smoothness of E-NN-dist curves for a set of prototypes
        :param include_prototypes: None (all) or list of str - name of prototypes (fcc ,bcc, sc, ...) to include
        :param exclude_prototypes: None (all) or list of str - name of prototypes (fcc ,bcc, sc, ...) to exclude
        :param max_cutoff: float, range of smoothness plots
        :param dx: float, step of smoothness plots
        :param plot: boolean, whether to plot a figures
        :param json_filename: str, filename to store the data or None, do not store the data
        :param plot_filename: str, filename to save the plot or None, do not save the plot
        """
        self.smoothness_structures_dict = get_structures_dictionary(
            elements=elements, include=include_prototypes, exclude=exclude_prototypes
        )
        for struct_name, struct_dict in tqdm(self.smoothness_structures_dict.items()):
            atoms = struct_dict["atoms"]
            if "nncalc" not in struct_dict:
                atoms.calc = ase_calculator
                nncalc = NearestNeighboursExpansionCalculator(
                    atoms,
                    nn_distance_range=[1.0, max_cutoff + 0.1],
                    num_of_point=int((max_cutoff - 1.0 + 0.1) / dx),
                )
                nncalc.calculate()
                struct_dict["nncalc"] = nncalc

        if json_filename:
            data_dict = self.prepare_smoothness_data(self.smoothness_structures_dict)
            save_to_json(data_dict, filename=json_filename)

        if plot:
            import matplotlib.pylab as plt

            plt.figure(figsize=(10, 21), dpi=100)
            plt.subplot(3, 1, 1)
            for struct_name, struct_dict in self.smoothness_structures_dict.items():
                atoms = struct_dict["atoms"]
                if "nncalc" not in struct_dict:
                    atoms.calc = ase_calculator
                    nncalc = NearestNeighboursExpansionCalculator(
                        atoms,
                        nn_distance_range=[1.0, max_cutoff + 0.1],
                        num_of_point=int((max_cutoff - 1.0 + 0.1) / 0.05),
                    )
                    nncalc.calculate()
                    struct_dict["nncalc"] = nncalc
                else:
                    nncalc = struct_dict["nncalc"]
                if plot:
                    plt.plot(
                        nncalc._value["nn_distances"],
                        nncalc._value["energy"] / len(atoms),
                        color=struct_dict["colour"],
                        label=struct_name,
                    )
            plt.ylim(-5, 0.5)
            plt.grid()
            plt.legend()

            plt.subplot(3, 1, 2)
            # plt.figure(figsize=(10, 7), dpi=100)
            for struct_name, struct_dict in self.smoothness_structures_dict.items():
                atoms = struct_dict["atoms"]
                if "nncalc" not in struct_dict:
                    atoms.calc = ase_calculator
                    nncalc = NearestNeighboursExpansionCalculator(
                        atoms,
                        nn_distance_range=[1.0, max_cutoff + 0.1],
                        num_of_point=int((max_cutoff - 1.0 + 0.1) / 0.1),
                    )
                    nncalc.calculate()
                    struct_dict["nncalc"] = nncalc
                else:
                    nncalc = struct_dict["nncalc"]
                plt.plot(
                    nncalc._value["nn_distances"],
                    nncalc._value["gradient"] / len(atoms),
                    color=struct_dict["colour"],
                    label=struct_name,
                )
            plt.ylim(-3, 3)
            plt.grid()
            plt.legend()

            plt.subplot(3, 1, 3)
            for struct_name, struct_dict in self.smoothness_structures_dict.items():
                atoms = struct_dict["atoms"]
                if "nncalc" not in struct_dict:
                    atoms.calc = ase_calculator
                    nncalc = NearestNeighboursExpansionCalculator(
                        atoms,
                        nn_distance_range=[1.0, max_cutoff + 0.1],
                        num_of_point=int((max_cutoff - 1.0 + 0.1) / 0.1),
                    )
                    nncalc.calculate()
                    struct_dict["nncalc"] = nncalc
                else:
                    nncalc = struct_dict["nncalc"]
                plt.plot(
                    nncalc._value["nn_distances"],
                    nncalc._value["stresses"][:, 0],
                    color=struct_dict["colour"],
                    label=struct_name,
                )
            plt.ylim(-3, 3)
            plt.grid()
            plt.legend()

            plt.tight_layout()
            if plot_filename:
                plt.savefig(plot_filename)

    def prepare_smoothness_data(self, smoothness_structures_dict):
        data_dict = {}
        for k, v in smoothness_structures_dict.items():
            v = v.copy()
            del v["atoms"]
            v["VALUE"] = v["nncalc"]._value
            del v["nncalc"]
            data_dict[k] = v
        return data_dict

    def validate_general_properties(
        self,
        ase_calculator,
        elements,
        include_prototypes=None,
        exclude_prototypes=("dimer"),
        properties=("Murnaghan", "Elastic", "Phonons", "Vacancy", "TransformationPath"),
        vacancy_prototypes=("fcc", "bcc", "hcp", "dhcp", "sh"),
        transformation_path_prototypes=("fcc", "bcc"),
        plot=True,
    ):
        self.pipeline_structures_dict = get_structures_dictionary(
            elements=elements, include=include_prototypes, exclude=exclude_prototypes
        )

        self.comp_prop_df = self.run_property_pipelines(
            ase_calculator,
            self.pipeline_structures_dict,
            properties=properties,
            vacancy_prototypes=vacancy_prototypes,
            transformation_path_prototypes=transformation_path_prototypes,
        )

        murn_df = self.comp_prop_df[self.comp_prop_df["job"] == "Murnaghan"].copy()
        dft_mur_props_list = []
        for calc in self.get_reference_calculators():
            dft_mur_props_list += self.storage.query_properties(
                MurnaghanProperty,
                composition=element + "-%",
                calculator=calc,
                generic_strukturbericht=murn_df["SB"].tolist(),
                filters_=[Property.VISIBLE_FOR_COMPARISON],
            )
        dft_murn_df = get_dataframe_from_props_list(dft_mur_props_list)
        tot_murn_df = pd.concat([murn_df, dft_murn_df])
        preprocess_murnaghan_df(tot_murn_df)
        min_energies_dict = tot_murn_df.groupby("CALCULATOR")["E0"].min().to_dict()
        if plot:
            import matplotlib.pylab as plt

            plt.figure(figsize=(10, 7), dpi=100)
            for _, row in tot_murn_df.iterrows():
                try:
                    if np.isnan(row["color"]):
                        color = "gray"
                except (TypeError, KeyError):
                    # Color is not numeric or doesn't exist - use as-is
                    color = row["color"]
                linestyle = "-" if color != "gray" else ":"
                label = row["SB"] if color != "gray" else None
                plt.plot(
                    row["volume"],
                    row["energy"] - min_energies_dict[row["CALCULATOR"]],
                    label=label,
                    ls=linestyle,
                    color=color,
                )
            plt.legend()

    def run_property_pipelines(
        self,
        ase_calculator,
        pipeline_structures_dict,
        properties=("Murnaghan", "Elastic", "Phonons", "Vacancy", "TransformationPath"),
        vacancy_prototypes=("fcc", "bcc", "hcp", "dhcp", "sh"),
        transformation_path_prototypes=("fcc", "bcc"),
    ):
        """ """
        if vacancy_prototypes is None:
            vacancy_prototypes = []
        if transformation_path_prototypes is None:
            transformation_path_prototypes = []

        for struct_name, struct_dict in tqdm(pipeline_structures_dict.items()):
            print("=" * 60)
            print("=" * 60)
            print("Processing structure:", struct_name)

            if not struct_dict["isbulk"]:
                print("Structure is not bulk, skipping")
                continue

            atoms = struct_dict["atoms"]

            if "pipeline" not in struct_dict:
                steps = {"Optimize": StepwiseOptimizer()}
                if "Murnaghan" in properties:
                    steps["Murnaghan"] = MurnaghanCalculator()
                if "Elastic" in properties:
                    steps["Elastic"] = ElasticMatrixCalculator(
                        eps_range=0.005
                    )
                if "Phonons" in properties:
                    steps["Phonons"] = PhonopyCalculator()

                if "Vacancy" in properties and struct_name in vacancy_prototypes:
                    steps["Vacancy"] = DefectFormationCalculator()

                if (
                    "TransformationPath" in properties
                    and struct_name in transformation_path_prototypes
                ):
                    steps["TransformationPath_tetragonal"] = (
                        TransformationPathCalculator(transformation_type="tetragonal")
                    )
                    steps["TransformationPath_trigonal"] = (
                        TransformationPathCalculator(transformation_type="trigonal")
                    )
                    steps["TransformationPath_hexagonal"] = (
                        TransformationPathCalculator(transformation_type="hexagonal")
                    )
                    steps["TransformationPath_orthogonal"] = (
                        TransformationPathCalculator(transformation_type="orthogonal")
                    )

                pipeline = Pipeline(steps=steps)
                struct_dict["pipeline"] = pipeline

                atoms.calc = ase_calculator
                pipeline.run(init_structure=atoms, engine=ase_calculator, verbose=True)

            else:
                print("Pipeline for this structure is already computed")

        computed_properties_data = []
        for struct_name, struct_dict in pipeline_structures_dict.items():
            atoms = struct_dict["atoms"]
            for step_name, step in struct_dict["pipeline"].steps.items():
                if step.status == "finished":
                    computed_properties_data.append(
                        [
                            struct_name,
                            atoms.GENERICPARENT.STRUKTURBERICHT,
                            step_name,
                            struct_dict["colour"],
                            step.job._value,
                            len(atoms),
                            "Potential",
                        ]
                    )

        self.comp_prop_df = pd.DataFrame(
            computed_properties_data,
            columns=["prototype", "SB", "job", "color", "VALUE", "n_at", "CALCULATOR"],
        )
        return self.comp_prop_df

    def validate(self, ase_calc, element):
        ref_calcs = self.get_reference_calculators()
