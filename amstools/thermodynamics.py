from collections import Counter
import logging
from typing import Optional, Dict

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from ase import Atoms

from amstools.validation.data import SINGLE_ATOM_ENERGY_DICT

E_FORMATION_PER_ATOM = "e_formation_per_atom"
E_CHULL_DIST_PER_ATOM = "e_chull_dist_per_atom"


def run_convex_hull_calculation(
    structure_dict: Dict[str, Atoms],
    calc,
    pipeline_dict: Optional[Dict] = None,
    df: Optional[pd.DataFrame] = None,
    optimizer_kwargs: Optional[Dict] = None,
    murnaghan_kwargs: Optional[Dict] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Process a dictionary of structures with StepwiseOptimizer + MurnaghanCalculator
    pipelines and compute convex hull distances.

    This function loops over structure_dict and:
    - If a structure has no corresponding pipeline in pipeline_dict, creates and runs one
    - If a pipeline exists but is not finished, runs it to completion
    - If a pipeline is already finished, skips it
    
    After processing all structures, collects results into a DataFrame and computes
    convex hull distances using compute_convexhull_dist.

    Args:
        structure_dict: Dictionary mapping structure names to ASE Atoms objects.
        calc: Calculator/engine to use for pipeline calculations.
        pipeline_dict: Optional dictionary of existing pipelines (modified in-place).
            If None, an empty dict is created internally.
        df: Optional existing DataFrame to append results to.
            If None, a new DataFrame is created.
        optimizer_kwargs: Optional kwargs for StepwiseOptimizer. Defaults to {}.
        murnaghan_kwargs: Optional kwargs for MurnaghanCalculator. Defaults to {}.
        verbose: Whether to print verbose output. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame with columns including 'ase_atoms', 'energy_per_atom',
            and convex hull distance information.
    """
    # Lazy imports to avoid circular dependencies
    from amstools.properties.relaxation import StepwiseOptimizer
    from amstools.properties.murnaghan import MurnaghanCalculator

    pipeline_dict =  pipeline_dict or {}
    optimizer_kwargs = optimizer_kwargs or {}
    murnaghan_kwargs =  murnaghan_kwargs or {}

    # Process each structure
    for name, atoms in structure_dict.items():
        if name not in pipeline_dict:
            # Create and run new pipeline
            try:
                logging.info(f"Processing structure '{name}'")
                pipeline = StepwiseOptimizer(**optimizer_kwargs) + MurnaghanCalculator(
                    **murnaghan_kwargs
                )
                pipeline.run(init_structure=atoms, engine=calc, verbose=verbose)
                pipeline_dict[name] = pipeline
            except Exception as e:
                logging.error(f"Failed to process structure '{name}': {e}")
                continue
        else:
            # Pipeline exists - check if finished
            pipeline = pipeline_dict[name]
            if not pipeline.is_finished():
                try:
                    logging.info(f"Pipeline for '{name}' is not finished, running...")
                    pipeline.run(verbose=verbose)
                except Exception as e:
                    logging.error(f"Failed to complete pipeline for '{name}': {e}")
                    continue
            else:
                logging.info(f"Pipeline for '{name}' is already finished")

    # Collect results to DataFrame
    data = []
    for name, pipeline in pipeline_dict.items():
        if not pipeline.is_finished():
            logging.info(f"Pipeline for '{name}' is not finished, skipping...")
            continue

        murnaghan_step = pipeline["murnaghan"]
        murnaghan_value = murnaghan_step.value

        # Get equilibrium energy and structure
        equilibrium_energy = murnaghan_value.get("equilibrium_energy")
        if equilibrium_energy is None:
            logging.warning(f"No equilibrium_energy found for '{name}', skipping")
            continue

        # Get optimized structure from murnaghan (equilibrium volume structure)
        final_structure = murnaghan_step.load_final_structure()
        if final_structure is None:
            final_structure = murnaghan_step.get_final_structure()

        n_atoms = len(final_structure) if final_structure is not None else 1
        energy_per_atom = equilibrium_energy / n_atoms

        data.append(
            {
                "name": name,
                "ase_atoms": final_structure,
                "energy": equilibrium_energy,
                "energy_per_atom": energy_per_atom,
                "equilibrium_volume": murnaghan_value.get("equilibrium_volume"),
                "equilibrium_bulk_modulus": murnaghan_value.get(
                    "equilibrium_bulk_modulus"
                ),
            }
        )

    new_df = pd.DataFrame(data)

    # Combine with existing DataFrame if provided
    if df is not None and not df.empty:
        result_df = pd.concat([df, new_df], ignore_index=True)
    else:
        result_df = new_df

    # Compute convex hull distances
    if not result_df.empty:
        compute_convexhull_dist(result_df, verbose=verbose)

    return result_df, pipeline_dict


def ensure_energy_per_atom_column(
    df,
    energy_per_atom_column="energy_per_atom",
    energy_column="energy",
    n_atoms_column="NUMBER_OF_ATOMS",
    ase_atoms_column="ase_atoms",
):
    """
    Ensures that the energy_per_atom column exists in the DataFrame.
    If it doesn't exist, it computes it from the energy and number of atoms columns.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        energy_per_atom_column (str, optional): The name of the column for energy per atom.
            Defaults to "energy_per_atom".
        energy_column (str, optional): The name of the column for total energy.
            Defaults to "energy".
        n_atoms_column (str, optional): The name of the column for the number of atoms.
            Defaults to "NUMBER_OF_ATOMS".
    """
    if energy_per_atom_column not in df.columns:

        if energy_column not in df.columns:
            raise ValueError(
                f"Cannot compute '{energy_per_atom_column}'. Missing required columns: '{energy_column}'."
            )

        if n_atoms_column not in df.columns:
            if ase_atoms_column not in df.columns:
                raise ValueError(
                    f"Cannot compute '{energy_per_atom_column}'. Missing required columns: '{n_atoms_column}' or '{ase_atoms_column}'."
                )
            else:
                df["NUMBER_OF_ATOMS"] = df[ase_atoms_column].map(len)

        df[energy_per_atom_column] = df[energy_column] / df[n_atoms_column]


def compdict_to_comptuple(comp_dict):
    """Converts a composition dictionary to a composition tuple.

    Args:
        comp_dict (dict): A dictionary representing the composition of a material,
            with element symbols as keys and their counts as values.

    Returns:
        tuple: A sorted tuple of (element, fraction) pairs.
    """
    n_atoms = sum([v for v in comp_dict.values()])
    return tuple(sorted([(k, v / n_atoms) for k, v in comp_dict.items()]))


def comptuple_to_str(comp_tuple):
    """Converts a composition tuple to a string.

    Args:
        comp_tuple (tuple): A tuple of (element, fraction) pairs.

    Returns:
        str: A string representation of the composition.
    """
    return " ".join(("{}_{:.3f}".format(e, c) for e, c in comp_tuple))


def compute_compositions(
    df, ase_atoms_column="ase_atoms", compute_composition_tuples=True
):
    """Generate new columns in a DataFrame with composition information.

    This function modifies the DataFrame in-place, adding the following columns:
    - 'comp_dict': A dictionary representing the composition.
    - 'NUMBER_OF_ATOMS': The total number of atoms in the structure.
    - 'comp_tuple': A sorted tuple of (element, fraction) pairs (optional).
    - 'n_ELEMENT': The number of atoms of each element.
    - 'c_ELEMENT': The concentration of each element.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        ase_atoms_column (str, optional): The name of the column containing ASE Atoms objects.
            Defaults to "ase_atoms".
        compute_composition_tuples (bool, optional): Whether to compute the 'comp_tuple' column.
            Defaults to True.

    Returns:
        list: A sorted list of unique elements present in the DataFrame.
    """
    df["comp_dict"] = df[ase_atoms_column].map(
        lambda atoms: Counter(atoms.get_chemical_symbols())
    )
    df["NUMBER_OF_ATOMS"] = df[ase_atoms_column].map(len)

    if compute_composition_tuples:
        df["comp_tuple"] = df["comp_dict"].map(compdict_to_comptuple)

    elements = extract_elements(df)

    for el in elements:
        df["n_" + el] = df["comp_dict"].map(lambda d: d.get(el, 0))
        df["c_" + el] = df["n_" + el] / df["NUMBER_OF_ATOMS"]

    return elements


def extract_elements(df, composition_dict_column="comp_dict"):
    """Extracts a sorted list of unique elements from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        composition_dict_column (str, optional): The name of the column containing
            composition dictionaries. Defaults to "comp_dict".

    Returns:
        list: A sorted list of unique element symbols.
    """
    elements_set = set()
    for cd in df[composition_dict_column]:
        elements_set.update(cd.keys())
    elements = sorted(elements_set)
    return elements


def compute_formation_energy(
    df,
    elements=None,
    epa_gs_dict=None,
    energy_per_atom_column="energy_per_atom",
    verbose=True,
):
    """Computes the formation energy per atom for each entry in a DataFrame.

    This function modifies the DataFrame in-place, adding the 'e_formation_per_atom' column.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        elements (list, optional): A list of element symbols. If None, they are
            extracted from the DataFrame. Defaults to None.
        epa_gs_dict (dict, optional): A dictionary mapping element symbols to their
            ground state energy per atom. If None, it is computed from the DataFrame.
            Defaults to None.
        energy_per_atom_column (str, optional): The name of the column containing the
            energy per atom. Defaults to "energy_per_atom".
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
    """
    if elements is None:
        elements = extract_elements(df)

    ensure_energy_per_atom_column(df, energy_per_atom_column=energy_per_atom_column)

    c_elements = ["c_" + el for el in elements]

    if epa_gs_dict is None:
        epa_gs_dict = {}
        for el in elements:
            subdf = df[df["c_" + el] == 1.0]
            if len(subdf) > 0:
                e_min_pa = subdf[energy_per_atom_column].min()
            else:
                e_min_pa = 0.0
                if verbose:
                    print(
                        "No pure element energy for {} is available, assuming 0  eV/atom".format(
                            el
                        )
                    )
            epa_gs_dict[el] = e_min_pa
    element_emin_array = np.array([epa_gs_dict[el] for el in elements])
    c_conc = df[c_elements].values
    e_formation_ideal = np.dot(c_conc, element_emin_array)
    df[E_FORMATION_PER_ATOM] = df[energy_per_atom_column] - e_formation_ideal


# TODO: write tests
def compute_convexhull_dist(
    df,
    ase_atoms_column="ase_atoms",
    energy_per_atom_column="energy_per_atom",
    verbose=True,
):
    """Computes the distance to the convex hull for each entry in a DataFrame.

    This function modifies the DataFrame in-place, adding the 'e_chull_dist_per_atom' column.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        ase_atoms_column (str, optional): The name of the column containing ASE Atoms objects.
            Defaults to "ase_atoms".
        energy_per_atom_column (str, optional): The name of the column containing the
            energy per atom. Defaults to "energy_per_atom".
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        list: A sorted list of unique elements present in the DataFrame.
    """
    elements = compute_compositions(df, ase_atoms_column=ase_atoms_column)
    ensure_energy_per_atom_column(df, energy_per_atom_column=energy_per_atom_column)
    c_elements = ["c_" + el for el in elements]

    compute_formation_energy(
        df, elements, energy_per_atom_column=energy_per_atom_column, verbose=verbose
    )

    # check if more than one unique compositions
    uniq_compositions = df["comp_tuple"].unique()
    # df.drop(columns=["comp_tuple"], inplace=True)

    if len(uniq_compositions) > 1:
        if verbose:
            print(
                "Structure dataset: multiple unique compositions found, trying to construct convex hull"
            )
        chull_values = df[c_elements[:-1] + [E_FORMATION_PER_ATOM]].values
        hull = ConvexHull(chull_values)
        ok = hull.equations[:, -2] < 0
        selected_simplices = hull.simplices[ok]
        selected_equations = hull.equations[ok]

        norms = selected_equations[:, :-1]
        offsets = selected_equations[:, -1]

        norms_c = norms[:, :-1]
        norms_e = norms[:, -1]

        e_chull_dist_list = []
        for p in chull_values:
            p_c = p[:-1]
            p_e = p[-1]
            e_simplex_projections = []
            for nc, ne, b, simplex in zip(
                norms_c, norms_e, offsets, selected_simplices
            ):
                if ne != 0:
                    e_simplex = (-b - np.dot(nc, p_c)) / ne
                    e_simplex_projections.append(e_simplex)
                elif (
                    np.abs(b + np.dot(nc, p_c)) < 2e-15
                ):  # ne*e_simplex + b + np.dot(nc,p_c), ne==0
                    e_simplex = p_e
                    e_simplex_projections.append(e_simplex)

            e_simplex_projections = np.array(e_simplex_projections)

            mask = e_simplex_projections < p_e + 1e-15

            e_simplex_projections = e_simplex_projections[mask]

            e_dist_to_chull = np.min(p_e - e_simplex_projections)

            e_chull_dist_list.append(e_dist_to_chull)

        e_chull_dist_list = np.array(e_chull_dist_list)
    else:
        if verbose:
            print(
                "Structure dataset: only single unique composition found, switching to cohesive energy reference"
            )
        emin = df[energy_per_atom_column].min()
        e_chull_dist_list = df[energy_per_atom_column] - emin

    df[E_CHULL_DIST_PER_ATOM] = e_chull_dist_list
    return elements


def compute_corrected_energy(
    df,
    esa_dict=None,
    calculator_name="VASP_PBE_500_0.125_0.1_NM",
    n_atoms_column="NUMBER_OF_ATOMS",
):
    """Computes the corrected energy for each entry in a DataFrame.

    This function modifies the DataFrame in-place, adding the 'energy_corrected' and
    'energy_corrected_per_atom' columns.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        esa_dict (dict, optional): A dictionary mapping element symbols to their
            elemental single atom energy. If None, it is loaded from
            SINGLE_ATOM_ENERGY_DICT. Defaults to None.
        calculator_name (str, optional): The name of the calculator to use for
            loading the esa_dict. Defaults to "VASP_PBE_500_0.125_0.1_NM".
        n_atoms_column (str, optional): The name of the column containing the number
            of atoms. Defaults to "NUMBER_OF_ATOMS".
    """
    if esa_dict is None:
        esa_dict = SINGLE_ATOM_ENERGY_DICT[calculator_name]
    elements = compute_compositions(df)
    n_elements = ["n_" + e for e in elements]
    esa_array = np.array([esa_dict[e] for e in elements])
    df["energy_corrected"] = df["energy"] - (df[n_elements] * esa_array).sum(axis=1)
    df["energy_corrected_per_atom"] = df["energy_corrected"] / df[n_atoms_column]


def plot_convex_hull(
    df_or_dict,
    chull_threshold=1e-3,
    plot_all_structures=False,
    style_dict=None,
    figsize=(8, 6),
    ase_atoms_column="ase_atoms",
    energy_per_atom_column="energy_per_atom",
):
    """
    Plots the convex hull for binary systems.

    Args:
        df_or_dict (pd.DataFrame or dict): A single DataFrame or a dictionary of
            DataFrames (name -> df) to plot.
        chull_threshold (float, optional): Threshold for identifying convex hull points.
            Defaults to 1e-3.
        plot_all_structures (bool, optional): Whether to plot projections of all structures.
            Defaults to False.
        style_dict (dict, optional): Style settings for each DataFrame.
            Format: {name: {color, linestyle, marker, label, ...}}.
        figsize (tuple, optional): Figure size. Defaults to (8, 6).
        ase_atoms_column (str, optional): Name of the ASE atoms column. Defaults to "ase_atoms".
        energy_per_atom_column (str, optional): Name of the energy per atom column.
            Defaults to "energy_per_atom".

    Returns:
        dict: A dictionary mapping element pairs (tuple) to matching matplotlib axes.
    """
    import matplotlib.pyplot as plt
    from itertools import combinations

    if isinstance(df_or_dict, pd.DataFrame):
        dfs = {"dataset": df_or_dict}
    else:
        dfs = df_or_dict

    style_dict = style_dict or {}

    # Extract all elements from all DataFrames
    all_elements = set()
    for df in dfs.values():
        compute_compositions(df, ase_atoms_column=ase_atoms_column)
        all_elements.update(extract_elements(df))
    elements = sorted(list(all_elements))

    # Generate binary pairs
    pairs = list(combinations(elements, 2))
    if not pairs:
        # Single element case
        if len(elements) == 1:
            logging.warning("Only one element found, cannot plot binary convex hull.")
        return {}

    axes_dict = {}

    for el1, el2 in pairs:
        fig, ax = plt.subplots(figsize=figsize)
        axes_dict[(el1, el2)] = ax

        for name, df in dfs.items():
            # Get style
            style = style_dict.get(name, {}).copy()
            color = style.pop("color", None)
            marker = style.pop("marker", "o")
            linestyle = style.pop("linestyle", "--")
            label = style.pop("label", name)

            # Columns for elements
            c1 = "c_" + el1
            c2 = "c_" + el2

            # All structures projection if requested
            if plot_all_structures:
                # Structures containing at least one of el1 or el2
                mask_all = (df[c1] + df[c2]) > 0
                df_all = df[mask_all].copy()
                if not df_all.empty:
                    # Projection: x = c2 / (c1 + c2)
                    x = df_all[c2] / (df_all[c1] + df_all[c2])
                    y = df_all[E_FORMATION_PER_ATOM]
                    ax.scatter(
                        x,
                        y,
                        marker=marker,
                        color=color,
                        alpha=0.2,
                        label=f"{label} (all)" if label else None,
                        **style,
                    )

            # Pure binary structures for the hull line
            # c1 + c2 == 1 (within tolerance)
            mask_binary = (df[c1] + df[c2]).between(1 - 1e-6, 1 + 1e-6)
            df_binary = df[mask_binary].copy()

            if not df_binary.empty:
                # Plot all scatter points in binary
                x_bin = df_binary[c2]
                y_bin = df_binary[E_FORMATION_PER_ATOM]
                scatter = ax.scatter(
                    x_bin,
                    y_bin,
                    marker=marker,
                    color=color,
                    label=label if not plot_all_structures else None,
                    **style,
                )
                if color is None:
                    color = scatter.get_facecolor()[0]

                # Identify and plot Hull points
                if E_CHULL_DIST_PER_ATOM in df_binary.columns:
                    df_hull = df_binary[
                        df_binary[E_CHULL_DIST_PER_ATOM] < chull_threshold
                    ].copy()
                else:
                    # Fallback if distance not computed: assume all points for drawing if single
                    df_hull = df_binary.copy()

                if not df_hull.empty:
                    df_hull = df_hull.sort_values(by=c2)
                    ax.plot(
                        df_hull[c2],
                        df_hull[E_FORMATION_PER_ATOM],
                        color=color,
                        linestyle=linestyle,
                        **style,
                    )

        ax.set_xlabel(f"Concentration of {el2}")
        ax.set_ylabel("Formation energy (eV/atom)")
        ax.set_title(f"Convex Hull: {el1}-{el2}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    return axes_dict
