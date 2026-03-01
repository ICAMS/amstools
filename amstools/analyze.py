import sys
from collections import defaultdict
from itertools import combinations_with_replacement

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ase

from ase.io import read
from matscipy.neighbours import neighbour_list
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# --- Configuration ---
# User-configurable parameter for the resolution of the 3D density grid.
BINS_PER_DIMENSION = 100

ASE_FORMAT = "lammps-data"  # ASE format for reading the trajectory
COLORMAP = "viridis"  # Recommended colormaps: 'viridis', 'plasma', 'inferno', 'magma'


def load_md_trajectory(filename, file_format):
    """
    Loads a full MD trajectory from a file using ASE.

    Args:
        filename (str): Path to the trajectory file.
        file_format (str): The format of the file for ASE to use.

    Returns:
        list: A list of ASE Atoms objects, one for each frame.
              Returns None if the file cannot be loaded.
    """
    try:
        # The index=':' argument is crucial for loading all frames.
        trajectory = read(filename, format=file_format, index=":")
        return trajectory
    except FileNotFoundError:
        print(f"-> Error: The file '{filename}' was not found.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"-> An error occurred while reading the file: {e}", file=sys.stderr)
        return None


def compute_3d_density_grid(atoms_frame, bins_per_dim):
    """
    Calculates a 3D histogram of atomic positions for a given ASE Atoms object.

    Args:
        atoms_frame (ase.Atoms): The ASE Atoms object for a single frame.
        bins_per_dim (int): The number of bins for histogramming each dimension.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The 3D grid representing atomic number density.
            - tuple: The box dimensions (Lx, Ly, Lz).
    """
    # Get atomic positions
    positions = atoms_frame.get_positions()

    # Get simulation box dimensions (assuming an orthogonal cell)
    cell_lengths = atoms_frame.get_cell_lengths_and_angles()[:3]
    Lx, Ly, Lz = cell_lengths

    # Define the bins and range for the 3D histogram
    bins = [bins_per_dim, bins_per_dim, bins_per_dim]
    hist_range = [[0, Lx], [0, Ly], [0, Lz]]

    # Calculate the 3D histogram. The result is a 3D grid of atom counts.
    density_grid_3d, _ = np.histogramdd(positions, bins=bins, range=hist_range)
    return density_grid_3d, (Lx, Ly, Lz)


def plot_density_projections(atoms_frame, bins_per_dim):
    """
    Computes and visualizes the 2D density projections for a single trajectory frame.

    Args:
        atoms_frame (ase.Atoms): The ASE Atoms object for the frame to analyze.
        bins_per_dim (int): The number of bins for histogramming each dimension.
    """
    # Step 2: Calculating the 3D density grid.
    density_grid_3d, (Lx, Ly, Lz) = compute_3d_density_grid(atoms_frame, bins_per_dim)

    # Step 3: Creating 2D Density Projections
    # XY Projection: Sum the 3D grid along the Z-axis (axis=2)
    density_xy = np.sum(density_grid_3d, axis=2)
    # YZ Projection: Sum the 3D grid along the X-axis (axis=0)
    density_yz = np.sum(density_grid_3d, axis=0)
    # XZ Projection: Sum the 3D grid along the Y-axis (axis=1)
    density_xz = np.sum(density_grid_3d, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("2D Projections of Atomic Number Density", fontsize=16)

    # Plot 1: XY Plane Projection
    im1 = axes[0].imshow(
        density_xy.T,
        cmap=COLORMAP,
        origin="lower",
        extent=[0, Lx, 0, Ly],
        aspect="equal",
    )
    axes[0].set_title("XY Plane Projection")
    axes[0].set_xlabel("X (Å)")
    axes[0].set_ylabel("Y (Å)")
    fig.colorbar(im1, ax=axes[0], label="Atom Count")

    # Plot 2: YZ Plane Projection
    im2 = axes[1].imshow(
        density_yz.T,
        cmap=COLORMAP,
        origin="lower",
        extent=[0, Ly, 0, Lz],
        aspect="equal",
    )
    axes[1].set_title("YZ Plane Projection")
    axes[1].set_xlabel("Y (Å)")
    axes[1].set_ylabel("Z (Å)")
    fig.colorbar(im2, ax=axes[1], label="Atom Count")

    # Plot 3: XZ Plane Projection
    # Note the transpose (.T) is used to orient the data correctly for imshow
    # such that the axes correspond to the labels.
    im3 = axes[2].imshow(
        density_xz.T,
        cmap=COLORMAP,
        origin="lower",
        extent=[0, Lx, 0, Lz],
        aspect="equal",
    )
    axes[2].set_title("XZ Plane Projection")
    axes[2].set_xlabel("X (Å)")
    axes[2].set_ylabel("Z (Å)")
    fig.colorbar(im3, ax=axes[2], label="Atom Count")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle


def plot_density_projections_by_type(atoms_frame, bins_per_dim, element_types):
    """
    Filters atoms by type and visualizes their 2D density projections.

    Args:
        atoms_frame (ase.Atoms): The ASE Atoms object for the frame to analyze.
        bins_per_dim (int): The number of bins for histogramming.
        element_types (str or list): The chemical symbol(s) of the element(s) to include (e.g., 'Ar', ['Si', 'O']).
    """
    # Ensure element_types is a list for consistent processing
    if not isinstance(element_types, (list, tuple)):
        element_types = [element_types]

    # Get indices of atoms matching the specified chemical symbols.
    symbols = np.array(atoms_frame.get_chemical_symbols())
    mask = np.isin(symbols, element_types)
    indices = np.where(mask)[0]

    if len(indices) == 0:
        print(
            f"\nWarning: No atoms found for element types {element_types}.",
            file=sys.stderr,
        )
        return

    # Create a new Atoms object containing only the selected atoms
    filtered_atoms = atoms_frame[indices]

    # We pass the original atoms_frame to compute_3d_density_grid to get the
    # full box dimensions, but we only use the positions of the filtered atoms.
    # A cleaner way is to pass the filtered atoms and the original cell.

    # Get simulation box dimensions from the original, full frame
    cell_lengths = atoms_frame.get_cell_lengths_and_angles()[:3]
    Lx, Ly, Lz = cell_lengths

    # Get positions of only the filtered atoms
    positions = filtered_atoms.get_positions()

    # Define the bins and range for the 3D histogram using the full box size
    bins = [bins_per_dim, bins_per_dim, bins_per_dim]
    hist_range = [[0, Lx], [0, Ly], [0, Lz]]

    # Calculate the 3D histogram using only the filtered positions
    density_grid_3d, _ = np.histogramdd(positions, bins=bins, range=hist_range)

    # Create 2D projections
    density_xy = np.sum(density_grid_3d, axis=2)
    density_yz = np.sum(density_grid_3d, axis=0)
    density_xz = np.sum(density_grid_3d, axis=1)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        f"2D Density Projections for Atom Type(s) {element_types}", fontsize=16
    )

    # Plot 1: XY Plane
    im1 = axes[0].imshow(
        density_xy.T,
        cmap=COLORMAP,
        origin="lower",
        extent=[0, Lx, 0, Ly],
        aspect="equal",
    )
    axes[0].set_title("XY Plane Projection")
    axes[0].set_xlabel("X (Å)")
    axes[0].set_ylabel("Y (Å)")
    fig.colorbar(im1, ax=axes[0], label="Atom Count")

    # Plot 2: YZ Plane
    im2 = axes[1].imshow(
        density_yz.T,
        cmap=COLORMAP,
        origin="lower",
        extent=[0, Ly, 0, Lz],
        aspect="equal",
    )
    axes[1].set_title("YZ Plane Projection")
    axes[1].set_xlabel("Y (Å)")
    axes[1].set_ylabel("Z (Å)")
    fig.colorbar(im2, ax=axes[1], label="Atom Count")

    # Plot 3: XZ Plane
    im3 = axes[2].imshow(
        density_xz.T,
        cmap=COLORMAP,
        origin="lower",
        extent=[0, Lx, 0, Lz],
        aspect="equal",
    )
    axes[2].set_title("XZ Plane Projection")
    axes[2].set_xlabel("X (Å)")
    axes[2].set_ylabel("Z (Å)")
    fig.colorbar(im3, ax=axes[2], label="Atom Count")

    plt.tight_layout(rect=[0, 0, 1, 0.96])


from ase.data import atomic_numbers, covalent_radii


def estimate_bond_cutoffs(atoms_frame, tolerance=0.4):
    """
    Estimates bond cutoffs for element pairs based on Covalent Radii.

    Formula: Cutoff = r_cov(A) + r_cov(B) + tolerance

    Args:
        atoms_frame (ase.Atoms): The Atoms object containing the elements.
        tolerance (float): Buffer distance in Angstroms to add to the sum of radii
                           (default=0.4 Å).

    Returns:
        dict: A dictionary of cutoffs for every unique pair of elements found
              in the frame.
    """
    # Get unique chemical symbols in the frame
    unique_symbols = sorted(list(set(atoms_frame.get_chemical_symbols())))
    cutoffs = {}

    print(f"\nEstimating bond cutoffs (Covalent Radii + {tolerance} Å)...")

    for i, sym1 in enumerate(unique_symbols):
        for sym2 in unique_symbols[i:]:
            # Get atomic numbers
            z1 = atomic_numbers[sym1]
            z2 = atomic_numbers[sym2]

            # Get covalent radii (ase.data.covalent_radii is indexed by atomic number)
            r1 = covalent_radii[z1]
            r2 = covalent_radii[z2]

            # Calculate cutoff
            cutoff = r1 + r2 + tolerance

            # Create sorted key
            pair_key = tuple(sorted((sym1, sym2)))
            cutoffs[pair_key] = round(cutoff, 3)

            # print(f"  -> Pair {sym1}-{sym2}: r_cov({r1:.2f}) + r_cov({r2:.2f}) + tol({tolerance}) = {cutoff:.2f} Å")

    return cutoffs


def compute_bond_distances(atoms_frame, cutoffs):
    """
    Computes bond distances below specified cutoffs for a given ASE Atoms object.
    Useful for extracting the first coordination shell.

    Args:
        atoms_frame (ase.Atoms): The ASE Atoms object for the frame to analyze.
        cutoffs (float or dict):
            - If float: A global cutoff for all pairs.
            - If dict: A dictionary where keys are sorted tuples of element symbols
              (e.g., ('O', 'Si')) and values are float cutoffs for that specific pair.
              Pairs not in the dict will not be processed.

    Returns:
        dict: A dictionary where keys are sorted tuples of element symbol pairs
              and values are lists of corresponding bond distances.
    """
    if neighbour_list is None:
        print(
            "Error: 'matscipy' is not installed. Cannot compute bond distances.",
            file=sys.stderr,
        )
        return {}

    # Determine the maximum search radius
    if isinstance(cutoffs, dict):
        if not cutoffs:
            print("Warning: Empty cutoff dictionary provided.")
            return {}
        max_search_cutoff = max(cutoffs.values())
        print(
            f"\nComputing bonds with custom per-pair cutoffs (Max search: {max_search_cutoff} Å)..."
        )
    else:
        max_search_cutoff = float(cutoffs)
        print(f"\nComputing bonds with global cutoff of {max_search_cutoff} Å...")

    bond_data = defaultdict(list)
    symbols = np.array(atoms_frame.get_chemical_symbols())

    # Efficient search using the largest necessary cutoff
    i_list, j_list, d_list = neighbour_list(
        "ijd", atoms=atoms_frame, cutoff=max_search_cutoff
    )

    for i, j, d in zip(i_list, j_list, d_list):
        sym_i = symbols[i]
        sym_j = symbols[j]

        # Ensure consistent key ordering (alphabetical)
        if sym_i < sym_j:
            pair_key = (sym_i, sym_j)
        else:
            pair_key = (sym_j, sym_i)

        # Determine the specific limit for this pair
        limit = None
        if isinstance(cutoffs, dict):
            # Check if this pair is in the dictionary
            if pair_key in cutoffs:
                limit = cutoffs[pair_key]
        else:
            limit = cutoffs

        # Filter based on the limit
        if limit is not None and d <= limit:
            bond_data[pair_key].append(d)

    print("-> Bond distance calculation complete.")

    return {pair_key: np.array(v) for pair_key, v in bond_data.items()}


def plot_bond_count_heatmap(bond_distances):
    """
    Plots a 2D heatmap of bond counts between different element types
    using Pandas and Matplotlib.

    The color and upper triangle annotations show the number of bonds.
    The lower triangle annotations show the average bond distance in Angstroms.
    Elements are clustered based on bonding frequency to group related elements.

    Args:
        bond_distances (dict): A dictionary from compute_bond_distances, where
                               keys are tuples of element pairs and values are
                               lists of distances.
    """
    if not bond_distances:
        print("-> No bond data to plot.", file=sys.stderr)
        return

    # 1. Convert the bond dictionary to a DataFrame with counts and averages
    bond_list = []
    for pair, distances in bond_distances.items():
        count = len(distances)
        avg_dist = np.mean(distances)
        bond_list.append(
            {"A": pair[0], "B": pair[1], "count": count, "avg_dist": avg_dist}
        )
        # Add symmetric entry for easier pivoting
        if pair[0] != pair[1]:
            bond_list.append(
                {"A": pair[1], "B": pair[0], "count": count, "avg_dist": avg_dist}
            )

    df = pd.DataFrame(bond_list)

    # 2. Create the pivot tables for counts and average distances
    try:
        pivot_counts = df.pivot_table(index="A", columns="B", values="count").fillna(0)
        pivot_avg_dist = df.pivot_table(
            index="A", columns="B", values="avg_dist"
        ).fillna(0)
    except IndexError:
        print("-> Not enough diverse bond types to create a heatmap.", file=sys.stderr)
        return

    # Get initial alphabetical list of symbols
    all_symbols = sorted(list(set(df["A"].unique()) | set(df["B"].unique())))
    pivot_counts = pivot_counts.reindex(
        index=all_symbols, columns=all_symbols, fill_value=0
    )

    # 3. Perform clustering to reorder the elements based on bonding frequency
    ordered_symbols = all_symbols  # Default to alphabetical order
    if len(all_symbols) > 2:
        # Convert similarity (bond counts) to a distance matrix
        distance_matrix = pivot_counts.max().max() - pivot_counts

        # *** FIX: Set the diagonal to zero to satisfy squareform requirement ***
        np.fill_diagonal(distance_matrix.values, 0)

        # Scipy's linkage function needs a condensed distance matrix (1D array)
        condensed_distance = squareform(distance_matrix)

        # Perform hierarchical clustering
        Z = linkage(condensed_distance, method="average")

        # Get the order of elements from the clustering result
        dn = dendrogram(Z, no_plot=True)
        ordered_indices = dn["leaves"]

    # 4. Reindex the pivot tables according to the new clustered order
    pivot_counts = pivot_counts.reindex(index=ordered_symbols, columns=ordered_symbols)
    pivot_avg_dist = pivot_avg_dist.reindex(
        index=ordered_symbols, columns=ordered_symbols, fill_value=0
    )

    # 5. Plotting the heatmap using Matplotlib
    fig, ax = plt.subplots(figsize=(15, 15))
    # The color will represent the number of bonds.
    im = ax.imshow(pivot_counts.values, cmap="YlGnBu")

    # Create the colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of Bonds", rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(ordered_symbols)))
    ax.set_yticks(np.arange(len(ordered_symbols)))
    ax.set_xticklabels(ordered_symbols)
    ax.set_yticklabels(ordered_symbols)
    ax.set_xlabel("Element")
    ax.set_ylabel("Element")
    ax.set_title(
        "Bond Counts (Color/Upper Triangle) & Avg. Distances (Å, Lower Triangle)"
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    threshold = im.get_clim()[1] / 2.0
    for i in range(len(ordered_symbols)):
        for j in range(len(ordered_symbols)):
            # Determine text color based on background
            color = "w" if pivot_counts.iloc[i, j] > threshold else "k"

            # Upper triangle (and diagonal) shows counts
            if j >= i:
                text_val = f"{pivot_counts.iloc[i, j]:.0f}"
            # Lower triangle shows average distance
            else:
                text_val = f"{pivot_avg_dist.iloc[i, j]:.2f}"

            ax.text(j, i, text_val, ha="center", va="center", color=color)

    fig.tight_layout()


def compute_rdf(atoms_frame, element_pairs=None, max_range=10, nbins=50):
    """
    Computes the Radial Distribution Function (RDF), g(r), for one or more
    pairs of element types.

    Args:
        atoms_frame (ase.Atoms or list): A single ASE Atoms object or a list of
                                         Atoms objects (e.g., a trajectory).
                                         If a list is provided, the RDF is averaged.
        element_pairs (list, optional): A list of tuples, where each tuple contains two
                              element symbols (e.g., `[('Si', 'O'), ('O', 'O')]`). Defaults to None.
                              If None, all unique pairs from the structure(s) are used.
                              Can also be a list of element symbols
                              (e.g., `['Si', 'O']`), from which all unique
                              pairs will be automatically generated.
        max_range (float): The maximum distance (in Angstroms) to compute the RDF.
        nbins (int): The number of bins to use for the distance histogram.

    Returns:
        dict: A dictionary where keys are sorted element pair tuples (e.g., ('O', 'Si'))
              and values are a dictionary.
              If a single frame is provided, the dictionary contains:
                - 'r': bin centers (np.ndarray)
                - 'g_r': RDF values (np.ndarray)
              If multiple frames are provided, the dictionary contains:
                - 'r': bin centers (np.ndarray)
                - 'g_r': Mean RDF values across frames (np.ndarray)
                - 'g_r_std': Standard deviation of RDF values (np.ndarray)
              Returns an empty dict if no pairs are found.
    """
    # --- Handle single frame vs. trajectory (list of frames) ---
    if not isinstance(atoms_frame, list):
        frames = [atoms_frame]
    else:
        frames = atoms_frame

    if not frames:
        return {}

    if element_pairs is None:
        # If no pairs are specified, find all unique elements and create all possible pairs.
        all_symbols = set()
        for frame in frames:
            all_symbols.update(frame.get_chemical_symbols())
        unique_elements = sorted(list(all_symbols))
        pairs_to_compute = list(combinations_with_replacement(unique_elements, 2))
    elif element_pairs and isinstance(element_pairs[0], str):
        pairs_to_compute = list(combinations_with_replacement(element_pairs, 2))
    else:
        # Assume it's already a list of pairs
        pairs_to_compute = element_pairs

    if not pairs_to_compute:
        print("-> No element pairs provided for RDF computation.", file=sys.stderr)
        return {}

    # Determine unique elements for filtering from the requested pairs
    unique_elements = sorted(list(set(el for pair in pairs_to_compute for el in pair)))

    # --- Process all frames ---
    all_g_r_data = defaultdict(list)
    r_bins = None

    for i, frame in enumerate(frames):

        # --- Optimization: Filter atoms for the current frame ---
        original_symbols = np.array(frame.get_chemical_symbols())
        mask = np.isin(original_symbols, unique_elements)
        filtered_atoms = frame[mask]

        if len(filtered_atoms) == 0:
            continue

        volume = frame.get_volume()

        # Run neighbor list calculation on the smaller, filtered structure
        all_distances = defaultdict(list)
        filtered_symbols = np.array(filtered_atoms.get_chemical_symbols())
        i_list, j_list, d_list = neighbour_list(
            "ijd", atoms=filtered_atoms, cutoff=float(max_range)
        )
        for i_idx, j_idx, d in zip(i_list, j_list, d_list):
            pair_key = tuple(sorted((filtered_symbols[i_idx], filtered_symbols[j_idx])))
            all_distances[pair_key].append(d)

        for el1, el2 in pairs_to_compute:
            pair_key = tuple(sorted((el1, el2)))
            distances = all_distances.get(pair_key, [])

            if not distances:
                continue

            hist, bin_edges = np.histogram(
                distances, bins=nbins, range=(0.0, max_range)
            )

            if r_bins is None:
                r_bins = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                dr = bin_edges[1] - bin_edges[0]
                shell_volumes = 4.0 * np.pi * r_bins**2 * dr

            n1 = np.sum(original_symbols == el1)
            n2 = np.sum(original_symbols == el2)

            if n1 == 0 or n2 == 0:
                continue

            if el1 == el2:
                norm_factor = (n1 * (n1 - 1)) / (2.0 * volume)
            else:
                norm_factor = (n1 * n2) / volume

            g_r = np.zeros_like(shell_volumes)
            non_zero_mask = (shell_volumes > 1e-9) & (norm_factor > 1e-9)
            g_r[non_zero_mask] = (
                hist[non_zero_mask] / (shell_volumes[non_zero_mask] * norm_factor) / 2
            )

            all_g_r_data[pair_key].append(g_r)

    # --- Finalize data: Averaging and standard deviation ---
    final_rdf_data = {}
    if not all_g_r_data:
        print(
            "-> No valid RDF data could be computed from the frames.", file=sys.stderr
        )
        return {}

    for pair_key, g_r_list in all_g_r_data.items():
        if len(frames) > 1:
            mean_g_r = np.mean(g_r_list, axis=0)
            std_g_r = np.std(g_r_list, axis=0)
            final_rdf_data[pair_key] = {
                "r": r_bins,
                "g_r": mean_g_r,
                "g_r_std": std_g_r,
            }
        else:
            # Only one frame, no averaging needed
            final_rdf_data[pair_key] = {"r": r_bins, "g_r": g_r_list[0]}

    return final_rdf_data

    return rdf_data


def plot_rdf(rdf_data, std_factor=1.0):
    """Plots the pre-computed RDF data."""
    plt.figure(figsize=(8, 6))
    for pair_tuple, data in rdf_data.items():
        r = data["r"]
        g_r = data["g_r"]
        label = f"{pair_tuple[0]}-{pair_tuple[1]}"
        plt.plot(r, g_r, label=label)
        # If standard deviation is available, plot it as a shaded region
        if "g_r_std" in data:
            g_r_std = data["g_r_std"]
            lower_bound = np.maximum(0, g_r - std_factor * g_r_std)
            plt.fill_between(r, lower_bound, g_r + std_factor * g_r_std, alpha=0.2)

    plt.xlabel("Distance, r (Å)")
    plt.ylabel("Radial Distribution Function, g(r)")
    plt.title("Radial Distribution Function (RDF)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.axhline(1.0, color="grey", linestyle="--")
    plt.legend()


def compute_and_plot_rdf(
    atoms_frame, element_pairs=None, max_range=10, nbins=50, std_factor=1.0
):
    """
    Computes and plots the Radial Distribution Function (RDF), g(r), for
    one or more pairs of element types.

    Args:
        atoms_frame (ase.Atoms or list): A single ASE Atoms object or a list of
                                         frames to be analyzed.
        element_pairs (list): A list of element symbols or a list of tuples, where each tuple contains two
                              element symbols (e.g., [('Si', 'O'), ('O', 'O')]).
        max_range (float): The maximum distance (in Angstroms) to compute the RDF.
        nbins (int): The number of bins to use for the distance histogram.
    """
    # Step 1: Compute the RDF data
    rdf_data = compute_rdf(atoms_frame, element_pairs, max_range, nbins)

    # Step 2: Plot the computed data
    if rdf_data:
        plot_rdf(rdf_data, std_factor=std_factor)
    else:
        print("-> No RDF data was generated, skipping plot.", file=sys.stderr)


def compute_cowley_sro(atoms, cutoff, central_element=None):
    """
    Computes the Warren-Cowley SRO parameter using a Global Coordination Number
    to ensure symmetry: alpha_ij = alpha_ji.
    """
    n_atoms = len(atoms)
    if n_atoms == 0:
        return {}

    symbols = np.array(atoms.get_chemical_symbols())
    unique_symbols = sorted(list(set(symbols)))

    # 1. Calculate Concentrations (c_j)
    # Returns a dict: {'Au': 0.5, 'Cu': 0.5}
    counts = pd.Series(symbols).value_counts()
    concentrations = (counts / n_atoms).to_dict()

    # 2. Get Neighbor List
    # i_list: indices of central atoms
    # j_list: indices of neighbor atoms
    i_list, j_list = neighbour_list("ij", atoms=atoms, cutoff=cutoff)

    # 3. Calculate Global Average Coordination Number (Z_global)
    # Total bonds / Total atoms.
    # Note: len(i_list) counts every bond A-B twice (once for A, once for B)
    Z_global = len(i_list) / n_atoms

    if Z_global == 0:
        return {}

    # --- Optimization: Use Pandas to count pairs instantly ---
    # Create a DataFrame of pairs (Symbol_i, Symbol_j)
    df = pd.DataFrame({"source": symbols[i_list], "target": symbols[j_list]})

    # Count occurrences of every pair type (e.g., how many Cu-Au bonds?)
    # This replaces the slow loop.
    pair_counts = df.groupby(["source", "target"]).size().unstack(fill_value=0)

    # Ensure all elements exist in the matrix (even if count is 0)
    for sym in unique_symbols:
        if sym not in pair_counts.index:
            pair_counts.loc[sym] = 0
        if sym not in pair_counts.columns:
            pair_counts[sym] = 0

    # Reorder for consistency
    pair_counts = pair_counts.loc[unique_symbols, unique_symbols]

    sro_results = {}

    elements_to_process = [central_element] if central_element else unique_symbols

    for i_sym in elements_to_process:
        sro_results[i_sym] = {}

        # Number of atoms of type i
        N_i = counts.get(i_sym, 0)

        if N_i == 0:
            continue

        for j_sym in unique_symbols:
            # Total number of bonds between type i and type j
            total_bonds_ij = pair_counts.loc[i_sym, j_sym]

            # Average number of j-neighbors around an i-atom (N_ij)
            N_ij = total_bonds_ij / N_i

            c_j = concentrations[j_sym]

            # 4. The Formula using Z_global
            # alpha_ij = 1 - (N_ij / (c_j * Z_global))
            alpha_ij = 1 - (N_ij / (c_j * Z_global))

            sro_results[i_sym][j_sym] = float(alpha_ij)

    return sro_results


def plot_sro_heatmap(sro_data, max_abs_val=None):
    """
    Plots the Cowley SRO parameters as a heatmap.

    Args:
        sro_data (dict): The nested dictionary returned by compute_cowley_sro.
    """
    if not sro_data:
        print("-> No SRO data to plot.", file=sys.stderr)
        return

    # Convert the nested dict to a DataFrame for easy plotting
    df = pd.DataFrame.from_dict(sro_data, orient="index").fillna(0)

    # Ensure the columns and index are in the same sorted order
    all_symbols = sorted(list(set(df.index) | set(df.columns)))
    df = df.reindex(index=all_symbols, columns=all_symbols, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use a diverging colormap since SRO can be positive or negative
    # Center the colormap at 0
    max_abs_val = (
        max(abs(df.min().min()), abs(df.max().max()))
        if max_abs_val is None
        else max_abs_val
    )
    im = ax.imshow(df.values, cmap="coolwarm", vmin=-max_abs_val, vmax=max_abs_val)

    # Create colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cowley SRO Parameter (alpha_ij)", rotation=-90, va="bottom")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(df.columns)))
    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticklabels(df.columns)
    ax.set_yticklabels(df.index)
    ax.set_xlabel("Neighbor Atom Type (j)")
    ax.set_ylabel("Central Atom Type (i)")
    ax.set_title("Warren-Cowley Short-Range Order (SRO)")

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.iloc[i, j]
            # Determine text color based on background
            color = "w" if abs(val) > max_abs_val / 2 else "k"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color)

    fig.tight_layout()


def atoms_to_dataframe(atoms_frame):
    """
    Converts an ASE Atoms object into a pandas DataFrame.

    The DataFrame will contain the chemical symbol and Cartesian coordinates
    for each atom.

    Args:
        atoms_frame (ase.Atoms): The ASE Atoms object to convert.

    Returns:
        pandas.DataFrame: A DataFrame with columns ['symbol', 'x', 'y', 'z'].
    """
    if atoms_frame is None:
        print("-> Input is not a valid ASE Atoms object.", file=sys.stderr)
        return None

    symbols = atoms_frame.get_chemical_symbols()
    positions = atoms_frame.get_positions()

    df = pd.DataFrame(positions, columns=["x", "y", "z"])
    df.insert(0, "symbol", symbols)

    return df


def plot_coordinate_histogram(atoms_df, axis="z", bins=100, elements=None):
    """
    Computes and plots a histogram of atomic coordinates along a specified axis.

    If `elements` are specified, it plots a separate, overlaid histogram for each
    element type. Otherwise, it plots a single histogram for all atoms.

    Args:
        atoms_df (pandas.DataFrame): DataFrame containing atomic coordinates,
                                     as created by `atoms_to_dataframe`.
        axis (str): The axis to plot the histogram for ('x', 'y', or 'z').
        bins (int): The number of bins for the histogram.
        elements (list or str, optional): A list of element symbols to plot.
                                          If None, all atoms are plotted.
                                          Defaults to None.
    """
    if atoms_df is None or atoms_df.empty:
        print("-> Input DataFrame is empty. Nothing to plot.", file=sys.stderr)
        return

    if axis not in list("xyz"):
        print(f"-> Invalid axis '{axis}'. Must be 'x', 'y', or 'z'.", file=sys.stderr)
        return

    plt.figure(figsize=(8, 6))

    if elements is None:
        plt.hist(atoms_df[axis], bins=bins, edgecolor="black", label="All Atoms")
        plt.title(f"Histogram of All Atomic Positions along the {axis}-axis")
    else:
        if isinstance(elements, str):
            elements = [elements]
        for elem in elements:
            df_elem = atoms_df[atoms_df["symbol"] == elem]
            plt.hist(df_elem[axis], bins=bins, alpha=0.7, label=elem)
        plt.title(
            f'Histogram of Atomic Positions for {", ".join(elements)} along the {axis}-axis'
        )
        plt.legend()

    plt.xlabel(f"Coordinate on {axis}-axis (Å)")
    plt.ylabel("Number of Atoms")
    plt.grid(True, linestyle="--", alpha=0.6)


def compute_atomic_distribution(
    atoms_df, axis="z", elements=None, bins=100, concentration=False
):
    """
    Computes the atomic distribution (histogram data) along a specified axis.

    This function groups atoms into bins and returns the counts and bin centers,
    which can then be used for plotting or further analysis.

    Args:
        atoms_df (pandas.DataFrame): DataFrame containing atomic coordinates,
                                     as created by `atoms_to_dataframe`.
        axis (str): The axis for which to compute the distribution ('x', 'y', or 'z'). Defaults to 'z'.
        elements (list or str, optional): A list of element symbols to plot.
                                          If None, all atoms are plotted together.
                                          If a list is provided, data for each element is computed.
        bins (int): The number of bins to use for the histogram. Defaults to 100.
        concentration (bool): If True, compute concentration in each bin instead of raw counts.
                              Defaults to False.

    Returns:
        dict: A dictionary where keys are element labels ('All Atoms' or element symbols)
              and values are another dictionary containing 'values' (counts or concentration), 'bin_centers',
              and 'bin_width'. Returns an empty dictionary if input is invalid.
    """
    if isinstance(atoms_df, ase.Atoms):
        atoms_df = atoms_to_dataframe(atoms_df)
    if atoms_df is None or atoms_df.empty:
        print("-> Input DataFrame is empty. Nothing to compute.", file=sys.stderr)
        return {}

    if axis not in ["x", "y", "z"]:
        print(f"-> Invalid axis '{axis}'. Must be 'x', 'y', or 'z'.", file=sys.stderr)
        return {}

    target_elements = []
    if elements is None:
        target_elements.append(("All Atoms", atoms_df))
    elif elements == "all":
        unique_elements = sorted(atoms_df["symbol"].unique())
        elements = unique_elements
    else:
        if isinstance(elements, str):
            elements = [elements]
        for elem in elements:
            df_elem = atoms_df[atoms_df["symbol"] == elem]
            if not df_elem.empty:
                target_elements.append((elem, df_elem))
            else:
                print(
                    f"-> Warning: No atoms of type '{elem}' found in DataFrame.",
                    file=sys.stderr,
                )

    # This block is now outside the 'else' to handle the case where elements="all"
    if isinstance(elements, list):
        for elem in elements:
            df_elem = atoms_df[atoms_df["symbol"] == elem]
            if not df_elem.empty:
                target_elements.append((elem, df_elem))
            else:
                print(
                    f"-> Warning: No atoms of type '{elem}' found in DataFrame.",
                    file=sys.stderr,
                )

    distribution_data = {}

    # First, get the total counts per bin for all atoms, needed for concentration
    total_counts, bin_edges = np.histogram(atoms_df[axis], bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # To avoid division by zero, replace zeros in total_counts with a small number or handle it later
    total_counts_for_division = np.where(total_counts == 0, 1, total_counts)

    for label, df_to_plot in target_elements:
        # Use the same bin_edges for all histograms to ensure alignment
        counts, _ = np.histogram(df_to_plot[axis], bins=bin_edges)

        if concentration:
            values = np.divide(
                counts, total_counts_for_division, where=total_counts != 0
            )
        else:
            values = counts

        distribution_data[label] = {
            "values": values,
            "bin_centers": bin_centers,
            "bin_width": bin_width,
        }

    return distribution_data


def plot_atomic_distribution(distribution_data, axis="z", ax=None):
    """
    Plots a pre-computed histogram of atomic distribution.

    Args:
        distribution_data (dict): The data computed by `compute_atomic_distribution`.
        axis (str): The axis label for the plot ('x', 'y', or 'z'). Defaults to 'z'.
        ax (matplotlib.axes.Axes, optional): An existing Axes object to plot on.
                                             If None, a new figure and axes are created.
    """
    if not distribution_data:
        print("-> No distribution data provided to plot.", file=sys.stderr)
        return

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    for label, data in distribution_data.items():
        ax.bar(
            data["bin_centers"],
            data["values"],
            width=data["bin_width"],
            alpha=0.7,
            label=label,
        )

    ax.set_title(f"Atomic Distribution along the {axis}-axis")
    ax.set_xlabel(f"Coordinate on {axis}-axis (Å)")

    # Check if the data is likely concentration or count to set the y-label
    is_concentration = any(
        np.any((data["values"] > 0) & (data["values"] < 1))
        for data in distribution_data.values()
    )
    ylabel = "Concentration" if is_concentration else "Element Count"
    ax.set_ylabel(ylabel)

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
