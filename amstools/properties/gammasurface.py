import os
from itertools import product
import numpy as np
from amstools.calculators.dft.base import AMSDFTBaseCalculator

from ase.optimize import (
    FIRE,
    BFGS,
    BFGSLineSearch,
    LBFGS,
    LBFGSLineSearch,
    GPMin,
    MDMin,
)
from ase.constraints import FixedLine
from amstools.properties.generalcalculator import GeneralCalculator

# eV/A2 to mJ/m2 to conversion factor
eV_A2_to_mJ_M2 = 1.602176565e-19 * 1e20 * 1e3


def apply_constraint(atoms):
    """
    Apply constraint: atom must relax along z
    :param atoms: ASE Atoms object
    """
    constraints = []
    for i in [atom.index for atom in atoms]:
        constraints.append(FixedLine(i, (0, 0, 1)))
    atoms.set_constraint(constraints)


def create_optimizer(optimizer, structure, logfile):
    """Create an optimizer
    :param optimizer: optimizer name
    :param structure: ASE Atoms object
    :param logfile: log file
    :return: ASE optimizer
    """
    if optimizer == "FIRE":
        relax = FIRE(structure, logfile=logfile)
    elif optimizer == "BFGS":
        relax = BFGS(structure, logfile=logfile)
    elif optimizer == "BFGSLineSearch":
        relax = BFGSLineSearch(structure, logfile=logfile)
    elif optimizer == "LBFGS":
        relax = LBFGS(structure, logfile=logfile)
    elif optimizer == "LBFGSLineSearch":
        relax = LBFGSLineSearch(structure, logfile=logfile)
    elif optimizer == "GPMin":
        relax = GPMin(structure, logfile=logfile)
    elif optimizer == "MDMin":
        relax = MDMin(structure, logfile=logfile)
    else:
        raise ValueError("Optimizer {} is not supported".format(optimizer))
    return relax


class GammaSurfaceCalculator(GeneralCalculator):
    """Calculation of a gamma line
    :param atoms: original ASE Atoms object to be used for calculation of gamma line energy
    :param shift_surface: default = [1,1] (shift of the surface in the X and Y directions)
    :param num_of_point_per_line: default = 10 (number of points in the gamma line)
    :param optimizer: default = FIRE (optimizer to be used for relaxation)
    :param fmax: default = 1e-2 (force criteria for optimizer)
    :param z_cut_level: default = 0.5, separation of upper and lower parts of atoms in Z direction, in relative units between 0 and 1
    :param auto_constraint: default = True, if True, the constraint to Z-only relaxation  will be set automatically

    Usage:
    >> block = FaceCenteredCubic(directions=[[-1, 1, 0], [1, 1, 2], [1, 1, -1]], size=(1, 1, 8),
                                       symbol="Al", pbc=(1, 1, 1), latticeconstant=4.05)
    >> block.calc = calc
    >> gamma_surface_calc = GammaSurfaceCalculator(block,
                                        shift_surface=[1, 1],
                                        num_of_point_per_line=10,
                                        fmax=1E-2,
                                        z_cut_level=0.5)
    >> gamma_surface_calc.calculate()
    >>
    >> gamma_surface_calc.plot()
    """

    property_name = "gammasurface"

    param_names = [
        "shift_surface",
        "num_of_point_per_line",
        "optimizer",
        "fmax",
        "z_cut_level",
    ]

    def __init__(
        self,
        atoms=None,
        shift_surface=(1, 1),
        num_of_point_per_line=10,
        optimizer="FIRE",
        fmax=1e-2,
        z_cut_level=0.5,
        auto_constraint=True,
        **kwargs,
    ):
        GeneralCalculator.__init__(self, atoms, **kwargs)
        self.shift_surface = shift_surface
        self.num_of_point_per_line = num_of_point_per_line
        self.optimizer = optimizer
        self.fmax = fmax
        self.z_cut_level = z_cut_level
        self.auto_constraint = auto_constraint
        self.real_shift_surface = [
            shift_surface[0] * np.linalg.norm(atoms.get_cell()[0][:2]),
            shift_surface[1] * np.linalg.norm(atoms.get_cell()[1][:2]),
        ]
        self._value = {
            "shift_surface": self.shift_surface,
            "num_of_point_per_line": self.num_of_point_per_line,
            "optimizer": self.optimizer,
            "fmax": self.fmax,
            "z_cut_level": self.z_cut_level,
            "real_shift_surface": self.real_shift_surface,
        }

    def generate_structures(self, verbose=None):
        basis_ref = self.basis_ref
        ns = np.arange(0, self.num_of_point_per_line)
        ms = np.arange(0, self.num_of_point_per_line)
        N = self.num_of_point_per_line
        for n, m in product(ns, ms):
            basis = basis_ref.copy()
            if self.auto_constraint:
                apply_constraint(basis)
            # Calculate the shift
            basis_cell = basis.get_cell()
            shift_vector = (
                n / N * self.shift_surface[0] * basis_cell[0, :2]
                + m / N * self.shift_surface[1] * basis_cell[1, :2]
            )
            # Apply the shift to the cell
            new_positions = []
            shift = [shift_vector[0], shift_vector[1], 0]
            for i, pos in enumerate(basis.get_positions()):
                if pos[2] > self.z_cut_level * np.linalg.norm(basis_cell[2]):
                    new_positions.append(pos + shift)
                else:
                    new_positions.append(pos)
            basis.set_positions(new_positions, apply_constraint=False)
            new_cell = basis_cell
            new_cell[2] = new_cell[2] + shift
            basis.set_cell(new_cell, scale_atoms=False, apply_constraint=False)

            jobname = self.subjob_name(n, m)
            self._structure_dict[jobname] = basis
        return self._structure_dict

    def analyse_structures(self, output_dict):
        basis_ref_get_cell = self.basis_ref.cell
        surface_area = np.linalg.det(basis_ref_get_cell[:2, :2])
        energy_list = []
        shift_list = []
        shift_list_X = []
        shift_list_Y = []
        for name, (e, shift) in output_dict.items():
            energy_list.append(e)
            shift_list.append(shift)
        for shift in shift_list:
            shift_list_X.append(shift[0])
            shift_list_Y.append(shift[1])
        energy_list = np.array(energy_list)
        shift_list = np.array(shift_list)

        energy_map = energy_list.copy() / surface_area * eV_A2_to_mJ_M2
        energy_map = energy_map.reshape(
            self.num_of_point_per_line, self.num_of_point_per_line
        )
        energy_map -= energy_map[0][0]

        self._value["shift"] = shift_list
        self._value["energy"] = energy_list
        self._value["energy_map(mJ/m2)"] = energy_map
        self._value["shift_map_X"] = np.reshape(shift_list_X, (self.num_of_point_per_line, self.num_of_point_per_line))
        self._value["shift_map_Y"] = np.reshape(shift_list_Y, (self.num_of_point_per_line, self.num_of_point_per_line))

    def get_structure_value(self, structure, name=None):
        shift = structure.cell[2] - self.basis_ref.cell[2]
        if isinstance(structure.calc, AMSDFTBaseCalculator):
            calc = structure.calc
            calc.optimize_atoms_only(ediff=-self.fmax, max_steps=100)
            # do calculations
            structure.get_potential_energy()
            structure = structure.calc.atoms
            structure.calc = calc
        else:
            logfile = "-" if self.verbose else "/dev/null"
            optimizer = create_optimizer(self.optimizer, structure, logfile)
            optimizer.run(fmax=self.fmax)
        en = structure.get_potential_energy(force_consistent=True)
        return (en, shift), structure

    @staticmethod
    def subjob_name(n, m):
        return "n_{}_m_{}".format(n, m)

    def plot(self, ax=None, **kwargs):
        """Plot the gamma surface energy map
        ax: matplotlib axis object
        kwargs: keyword arguments for matplotlib.pyplot.contourf
        """
        import matplotlib.pyplot as plt

        fig = plt.gcf()
        # use original axis if ax is not given
        if ax is None:
            ax = fig.gca()
        c = ax.contourf(
            self._value["shift_map_X"],
            self._value["shift_map_Y"],
            self._value["energy_map(mJ/m2)"],
            **kwargs
        )
        ax.set_xlabel("X, A")
        ax.set_ylabel("Y, A")
        # add colorbar
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label("Energy (mJ/m2)")

    def plot_minima(self, ax=None, zoom_factor=10, min_distance=None, **kwargs):
        """
        Plot the gamma surface contour with all local minima highlighted and labeled.
        
        Use this method to visually identify minima indices, then pass the desired
        indices to plot_mep(path=[...]) to specify a custom MEP path.

        :param ax: matplotlib axis object for the plot
        :param zoom_factor: factor by which to upscale the data for smoother visualization
        :param min_distance: minimum distance between detected minima (default: zoom_factor)
        :param kwargs: keyword arguments for matplotlib.pyplot.contourf
        :return: tuple of (fig, ax, minima_info) where minima_info is a list of dicts
                 containing 'index', 'pixel_coords', 'phys_coords', 'energy' for each minimum
        """
        from scipy.ndimage import zoom
        from skimage.feature import peak_local_max
        import matplotlib.pyplot as plt

        # --- 1. Load Data ---
        emap = self._value["energy_map(mJ/m2)"]
        X_map = self._value["shift_map_X"]
        Y_map = self._value["shift_map_Y"]

        # --- 2. Upscale Data (Zoom) ---
        emap_highres = zoom(emap, zoom_factor, order=3)
        X_highres = zoom(X_map, zoom_factor, order=1)
        Y_highres = zoom(Y_map, zoom_factor, order=1)

        # --- 3. Find All Local Minima ---
        if min_distance is None:
            min_distance = zoom_factor
        coordinates = peak_local_max(-emap_highres, min_distance=min_distance, exclude_border=False)
        energies = emap_highres[coordinates[:, 0], coordinates[:, 1]]
        sorted_indices = np.argsort(energies)
        minima_sorted = coordinates[sorted_indices]
        energies_sorted = energies[sorted_indices]

        # --- 4. Build minima_info list ---
        minima_info = []
        for idx, (coord, energy) in enumerate(zip(minima_sorted, energies_sorted)):
            phys_x = X_highres[coord[0], coord[1]]
            phys_y = Y_highres[coord[0], coord[1]]
            minima_info.append({
                'index': idx,
                'pixel_coords': tuple(coord),
                'phys_coords': (phys_x, phys_y),
                'energy': energy
            })

        # --- 5. Plotting ---
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        else:
            fig = ax.figure

        levels = kwargs.pop("levels", 25)
        cmap = kwargs.pop("cmap", "viridis")
        cntr = ax.contourf(X_highres, Y_highres, emap_highres, levels=levels, cmap=cmap, **kwargs)
        fig.colorbar(cntr, ax=ax, label="Energy (mJ/m2)")

        # Plot all minima with their indices
        for info in minima_info:
            phys_x, phys_y = info['phys_coords']
            idx = info['index']
            ax.scatter(phys_x, phys_y, c="red", marker="o", s=100, edgecolors="white", linewidths=1.5, zorder=5)
            ax.annotate(
                str(idx),
                (phys_x, phys_y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=10,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7),
                zorder=6
            )

        ax.set_xlabel("X (Angstrom)")
        ax.set_ylabel("Y (Angstrom)")
        ax.set_title("Gamma Surface - Local Minima (sorted by energy)")
        ax.set_aspect(1.0)

        # Print minima info
        print("Found {} local minima (sorted by energy):".format(len(minima_info)))
        print("-" * 60)
        for info in minima_info:
            print("  Index {:2d}: E = {:8.2f} mJ/m², Position = ({:.3f}, {:.3f}) Å".format(
                info['index'], info['energy'], info['phys_coords'][0], info['phys_coords'][1]))
        print("-" * 60)
        print("Use plot_mep(path=[i, j, ...]) to specify a custom path through minima.")

        plt.tight_layout()
        return fig, ax, minima_info

    def calculate_mep(self, zoom_factor=10, path=None, min_distance=None):
        """
        Calculate the Minimum Energy Path (MEP) on the gamma surface.

        :param zoom_factor: factor by which to upscale the data for smoother path finding
        :param path: optional list of minima indices to use as waypoints for the path.
        :param min_distance: minimum distance between detected minima (default: zoom_factor)
        :return: dict containing MEP data
        """
        from scipy.ndimage import zoom, map_coordinates
        from scipy.interpolate import splprep, splev
        from skimage.feature import peak_local_max
        from skimage.graph import route_through_array

        # --- 1. Load Data ---
        emap = self._value["energy_map(mJ/m2)"]
        X_map = self._value["shift_map_X"]
        Y_map = self._value["shift_map_Y"]

        # --- 2. Upscale Data (Zoom) ---
        emap_highres = zoom(emap, zoom_factor, order=3)
        X_highres = zoom(X_map, zoom_factor, order=1)
        Y_highres = zoom(Y_map, zoom_factor, order=1)

        # --- 3. Find Minima & Determine Path Nodes ---
        if min_distance is None:
            min_distance = zoom_factor
        coordinates = peak_local_max(-emap_highres, min_distance=min_distance, exclude_border=False)
        energies = emap_highres[coordinates[:, 0], coordinates[:, 1]]
        sorted_indices = np.argsort(energies)
        minima_sorted = coordinates[sorted_indices]

        if path is not None:
            # User-specified path through minima
            if len(path) < 2:
                raise ValueError("path must contain at least 2 minima indices")
            for idx in path:
                if idx < 0 or idx >= len(minima_sorted):
                    raise ValueError(f"Invalid minima index {idx}. Valid range: 0-{len(minima_sorted)-1}")
            path_nodes = [tuple(minima_sorted[idx]) for idx in path]
        else:
            # Auto-detect path between two lowest-energy minima
            if len(minima_sorted) < 2:
                # Fallback if not enough minima found
                path_nodes = [(0, 0), (emap_highres.shape[0] - 1, emap_highres.shape[1] - 1)]
            else:
                path_nodes = [tuple(minima_sorted[0]), tuple(minima_sorted[1])]

        # --- 4. Find Rough Path (Graph Search) through all waypoints ---
        cost_function = np.exp(emap_highres / np.max(emap_highres) * 10)

        all_path_indices = []
        for i in range(len(path_nodes) - 1):
            start_node = path_nodes[i]
            end_node = path_nodes[i + 1]
            indices, weight = route_through_array(
                cost_function, start_node, end_node, fully_connected=True, geometric=True
            )
            # Avoid duplicating waypoints when concatenating segments
            if i > 0 and len(all_path_indices) > 0:
                indices = indices[1:]  # Skip first point (same as last of previous segment)
            all_path_indices.extend(indices)

        all_path_indices = np.array(all_path_indices)
        path_y, path_x = all_path_indices[:, 0], all_path_indices[:, 1]

        # --- 5. Smooth Path (B-Spline) ---
        step_size = zoom_factor * 2
        x_anchors = path_x[::step_size]
        y_anchors = path_y[::step_size]

        if path_x[-1] != x_anchors[-1]:
            x_anchors = np.append(x_anchors, path_x[-1])
            y_anchors = np.append(y_anchors, path_y[-1])

        # Ensure we have enough points for spline (k=3 needs at least 4 points)
        if len(x_anchors) < 4:
            k = len(x_anchors) - 1
        else:
            k = 3

        if k > 0:
            tck, u = splprep([x_anchors, y_anchors], s=1, k=k)
            u_fine = np.linspace(0, 1, 100)
            x_smooth_idx, y_smooth_idx = splev(u_fine, tck)
        else:
            x_smooth_idx, y_smooth_idx = path_x, path_y

        # --- 6. Extract Physical Data ---
        phys_x_path = map_coordinates(X_highres, [y_smooth_idx, x_smooth_idx], order=1)
        phys_y_path = map_coordinates(Y_highres, [y_smooth_idx, x_smooth_idx], order=1)
        energy_profile = map_coordinates(emap_highres, [y_smooth_idx, x_smooth_idx], order=3)

        dx = np.diff(phys_x_path)
        dy = np.diff(phys_y_path)
        dist_steps = np.sqrt(dx**2 + dy**2)
        reaction_coordinate = np.concatenate(([0], np.cumsum(dist_steps)))

        return {
            "reaction_coordinate": reaction_coordinate,
            "energy_profile": energy_profile,
            "phys_x_path": phys_x_path,
            "phys_y_path": phys_y_path,
            "path_nodes": path_nodes,
            "X_highres": X_highres,
            "Y_highres": Y_highres,
            "emap_highres": emap_highres,
        }

    def get_mep_structures(self, zoom_factor=10, path=None, min_distance=None, n_images=None):
        """
        Generate atomic structures along the Minimum Energy Path.

        :param zoom_factor: factor by which to upscale the data
        :param path: optional list of minima indices to use as waypoints
        :param min_distance: minimum distance between detected minima
        :param n_images: number of structures to generate along the path. 
                         If None, generates structures for all points in the calculated path (approx 100).
        :return: list of ASE Atoms objects
        """
        mep_data = self.calculate_mep(zoom_factor, path, min_distance)
        
        phys_x = mep_data["phys_x_path"]
        phys_y = mep_data["phys_y_path"]
        reaction_coord = mep_data["reaction_coordinate"]

        if n_images is not None:
             # Resample path to n_images
            from scipy.interpolate import interp1d
            
            target_rc = np.linspace(reaction_coord[0], reaction_coord[-1], n_images)
            
            # Interpolate X and Y coordinates over reaction coordinate
            f_x = interp1d(reaction_coord, phys_x, kind='linear')
            f_y = interp1d(reaction_coord, phys_y, kind='linear')
            
            phys_x = f_x(target_rc)
            phys_y = f_y(target_rc)

        structures = []
        basis_ref = self.basis_ref
        basis_cell = basis_ref.get_cell()

        for x, y in zip(phys_x, phys_y):
            basis = basis_ref.copy()
            if self.auto_constraint:
                apply_constraint(basis)
            
            shift = np.array([x, y, 0.0])
            
            new_positions = []
            for pos in basis.get_positions():
                if pos[2] > self.z_cut_level * np.linalg.norm(basis_cell[2]):
                    new_positions.append(pos + shift)
                else:
                    new_positions.append(pos)
            
            basis.set_positions(new_positions, apply_constraint=False)
            
            new_cell = basis_cell.copy()
            new_cell[2] = new_cell[2] + shift
            basis.set_cell(new_cell, scale_atoms=False, apply_constraint=False)
            
            structures.append(basis)

        return structures

    def plot_mep(self, ax1=None, ax2=None, zoom_factor=10, path=None, min_distance=None, **kwargs):
        """
        Find and plot the Minimum Energy Path (MEP) on the gamma surface.

        :param ax1: matplotlib axis object for the map plot
        :param ax2: matplotlib axis object for the energy profile plot
        :param zoom_factor: factor by which to upscale the data for smoother path finding
        :param path: optional list of minima indices to use as waypoints for the path.
        :param min_distance: minimum distance between detected minima (default: zoom_factor)
        :param kwargs: keyword arguments for matplotlib.pyplot.contourf (map plot)
        """
        import matplotlib.pyplot as plt

        # --- 1. Get MEP Data ---
        mep_data = self.calculate_mep(zoom_factor, path, min_distance)
        
        reaction_coordinate = mep_data["reaction_coordinate"]
        energy_profile = mep_data["energy_profile"]
        phys_x_path = mep_data["phys_x_path"]
        phys_y_path = mep_data["phys_y_path"]
        path_nodes = mep_data["path_nodes"]
        X_highres = mep_data["X_highres"]
        Y_highres = mep_data["Y_highres"]
        emap_highres = mep_data["emap_highres"]

        start_node = path_nodes[0]
        end_node = path_nodes[-1]
        start_phys_x = X_highres[start_node[0], start_node[1]]
        start_phys_y = Y_highres[start_node[0], start_node[1]]
        end_phys_x = X_highres[end_node[0], end_node[1]]
        end_phys_y = Y_highres[end_node[0], end_node[1]]

        # --- 2. Plotting ---
        if ax1 is None or ax2 is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        else:
            fig = ax1.figure

        # MAP (Left)
        levels = kwargs.pop("levels", 25)
        cmap = kwargs.pop("cmap", "viridis")
        cntr = ax1.contourf(X_highres, Y_highres, emap_highres, levels=levels, cmap=cmap, **kwargs)
        fig.colorbar(cntr, ax=ax1, label="Energy (mJ/m2)")

        ax1.plot(phys_x_path, phys_y_path, color="white", linestyle="--", linewidth=2.5, label="MEP")
        ax1.scatter(start_phys_x, start_phys_y, c="green", marker="o", s=150, edgecolors="k", label="Start")
        ax1.scatter(end_phys_x, end_phys_y, c="red", marker="*", s=200, edgecolors="k", label="End")
        
        # Plot intermediate waypoints if path was specified
        if path is not None and len(path) > 2:
            for i, node in enumerate(path_nodes[1:-1], start=1):
                wp_x = X_highres[node[0], node[1]]
                wp_y = Y_highres[node[0], node[1]]
                ax1.scatter(wp_x, wp_y, c="yellow", marker="D", s=100, edgecolors="k", zorder=5)
                ax1.annotate(str(path[i]), (wp_x, wp_y), textcoords="offset points",
                           xytext=(5, 5), fontsize=9, color="white", fontweight="bold")

        ax1.set_xlabel("X (Angstrom)")
        ax1.set_ylabel("Y (Angstrom)")
        ax1.set_title("Gamma Surface MEP")
        ax1.legend()
        ax1.set_aspect(1.0)

        # Energy Profile (Right)
        ax2.plot(reaction_coordinate, energy_profile, linewidth=2, color="navy")
        ax2.fill_between(reaction_coordinate, energy_profile, np.min(energy_profile), color="navy", alpha=0.1)

        ax2.scatter(reaction_coordinate[0], energy_profile[0], c="green", s=100, label="Start", zorder=5)
        ax2.scatter(reaction_coordinate[-1], energy_profile[-1], c="red", marker="*", s=100, label="End", zorder=5)

        # Plot intermediate waypoints on energy profile if path was specified
        if path is not None and len(path) > 2:
            # Calculate reaction coordinate positions for each waypoint
            for i, node in enumerate(path_nodes[1:-1], start=1):
                wp_phys_x = X_highres[node[0], node[1]]
                wp_phys_y = Y_highres[node[0], node[1]]
                # Find closest point on the smoothed path
                distances = np.sqrt((phys_x_path - wp_phys_x)**2 + (phys_y_path - wp_phys_y)**2)
                closest_idx = np.argmin(distances)
                wp_rc = reaction_coordinate[closest_idx]
                wp_energy = energy_profile[closest_idx]
                ax2.scatter(wp_rc, wp_energy, c="yellow", marker="D", s=80, edgecolors="k", zorder=5)
                ax2.annotate(str(path[i]), (wp_rc, wp_energy), textcoords="offset points",
                           xytext=(5, 5), fontsize=9, color="black", fontweight="bold")

        ax2.set_xlabel("Reaction Coordinate (Angstrom)")
        ax2.set_ylabel("Energy (mJ/m2)")
        ax2.set_title("Energy Profile along MEP")
        ax2.grid(True, linestyle="--", alpha=0.7)
        if "ylim" in kwargs:
            ax2.set_ylim(kwargs["ylim"])
        ax2.legend()

        plt.tight_layout()
        return fig, (ax1, ax2)


class GammaLineCalculator(GeneralCalculator):
    """Calculation of a gamma line

    :param atoms: original ASE Atoms object to be used for calculation of gamma line energy
    :param shift_vector: default = [0,1] (shift in x and y direction)
    :param num_of_point: default = 10 (number of points in the gamma line)
    :param optimizer: default = FIRE (optimizer to be used for relaxation)
    :param fmax: default = 1e-2 (force criteria for optimizer)
    :param z_cut_level: default = 0.5, separation of upper and lower parts of atoms in Z direction, in relative units between 0 and 1
    :param auto_constraint: default = True, if True, the constraint to Z-only relaxation  will be set automatically

    Usage:
    >> block = FaceCenteredCubic(directions=[[1, -1, 0], [1, 1, -2], [1, 1, 1]], size=(1, 1, 4),
                                       symbol="Al", pbc=(1, 1, 1), latticeconstant=4.05)
    >> block.calc = calc
    >> gamma_line = GammaLineCalculator(bloc, shift_vector=[0,1.0],
                                num_of_point=21,
                                fmax=1E-2,
                                z_cut_level=0.5)
    >> gamma_line.calculate()
    >> E=gamma_line_calc.value['energy_map(mJ/m2)']
    >> gamma_line.plot()
    """

    property_name = "gammaline"

    param_names = [
        "shift_vector",
        "num_of_point",
        "optimizer",
        "fmax",
        "z_cut_level",
        "auto_constraint",
    ]

    def __init__(
        self,
        atoms=None,
        shift_vector=(0, 1),
        num_of_point=10,
        optimizer="FIRE",
        fmax=1e-2,
        z_cut_level=0.5,
        auto_constraint=True,
        **kwargs,
    ):
        GeneralCalculator.__init__(self, atoms, **kwargs)
        self.shift_vector = shift_vector
        self.num_of_point = num_of_point
        self.optimizer = optimizer
        self.fmax = fmax
        self.z_cut_level = z_cut_level
        cell = atoms.get_cell()
        self.real_shift_vector = shift_vector[0] * cell[0] + shift_vector[1] * cell[1]
        self.auto_constraint = auto_constraint
        self._value = {
            "shift_vector": self.shift_vector,
            "num_of_point": self.num_of_point,
            "optimizer": self.optimizer,
            "fmax": self.fmax,
            "z_cut_level": self.z_cut_level,
            "real_shift_vector": self.real_shift_vector,
        }

    def generate_structures(self, verbose=None):
        basis_ref = self.basis_ref
        real_shift_norm = np.linalg.norm(self.real_shift_vector)

        for n in np.arange(self.num_of_point):
            basis = basis_ref.copy()

            if self.auto_constraint:
                apply_constraint(basis)

            # Calculate the shift
            L = n / self.num_of_point * real_shift_norm
            if self.real_shift_vector[1] == 0:
                theta = 0
            elif self.real_shift_vector[0] == 0:
                theta = np.pi / 2
            else:
                theta = np.arctan(self.real_shift_vector[1] / self.real_shift_vector[0])
            shift = L * np.array([np.cos(theta), np.sin(theta), 0])
            # Apply the shift to the cell
            new_positions = []
            for i, pos in enumerate(basis.get_positions()):
                if pos[2] > self.z_cut_level * np.linalg.norm(basis.get_cell()[2]):
                    new_positions.append(pos + shift)
                else:
                    new_positions.append(pos)
            basis.set_positions(new_positions, apply_constraint=False)
            new_cell = basis.get_cell()
            new_cell[2] = new_cell[2] + shift
            basis.set_cell(new_cell, scale_atoms=False, apply_constraint=False)

            jobname = self.subjob_name(self.shift_vector, n)
            self._structure_dict[jobname] = basis
        return self._structure_dict

    def analyse_structures(self, output_dict):
        surf_area = np.linalg.det(self.basis_ref.get_cell()[:2, :2])

        energy_list = []
        shift_list = []
        for name, (e, s) in output_dict.items():
            energy_list.append(e)
            shift_list.append(s)
        shift_list = np.array(shift_list)
        energy_list = np.array(energy_list)
        energy_map = energy_list / surf_area * eV_A2_to_mJ_M2
        energy_map -= energy_map[0]

        ind_order = np.argsort(shift_list)
        shift_list = shift_list[ind_order]
        energy_list = energy_list[ind_order]

        self._value["shift"] = shift_list
        self._value["energy"] = energy_list
        self._value["energy_map(mJ/m2)"] = energy_map

    def get_structure_value(self, structure, name=None):
        if isinstance(structure.calc, AMSDFTBaseCalculator):
            calc = structure.calc
            calc.optimize_atoms_only(ediff=-self.fmax, max_steps=100)
            # do calculations
            structure.get_potential_energy()
            structure = structure.calc.atoms
            structure.calc = calc
        else:
            shift = np.linalg.norm(
                structure.get_cell()[2] - self.basis_ref.get_cell()[2]
            )
            logfile = "-" if self.verbose else "/dev/null"
            optimizer = create_optimizer(self.optimizer, structure, logfile)
            optimizer.run(fmax=self.fmax)
        en = structure.get_potential_energy(force_consistent=True)
        return (en, shift), structure

    @staticmethod
    def subjob_name(shift, n):
        return "X_{}_Y_{}_n_{}".format(
            str(shift[0]).replace(".", "_"), str(shift[1]).replace(".", "_"), n
        )

    def plot(self, ax=None, **kwargs):
        """Plot gamma line energy vs displacement
        ax: matplotlib axis
        **kwargs: keyword arguments for ax.plot function
        """
        from matplotlib import pyplot as plt

        if ax is None:
            ax = plt.gca()
        ax.plot(self._value["shift"], self._value["energy_map(mJ/m2)"], **kwargs)
        ax.set_xlabel("Displacement (A)")
        ax.set_ylabel("Energy, mJ/m2")
