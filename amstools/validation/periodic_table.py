import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _get_element_layout(max_atomic_number=118):
    """
    Determines the layout coordinates for elements in a periodic table plot.

    Args:
        max_atomic_number (int): The maximum atomic number to include.

    Returns:
        list: A list of tuples, where each tuple contains:
              - element (pymatgen.core.periodic_table.Element): The element object.
              - x (int): The x-coordinate (group).
              - y (int): The y-coordinate (period).
              - ax_name (str): The name of the axes to plot on ('main' or 'f_block').
    """
    from pymatgen.core.periodic_table import Element

    layout = []
    elements = [Element.from_Z(i) for i in range(1, max_atomic_number + 1)]

    for el in elements:
        # Main block elements
        if el.row is not None and 1 <= el.row <= 7 and el.group is not None:
            if 57 <= el.Z <= 71 or 89 <= el.Z <= 103:  # f-block elements
                continue
            x = el.group - 1
            y = 7 - el.row
            layout.append((el, x, y, "main"))

        # f-block elements
        elif el.row is not None and el.row >= 8:
            if 57 <= el.Z <= 71:  # Lanthanides
                x = (el.Z - 57) + 2.5
                y = 1
                layout.append((el, x, y, "f_block"))
            elif 89 <= el.Z <= 103:  # Actinides
                x = (el.Z - 89) + 2.5
                y = 0
                layout.append((el, x, y, "f_block"))
    return layout


def default_plot_patch(el, x, y, ax, val_fn, color_fn, text_color_fn, cw=0.95, ch=0.95):
    """
    Default function to plot a patch for an element in the periodic table.

    Displays the element's symbol, atomic number, and a given value.

    Args:
        el (pymatgen.core.periodic_table.Element): The element to plot.
        x (float): The x-coordinate of the patch.
        y (float): The y-coordinate of the patch.
        ax (matplotlib.axes.Axes): The axes to plot on.
        val_fn (callable): A function that takes an element and returns a value.
        color_fn (callable): A function that takes an element and returns a color.
        text_color_fn (callable): A function that takes an element and returns a text color.
        cw (float): The width of the patch.
        ch (float): The height of the patch.
    """
    text_color = text_color_fn(el)

    # Atomic symbol
    ax.text(
        x + cw * 0.5,
        y + ch * 0.5,
        el.symbol,
        ha="center",
        va="center",
        color=text_color,
        fontsize="small",
    )

    # Atomic number Z
    ax.text(
        x + cw * 0.5,
        y + ch * 0.85,
        f"{el.Z}",
        ha="center",
        va="center",
        color=text_color,
        fontsize="x-small",
    )

    # Value
    val = val_fn(el)
    val_txt = f"{val:.2f}" if val is not None else "-"
    ax.text(
        x + cw * 0.5,
        y + ch * 0.15,
        val_txt,
        ha="center",
        va="center",
        color=text_color,
        fontsize="x-small",
    )


def plot_periodic_table(
    val_fn,
    plot_patch_fn=default_plot_patch,
    val_min=0,
    val_max=1,
    cmap=plt.cm.viridis,
    figsize=(12, 6),
    text_color_inverted=False,
    dpi=150,
    label=None,
    max_atomic_number=118,
):
    """
    Plots a periodic table with cells colored by a user-defined value function.

    Args:
        val_fn (callable): A function that takes a pymatgen Element object and returns a
                           numerical value. This value determines the color of the element's cell.
                           Should return None for elements without a value.
        plot_patch_fn (callable, optional): A function to draw the content of each element's cell.
                                            Defaults to `default_plot_patch`.
        val_min (float, optional): The minimum value for the color scale. Defaults to 0.
        val_max (float, optional): The maximum value for the color scale. Defaults to 1.
        cmap (matplotlib.colors.Colormap, optional): The colormap to use. Defaults to `plt.cm.viridis`.
        figsize (tuple, optional): The figure size. Defaults to (12, 6).
        text_color_inverted (bool, optional): If True, inverts the text color for better contrast
                                              on the colormap. Defaults to False.
        dpi (int, optional): The resolution of the figure. Defaults to 150.
        label (str, optional): The label for the colorbar. Defaults to None.
        max_atomic_number (int, optional): The maximum atomic number to display. Defaults to 118.
    """
    element_layout = _get_element_layout(max_atomic_number)

    def normalized_val_fn(el):
        val = val_fn(el)
        if val is not None:
            return (val - val_min) / (val_max - val_min)
        return None

    def color_fn(el):
        normalized_val = normalized_val_fn(el)
        if normalized_val is not None:
            return cmap(normalized_val)
        return (0.5, 0.5, 0.5, 1.0)

    def text_color_fn(el):
        normalized_val = normalized_val_fn(el)
        if normalized_val is not None:
            if text_color_inverted:
                return "white" if normalized_val > 0.5 else "black"
            return "white" if normalized_val < 0.5 else "black"
        return "black"

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax_main = fig.add_axes([0, 0.2, 1, 0.75])
    ax_f_block = fig.add_axes([0, 0, 1, 0.2])
    axes = {"main": ax_main, "f_block": ax_f_block}

    def plot_patch_wrapper(el, x, y, ax, cw=0.95, ch=0.95):
        ax.add_patch(
            patches.Rectangle((x, y), cw, ch, facecolor=color_fn(el), edgecolor="k")
        )
        plot_patch_fn(el, x, y, ax, val_fn, color_fn, text_color_fn, cw=cw, ch=ch)

    for el, x, y, ax_name in element_layout:
        plot_patch_wrapper(el, x, y, axes[ax_name])

    ax_main.set_xlim(0, 18)
    ax_main.set_ylim(0, 7)
    ax_main.axis("off")

    ax_f_block.set_xlim(0, 18)
    ax_f_block.set_ylim(0, 2)
    ax_f_block.axis("off")

    # Colorbar
    norm = mpl.colors.Normalize(vmin=val_min, vmax=val_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar_ax = fig.add_axes([3 / 18, (7 - 1.5) / 7, 8 / 18, 0.05])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")

    if label:
        cbar.ax.xaxis.set_label_position("top")
        cbar.set_label(label, labelpad=10, fontsize=12)

    fig.subplots_adjust(bottom=0.15)
    plt.show()
