"""Materials Project interface for downloading crystal structures.

Structures are cached as CIF files in per-chemsys subfolders so that
previously downloaded sub-systems can be reused across queries.

Cache layout::

    mp_cache/
      Al/
        Al__mp-2.cif
        metadata.json          ← written by fetch_mp_reference_df
      Ni/
        Ni__mp-23.cif
        metadata.json
      Al-Ni/
        AlNi__mp-81.cif
        AlNi3__mp-2658.cif
        metadata.json

Each chemsys subfolder is the unit of caching:

- :func:`fetch_structures` — considers a folder cached when it contains
  any file.
- :func:`fetch_mp_reference_df` — requires ``metadata.json`` to be present
  (so that energetics are always available).  CIF-only folders produced by
  ``fetch_structures`` will be re-downloaded to fetch the energetics.

After ``fetch_mp_reference_df`` runs, ``fetch_structures`` will also find
the cached CIFs without re-downloading.
"""

import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

from ase import Atoms


def get_chemsys(elements: List[str]) -> str:
    """Return the canonical ``'Al-Ni'`` style chemsys string for *elements*.

    Elements are sorted alphabetically so the result is deterministic
    regardless of the order in which they are supplied.

    Parameters
    ----------
    elements:
        List of element symbols, e.g. ``["Ni", "Al"]``.

    Returns
    -------
    str
        Hyphen-joined sorted element string, e.g. ``"Al-Ni"``.
    """
    return "-".join(sorted(elements))


def get_all_chemsys(elements: List[str]) -> List[str]:
    """Return all sub-system chemsys strings for a multinary element set.

    For example, ``["Al", "Ni", "Ti"]`` yields
    ``["Al", "Ni", "Ti", "Al-Ni", "Al-Ti", "Ni-Ti", "Al-Ni-Ti"]``.

    Parameters
    ----------
    elements:
        List of element symbols.

    Returns
    -------
    list of str
        All sub-system chemsys strings, from unaries up to the full system,
        each in sorted alphabetical order.
    """
    sorted_els = sorted(elements)
    result = []
    for r in range(1, len(sorted_els) + 1):
        for combo in combinations(sorted_els, r):
            result.append("-".join(combo))
    return result


def save_structures_cif(structure_dict: Dict[str, Atoms], folder: Path) -> Path:
    """Write *structure_dict* entries as CIF files into *folder*.

    Parameters
    ----------
    structure_dict:
        Mapping of structure name → ASE :class:`~ase.Atoms`.
    folder:
        Directory to write CIF files into.  Created if it does not exist.

    Returns
    -------
    Path
        The (possibly newly created) *folder* path.
    """
    import ase.io

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, atoms in structure_dict.items():
        cif_path = folder / f"{name}.cif"
        ase.io.write(str(cif_path), atoms, format="cif")
    return folder


def load_structures_folder(folder: Path) -> Dict[str, Atoms]:
    """Load all structure files from *folder* using ASE auto-detection.

    ASE infers the format from the file extension, so the folder may
    contain a mix of ``.cif``, ``.vasp``, ``.xyz``, ``.extxyz``,
    ``.json``, etc.

    Parameters
    ----------
    folder:
        Directory to read structures from.

    Returns
    -------
    dict
        Mapping of filename stem → :class:`~ase.Atoms`.  Returns an empty
        dict if *folder* does not exist.
    """
    import ase.io

    folder = Path(folder)
    if not folder.is_dir():
        return {}

    result = {}
    for filepath in folder.iterdir():
        if not filepath.is_file():
            continue
        try:
            atoms = ase.io.read(str(filepath))
            result[filepath.stem] = atoms
        except Exception:
            pass
    return result


_METADATA_FILENAME = "metadata.json"


def _save_metadata(metadata: dict, folder: Path) -> None:
    """Write *metadata* dict to ``metadata.json`` inside *folder*.

    Parameters
    ----------
    metadata:
        Mapping of structure name → energy dict, e.g.::

            {"AlNi__mp-81": {"e_formation_per_atom": -0.42,
                             "e_chull_dist_per_atom": 0.0}}
    folder:
        Target directory.  Created if it does not exist.
    """
    import json

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / _METADATA_FILENAME, "w") as fh:
        json.dump(metadata, fh)


def _load_metadata(folder: Path) -> dict:
    """Return the metadata dict from ``metadata.json`` in *folder*.

    Returns an empty dict if the file is absent.
    """
    import json

    path = Path(folder) / _METADATA_FILENAME
    if not path.exists():
        return {}
    with open(path) as fh:
        return json.load(fh)


def _pymatgen_structure_to_atoms(structure) -> Atoms:
    """Convert a pymatgen Structure to ASE Atoms without importing pymatgen.

    Uses only the public attributes that every pymatgen Structure exposes,
    so no explicit ``import pymatgen`` is required — pymatgen is already
    present as a transitive dependency of ``mp-api``.
    """
    return Atoms(
        symbols=[str(site.specie) for site in structure],
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        pbc=True,
    )


def fetch_structures(
    elements: List[str],
    e_above_hull: float = 0.1,
    max_atoms: Optional[int] = None,
    mp_api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_redownload: bool = False,
    thermo_types: Optional[List[str]] = None,
) -> Dict[str, Atoms]:
    """Download crystal structures from the Materials Project and cache locally.

    All sub-systems (unaries, binaries, …) of *elements* are included.
    Previously downloaded chemsys subfolders are reused from the cache
    without contacting the MP API.

    Parameters
    ----------
    elements:
        Element symbols to query, e.g. ``["Al", "Ni"]``.
    e_above_hull:
        Maximum energy above the convex hull in eV/atom (inclusive).
        Defaults to ``0.1``.
    max_atoms:
        If given, only structures with at most this many atoms in the unit
        cell are returned.  The limit is applied both in the MP query (to
        avoid downloading oversized structures) and as a post-filter when
        loading from the cache (so existing cache folders built without this
        limit are handled correctly).  Set ``force_redownload=True`` if you
        previously cached without a limit and now want the cache to reflect
        the stricter filter.
    mp_api_key:
        Materials Project API key.  Falls back to the ``MP_API_KEY``
        environment variable when not given.
    cache_dir:
        Root directory for the per-chemsys cache folders.  Defaults to the
        current working directory.
    force_redownload:
        When ``True``, re-query MP for every chemsys even if a cached
        subfolder already exists.  Useful when widening *e_above_hull* or
        *max_atoms*, or when refreshing stale data.
    thermo_types:
        MP thermodynamic dataset to use.  Defaults to ``["GGA_GGA+U"]``
        (PBE functional with GGA+U for correlated oxides), which is MP's
        primary dataset.  Pass ``["R2SCAN"]`` to use the r²SCAN dataset
        instead.

    Returns
    -------
    dict
        Mapping of ``"{formula}__{mp-id}"`` → :class:`~ase.Atoms`, covering
        all sub-systems of *elements* up to *e_above_hull* (and *max_atoms*
        if given).

    Raises
    ------
    ImportError
        If ``mp-api`` is not installed (``pymatgen`` is pulled in automatically
        as a dependency of ``mp-api`` and does not need to be installed separately).
    ValueError
        If no API key is available.
    """
    if thermo_types is None:
        thermo_types = ["GGA_GGA+U"]
    # -- Step 1: compute the full list of required chemsys ----------------
    all_chemsys = get_all_chemsys(elements)

    # -- Step 2: split into cached vs. to-download ------------------------
    cache_root = Path(cache_dir) if cache_dir else Path.cwd()

    if force_redownload:
        to_download = list(all_chemsys)
    else:
        to_download = [
            cs
            for cs in all_chemsys
            if not any(f for f in (cache_root / cs).glob("*") if f.is_file())
        ]
    cached = [cs for cs in all_chemsys if cs not in to_download]

    # -- Step 3: download missing chemsys from MP -------------------------
    if to_download:
        try:
            from mp_api.client import MPRester
        except ImportError as exc:
            raise ImportError(
                "mp-api is required to query the Materials Project. "
                "Install it with: pip install mp-api"
            ) from exc

        api_key = mp_api_key or os.environ.get("MP_API_KEY")
        if not api_key:
            raise ValueError(
                "No Materials Project API key found. "
                "Pass mp_api_key= or set the MP_API_KEY environment variable."
            )

        search_kwargs = dict(
            chemsys=to_download,
            energy_above_hull=(0, e_above_hull),
            thermo_types=thermo_types,
            fields=[
                "material_id",
                "formula_pretty",
                "structure",
                "energy_above_hull",
                "chemsys",
            ],
        )
        if max_atoms is not None:
            search_kwargs["num_sites"] = (1, max_atoms)

        with MPRester(api_key) as mpr:
            docs = mpr.summary.search(**search_kwargs)

        # Group docs by chemsys, then save each group to its subfolder
        by_chemsys: Dict[str, Dict[str, Atoms]] = {cs: {} for cs in to_download}
        for doc in docs:
            cs = doc.chemsys
            # Normalise to our sorted convention
            cs_norm = get_chemsys(cs.split("-"))
            if cs_norm not in by_chemsys:
                by_chemsys[cs_norm] = {}
            mp_id = str(doc.material_id).replace("mp-", "")
            name = f"{doc.formula_pretty}__mp-{mp_id}"
            atoms = _pymatgen_structure_to_atoms(doc.structure)
            by_chemsys[cs_norm][name] = atoms

        for cs, struct_dict in by_chemsys.items():
            save_structures_cif(struct_dict, cache_root / cs)

    # -- Step 4: load everything from cache --------------------------------
    structure_dict: Dict[str, Atoms] = {}
    for cs in all_chemsys:
        structure_dict.update(load_structures_folder(cache_root / cs))

    if max_atoms is not None:
        structure_dict = {k: v for k, v in structure_dict.items() if len(v) <= max_atoms}

    return structure_dict


def fetch_mp_reference_df(
    elements: List[str],
    e_above_hull: float = 0.1,
    max_atoms: Optional[int] = None,
    mp_api_key: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_redownload: bool = False,
    thermo_types: Optional[List[str]] = None,
):
    """Download MP structures and energetics, returning a DataFrame for hull comparison.

    The returned DataFrame has five columns:

    ========================= =============================================
    ``name``                  ``"{formula}__mp-{id}"`` key
    ``ase_atoms``             :class:`~ase.Atoms` structure
    ``e_per_atom``            Total DFT energy per atom in eV/atom
    ``e_formation_per_atom``  Formation energy per atom in eV/atom
    ``e_chull_dist_per_atom`` Energy above hull in eV/atom
    ========================= =============================================

    This DataFrame can be passed directly to
    :func:`~amstools.thermodynamics.plot_convex_hull` alongside a
    calculated DataFrame for visual comparison::

        mp_df  = fetch_mp_reference_df(["Al", "Ni"], cache_dir="./mp_cache")
        my_df, _ = run_convex_hull_calculation(fetch_structures(...), calc)
        plot_convex_hull({"My calculator": my_df, "MP reference": mp_df})

    A chemsys subfolder is considered cached only when ``metadata.json``
    exists inside it.  CIF-only folders produced by :func:`fetch_structures`
    do **not** satisfy this check and will be re-downloaded so that
    energetics are retrieved.  After this function runs, :func:`fetch_structures`
    can reuse the cached CIF files without a separate download.

    Parameters
    ----------
    elements:
        Element symbols to query, e.g. ``["Al", "Ni"]``.
    e_above_hull:
        Maximum energy above the convex hull in eV/atom (inclusive).
        Defaults to ``0.1``.
    max_atoms:
        If given, only structures with at most this many atoms in the unit
        cell are returned.  Applied both in the MP query and as a
        post-filter when loading from cache.
    mp_api_key:
        Materials Project API key.  Falls back to the ``MP_API_KEY``
        environment variable when not given.
    cache_dir:
        Root directory for the per-chemsys cache folders.  Defaults to the
        current working directory.
    force_redownload:
        When ``True``, re-query MP for every chemsys even if ``metadata.json``
        already exists.
    thermo_types:
        MP thermodynamic dataset to use.  Defaults to ``["GGA_GGA+U"]``
        (PBE functional with GGA+U for correlated oxides), which is MP's
        primary dataset.  Pass ``["R2SCAN"]`` to use the r²SCAN dataset
        instead.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``name``, ``ase_atoms``, ``e_per_atom``,
        ``e_formation_per_atom``, ``e_chull_dist_per_atom``.
        Rows whose metadata entry is missing have ``NaN`` for the energy
        columns.

    Raises
    ------
    ImportError
        If ``mp-api`` or ``pandas`` is not installed.
    ValueError
        If no API key is available.
    """
    if thermo_types is None:
        thermo_types = ["GGA_GGA+U"]
    import pandas as pd

    # -- Step 1: compute the full list of required chemsys ----------------
    all_chemsys = get_all_chemsys(elements)

    # -- Step 2: split into cached vs. to-download ------------------------
    cache_root = Path(cache_dir) if cache_dir else Path.cwd()

    if force_redownload:
        to_download = list(all_chemsys)
    else:
        to_download = [
            cs
            for cs in all_chemsys
            if not (cache_root / cs / _METADATA_FILENAME).exists()
        ]

    # -- Step 3: download missing chemsys from MP -------------------------
    if to_download:
        try:
            from mp_api.client import MPRester
        except ImportError as exc:
            raise ImportError(
                "mp-api is required to query the Materials Project. "
                "Install it with: pip install mp-api"
            ) from exc

        api_key = mp_api_key or os.environ.get("MP_API_KEY")
        if not api_key:
            raise ValueError(
                "No Materials Project API key found. "
                "Pass mp_api_key= or set the MP_API_KEY environment variable."
            )

        search_kwargs = dict(
            chemsys=to_download,
            energy_above_hull=(0, e_above_hull),
            thermo_types=thermo_types,
            fields=[
                "material_id",
                "formula_pretty",
                "structure",
                "energy_per_atom",
                "energy_above_hull",
                "formation_energy_per_atom",
                "chemsys",
            ],
        )
        if max_atoms is not None:
            search_kwargs["num_sites"] = (1, max_atoms)

        with MPRester(api_key) as mpr:
            docs = mpr.summary.search(**search_kwargs)

        # Group docs by chemsys
        by_chemsys: Dict[str, dict] = {
            cs: {"structs": {}, "meta": {}} for cs in to_download
        }
        for doc in docs:
            cs_norm = get_chemsys(doc.chemsys.split("-"))
            if cs_norm not in by_chemsys:
                by_chemsys[cs_norm] = {"structs": {}, "meta": {}}
            mp_id = str(doc.material_id).replace("mp-", "")
            name = f"{doc.formula_pretty}__mp-{mp_id}"
            atoms = _pymatgen_structure_to_atoms(doc.structure)
            by_chemsys[cs_norm]["structs"][name] = atoms
            by_chemsys[cs_norm]["meta"][name] = {
                "e_per_atom": doc.energy_per_atom,
                "e_formation_per_atom": doc.formation_energy_per_atom,
                "e_chull_dist_per_atom": doc.energy_above_hull,
            }

        for cs, data in by_chemsys.items():
            folder = cache_root / cs
            save_structures_cif(data["structs"], folder)
            _save_metadata(data["meta"], folder)

    # -- Step 4: load everything from cache --------------------------------
    rows = []
    for cs in all_chemsys:
        folder = cache_root / cs
        structures = load_structures_folder(folder)
        metadata = _load_metadata(folder)
        for name, atoms in structures.items():
            meta = metadata.get(name, {})
            rows.append(
                {
                    "name": name,
                    "ase_atoms": atoms,
                    "e_per_atom": meta.get("e_per_atom", float("nan")),
                    "e_formation_per_atom": meta.get(
                        "e_formation_per_atom", float("nan")
                    ),
                    "e_chull_dist_per_atom": meta.get(
                        "e_chull_dist_per_atom", float("nan")
                    ),
                }
            )

    if max_atoms is not None:
        rows = [r for r in rows if len(r["ase_atoms"]) <= max_atoms]

    return pd.DataFrame(
        rows,
        columns=[
            "name",
            "ase_atoms",
            "e_per_atom",
            "e_formation_per_atom",
            "e_chull_dist_per_atom",
        ],
    )
