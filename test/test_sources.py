"""Tests for amstools.sources.materials_project."""

import sys
from unittest.mock import MagicMock, patch

import pytest
from ase import Atoms
from ase.build import bulk

from amstools.sources.materials_project import (
    fetch_structures,
    fetch_mp_reference_df,
    get_all_chemsys,
    get_chemsys,
    load_structures_folder,
    save_structures_cif,
    _save_metadata,
    _load_metadata,
    _METADATA_FILENAME,
)


# ---------------------------------------------------------------------------
# get_chemsys
# ---------------------------------------------------------------------------


def test_get_chemsys_sorts_alphabetically():
    assert get_chemsys(["Ni", "Al"]) == "Al-Ni"


def test_get_chemsys_single_element():
    assert get_chemsys(["Al"]) == "Al"


def test_get_chemsys_ternary():
    assert get_chemsys(["Ni", "Li", "Al"]) == "Al-Li-Ni"


def test_get_chemsys_already_sorted():
    assert get_chemsys(["Al", "Ni"]) == "Al-Ni"


# ---------------------------------------------------------------------------
# get_all_chemsys
# ---------------------------------------------------------------------------


def test_get_all_chemsys_single():
    assert get_all_chemsys(["Al"]) == ["Al"]


def test_get_all_chemsys_binary():
    assert get_all_chemsys(["Ni", "Al"]) == ["Al", "Ni", "Al-Ni"]


def test_get_all_chemsys_ternary_length():
    assert len(get_all_chemsys(["Al", "Ni", "Li"])) == 7


def test_get_all_chemsys_ternary_contains_all_subsystems():
    result = get_all_chemsys(["Ni", "Al", "Li"])
    for expected in ["Al", "Li", "Ni", "Al-Li", "Al-Ni", "Li-Ni", "Al-Li-Ni"]:
        assert expected in result


def test_get_all_chemsys_entries_are_sorted():
    for entry in get_all_chemsys(["Ni", "Al", "Li"]):
        parts = entry.split("-")
        assert parts == sorted(parts)


def test_get_all_chemsys_unaries_before_binaries():
    result = get_all_chemsys(["Al", "Ni"])
    unary_indices = [i for i, x in enumerate(result) if "-" not in x]
    binary_indices = [i for i, x in enumerate(result) if "-" in x]
    assert max(unary_indices) < min(binary_indices)


def test_get_all_chemsys_full_system_is_last():
    result = get_all_chemsys(["Al", "Ni", "Li"])
    assert result[-1] == "Al-Li-Ni"


# ---------------------------------------------------------------------------
# save_structures_cif
# ---------------------------------------------------------------------------


def test_save_structures_cif_creates_file(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05)
    save_structures_cif({"Al__mp-2": atoms}, tmp_path)
    assert (tmp_path / "Al__mp-2.cif").exists()


def test_save_structures_cif_creates_folder_if_missing(tmp_path):
    atoms = bulk("Al", "fcc", a=4.05)
    target = tmp_path / "new_folder" / "Al"
    save_structures_cif({"Al__mp-2": atoms}, target)
    assert target.is_dir()
    assert (target / "Al__mp-2.cif").exists()


def test_save_structures_cif_multiple_structures(tmp_path):
    save_structures_cif(
        {"Al__mp-2": bulk("Al", "fcc", a=4.05), "Ni__mp-23": bulk("Ni", "fcc", a=3.52)},
        tmp_path,
    )
    assert (tmp_path / "Al__mp-2.cif").exists()
    assert (tmp_path / "Ni__mp-23.cif").exists()


def test_save_structures_cif_returns_folder(tmp_path):
    result = save_structures_cif({"Al__mp-2": bulk("Al", "fcc", a=4.05)}, tmp_path)
    assert result == tmp_path


def test_save_structures_cif_empty_dict(tmp_path):
    result = save_structures_cif({}, tmp_path)
    assert result == tmp_path
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# load_structures_folder
# ---------------------------------------------------------------------------


def test_load_structures_folder_missing_dir_returns_empty(tmp_path):
    result = load_structures_folder(tmp_path / "nonexistent")
    assert result == {}


def test_load_structures_folder_round_trip(tmp_path):
    al = bulk("Al", "fcc", a=4.05)
    save_structures_cif({"Al__mp-2": al}, tmp_path)
    result = load_structures_folder(tmp_path)
    assert "Al__mp-2" in result
    assert isinstance(result["Al__mp-2"], Atoms)
    assert "Al" in result["Al__mp-2"].get_chemical_symbols()


def test_load_structures_folder_multiple(tmp_path):
    save_structures_cif(
        {"Al__mp-2": bulk("Al", "fcc", a=4.05), "Ni__mp-23": bulk("Ni", "fcc", a=3.52)},
        tmp_path,
    )
    result = load_structures_folder(tmp_path)
    assert "Al__mp-2" in result
    assert "Ni__mp-23" in result


def test_load_structures_folder_skips_unreadable_files(tmp_path):
    save_structures_cif({"Al__mp-2": bulk("Al", "fcc", a=4.05)}, tmp_path)
    (tmp_path / "garbage.cif").write_text("this is not valid CIF content!!!")
    result = load_structures_folder(tmp_path)
    assert "Al__mp-2" in result
    assert "garbage" not in result


def test_load_structures_folder_skips_subdirectories(tmp_path):
    save_structures_cif({"Al__mp-2": bulk("Al", "fcc", a=4.05)}, tmp_path)
    (tmp_path / "subdir").mkdir()
    result = load_structures_folder(tmp_path)
    assert "subdir" not in result


def test_load_structures_folder_empty_dir_returns_empty(tmp_path):
    result = load_structures_folder(tmp_path)
    assert result == {}


# ---------------------------------------------------------------------------
# fetch_structures — error handling
# ---------------------------------------------------------------------------


def test_fetch_structures_raises_import_error_when_mp_api_missing(tmp_path):
    with patch.dict(
        sys.modules,
        {
            "mp_api": None,
            "mp_api.client": None,
        },
    ):
        with pytest.raises(ImportError, match="pip install mp-api"):
            fetch_structures(["Al"], mp_api_key="key", cache_dir=str(tmp_path))


def test_fetch_structures_raises_value_error_without_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    mock_mp_client = MagicMock()
    with patch.dict(sys.modules, {"mp_api": MagicMock(), "mp_api.client": mock_mp_client}):
        with pytest.raises(ValueError, match="MP_API_KEY"):
            fetch_structures(["Al"], cache_dir=str(tmp_path))


def test_fetch_structures_accepts_env_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "env_test_key")
    al = bulk("Al", "fcc", a=4.05)
    docs = [_mock_doc("Al", "mp-2", "Al", al)]
    with _patch_mp_modules(docs, al):
        result = fetch_structures(["Al"], cache_dir=str(tmp_path))
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# fetch_structures — caching logic
# ---------------------------------------------------------------------------


def test_fetch_structures_uses_cache_avoids_mp_call(tmp_path):
    """Fully-cached chemsys folders require no API key at all."""
    al = bulk("Al", "fcc", a=4.05)
    save_structures_cif({"Al__mp-2": al}, tmp_path / "Al")

    # No mp_api_key — a download attempt would raise ValueError
    result = fetch_structures(["Al"], cache_dir=str(tmp_path))
    assert "Al__mp-2" in result
    assert isinstance(result["Al__mp-2"], Atoms)


def test_fetch_structures_returns_ase_atoms(tmp_path, monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    ni = bulk("Ni", "fcc", a=3.52)
    docs = [
        _mock_doc("Al", "mp-2", "Al", al),
        _mock_doc("Ni", "mp-23", "Ni", ni),
        _mock_doc("Al-Ni", "mp-81", "AlNi", al),
    ]
    with _patch_mp_modules(docs, al):
        result = fetch_structures(["Al", "Ni"], mp_api_key="key", cache_dir=str(tmp_path))
    assert len(result) > 0
    assert all(isinstance(a, Atoms) for a in result.values())


def test_fetch_structures_name_format(tmp_path, monkeypatch):
    """Returned keys follow the '{formula}__mp-{id}' convention."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    docs = [_mock_doc("Al", "mp-2", "Al", al)]
    with _patch_mp_modules(docs, al):
        result = fetch_structures(["Al"], mp_api_key="key", cache_dir=str(tmp_path))
    assert "Al__mp-2" in result


def test_fetch_structures_groups_by_chemsys_folder(tmp_path, monkeypatch):
    """Structures are saved into per-chemsys subfolders."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    ni = bulk("Ni", "fcc", a=3.52)
    docs = [
        _mock_doc("Al", "mp-2", "Al", al),
        _mock_doc("Ni", "mp-23", "Ni", ni),
    ]
    with _patch_mp_modules(docs, al):
        fetch_structures(["Al", "Ni"], mp_api_key="key", cache_dir=str(tmp_path))
    assert any((tmp_path / "Al").glob("*.cif"))
    assert any((tmp_path / "Ni").glob("*.cif"))


def test_fetch_structures_partial_cache_only_downloads_missing(tmp_path, monkeypatch):
    """Only uncached chemsys are included in the MP query."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    # Pre-cache "Al"
    save_structures_cif({"Al__mp-2": al}, tmp_path / "Al")

    ni = bulk("Ni", "fcc", a=3.52)
    docs = [
        _mock_doc("Ni", "mp-23", "Ni", ni),
        _mock_doc("Al-Ni", "mp-81", "AlNi", al),
    ]
    mock_mpr, mock_mp_client = _build_mocks(docs, al)
    with patch.dict(
        sys.modules,
        {
            "mp_api": MagicMock(),
            "mp_api.client": mock_mp_client,
        },
    ):
        result = fetch_structures(["Al", "Ni"], mp_api_key="key", cache_dir=str(tmp_path))

    queried = mock_mpr.summary.search.call_args[1]["chemsys"]
    assert "Al" not in queried
    assert "Ni" in queried
    # Cached Al structure is still in result
    assert "Al__mp-2" in result


def test_fetch_structures_force_redownload_queries_cached_chemsys(tmp_path, monkeypatch):
    """force_redownload=True calls MP even when the cache folder exists."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    save_structures_cif({"Al__mp-2": al}, tmp_path / "Al")

    docs = [_mock_doc("Al", "mp-2", "Al", al)]
    mock_mpr, mock_mp_client = _build_mocks(docs, al)
    with patch.dict(
        sys.modules,
        {
            "mp_api": MagicMock(),
            "mp_api.client": mock_mp_client,
        },
    ):
        fetch_structures(
            ["Al"], mp_api_key="key", cache_dir=str(tmp_path), force_redownload=True
        )

    queried = mock_mpr.summary.search.call_args[1]["chemsys"]
    assert "Al" in queried


def test_fetch_structures_caches_to_disk(tmp_path, monkeypatch):
    """After a download, CIF files exist on disk for reuse."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    docs = [_mock_doc("Al", "mp-2", "Al", al)]
    with _patch_mp_modules(docs, al):
        fetch_structures(["Al"], mp_api_key="key", cache_dir=str(tmp_path))
    assert any((tmp_path / "Al").glob("*.cif"))


def test_fetch_structures_second_call_uses_cache(tmp_path, monkeypatch):
    """A second call for the same elements reads from disk, no MP query."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    docs = [_mock_doc("Al", "mp-2", "Al", al)]

    mock_mpr, mock_mp_client = _build_mocks(docs, al)
    with patch.dict(
        sys.modules,
        {
            "mp_api": MagicMock(),
            "mp_api.client": mock_mp_client,
        },
    ):
        fetch_structures(["Al"], mp_api_key="key", cache_dir=str(tmp_path))
        assert mock_mpr.summary.search.call_count == 1
        # Second call — no MP key needed because cache is populated
        result2 = fetch_structures(["Al"], cache_dir=str(tmp_path))
        assert mock_mpr.summary.search.call_count == 1  # not called again

    assert "Al__mp-2" in result2


# ---------------------------------------------------------------------------
# fetch_structures — max_atoms filter
# ---------------------------------------------------------------------------


def test_fetch_structures_max_atoms_filters_loaded_structures(tmp_path, monkeypatch):
    """Structures with more atoms than max_atoms are excluded from the result."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    small = bulk("Al", "fcc", a=4.05)          # 1 atom
    large = bulk("Al", "fcc", a=4.05) * (2, 2, 2)  # 8 atoms
    save_structures_cif({"Al__mp-2": small, "Al__mp-1000": large}, tmp_path / "Al")

    result = fetch_structures(["Al"], max_atoms=4, cache_dir=str(tmp_path))
    assert "Al__mp-2" in result
    assert "Al__mp-1000" not in result


def test_fetch_structures_max_atoms_none_returns_all(tmp_path, monkeypatch):
    """max_atoms=None (default) imposes no filter."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    small = bulk("Al", "fcc", a=4.05)
    large = bulk("Al", "fcc", a=4.05) * (2, 2, 2)
    save_structures_cif({"Al__mp-2": small, "Al__mp-1000": large}, tmp_path / "Al")

    result = fetch_structures(["Al"], cache_dir=str(tmp_path))
    assert "Al__mp-2" in result
    assert "Al__mp-1000" in result


def test_fetch_structures_max_atoms_passed_to_mp_query(tmp_path, monkeypatch):
    """nsites is included in the MP query when max_atoms is set."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    docs = [_mock_doc("Al", "mp-2", "Al", al)]
    mock_mpr, mock_mp_client = _build_mocks(docs, al)

    with patch.dict(
        sys.modules,
        {"mp_api": MagicMock(), "mp_api.client": mock_mp_client},
    ):
        fetch_structures(["Al"], max_atoms=10, mp_api_key="key", cache_dir=str(tmp_path))

    call_kwargs = mock_mpr.summary.search.call_args[1]
    assert "num_sites" in call_kwargs
    assert call_kwargs["num_sites"] == (1, 10)


def test_fetch_structures_thermo_types_default(tmp_path, monkeypatch):
    """thermo_types defaults to GGA_GGA+U in the MP query."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    docs = [_mock_doc("Al", "mp-2", "Al", al)]
    mock_mpr, mock_mp_client = _build_mocks(docs, al)

    with patch.dict(
        sys.modules,
        {"mp_api": MagicMock(), "mp_api.client": mock_mp_client},
    ):
        fetch_structures(["Al"], mp_api_key="key", cache_dir=str(tmp_path))

    call_kwargs = mock_mpr.summary.search.call_args[1]
    assert call_kwargs["thermo_types"] == ["GGA_GGA+U"]


def test_fetch_structures_thermo_types_custom(tmp_path, monkeypatch):
    """Custom thermo_types are forwarded to the MP query."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    docs = [_mock_doc("Al", "mp-2", "Al", al)]
    mock_mpr, mock_mp_client = _build_mocks(docs, al)

    with patch.dict(
        sys.modules,
        {"mp_api": MagicMock(), "mp_api.client": mock_mp_client},
    ):
        fetch_structures(
            ["Al"], mp_api_key="key", cache_dir=str(tmp_path), thermo_types=["R2SCAN"]
        )

    call_kwargs = mock_mpr.summary.search.call_args[1]
    assert call_kwargs["thermo_types"] == ["R2SCAN"]


def test_fetch_structures_no_max_atoms_omits_nsites_from_query(tmp_path, monkeypatch):
    """nsites is not sent to MP when max_atoms is not set."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    docs = [_mock_doc("Al", "mp-2", "Al", al)]
    mock_mpr, mock_mp_client = _build_mocks(docs, al)

    with patch.dict(
        sys.modules,
        {"mp_api": MagicMock(), "mp_api.client": mock_mp_client},
    ):
        fetch_structures(["Al"], mp_api_key="key", cache_dir=str(tmp_path))

    call_kwargs = mock_mpr.summary.search.call_args[1]
    assert "num_sites" not in call_kwargs


def test_fetch_structures_max_atoms_exact_boundary(tmp_path, monkeypatch):
    """A structure with exactly max_atoms atoms is included."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    atoms_4 = bulk("Al", "fcc", a=4.05) * (2, 2, 1)  # 4 atoms
    save_structures_cif({"Al__mp-5": atoms_4}, tmp_path / "Al")

    result = fetch_structures(["Al"], max_atoms=4, cache_dir=str(tmp_path))
    assert "Al__mp-5" in result


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _mock_doc(chemsys, material_id, formula_pretty, atoms):
    """Create a minimal mock Materials Project document."""
    doc = MagicMock()
    doc.chemsys = chemsys
    doc.material_id = material_id
    doc.formula_pretty = formula_pretty
    doc.structure = MagicMock()
    doc.structure._atoms = atoms
    return doc


def _build_mocks(docs, fallback_atoms=None):
    """Return (mock_mpr, mock_mp_client) for a doc list.

    Each mock doc's ``.structure`` attribute is a plain object with the
    attributes that ``_pymatgen_structure_to_atoms`` reads, so no pymatgen
    import is required.
    """
    fallback = fallback_atoms or bulk("Al", "fcc", a=4.05)
    # Patch each doc's structure to look like a pymatgen Structure
    for doc in docs:
        atoms = getattr(doc.structure, "_atoms", fallback)
        doc.structure.lattice = MagicMock()
        doc.structure.lattice.matrix = atoms.get_cell()
        doc.structure.cart_coords = atoms.get_positions()
        doc.structure.__iter__ = MagicMock(
            return_value=iter(
                [_MockSite(sym) for sym in atoms.get_chemical_symbols()]
            )
        )

    mock_mpr = MagicMock()
    mock_mpr.__enter__ = MagicMock(return_value=mock_mpr)
    mock_mpr.__exit__ = MagicMock(return_value=False)
    mock_mpr.summary.search.return_value = docs

    mock_mp_client = MagicMock()
    mock_mp_client.MPRester.return_value = mock_mpr

    return mock_mpr, mock_mp_client


class _MockSite:
    """Minimal stand-in for a pymatgen PeriodicSite."""

    def __init__(self, symbol):
        self.specie = symbol

    def __str__(self):
        return self.specie


from contextlib import contextmanager


@contextmanager
def _patch_mp_modules(docs, fallback_atoms=None):
    """Context manager: patch sys.modules with a fully mocked MP backend."""
    _, mock_mp_client = _build_mocks(docs, fallback_atoms)
    with patch.dict(
        sys.modules,
        {
            "mp_api": MagicMock(),
            "mp_api.client": mock_mp_client,
        },
    ):
        yield


# ---------------------------------------------------------------------------
# helpers shared by fetch_mp_reference_df tests
# ---------------------------------------------------------------------------


def _mock_doc_with_energetics(
    chemsys,
    material_id,
    formula_pretty,
    atoms,
    formation_energy_per_atom=-0.42,
    energy_above_hull=0.0,
    energy_per_atom=-3.70,
):
    """Extend ``_mock_doc`` with MP energetics attributes."""
    doc = _mock_doc(chemsys, material_id, formula_pretty, atoms)
    doc.energy_per_atom = energy_per_atom
    doc.formation_energy_per_atom = formation_energy_per_atom
    doc.energy_above_hull = energy_above_hull
    return doc


@contextmanager
def _patch_mp_modules_energetics(docs, fallback_atoms=None):
    """Like ``_patch_mp_modules`` but for docs that include energetics."""
    _, mock_mp_client = _build_mocks(docs, fallback_atoms)
    with patch.dict(
        sys.modules,
        {
            "mp_api": MagicMock(),
            "mp_api.client": mock_mp_client,
        },
    ):
        yield


# ---------------------------------------------------------------------------
# fetch_mp_reference_df — basic return type
# ---------------------------------------------------------------------------


def test_fetch_mp_reference_df_returns_dataframe(tmp_path, monkeypatch):
    """Function returns a pandas DataFrame."""
    import pandas as pd

    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    docs = [_mock_doc_with_energetics("Al", "mp-2", "Al", al)]
    with _patch_mp_modules_energetics(docs, al):
        result = fetch_mp_reference_df(["Al"], mp_api_key="key", cache_dir=str(tmp_path))
    assert isinstance(result, pd.DataFrame)


def test_fetch_mp_reference_df_has_required_columns(tmp_path, monkeypatch):
    """DataFrame has all five expected columns."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    docs = [_mock_doc_with_energetics("Al", "mp-2", "Al", al)]
    with _patch_mp_modules_energetics(docs, al):
        result = fetch_mp_reference_df(["Al"], mp_api_key="key", cache_dir=str(tmp_path))
    for col in ("name", "ase_atoms", "e_per_atom", "e_formation_per_atom", "e_chull_dist_per_atom"):
        assert col in result.columns


# ---------------------------------------------------------------------------
# fetch_mp_reference_df — energetics values
# ---------------------------------------------------------------------------


def test_fetch_mp_reference_df_energetics_match_mp_data(tmp_path, monkeypatch):
    """All three energy columns match the MP document values."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    docs = [
        _mock_doc_with_energetics(
            "Al", "mp-2", "Al", al,
            formation_energy_per_atom=-0.55,
            energy_above_hull=0.03,
            energy_per_atom=-3.70,
        )
    ]
    with _patch_mp_modules_energetics(docs, al):
        result = fetch_mp_reference_df(["Al"], mp_api_key="key", cache_dir=str(tmp_path))
    row = result[result["name"] == "Al__mp-2"].iloc[0]
    assert abs(row["e_per_atom"] - (-3.70)) < 1e-9
    assert abs(row["e_formation_per_atom"] - (-0.55)) < 1e-9
    assert abs(row["e_chull_dist_per_atom"] - 0.03) < 1e-9


def test_fetch_mp_reference_df_ase_atoms_column_contains_atoms(tmp_path, monkeypatch):
    """ase_atoms column entries are ASE Atoms instances."""
    from ase import Atoms

    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    docs = [_mock_doc_with_energetics("Al", "mp-2", "Al", al)]
    with _patch_mp_modules_energetics(docs, al):
        result = fetch_mp_reference_df(["Al"], mp_api_key="key", cache_dir=str(tmp_path))
    assert all(isinstance(a, Atoms) for a in result["ase_atoms"])


# ---------------------------------------------------------------------------
# fetch_mp_reference_df — caching logic
# ---------------------------------------------------------------------------


def test_fetch_mp_reference_df_uses_metadata_cache(tmp_path, monkeypatch):
    """When metadata.json is present, no MP API call is made."""
    al = bulk("Al", "fcc", a=4.05)
    folder = tmp_path / "Al"
    save_structures_cif({"Al__mp-2": al}, folder)
    _save_metadata(
        {"Al__mp-2": {"e_per_atom": -3.7, "e_formation_per_atom": -0.42, "e_chull_dist_per_atom": 0.0}},
        folder,
    )

    # No api_key — a download attempt would raise ValueError
    result = fetch_mp_reference_df(["Al"], cache_dir=str(tmp_path))
    assert "Al__mp-2" in result["name"].values


def test_fetch_mp_reference_df_downloads_when_no_metadata(tmp_path, monkeypatch):
    """If CIFs exist but metadata.json is absent, MP is still queried."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    # Pre-populate CIF only — no metadata.json
    save_structures_cif({"Al__mp-2": al}, tmp_path / "Al")

    docs = [_mock_doc_with_energetics("Al", "mp-2", "Al", al)]
    mock_mpr, mock_mp_client = _build_mocks(docs, al)
    # Attach energetics to the mock docs
    for doc in docs:
        doc.formation_energy_per_atom = -0.42
        doc.energy_above_hull = 0.0

    with patch.dict(
        sys.modules,
        {"mp_api": MagicMock(), "mp_api.client": mock_mp_client},
    ):
        fetch_mp_reference_df(["Al"], mp_api_key="key", cache_dir=str(tmp_path))

    assert mock_mpr.summary.search.call_count == 1


def test_fetch_mp_reference_df_force_redownload(tmp_path, monkeypatch):
    """force_redownload=True queries MP even when metadata.json is cached."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    folder = tmp_path / "Al"
    save_structures_cif({"Al__mp-2": al}, folder)
    _save_metadata(
        {"Al__mp-2": {"e_formation_per_atom": -0.42, "e_chull_dist_per_atom": 0.0}},
        folder,
    )

    docs = [_mock_doc_with_energetics("Al", "mp-2", "Al", al)]
    mock_mpr, mock_mp_client = _build_mocks(docs, al)
    for doc in docs:
        doc.formation_energy_per_atom = -0.42
        doc.energy_above_hull = 0.0

    with patch.dict(
        sys.modules,
        {"mp_api": MagicMock(), "mp_api.client": mock_mp_client},
    ):
        fetch_mp_reference_df(
            ["Al"], mp_api_key="key", cache_dir=str(tmp_path), force_redownload=True
        )

    assert mock_mpr.summary.search.call_count == 1


# ---------------------------------------------------------------------------
# fetch_mp_reference_df — max_atoms filter
# ---------------------------------------------------------------------------


def test_fetch_mp_reference_df_max_atoms_filters(tmp_path):
    """max_atoms excludes structures with too many atoms."""
    small = bulk("Al", "fcc", a=4.05)           # 1 atom
    large = bulk("Al", "fcc", a=4.05) * (2, 2, 2)  # 8 atoms
    folder = tmp_path / "Al"
    save_structures_cif({"Al__mp-2": small, "Al__mp-1000": large}, folder)
    _save_metadata(
        {
            "Al__mp-2": {"e_per_atom": -3.7, "e_formation_per_atom": -0.42, "e_chull_dist_per_atom": 0.0},
            "Al__mp-1000": {"e_per_atom": -3.5, "e_formation_per_atom": -0.10, "e_chull_dist_per_atom": 0.05},
        },
        folder,
    )

    result = fetch_mp_reference_df(["Al"], max_atoms=4, cache_dir=str(tmp_path))
    assert "Al__mp-2" in result["name"].values
    assert "Al__mp-1000" not in result["name"].values


def test_fetch_mp_reference_df_max_atoms_none_returns_all(tmp_path):
    """max_atoms=None (default) applies no filter."""
    small = bulk("Al", "fcc", a=4.05)
    large = bulk("Al", "fcc", a=4.05) * (2, 2, 2)
    folder = tmp_path / "Al"
    save_structures_cif({"Al__mp-2": small, "Al__mp-1000": large}, folder)
    _save_metadata(
        {
            "Al__mp-2": {"e_per_atom": -3.7, "e_formation_per_atom": -0.42, "e_chull_dist_per_atom": 0.0},
            "Al__mp-1000": {"e_per_atom": -3.5, "e_formation_per_atom": -0.10, "e_chull_dist_per_atom": 0.05},
        },
        folder,
    )

    result = fetch_mp_reference_df(["Al"], cache_dir=str(tmp_path))
    assert "Al__mp-2" in result["name"].values
    assert "Al__mp-1000" in result["name"].values


# ---------------------------------------------------------------------------
# fetch_mp_reference_df — metadata.json written to disk
# ---------------------------------------------------------------------------


def test_fetch_mp_reference_df_saves_metadata_json(tmp_path, monkeypatch):
    """After a download, metadata.json is written to the chemsys subfolder."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    docs = [_mock_doc_with_energetics("Al", "mp-2", "Al", al, -0.42, 0.0)]
    with _patch_mp_modules_energetics(docs, al):
        fetch_mp_reference_df(["Al"], mp_api_key="key", cache_dir=str(tmp_path))
    assert (tmp_path / "Al" / _METADATA_FILENAME).exists()


def test_fetch_mp_reference_df_metadata_json_content(tmp_path, monkeypatch):
    """The written metadata.json maps the structure name to all three energetics."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    docs = [_mock_doc_with_energetics("Al", "mp-2", "Al", al, -0.42, 0.0, -3.70)]
    with _patch_mp_modules_energetics(docs, al):
        fetch_mp_reference_df(["Al"], mp_api_key="key", cache_dir=str(tmp_path))
    meta = _load_metadata(tmp_path / "Al")
    assert "Al__mp-2" in meta
    assert abs(meta["Al__mp-2"]["e_per_atom"] - (-3.70)) < 1e-9
    assert abs(meta["Al__mp-2"]["e_formation_per_atom"] - (-0.42)) < 1e-9
    assert abs(meta["Al__mp-2"]["e_chull_dist_per_atom"] - 0.0) < 1e-9


def test_fetch_mp_reference_df_thermo_types_default(tmp_path, monkeypatch):
    """thermo_types defaults to GGA_GGA+U in the MP query."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    docs = [_mock_doc_with_energetics("Al", "mp-2", "Al", al)]
    mock_mpr, mock_mp_client = _build_mocks(docs, al)
    for doc in docs:
        doc.energy_per_atom = -3.70
        doc.formation_energy_per_atom = -0.42
        doc.energy_above_hull = 0.0

    with patch.dict(
        sys.modules,
        {"mp_api": MagicMock(), "mp_api.client": mock_mp_client},
    ):
        fetch_mp_reference_df(["Al"], mp_api_key="key", cache_dir=str(tmp_path))

    call_kwargs = mock_mpr.summary.search.call_args[1]
    assert call_kwargs["thermo_types"] == ["GGA_GGA+U"]


def test_fetch_mp_reference_df_thermo_types_custom(tmp_path, monkeypatch):
    """Custom thermo_types are forwarded to the MP query."""
    monkeypatch.delenv("MP_API_KEY", raising=False)
    al = bulk("Al", "fcc", a=4.05)
    docs = [_mock_doc_with_energetics("Al", "mp-2", "Al", al)]
    mock_mpr, mock_mp_client = _build_mocks(docs, al)
    for doc in docs:
        doc.energy_per_atom = -3.70
        doc.formation_energy_per_atom = -0.42
        doc.energy_above_hull = 0.0

    with patch.dict(
        sys.modules,
        {"mp_api": MagicMock(), "mp_api.client": mock_mp_client},
    ):
        fetch_mp_reference_df(
            ["Al"], mp_api_key="key", cache_dir=str(tmp_path), thermo_types=["R2SCAN"]
        )

    call_kwargs = mock_mpr.summary.search.call_args[1]
    assert call_kwargs["thermo_types"] == ["R2SCAN"]


# ---------------------------------------------------------------------------
# fetch_mp_reference_df — graceful handling of missing metadata
# ---------------------------------------------------------------------------


def test_fetch_mp_reference_df_missing_metadata_entry_gives_nan(tmp_path):
    """A structure with no metadata entry has NaN in the energy columns."""
    import math

    al = bulk("Al", "fcc", a=4.05)
    folder = tmp_path / "Al"
    save_structures_cif({"Al__mp-2": al}, folder)
    # metadata.json exists but is missing the key for Al__mp-2
    _save_metadata({}, folder)

    result = fetch_mp_reference_df(["Al"], cache_dir=str(tmp_path))
    row = result[result["name"] == "Al__mp-2"].iloc[0]
    assert math.isnan(row["e_formation_per_atom"])
    assert math.isnan(row["e_chull_dist_per_atom"])
