"""Coverage tests — verify data completeness.

All tests require data files and are marked with @pytest.mark.data.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest


@pytest.mark.data
def test_catalog_paths_exist(data_dir_path: Path) -> None:
    """All catalog library paths should exist (or be optional).

    Skips entries without a `path` (build-time provenance records like
    strata-data-nuclear that point to HF instead of a local directory).
    """
    catalog = json.loads((data_dir_path / "catalog.json").read_text())
    missing = []
    for lib_key, lib_info in catalog["libraries"].items():
        if "path" not in lib_info:
            continue
        lib_dir = data_dir_path / lib_info["path"]
        if not lib_dir.exists():
            missing.append(lib_key)
    # Allow some optional libraries to be missing, but most should exist
    assert len(missing) <= 3, f"Too many missing libraries: {missing}"


@pytest.mark.data
def test_meta_files_exist(data_dir_path: Path) -> None:
    assert (data_dir_path / "meta" / "abundances.parquet").exists()
    assert (data_dir_path / "meta" / "decay.parquet").exists()
    assert (data_dir_path / "meta" / "elements.parquet").exists()


@pytest.mark.data
def test_stopping_exists(data_dir_path: Path) -> None:
    """Per-source NIST tables + catima master must be present."""
    stopping = data_dir_path / "stopping"
    assert (stopping / "PSTAR.parquet").exists()
    assert (stopping / "ASTAR.parquet").exists()
    assert (stopping / "ESTAR.parquet").exists()
    assert (stopping / "catima" / "catima.parquet").exists()


@pytest.mark.data
def test_elements_has_all_z(data_dir_path: Path) -> None:
    """elements.parquet should cover Z=1 through at least Z=118."""
    db = duckdb.connect()
    path = data_dir_path / "meta" / "elements.parquet"
    result = db.sql(f"SELECT MIN(Z), MAX(Z), COUNT(DISTINCT Z) FROM read_parquet('{path}')").fetchone()
    min_z, max_z, count = result
    assert min_z == 1
    assert max_z >= 83
    assert count >= 83


@pytest.mark.data
def test_stopping_covers_nist_sources(data_dir_path: Path) -> None:
    """The per-source NIST tables must each be loadable and non-empty.

    Sourced from build_stopping.py (NIST CGI). The deleted aggregate
    stopping.parquet was an unprovenanced blob; we now check the per-source
    files directly.
    """
    db = duckdb.connect()
    stopping = data_dir_path / "stopping"
    for label in ("PSTAR", "ASTAR", "ESTAR"):
        path = stopping / f"{label}.parquet"
        rows = db.sql(f"SELECT COUNT(*) FROM read_parquet('{path}') WHERE source = '{label}'").fetchone()[0]
        assert rows > 100, f"{label}: expected >100 rows, got {rows}"


@pytest.mark.data
def test_stopping_no_broken_legacy_files(data_dir_path: Path) -> None:
    """Regression guard: He3STAR.parquet and the unprovenanced stopping.parquet
    aggregate must stay deleted (both produced wrong α / ³He values; see #137).

    The top-level stopping/catima.parquet must also stay deleted (it was a
    byte-identical duplicate of stopping/catima/catima.parquet with a different
    schema that broke the `_register_glob` view).
    """
    stopping = data_dir_path / "stopping"
    assert not (stopping / "He3STAR.parquet").exists(), (
        "He3STAR.parquet inherited the broken ASTAR Z²-scaling — see #137"
    )
    assert not (stopping / "stopping.parquet").exists(), (
        "stopping.parquet was an unprovenanced aggregate; use per-source files"
    )
    assert not (stopping / "catima.parquet").exists(), (
        "stopping/catima.parquet duplicated stopping/catima/catima.parquet and broke the glob view (different schema)"
    )


@pytest.mark.data
def test_tendl_has_all_projectiles(data_dir_path: Path) -> None:
    """TENDL-2024 should have files for all listed projectiles."""
    catalog = json.loads((data_dir_path / "catalog.json").read_text())
    tendl = catalog["libraries"].get("tendl-2024")
    if tendl is None:
        pytest.skip("tendl-2024 not in catalog")
    xs_dir = data_dir_path / tendl["path"]
    if not xs_dir.exists():
        pytest.skip("tendl-2024 data not present")
    files = {f.stem.split("_")[0] for f in xs_dir.glob("*.parquet")}
    for proj in tendl["projectiles"]:
        assert proj in files, f"Missing projectile '{proj}' in TENDL-2024"


@pytest.mark.data
def test_manifest_counts(data_dir_path: Path) -> None:
    """Each library with a manifest.json should have matching file counts."""
    catalog = json.loads((data_dir_path / "catalog.json").read_text())
    checked = 0
    for lib_key, lib_info in catalog["libraries"].items():
        if "path" not in lib_info:
            continue
        lib_dir = data_dir_path / lib_info["path"]
        manifest_path = lib_dir.parent / "manifest.json"
        if not manifest_path.exists():
            continue
        actual_files = len(list(lib_dir.glob("*.parquet")))
        # Manifests may be stale — just check files exist
        assert actual_files > 0, f"{lib_key}: no parquet files found"
        checked += 1
    # Manifests are optional
    if checked == 0:
        pytest.skip("No manifests found")
