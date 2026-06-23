"""Integrity spot-checks against known published values.

All tests require data files and are marked with @pytest.mark.data.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest


@pytest.mark.data
def test_cu63_abundance(data_dir_path: Path) -> None:
    """Cu-63 natural abundance should be ~69.15%."""
    db = duckdb.connect()
    path = data_dir_path / "meta" / "abundances.parquet"
    result = db.sql(f"SELECT abundance FROM read_parquet('{path}') WHERE Z=29 AND A=63").fetchone()
    assert result is not None, "Cu-63 not found in abundances"
    assert result[0] == pytest.approx(0.6915, abs=0.005)


@pytest.mark.data
def test_co60_half_life(data_dir_path: Path) -> None:
    """Co-60 (ground state) half-life should be ~5.2714 years = ~1.663e8 s.

    Filter `state=''` explicitly — Co-60m also exists (~10.5 min) and sharing
    `(Z, A)` would make an unfiltered `LIMIT 1` non-deterministic.
    """
    db = duckdb.connect()
    path = data_dir_path / "meta" / "decay.parquet"
    result = db.sql(
        f"SELECT half_life_s FROM read_parquet('{path}') WHERE Z=27 AND A=60 AND state='' LIMIT 1"
    ).fetchone()
    assert result is not None, "Co-60 ground state not found in decay data"
    expected_s = 5.2714 * 365.25 * 24 * 3600  # ~1.663e8 s
    assert result[0] == pytest.approx(expected_s, rel=0.05)


@pytest.mark.data
def test_proton_stopping_cu(data_dir_path: Path) -> None:
    """PSTAR stopping power for protons in Cu at 10 MeV: NIST ICRU-49 value ≈ 27."""
    db = duckdb.connect()
    path = data_dir_path / "stopping" / "PSTAR.parquet"
    result = db.sql(
        f"SELECT dedx FROM read_parquet('{path}') "
        "WHERE target_Z=29 AND energy_MeV BETWEEN 9.0 AND 11.0 "
        "ORDER BY ABS(energy_MeV - 10.0) LIMIT 1"
    ).fetchone()
    assert result is not None, "No PSTAR data for Cu near 10 MeV"
    assert 20.0 < result[0] < 35.0, f"Unexpected stopping power: {result[0]}"


@pytest.mark.data
def test_element_symbols(data_dir_path: Path) -> None:
    """Spot-check element symbols."""
    db = duckdb.connect()
    path = data_dir_path / "meta" / "elements.parquet"
    checks = {1: "H", 6: "C", 26: "Fe", 29: "Cu", 79: "Au", 92: "U"}
    for z, expected_sym in checks.items():
        result = db.sql(f"SELECT symbol FROM read_parquet('{path}') WHERE Z={z}").fetchone()
        assert result is not None, f"Z={z} not found"
        assert result[0] == expected_sym, f"Z={z}: expected {expected_sym}, got {result[0]}"


@pytest.mark.data
def test_cu63_xs_tendl(data_dir_path: Path) -> None:
    """TENDL-2024 should have cross-section data for Cu-63(p,n)Zn-63."""
    xs_path = data_dir_path / "tendl-2024" / "xs" / "p_Cu.parquet"
    if not xs_path.exists():
        pytest.skip("TENDL-2024 p_Cu.parquet not present")
    db = duckdb.connect()
    result = db.sql(
        f"SELECT COUNT(*), MAX(xs_mb) FROM read_parquet('{xs_path}') "
        "WHERE target_A=63 AND residual_Z=30 AND residual_A=63"
    ).fetchone()
    count, max_xs = result
    assert count > 10, f"Expected >10 data points, got {count}"
    assert max_xs > 100, f"Expected max xs > 100 mb, got {max_xs}"


@pytest.mark.data
def test_spectrum_xs_thermal_cu63(data_dir_path: Path) -> None:
    """Cu-63(n,γ) thermal spectrum-averaged XS from activation libraries should be ~3500–5500 mb."""
    db = duckdb.connect()
    path = data_dir_path / "meta" / "spectrum_xs.parquet"
    if not path.exists():
        pytest.skip("spectrum_xs.parquet not built")
    result = db.sql(
        f"SELECT xs_avg_mb FROM read_parquet('{path}') "
        "WHERE target_Z=29 AND target_A=63 AND residual_Z=29 AND residual_A=64 "
        "AND spectrum='thermal' LIMIT 1"
    ).fetchone()
    if result is None:
        pytest.skip("Cu-63(n,γ) thermal XS not available (no activation library data)")
    assert 3000 < result[0] < 6000, f"Unexpected thermal XS: {result[0]} mb"


@pytest.mark.data
def test_neutron_total_xs_sanity():
    """Cu-63 total XS at ~1 MeV should be ~4-5 barn (4000-5000 mb)."""
    import nucl_parquet

    db = nucl_parquet.connect()
    row = db.sql(
        "SELECT xs_total_mb FROM neutron_total "
        "WHERE Z=29 AND A=63 AND energy_MeV BETWEEN 0.9 AND 1.1 "
        "ORDER BY energy_MeV LIMIT 1"
    ).fetchone()
    assert row is not None and 2000 < row[0] < 15000, f"Unexpected Cu-63 total XS: {row}"


@pytest.mark.data
def test_no_talys_sentinels(data_dir_path: Path) -> None:
    """No xs library should contain TALYS overflow sentinel values (>1e10 mb).

    TALYS emits ~1.99e38 (near FLT_MAX) when a reaction is undefined or the
    calculation diverges.  These must be removed at ingestion; any recurrence
    here means a new library was added without cleaning.
    """
    db = duckdb.connect()
    for lib_dir in sorted(data_dir_path.iterdir()):
        xs_dir = lib_dir / "xs"
        if not xs_dir.is_dir():
            continue
        parquets = list(xs_dir.glob("*.parquet"))
        if not parquets:
            continue
        glob = str(xs_dir / "*.parquet")
        # Threshold 1e30 mb safely separates TALYS overflow sentinels
        # (~1.99e38, near FLT_MAX) from the largest legitimate cross-sections
        # (~90 Gb = 9e10 mb, e.g. Xe-135 neutron capture at cold energies).
        n = db.sql(
            f"SELECT COUNT(*) FROM read_parquet('{glob}') WHERE xs_mb > 1e30 OR isnan(xs_mb) OR xs_mb IS NULL"
        ).fetchone()[0]
        assert n == 0, (
            f"{lib_dir.name}: found {n} invalid xs_mb rows (sentinel >1e30, NaN, or NULL) — clean at parquet level"
        )


@pytest.mark.data
def test_catima_straggling(data_dir_path: Path) -> None:
    """Federated catima shards should have a positive straggling column."""
    import polars as pl

    df = pl.read_parquet(str(data_dir_path / "stopping" / "catima_*.parquet"))
    assert "straggling" in df.columns, "straggling column missing"
    # C-12 in Cu: straggling should be positive
    row = df.filter((pl.col("proj_Z") == 6) & (pl.col("target_Z") == 29)).head(1)
    assert row["straggling"][0] > 0, "Expected positive straggling for C-12 in Cu"
