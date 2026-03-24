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
    """Co-60 half-life should be ~5.2714 years = ~1.663e8 s."""
    db = duckdb.connect()
    path = data_dir_path / "meta" / "decay.parquet"
    result = db.sql(f"SELECT half_life_s FROM read_parquet('{path}') WHERE Z=27 AND A=60 LIMIT 1").fetchone()
    assert result is not None, "Co-60 not found in decay data"
    expected_s = 5.2714 * 365.25 * 24 * 3600  # ~1.663e8 s
    assert result[0] == pytest.approx(expected_s, rel=0.05)


@pytest.mark.data
def test_proton_stopping_cu(data_dir_path: Path) -> None:
    """PSTAR stopping power for protons in Cu at 10 MeV should be ~30-60 MeV cm2/g."""
    db = duckdb.connect()
    path = data_dir_path / "stopping" / "PSTAR.parquet"
    result = db.sql(
        f"SELECT dedx FROM read_parquet('{path}') "
        "WHERE target_Z=29 "
        "AND energy_MeV BETWEEN 9.0 AND 11.0 "
        "ORDER BY ABS(energy_MeV - 10.0) LIMIT 1"
    ).fetchone()
    assert result is not None, "No PSTAR data for Cu near 10 MeV"
    assert 20.0 < result[0] < 80.0, f"Unexpected stopping power: {result[0]}"


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
def test_no_talys_sentinels(data_dir_path: Path) -> None:
    """No xs_mb values should be TALYS overflow sentinels (~1.99e38) or NaN/NULL."""
    db = duckdb.connect()
    for lib_dir in sorted(data_dir_path.iterdir()):
        xs_dir = lib_dir / "xs"
        if not xs_dir.is_dir():
            continue
        parquets = list(xs_dir.glob("*.parquet"))
        if not parquets:
            continue
        glob = str(xs_dir / "*.parquet")
        # 1e30 mb separates TALYS sentinels (~1.99e38) from legitimate values
        # (largest known physical xs: Xe-135 thermal capture ~90 Gb = 9e10 mb)
        n = db.sql(
            f"SELECT COUNT(*) FROM read_parquet('{glob}') "
            "WHERE xs_mb > 1e30 OR isnan(xs_mb) OR xs_mb IS NULL"
        ).fetchone()[0]
        assert n == 0, (
            f"{lib_dir.name}: found {n} invalid xs_mb rows "
            "(sentinel >1e30, NaN, or NULL)"
        )


@pytest.mark.data
def test_hi_xs_prod_coverage(data_dir_path: Path) -> None:
    """Geant4 fragment production XS should cover all 6 projectiles × 92 targets."""
    xs_dir = data_dir_path / "hi-xs-prod" / "xs"
    if not xs_dir.is_dir():
        pytest.skip("hi-xs-prod not present")
    db = duckdb.connect()
    glob = str(xs_dir / "*.parquet")

    # All 552 files should be present
    n_files = db.sql(
        f"SELECT COUNT(DISTINCT filename) FROM read_parquet('{glob}', filename=true)"
    ).fetchone()[0]
    assert n_files == 552, f"Expected 552 files (6 proj × 92 targets), got {n_files}"

    # No NULL/NaN/negative/sentinel values
    n_bad = db.sql(
        f"SELECT COUNT(*) FROM read_parquet('{glob}') "
        "WHERE xs_mb IS NULL OR isnan(xs_mb) OR xs_mb < 0 OR xs_mb > 1e10"
    ).fetchone()[0]
    assert n_bad == 0, f"Found {n_bad} invalid xs_mb rows in hi-xs-prod"

    # Spot-check: C12 on Cu should produce He-4 fragments with xs > 100 mb
    c12_cu = xs_dir / "c12_Cu.parquet"
    result = db.sql(
        f"SELECT MAX(xs_mb) FROM read_parquet('{c12_cu}') "
        "WHERE residual_Z=2 AND residual_A=4"
    ).fetchone()
    assert result is not None and result[0] is not None, "No He-4 production in C12+Cu"
    assert result[0] > 100, f"C12+Cu He-4 max xs={result[0]:.1f} mb, expected >100 mb"


@pytest.mark.data
def test_light_ion_stopping_velocity_scaling(data_dir_path: Path) -> None:
    """dSTAR/tSTAR/He3STAR should match PSTAR/ASTAR at the same velocity."""
    db = duckdb.connect()
    stopping_dir = data_dir_path / "stopping"
    pstar_path = stopping_dir / "PSTAR.parquet"
    dstar_path = stopping_dir / "dSTAR.parquet"
    if not pstar_path.exists() or not dstar_path.exists():
        pytest.skip("PSTAR.parquet or dSTAR.parquet not present")

    # Deuteron at 20 MeV == proton at 10 MeV (same velocity)
    p10 = db.sql(
        f"SELECT dedx FROM read_parquet('{pstar_path}') "
        "WHERE target_Z=29 AND energy_MeV=10.0"
    ).fetchone()
    d20 = db.sql(
        f"SELECT dedx FROM read_parquet('{dstar_path}') "
        "WHERE target_Z=29 AND energy_MeV=20.0"
    ).fetchone()
    assert p10 is not None and d20 is not None, "Missing PSTAR/dSTAR Cu data"
    assert p10[0] == pytest.approx(d20[0], rel=1e-9), (
        f"dSTAR velocity scaling broken: p@10={p10[0]}, d@20={d20[0]}"
    )


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
