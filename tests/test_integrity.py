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
    path = data_dir_path / "stopping" / "stopping.parquet"
    result = db.sql(
        f"SELECT dedx FROM read_parquet('{path}') "
        "WHERE source='PSTAR' AND target_Z=29 "
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
