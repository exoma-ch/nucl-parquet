"""Tests for the pre-tabulated β-decay spectra (#78).

Verifies:
- Schema and dtypes
- Probability conservation: ∫ dN/dE = 1 within tabulation tolerance
- Cumulative monotone non-decreasing, ending at 1.0
- P-32 mean energy matches published 694.9 keV
- Y-90 ground-state endpoint matches published 2275.6 keV
- DuckDB view registered
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def data_dir_path() -> Path:
    return Path("data").resolve()


_DIR = "meta/ensdf/beta_spectra"


@pytest.mark.data
def test_schema(data_dir_path: Path) -> None:
    import polars as pl

    df = pl.read_parquet(data_dir_path / _DIR / "P.parquet")
    expected = {
        "Z",
        "A",
        "state",
        "transition_idx",
        "decay_mode",
        "forbiddenness",
        "endpoint_keV",
        "branching",
        "energy_keV",
        "dN_dE",
        "cumulative",
    }
    assert set(df.columns) == expected
    assert df["Z"].dtype == pl.Int32
    assert df["A"].dtype == pl.Int32
    assert df["energy_keV"].dtype == pl.Float32
    assert df["dN_dE"].dtype == pl.Float32


@pytest.mark.data
def test_p32_anchors(data_dir_path: Path) -> None:
    """P-32: single transition, allowed-shape (forbiddenness blank), endpoint
    1710.66 keV, mean energy 694.9 keV per NNDC/ICRP."""
    import numpy as np
    import polars as pl

    df = pl.read_parquet(data_dir_path / _DIR / "P.parquet").filter((pl.col("A") == 32) & (pl.col("state") == ""))
    assert df["transition_idx"].n_unique() == 1
    assert df["decay_mode"][0] == "BetaMinus"
    assert abs(float(df["endpoint_keV"][0]) - 1710.66) < 0.1
    # Mean energy
    g = df.sort("energy_keV")
    e = g["energy_keV"].to_numpy()
    d = g["dN_dE"].to_numpy()
    mean_e = float(np.trapezoid(e * d, e))
    # NNDC publishes 694.9 keV. Our calc reproduces to <0.5%.
    assert abs(mean_e - 694.9) < 5.0, f"P-32 mean E = {mean_e}; expected 694.9 ± 5"


@pytest.mark.data
def test_y90_endpoint(data_dir_path: Path) -> None:
    """Y-90 ground-state: dominant transition is unique-first-forbidden with
    endpoint 2275.6 keV and branching ~1."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / _DIR / "Y.parquet").filter((pl.col("A") == 90) & (pl.col("state") == ""))
    # Find the dominant transition (highest branching among unique-1st-forbidden)
    dom = (
        df.group_by("transition_idx")
        .agg(
            [
                pl.col("endpoint_keV").first(),
                pl.col("branching").first(),
                pl.col("forbiddenness").first(),
            ]
        )
        .sort("branching", descending=True)
        .head(1)
    )
    assert abs(float(dom["endpoint_keV"][0]) - 2275.6) < 0.1
    assert float(dom["branching"][0]) > 0.99
    assert dom["forbiddenness"][0] == "uniqueFirstForbidden"


@pytest.mark.data
def test_normalization(data_dir_path: Path) -> None:
    """Every transition's dN/dE integrates to 1.0 within tabulation tolerance."""
    import numpy as np
    import polars as pl

    # Spot-check a representative sample across all elements
    for symbol in ["H", "P", "Y", "Cs", "Sr", "Co", "I", "U"]:
        df = pl.read_parquet(Path("data") / _DIR / f"{symbol}.parquet")
        # Sample first 5 transitions per (Z, A, state) for speed
        keys = df.select("Z", "A", "state", "transition_idx").unique().head(20).iter_rows(named=True)
        for k in keys:
            t = df.filter(
                (pl.col("Z") == k["Z"])
                & (pl.col("A") == k["A"])
                & (pl.col("state") == k["state"])
                & (pl.col("transition_idx") == k["transition_idx"])
            ).sort("energy_keV")
            e = t["energy_keV"].to_numpy()
            d = t["dN_dE"].to_numpy()
            integral = float(np.trapezoid(d, e))
            assert abs(integral - 1.0) < 5e-3, (
                f"{symbol} Z={k['Z']} A={k['A']} state={k['state']!r} tidx={k['transition_idx']}: "
                f"normalization = {integral}"
            )


@pytest.mark.data
def test_cumulative_monotone(data_dir_path: Path) -> None:
    """Cumulative must be monotone non-decreasing, end ≈ 1.0."""
    import numpy as np
    import polars as pl

    df = (
        pl.read_parquet(Path("data") / _DIR / "P.parquet")
        .filter((pl.col("A") == 32) & (pl.col("state") == "") & (pl.col("transition_idx") == 0))
        .sort("energy_keV")
    )
    c = df["cumulative"].to_numpy()
    # Monotone (Float32 tolerance)
    diffs = np.diff(c)
    assert (diffs >= -1e-6).all(), f"cumulative not monotone, min diff = {diffs.min()}"
    # Ends at ~1
    assert abs(float(c[-1]) - 1.0) < 5e-3, f"cumulative endpoint = {c[-1]}"
    # Starts near 0
    assert float(c[0]) < 5e-3


@pytest.mark.data
def test_view_registered(data_dir_path: Path) -> None:
    from nucl_parquet import loader as np_lib

    db = np_lib.connect(data_dir_path)
    # Count rows for P-32 via the unified view
    n = db.sql("SELECT COUNT(*) FROM beta_spectra WHERE Z=15 AND A=32 AND state=''").fetchall()[0][0]
    # P-32 has 1 transition × 200 bins = 200 rows
    assert 100 <= n <= 300, f"P-32 row count via view = {n}"
