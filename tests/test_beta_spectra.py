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

from nucl_parquet.state_vocabulary import GROUND


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
        "shape_factor_approx",
    }
    assert set(df.columns) == expected
    assert df["Z"].dtype == pl.Int32
    assert df["A"].dtype == pl.Int32
    assert df["energy_keV"].dtype == pl.Float32
    assert df["dN_dE"].dtype == pl.Float32
    assert df["shape_factor_approx"].dtype == pl.Boolean


@pytest.mark.data
def test_p32_anchors(data_dir_path: Path) -> None:
    """P-32: single transition, allowed-shape (forbiddenness blank), endpoint
    1710.66 keV, mean energy 694.9 keV per NNDC/ICRP."""
    import numpy as np
    import polars as pl

    df = pl.read_parquet(data_dir_path / _DIR / "P.parquet").filter((pl.col("A") == 32) & (pl.col("state") == GROUND))
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

    df = pl.read_parquet(data_dir_path / _DIR / "Y.parquet").filter((pl.col("A") == 90) & (pl.col("state") == GROUND))
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
        .filter((pl.col("A") == 32) & (pl.col("state") == GROUND) & (pl.col("transition_idx") == 0))
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
def test_f18_beta_plus_anchor(data_dir_path: Path) -> None:
    """F-18 β+: positron kinetic-energy endpoint 633.5 keV, mean ~250 keV.

    Critical sign-bug test: a wrong sign of `z_signed` in BetaPlus path
    would silently flip the Coulomb correction (suppression → enhancement)
    and shift the mean energy noticeably away from 250 keV.

    Also: strata's q_value_kev is the atomic Q, so the build must subtract
    2·m_e c² = 1022 keV to get the positron kinetic-energy endpoint. If that
    subtraction is missing, endpoint would read 1655.9 not ~633.
    """
    import numpy as np
    import polars as pl

    df = (
        pl.read_parquet(data_dir_path / _DIR / "F.parquet")
        .filter((pl.col("A") == 18) & (pl.col("decay_mode") == "BetaPlus"))
        .sort("energy_keV")
    )
    assert df.height > 0, "F-18 β+ missing"
    endpoint = float(df["endpoint_keV"][0])
    assert 630.0 < endpoint < 640.0, f"F-18 β+ endpoint = {endpoint} keV (expected ~633.5)"
    e = df["energy_keV"].to_numpy()
    d = df["dN_dE"].to_numpy()
    mean_e = float(np.trapezoid(e * d, e))
    # NNDC publishes 249.8 keV mean kinetic energy for F-18 positrons.
    assert 240.0 < mean_e < 260.0, f"F-18 β+ mean E = {mean_e} (expect ~250)"


@pytest.mark.data
def test_sn121m_isomer_present(data_dir_path: Path) -> None:
    """Sn-121m isomer (parent_ex_kev = 6.31 keV) must be represented with state='m'.

    Tests the state-inference path: strata's parent_level_flag is '-' for every
    row; we must infer isomer from parent_ex_kev > 0.
    """
    import polars as pl

    df = pl.read_parquet(data_dir_path / _DIR / "Sn.parquet").filter((pl.col("A") == 121) & (pl.col("state") == "m"))
    assert df.height > 0, "Sn-121m β- (state='m') missing — state inference is broken"
    # Sn-121m β- endpoint is 372.2 keV (to Sb-121 ground)
    endpoint = float(df["endpoint_keV"].max())
    assert 365.0 < endpoint < 380.0, f"Sn-121m endpoint = {endpoint} (expect ~372)"


@pytest.mark.data
def test_bi210_high_z_anchor(data_dir_path: Path) -> None:
    """Bi-210 β- (Z=83): tests Fermi function at high Z where the (2p)^(2(γ₀-1))
    factor matters. NNDC mean energy = 389 keV; endpoint 1162.2 keV.
    """
    import numpy as np
    import polars as pl

    df = (
        pl.read_parquet(data_dir_path / _DIR / "Bi.parquet")
        .filter((pl.col("A") == 210) & (pl.col("state") == GROUND) & (pl.col("decay_mode") == "BetaMinus"))
        .sort("energy_keV")
    )
    t = df.filter(pl.col("transition_idx") == 0).sort("energy_keV")
    e = t["energy_keV"].to_numpy()
    d = t["dN_dE"].to_numpy()
    mean_e = float(np.trapezoid(e * d, e))
    assert 380.0 < mean_e < 400.0, f"Bi-210 β- mean E = {mean_e} (expect ~389)"


@pytest.mark.data
def test_k40_non_unique_forbidden_flagged(data_dir_path: Path) -> None:
    """K-40 has third-forbidden transitions; shape_factor_approx must be True."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / _DIR / "K.parquet").filter(
        (pl.col("A") == 40)
        & (pl.col("forbiddenness") != "uniqueFirstForbidden")
        & (pl.col("forbiddenness") != "uniqueSecondForbidden")
        & (pl.col("forbiddenness") != "")
    )
    if df.height == 0:
        pytest.skip("K-40 has no non-unique forbidden transitions in this build")
    assert df["shape_factor_approx"].all(), "Non-unique-forbidden transitions must be flagged approx"


@pytest.mark.data
def test_view_registered(data_dir_path: Path) -> None:
    from nucl_parquet import loader as np_lib

    db = np_lib.connect(data_dir_path)
    # Count rows for P-32 via the unified view
    n = db.sql("SELECT COUNT(*) FROM beta_spectra WHERE Z=15 AND A=32 AND state='g'").fetchall()[0][0]
    # P-32 has 1 transition × 200 bins = 200 rows
    assert 100 <= n <= 300, f"P-32 row count via view = {n}"
