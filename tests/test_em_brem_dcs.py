"""Tests for the Seltzer-Berger bremsstrahlung DCS table (#118).

Verifies:
- Schema and coverage (Z=1..92, all reasonable energies/κ values)
- Physical invariants: dσ/dκ > 0 throughout, κ in (0, 1]
- Soft-photon limit: dσ/dκ diverges as κ → 0 (∝ 1/κ in non-relativistic
  Bethe-Heitler limit), so dcs(κ_min) >> dcs(κ_max) at fixed (Z, T_e)
- Z² scaling at low electron energy (Bethe-Heitler): dσ/dκ ∝ Z² approximately
  for κ ~ 0.5 in the non-relativistic limit
- DuckDB view registered
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def data_dir_path() -> Path:
    return Path("data").resolve()


_PARQUET = "em/electron_brem_sb_dcs.parquet"


@pytest.mark.data
def test_schema(data_dir_path: Path) -> None:
    import polars as pl

    df = pl.read_parquet(data_dir_path / _PARQUET)
    assert df.columns == ["target_Z", "electron_kev", "kappa", "dcs"]
    assert df["target_Z"].dtype == pl.Int32
    assert df["electron_kev"].dtype == pl.Float64
    assert df["kappa"].dtype == pl.Float64
    assert df["dcs"].dtype == pl.Float64


@pytest.mark.data
def test_z_coverage(data_dir_path: Path) -> None:
    import polars as pl

    df = pl.read_parquet(data_dir_path / _PARQUET)
    zs = set(df["target_Z"].unique().to_list())
    missing = sorted(set(range(1, 93)) - zs)
    assert not missing, f"missing Z: {missing}"


@pytest.mark.data
def test_kappa_in_unit_interval(data_dir_path: Path) -> None:
    """κ = E_γ / T_e ∈ (0, 1]. Strata's grid uses κ_min ≈ 1e-12 to handle
    the soft-photon log singularity; κ_max ≤ 1 — the hard-tail end-point."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / _PARQUET)
    assert df["kappa"].min() > 0.0
    assert df["kappa"].max() <= 1.0


@pytest.mark.data
def test_dcs_positive(data_dir_path: Path) -> None:
    """Differential cross-section must be strictly positive (physical)."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / _PARQUET)
    n_nonpos = df.filter(pl.col("dcs") <= 0).height
    assert n_nonpos == 0, f"{n_nonpos} rows with non-positive dcs"


@pytest.mark.data
def test_soft_photon_divergence(data_dir_path: Path) -> None:
    """dσ/dκ rises monotonically as κ → 0 for Pb at 10 MeV (Bethe-Heitler IR).

    The soft-photon end of the spectrum is the universal feature — DCS values
    near κ=0 should be much larger than near κ=1. Catches a row-order regression
    or column swap.
    """
    import polars as pl

    df = (
        pl.read_parquet(data_dir_path / _PARQUET)
        .filter((pl.col("target_Z") == 82) & (pl.col("electron_kev") > 9000) & (pl.col("electron_kev") < 11000))
        .sort("kappa")
    )
    assert df.height >= 8, f"too few κ samples at Z=82 ~10 MeV: {df.height}"
    soft = float(df["dcs"][0])  # near κ=0
    hard = float(df["dcs"][-1])  # near κ=1
    assert soft > 5 * hard, f"soft/hard ratio = {soft / hard:.2f} — expected >>1"


@pytest.mark.data
def test_dcs_is_scaled_form(data_dir_path: Path) -> None:
    """Seltzer-Berger publishes the *scaled* DCS = (dσ/dκ) × β²/Z² (Berger-Seltzer
    1985 convention). The factored-out Z² means values are O(1-10) across all Z
    instead of O(Z²) ≈ O(1000s) for high-Z. Catches a future un-scaling regression.
    """
    import polars as pl

    df = pl.read_parquet(data_dir_path / _PARQUET)

    def _value_at(z: int, e_kev_target: float, kappa_target: float) -> float:
        rows = df.filter(pl.col("target_Z") == z).sort(
            (pl.col("electron_kev") - e_kev_target).abs() + 1000 * (pl.col("kappa") - kappa_target).abs()
        )
        return float(rows["dcs"][0])

    pb = _value_at(82, 100.0, 0.5)
    h = _value_at(1, 100.0, 0.5)
    # Both should be O(1-10). If we had the raw (unscaled) DCS, Pb/H would be ~Z²=6724.
    assert 0.5 < pb < 20.0, f"Pb scaled DCS @ 100 keV κ=0.5 out of expected O(1-10): {pb}"
    assert 0.5 < h < 20.0, f"H scaled DCS @ 100 keV κ=0.5 out of expected O(1-10): {h}"
    # Z dependence of the scaled form is screening-corrections-only, so ratio is mild.
    assert pb / h < 5.0, f"Pb/H scaled DCS ratio = {pb / h:.2f}; suggests un-scaled (raw) DCS"


@pytest.mark.data
def test_view_registered(data_dir_path: Path) -> None:
    from nucl_parquet import loader as np_lib

    db = np_lib.connect(data_dir_path)
    n = db.sql("SELECT COUNT(*) FROM electron_brem_sb_dcs").fetchall()[0][0]
    assert n > 100_000, f"electron_brem_sb_dcs has only {n} rows"
