"""Tests for compound_compositions.parquet (#113 P0 — data only).

Verifies:
- Schema and dtypes
- Coverage matches xcom_compounds (33 materials)
- weight_fraction sums to 1.0 ± 1e-6 per material
- Spot checks against published NIST/ICRU values for water, air, polyethylene
- Bragg-additive reconstruction: ∑ wᵢ × elemental_µ/ρ ≈ compound_µ/ρ for water
  at 100 keV within tabulation tolerance (Compton-dominated, no big screening)
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def data_dir_path() -> Path:
    return Path("data").resolve()


@pytest.mark.data
def test_schema(data_dir_path: Path) -> None:
    import polars as pl

    df = pl.read_parquet(data_dir_path / "meta" / "compound_compositions.parquet")
    assert df.columns == ["material", "Z", "weight_fraction"]
    assert df["material"].dtype == pl.String
    assert df["Z"].dtype == pl.Int32
    assert df["weight_fraction"].dtype == pl.Float64


@pytest.mark.data
def test_covers_all_xcom_compounds(data_dir_path: Path) -> None:
    """Every material in xcom_compounds.parquet must have a composition."""
    import polars as pl

    comp = pl.read_parquet(data_dir_path / "meta" / "compound_compositions.parquet")
    xcom = pl.read_parquet(data_dir_path / "meta" / "xcom_compounds.parquet")
    xcom_materials = set(xcom["material"].unique().to_list())
    comp_materials = set(comp["material"].unique().to_list())
    missing = xcom_materials - comp_materials
    assert not missing, f"materials in xcom_compounds without composition: {sorted(missing)}"


@pytest.mark.data
def test_weight_fractions_sum_to_one(data_dir_path: Path) -> None:
    """For each material, ∑ weight_fraction = 1.0 within float tolerance."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "meta" / "compound_compositions.parquet")
    sums = df.group_by("material").agg(pl.col("weight_fraction").sum())
    for r in sums.iter_rows(named=True):
        assert abs(r["weight_fraction"] - 1.0) < 1e-6, f"{r['material']} weight fractions sum to {r['weight_fraction']}"


@pytest.mark.data
def test_water_composition(data_dir_path: Path) -> None:
    """Water: H=0.111894, O=0.888106 per NIST PML (H2O atomic-mass weighted)."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "meta" / "compound_compositions.parquet").filter(pl.col("material") == "water")
    rows = {int(r["Z"]): float(r["weight_fraction"]) for r in df.iter_rows(named=True)}
    assert set(rows.keys()) == {1, 8}
    assert abs(rows[1] - 0.111894) < 1e-5
    assert abs(rows[8] - 0.888106) < 1e-5


@pytest.mark.data
def test_air_composition(data_dir_path: Path) -> None:
    """Dry air (NIST std): C=0.000124, N=0.755268, O=0.231781, Ar=0.012827."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "meta" / "compound_compositions.parquet").filter(pl.col("material") == "air")
    rows = {int(r["Z"]): float(r["weight_fraction"]) for r in df.iter_rows(named=True)}
    assert set(rows.keys()) == {6, 7, 8, 18}
    assert abs(rows[7] - 0.755268) < 1e-5  # N2 dominant
    assert abs(rows[8] - 0.231781) < 1e-5
    assert abs(rows[18] - 0.012827) < 1e-5


@pytest.mark.data
def test_polyethylene_composition(data_dir_path: Path) -> None:
    """(C2H4)n → H=0.143711, C=0.856289."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "meta" / "compound_compositions.parquet").filter(
        pl.col("material") == "polyethylene"
    )
    rows = {int(r["Z"]): float(r["weight_fraction"]) for r in df.iter_rows(named=True)}
    assert set(rows.keys()) == {1, 6}
    assert abs(rows[1] - 0.143711) < 1e-5
    assert abs(rows[6] - 0.856289) < 1e-5


@pytest.mark.data
@pytest.mark.parametrize(
    ("material", "energy_MeV", "tolerance"),
    [
        # Compton-dominated low-Z: tightest agreement
        ("water", 0.1, 0.05),
        # PE-dominated low-Z compound (carbon-rich) at low energy
        ("polyethylene", 0.05, 0.05),
        # High-Z mix (lead glass) at 1 MeV — Compton regime, screening corrections
        # are still well-approximated by Bragg additivity here
        ("glass", 1.0, 0.05),
    ],
)
def test_bragg_reconstruction_anchors(data_dir_path: Path, material: str, energy_MeV: float, tolerance: float) -> None:
    """Bragg-additive ∑ wᵢ × xcom_elements.mu_rho reconstructs xcom_compounds.mu_rho
    for multiple representative materials across the periodic table."""
    from nucl_parquet import loader as np_lib

    db = np_lib.connect(data_dir_path)
    bragg = float(
        db.sql(
            """
            SELECT SUM(c.weight_fraction * x.mu_rho_cm2_g) AS bragg_sum
            FROM compound_compositions c
            JOIN xcom_elements x ON c.Z = x.Z
            WHERE c.material = ? AND x.energy_MeV = ?
            """,
            params=[material, energy_MeV],
        ).fetchall()[0][0]
    )
    direct = float(
        db.sql(
            "SELECT mu_rho_cm2_g FROM xcom_compounds WHERE material = ? AND energy_MeV = ?",
            params=[material, energy_MeV],
        ).fetchall()[0][0]
    )
    rel = abs(bragg - direct) / direct
    assert rel < tolerance, f"{material} @ {energy_MeV} MeV: Bragg {bragg} vs direct {direct}, rel {rel:.3f}"


@pytest.mark.data
def test_view_registered(data_dir_path: Path) -> None:
    from nucl_parquet import loader as np_lib

    db = np_lib.connect(data_dir_path)
    n = db.sql("SELECT COUNT(*) FROM compound_compositions").fetchall()[0][0]
    assert n > 100, f"compound_compositions has only {n} rows"
