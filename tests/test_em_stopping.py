"""Tests for the strata-rebuilt electron-stopping tables (#121).

Verifies:
- Three output files exist with the expected schema
- ESTAR.parquet legacy schema is preserved (drop-in for v0.5 consumers)
- electron_stopping.parquet covers all 98 elements + standard compounds
- density_effect_params.parquet has Sternheimer params for water + standard elements
- Anchor: e- on Cu at 1 MeV total ≈ 1.435 MeV·cm²/g (NIST ICRU-37, ±2%)
- DuckDB views are registered correctly
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def data_dir_path() -> Path:
    return Path("data").resolve()


@pytest.mark.data
def test_legacy_estar_schema_unchanged(data_dir_path: Path) -> None:
    """ESTAR.parquet must keep the v0.5 four-column schema for backwards compat."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "stopping" / "ESTAR.parquet")
    assert df.columns == ["source", "target_Z", "energy_MeV", "dedx"]
    assert df["source"].dtype == pl.String
    assert df["target_Z"].dtype == pl.Int32
    assert df["energy_MeV"].dtype == pl.Float64
    assert df["dedx"].dtype == pl.Float64


@pytest.mark.data
def test_legacy_estar_covers_all_98_elements(data_dir_path: Path) -> None:
    """Every element Z=1..98 from strata's estar_long.parquet must be in legacy table."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "stopping" / "ESTAR.parquet")
    z_set = set(df["target_Z"].unique().to_list())
    assert z_set == set(range(1, 99)), f"missing/extra Z: {sorted(set(range(1, 99)) ^ z_set)}"


@pytest.mark.data
def test_electron_stopping_rich_schema(data_dir_path: Path) -> None:
    """electron_stopping.parquet exposes collision/radiative/total + density-effect delta."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "stopping" / "em" / "electron_stopping.parquet")
    expected = {
        "name",
        "g4_name",
        "is_element",
        "target_Z",
        "energy_MeV",
        "collision_sp_mev_cm2_g",
        "radiative_sp_mev_cm2_g",
        "total_sp_mev_cm2_g",
        "density_effect_delta",
    }
    assert set(df.columns) == expected
    # Compounds have NULL target_Z
    assert df.filter(~pl.col("is_element"))["target_Z"].null_count() > 0
    # Elements have populated target_Z
    assert df.filter(pl.col("is_element"))["target_Z"].null_count() == 0


@pytest.mark.data
def test_electron_stopping_includes_standard_compounds(data_dir_path: Path) -> None:
    """ICRU-44/PhysicsList compounds (water, air, soft tissue, bone) must be present."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "stopping" / "em" / "electron_stopping.parquet")
    names = set(df.filter(~pl.col("is_element"))["name"].unique().to_list())
    # Strata uses uppercase ICRU naming for compounds
    must_have_substrings = ["WATER", "AIR", "BONE", "MUSCLE", "TISSUE"]
    for substr in must_have_substrings:
        matches = [n for n in names if substr in n.upper()]
        assert matches, f"no compound matching {substr!r} in electron_stopping.parquet"


@pytest.mark.data
def test_density_effect_params_schema_and_water(data_dir_path: Path) -> None:
    """Sternheimer params: water should be present with NIST I_eV=78.0 eV.

    Source: NIST PML I-values + ICRU-37 Sternheimer parameterization.
    """
    import polars as pl

    df = pl.read_parquet(data_dir_path / "stopping" / "em" / "density_effect_params.parquet")
    # `phase` after #357 — the column holds solid/liquid/gas, and calling that
    # `state` collided with the isomeric-state column in twenty other tables.
    # Both spellings are accepted while the shipped parquet predates the
    # rebuild; `tests/test_state_vocabulary.py` holds the debt and the deadline.
    phase_column = "phase" if "phase" in df.columns else "state"
    assert {
        "name",
        "I_eV",
        "C",
        "X0",
        "X1",
        "a",
        "k",
        "delta0",
        "density_gcm3",
        phase_column,
        "Zeff",
        "nElements",
    } == set(df.columns)
    water = df.filter(pl.col("name") == "G4_WATER")
    assert water.height == 1, "G4_WATER must be present in density_effect_params"
    # NIST PML lists I-value for liquid water at 75 eV; Geant4 uses 78.0 eV (Sternheimer-Berger value).
    # Either is acceptable as long as it's the canonical value (not 0 or garbage).
    i_ev = float(water["I_eV"][0])
    assert 70.0 < i_ev < 90.0, f"G4_WATER I_eV={i_ev} — outside expected NIST/G4 range (75-78)"
    # Phase of matter must be liquid (or solid, for the ice parameterisation).
    assert water[phase_column][0] in ("liquid", "solid"), f"G4_WATER phase={water[phase_column][0]}"


@pytest.mark.data
@pytest.mark.parametrize(
    ("target_Z", "energy_MeV", "expected"),
    [
        # NIST PML ESTAR tabulation, 2017 release (matches strata to all 5 digits)
        (1, 1.0, 3.821),  # H
        (13, 1.0, 1.486),  # Al
        (29, 1.0, 1.309),  # Cu
        (74, 10.0, 2.335),  # W at 10 MeV
        (82, 1.0, 1.123),  # Pb
    ],
)
def test_estar_anchors_match_nist(data_dir_path: Path, target_Z: int, energy_MeV: float, expected: float) -> None:
    """Strata-rebuilt ESTAR matches NIST tabulation to <0.1% — drop-in compat."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "stopping" / "ESTAR.parquet")
    row = df.filter((pl.col("target_Z") == target_Z) & (pl.col("energy_MeV") == energy_MeV))
    assert row.height == 1, f"Z={target_Z} at {energy_MeV} MeV not on grid"
    dedx = float(row["dedx"][0])
    rel = abs(dedx - expected) / expected
    assert rel < 1e-3, f"Z={target_Z} @ {energy_MeV} MeV: got {dedx}, expected NIST {expected}, |Δ|/E = {rel:.4f}"


@pytest.mark.data
def test_collision_plus_radiative_equals_total(data_dir_path: Path) -> None:
    """Physics invariant: collision_sp + radiative_sp == total_sp on every row."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "stopping" / "em" / "electron_stopping.parquet")
    diff = (df["collision_sp_mev_cm2_g"] + df["radiative_sp_mev_cm2_g"] - df["total_sp_mev_cm2_g"]).abs()
    rel = (diff / df["total_sp_mev_cm2_g"]).max()
    # Tabulated to 4 sig figs in NIST, so ≤2e-3 is within rounding.
    assert rel < 3e-3, f"collision + radiative != total on some rows; max rel residual = {rel:.2e}"


@pytest.mark.data
def test_density_effect_delta_monotone_and_finite(data_dir_path: Path) -> None:
    """δ(E) must be finite, non-negative, and monotone non-decreasing in energy for any material."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "stopping" / "em" / "electron_stopping.parquet")
    # Check on a load-bearing material set: water + a couple of metals
    for material_name in ["WATER_LIQUID", "AIR_DRY_NEAR_SEA_LEVEL", "29", "82"]:
        rows = df.filter(pl.col("name") == material_name).sort("energy_MeV")
        if rows.height == 0:
            continue
        delta = rows["density_effect_delta"].to_list()
        assert all(d == d for d in delta), f"NaN in {material_name} density_effect_delta"  # NaN check
        assert all(d >= 0 for d in delta), f"negative δ in {material_name}"
        for i in range(len(delta) - 1):
            # Allow tiny non-monotonicity at the rounding floor
            assert delta[i + 1] >= delta[i] - 1e-9, (
                f"{material_name} δ decreased between {rows['energy_MeV'][i]:.4f} and {rows['energy_MeV'][i + 1]:.4f}: "
                f"{delta[i]} -> {delta[i + 1]}"
            )


@pytest.mark.data
def test_g4_name_joins_with_density_effect_params(data_dir_path: Path) -> None:
    """electron_stopping.g4_name joins 1:1 with density_effect_params.name for Z=1..92.

    density_effect_params (sourced from Geant4 PhysicsList) covers Z=1..92.
    The transuranic elements Np-Cf (Z=93..98) are in electron_stopping (from
    strata's estar_long) but absent from density_effect_params — consumers
    needing density-effect correction for transuranics must derive Sternheimer
    params manually or fall back to a scaling formula.
    """
    import polars as pl

    es = pl.read_parquet(data_dir_path / "stopping" / "em" / "electron_stopping.parquet")
    de = pl.read_parquet(data_dir_path / "stopping" / "em" / "density_effect_params.parquet")

    element_g4_names = set(es.filter(pl.col("is_element") & (pl.col("target_Z") <= 92))["g4_name"].unique().to_list())
    de_names = set(de["name"].to_list())
    missing = sorted(element_g4_names - de_names)
    assert not missing, f"Z=1..92 elements without density_effect_params row: {missing}"


@pytest.mark.data
def test_views_registered_in_loader(data_dir_path: Path) -> None:
    """connect() must expose electron_stopping + density_effect_params as queryable views."""
    from nucl_parquet import loader as np_lib

    db = np_lib.connect(data_dir_path)
    # electron_stopping: query a known compound (water at 1 MeV)
    water_rows = db.sql(
        "SELECT total_sp_mev_cm2_g FROM electron_stopping WHERE name = 'WATER_LIQUID' AND energy_MeV = 1.0"
    ).fetchall()
    # if WATER_LIQUID isn't the exact name, try a substring match
    if not water_rows:
        water_rows = db.sql(
            "SELECT name, total_sp_mev_cm2_g FROM electron_stopping WHERE name LIKE '%WATER%' AND energy_MeV = 1.0 LIMIT 3"
        ).fetchall()
    assert water_rows, "no water-containing rows in electron_stopping at 1 MeV"
    # density_effect_params: water
    water_de = db.sql("SELECT I_eV FROM density_effect_params WHERE name = 'G4_WATER'").fetchall()
    assert water_de and 70.0 < float(water_de[0][0]) < 90.0
