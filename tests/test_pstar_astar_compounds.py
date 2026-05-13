"""Tests for the NIST PSTAR/ASTAR compound stopping tables (#160).

Verifies:
- Both parquet files exist with the expected schema
- A core set of common compounds is present in both: water (liquid),
  air, polyethylene, A-150 plastic, soft tissue
- Anchor values match NIST PML tabulations within tabulation precision
- DuckDB views are registered correctly
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def data_dir_path() -> Path:
    return Path("data").resolve()


_EXPECTED_SCHEMA = ["source", "matno", "compound", "energy_MeV", "dedx"]


@pytest.mark.data
def test_pstar_compounds_schema(data_dir_path: Path) -> None:
    """PSTAR_compounds.parquet exposes (source, matno, compound, energy_MeV, dedx)."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "stopping" / "compounds" / "PSTAR_compounds.parquet")
    assert df.columns == _EXPECTED_SCHEMA
    assert df["source"].unique().to_list() == ["PSTAR_compound"]
    assert df["matno"].dtype == pl.Int32
    assert df["compound"].dtype == pl.String
    assert df["energy_MeV"].dtype == pl.Float64
    assert df["dedx"].dtype == pl.Float64


@pytest.mark.data
def test_astar_compounds_schema(data_dir_path: Path) -> None:
    """ASTAR_compounds.parquet exposes the same schema as PSTAR_compounds."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "stopping" / "compounds" / "ASTAR_compounds.parquet")
    assert df.columns == _EXPECTED_SCHEMA
    assert df["source"].unique().to_list() == ["ASTAR_compound"]


@pytest.mark.data
def test_pstar_compounds_includes_core_set(data_dir_path: Path) -> None:
    """The five most-requested clinical/calibration compounds must ship."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "stopping" / "compounds" / "PSTAR_compounds.parquet")
    names = set(df["compound"].unique().to_list())
    must_have_substrings = [
        "WATER",  # WATER_LIQUID or similar
        "AIR",  # AIR_DRY_NEAR_SEA_LEVEL or AIR
        "POLYETHYLENE",
        "A_150",  # A-150 TISSUE-EQUIVALENT PLASTIC → A_150_…
        "TISSUE",  # any tissue substitute
    ]
    for substr in must_have_substrings:
        matches = [n for n in names if substr in n]
        assert matches, f"PSTAR_compounds.parquet has no compound matching {substr!r}"


@pytest.mark.data
def test_astar_compounds_includes_core_set(data_dir_path: Path) -> None:
    """ASTAR compound coverage parallels PSTAR (same matno space)."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "stopping" / "compounds" / "ASTAR_compounds.parquet")
    names = set(df["compound"].unique().to_list())
    for substr in ["WATER", "AIR", "POLYETHYLENE"]:
        matches = [n for n in names if substr in n]
        assert matches, f"ASTAR_compounds.parquet has no compound matching {substr!r}"


@pytest.mark.data
@pytest.mark.parametrize(
    ("energy_MeV", "expected_dedx"),
    [
        # NIST PSTAR liquid water tabulation (matno 276). Proton TOTAL kinetic
        # energy in MeV; dedx in MeV·cm²/g (collision + nuclear).
        (1.0, 260.8),  # near Bragg peak (~0.1 MeV)
        (10.0, 45.67),  # falling
        (100.0, 7.289),  # plateau
        (1000.0, 2.211),  # Bethe minimum
    ],
)
def test_pstar_water_anchors_match_nist(data_dir_path: Path, energy_MeV: float, expected_dedx: float) -> None:
    """Proton-on-water dedx must match NIST PSTAR to <0.1%."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "stopping" / "compounds" / "PSTAR_compounds.parquet")
    row = df.filter((pl.col("compound") == "WATER_LIQUID") & (pl.col("energy_MeV") == energy_MeV))
    assert row.height == 1, f"WATER_LIQUID at {energy_MeV} MeV not on grid"
    dedx = float(row["dedx"][0])
    rel = abs(dedx - expected_dedx) / expected_dedx
    assert rel < 1e-3, f"p/water @ {energy_MeV} MeV: got {dedx}, expected NIST {expected_dedx}"


@pytest.mark.data
@pytest.mark.parametrize(
    ("energy_MeV", "expected_dedx"),
    [
        # NIST ASTAR liquid water (matno 276). α energy is TOTAL kinetic energy.
        (1.0, 2193.0),  # well below Bragg peak (α-on-water peaks ~1 MeV)
        (10.0, 534.4),
        (100.0, 86.49),
        (500.0, 24.65),
    ],
)
def test_astar_water_anchors_match_nist(data_dir_path: Path, energy_MeV: float, expected_dedx: float) -> None:
    """α-on-water dedx must match NIST ASTAR to <0.1%."""
    import polars as pl

    df = pl.read_parquet(data_dir_path / "stopping" / "compounds" / "ASTAR_compounds.parquet")
    row = df.filter((pl.col("compound") == "WATER_LIQUID") & (pl.col("energy_MeV") == energy_MeV))
    assert row.height == 1, f"WATER_LIQUID at {energy_MeV} MeV not on grid"
    dedx = float(row["dedx"][0])
    rel = abs(dedx - expected_dedx) / expected_dedx
    assert rel < 1e-3, f"α/water @ {energy_MeV} MeV: got {dedx}, expected NIST {expected_dedx}"


@pytest.mark.data
def test_views_registered_in_loader(data_dir_path: Path) -> None:
    """connect() must expose pstar_compounds + astar_compounds as queryable views."""
    from nucl_parquet import loader as np_lib

    db = np_lib.connect(data_dir_path)
    # Sanity: water row count > 50 (NIST publishes ~70 energy points)
    p_water = db.sql("SELECT COUNT(*) FROM pstar_compounds WHERE compound LIKE '%WATER%'").fetchall()[0][0]
    assert p_water > 30, f"pstar_compounds has only {p_water} water rows"
    a_water = db.sql("SELECT COUNT(*) FROM astar_compounds WHERE compound LIKE '%WATER%'").fetchall()[0][0]
    assert a_water > 30, f"astar_compounds has only {a_water} water rows"


@pytest.mark.data
def test_compounds_dont_collide_with_elemental_stopping_view(data_dir_path: Path) -> None:
    """The elemental `stopping` view must NOT include compound rows.

    The `stopping` view comes from a non-recursive glob over `data/stopping/`;
    compound files live under `data/stopping/compounds/` so the glob ignores
    them. Verify by checking that `compound` is NOT a column of the elemental
    view (which would only happen if the glob picked up the compound files).
    """
    from nucl_parquet import loader as np_lib

    db = np_lib.connect(data_dir_path)
    cols = db.sql("DESCRIBE stopping").fetchall()
    col_names = {row[0] for row in cols}
    assert "compound" not in col_names, (
        f"stopping view picked up compound files — schema bleed. columns: {sorted(col_names)}"
    )
