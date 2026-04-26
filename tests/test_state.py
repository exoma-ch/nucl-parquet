"""Tests for the state/isomer feature — `state` column on `radiation`, the
`gamma_lines()` / `identify_gamma()` helpers, and the `build_radiation_state`
assigner.

Two layers:
  * Synthetic unit tests using an in-memory DuckDB over a hand-rolled
    `radiation` / `nuclides` pair covering the full `(Z, A, state)` matrix.
  * `@pytest.mark.data` invariants + structural spot-checks over the real
    shipped parquet files.

Design notes:
  - We assert *structural* facts (row with a given state exists) rather than
    intensities — ENSDF republishes intensities between releases.
  - Row-count floors are intentionally absent; they pass for a build that
    assigns every isomer to `state=''` (the exact bug v0.9.0 fixed).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pytest

from nucl_parquet.build_radiation_state import _assert_unique_levels, _assign_states
from nucl_parquet.loader import gamma_lines, identify_gamma

# ---------------------------------------------------------------------------
# Synthetic fixture — one ground-state nuclide plus two isomers (m, m2).
# ---------------------------------------------------------------------------


@pytest.fixture()
def iso_db() -> duckdb.DuckDBPyConnection:
    """DuckDB with `radiation` and `nuclides` tables built via SQL.

    Isotope Z=99, A=199 — fictional. Gamma lines:
      100 keV, 14% — ground only
      200 keV,  8% — ground (and 2% on the m isomer: different intensity)
      300 keV, 20% — m only
      400 keV, 50% — m2 only
    """
    db = duckdb.connect()
    db.execute(
        """
        CREATE TABLE radiation (
            Z INTEGER, A INTEGER, state VARCHAR, rad_type VARCHAR,
            energy_keV DOUBLE, intensity_pct DOUBLE,
            decay_mode VARCHAR, rad_subtype VARCHAR, dose_MeV_per_Bq_s DOUBLE,
            parent_level_keV DOUBLE
        );
        INSERT INTO radiation VALUES
            (99, 199, '',   'gamma', 100.0, 14.0, 'B-', NULL, 0.0,   0.0),
            (99, 199, '',   'gamma', 200.0,  8.0, 'B-', NULL, 0.0,   0.0),
            (99, 199, 'm',  'gamma', 200.0,  2.0, 'B-', NULL, 0.0,  45.0),
            (99, 199, 'm',  'gamma', 300.0, 20.0, 'B-', NULL, 0.0,  45.0),
            (99, 199, 'm2', 'gamma', 400.0, 50.0, 'B-', NULL, 0.0, 150.0);

        CREATE TABLE nuclides (
            Z INTEGER, A INTEGER, state VARCHAR,
            symbol VARCHAR, half_life_s DOUBLE
        );
        INSERT INTO nuclides VALUES
            (99, 199, '',   'Xx', 1e7),
            (99, 199, 'm',  'Xx', 3600.0),
            (99, 199, 'm2', 'Xx', 60.0);
        """
    )
    return db


def test_gamma_lines_default_is_ground(iso_db: duckdb.DuckDBPyConnection) -> None:
    """Default `state=""` must not leak isomer lines — the v0.9.0 bug."""
    rows = gamma_lines(iso_db, z=99, a=199).fetchall()
    energies = sorted(r[4] for r in rows)
    assert energies == [100.0, 200.0]
    # Intensity for the shared 200 keV line is the ground-state 8%, not the m's 2%
    e200 = [r for r in rows if r[4] == 200.0][0]
    assert e200[5] == pytest.approx(8.0)


def test_gamma_lines_isomer_explicit(iso_db: duckdb.DuckDBPyConnection) -> None:
    rows = gamma_lines(iso_db, z=99, a=199, state="m").fetchall()
    energies = sorted(r[4] for r in rows)
    assert energies == [200.0, 300.0]
    # All rows are tagged as the m state, not ground
    assert {r[2] for r in rows} == {"m"}


def test_gamma_lines_second_isomer(iso_db: duckdb.DuckDBPyConnection) -> None:
    rows = gamma_lines(iso_db, z=99, a=199, state="m2").fetchall()
    assert len(rows) == 1
    assert rows[0][4] == 400.0
    assert rows[0][2] == "m2"


def test_gamma_lines_min_intensity_filter(iso_db: duckdb.DuckDBPyConnection) -> None:
    rows = gamma_lines(iso_db, z=99, a=199, state="m", min_intensity=10.0).fetchall()
    assert [r[4] for r in rows] == [300.0]


def test_identify_gamma_ground_excludes_isomer(iso_db: duckdb.DuckDBPyConnection) -> None:
    """A 300 keV query at ground must not hit the m-only 300 keV line."""
    rows = identify_gamma(iso_db, 300.0, tolerance=1.0).fetchall()
    assert rows == [] or all(r[2] == "" for r in rows)


def test_identify_gamma_isomer_finds_line(iso_db: duckdb.DuckDBPyConnection) -> None:
    rows = identify_gamma(iso_db, 300.0, tolerance=1.0, state="m").fetchall()
    assert len(rows) == 1
    assert rows[0][4] == 300.0
    assert rows[0][2] == "m"


# ---------------------------------------------------------------------------
# Assigner unit tests — _assign_states is pure given a state_map.
# ---------------------------------------------------------------------------


def _mk_rad(rows: list[dict]) -> pl.DataFrame:
    """Minimal DataFrame shape expected by `_assign_states`."""
    return pl.DataFrame(rows)


def test_assigner_null_parent_level_is_ground() -> None:
    df = _mk_rad(
        [
            {"Z": 50, "A": 120, "parent_level_keV": None},
            {"Z": 50, "A": 120, "parent_level_keV": 0.0},
        ]
    )
    orphans: set = set()
    unlabeled: list = []
    assert _assign_states(df, state_map={}, orphans=orphans, unlabeled_excited=unlabeled) == ["", ""]
    assert orphans == set()
    assert unlabeled == []


def test_assigner_orphan_with_parent_level_is_ground() -> None:
    """Orphan (Z,A) — no entry in state_map — labels '' (ground), not 'm'.

    Regression for #49 Bug 2: previously fabricated 'm' for unknown nuclides,
    which then failed the nuclides-subset invariant downstream.
    """
    df = _mk_rad([{"Z": 77, "A": 177, "parent_level_keV": 500.0}])
    orphans: set = set()
    unlabeled: list = []
    states = _assign_states(df, state_map={}, orphans=orphans, unlabeled_excited=unlabeled)
    assert states == [""]
    assert (77, 177) in orphans
    assert unlabeled == []


def test_assigner_near_degenerate_levels_picks_nearest() -> None:
    state_map = {(1, 1): [("m", 45.0), ("m2", 46.0)]}
    df = _mk_rad(
        [
            {"Z": 1, "A": 1, "parent_level_keV": 45.1},
            {"Z": 1, "A": 1, "parent_level_keV": 45.9},
        ]
    )
    unlabeled: list = []
    assert _assign_states(df, state_map=state_map, orphans=set(), unlabeled_excited=unlabeled) == ["m", "m2"]
    assert unlabeled == []  # both within tolerance


def test_assigner_non_isomeric_parent_within_warning_threshold_is_ground() -> None:
    """Distinct non-isomeric excited parent within the *old* 2.0 keV warning
    band but beyond the *new* 0.5 keV match threshold is labeled '' (ground)
    and recorded for diagnostics. Regression for #49 Bug 1: previously these
    silently inherited the nearest isomer's label.

    Concrete example mirrors Mn-60 (m2 at 271.8) vs the radiation file's
    distinct 271.2 keV parent level.
    """
    state_map = {(25, 60): [("m", 114.0), ("m2", 271.8)]}
    df = _mk_rad([{"Z": 25, "A": 60, "parent_level_keV": 271.2}])
    unlabeled: list = []
    states = _assign_states(df, state_map=state_map, orphans=set(), unlabeled_excited=unlabeled)
    assert states == [""]
    assert len(unlabeled) == 1
    assert unlabeled[0][0:2] == (25, 60)
    assert unlabeled[0][3] == "m2"  # nearest isomer recorded for the diagnostic


def test_assigner_far_match_is_ground_not_isomer() -> None:
    state_map = {(1, 1): [("m", 45.0)]}
    df = _mk_rad([{"Z": 1, "A": 1, "parent_level_keV": 100.0}])
    unlabeled: list = []
    states = _assign_states(df, state_map=state_map, orphans=set(), unlabeled_excited=unlabeled)
    assert states == [""]
    assert len(unlabeled) == 1
    assert unlabeled[0][0:2] == (1, 1)


def test_assert_unique_levels_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="Duplicate isomeric level"):
        _assert_unique_levels({(63, 152): [("m", 45.6), ("m2", 45.6)]})


def test_assert_unique_levels_accepts_distinct() -> None:
    _assert_unique_levels({(63, 152): [("m", 45.6), ("m2", 147.9)]})


# ---------------------------------------------------------------------------
# Invariants + spot-checks over real data (@pytest.mark.data).
# ---------------------------------------------------------------------------


@pytest.mark.data
@pytest.mark.xfail(
    reason="Known data inconsistency: get_state_map() fabricates isomeric "
    "labels from radiation.parent_level_keV values that aren't backed by a "
    "real isomer in decay.parquet/nuclides.parquet — ~134 orphan (Z,A,state) "
    "triples (down from 135 after #49 fixed the assigner-side mislabels). "
    "Flips to passing once get_state_map() sources from nuclides.parquet "
    "and/or nuclides.parquet covers every isomer that radiation references.",
    strict=False,
)
def test_radiation_state_subset_of_nuclides(data_dir_path: Path) -> None:
    """Invariant: for every (Z, A) emitting radiation, every distinct `state`
    value must also exist in `nuclides.parquet` for that (Z, A). Catches
    state-label drift, orphan nuclides, and typos — one query for ~3500
    nuclides.
    """
    db = duckdb.connect()
    rad_glob = data_dir_path / "meta" / "ensdf" / "radiation" / "*.parquet"
    nuc_path = data_dir_path / "meta" / "ensdf" / "nuclides.parquet"
    bad = db.sql(
        f"""
        WITH rad AS (
            SELECT DISTINCT Z, A, state FROM read_parquet('{rad_glob}')
            WHERE Z > 0  -- skip spontaneous-fission fragments (Z=0 pseudo-rows)
        ),
        nuc AS (
            SELECT DISTINCT Z, A, state FROM read_parquet('{nuc_path}')
        )
        SELECT rad.Z, rad.A, rad.state FROM rad
        LEFT JOIN nuc USING (Z, A, state)
        WHERE nuc.Z IS NULL
        """
    ).fetchall()
    assert bad == [], (
        f"{len(bad)} (Z,A,state) triples present in radiation but missing from nuclides. First 5: {bad[:5]}"
    )


@pytest.mark.data
def test_eu152_ground_and_isomer_distinguished(data_dir_path: Path) -> None:
    """Eu-152 must have a 121.78 keV line in the ground state AND an 841.63 keV
    line labelled 'm' (the 9.31 h isomer). Assert existence only — not intensity.
    """
    db = duckdb.connect()
    rad_glob = data_dir_path / "meta" / "ensdf" / "radiation" / "*.parquet"

    ground = db.sql(
        f"""SELECT COUNT(*) FROM read_parquet('{rad_glob}')
            WHERE Z=63 AND A=152 AND state=''
              AND rad_type='gamma' AND ABS(energy_keV - 121.78) < 0.1"""
    ).fetchone()[0]
    assert ground >= 1, "Eu-152 ground-state 121.78 keV line missing"

    isomer_841 = db.sql(
        f"""SELECT COUNT(*) FROM read_parquet('{rad_glob}')
            WHERE Z=63 AND A=152 AND state='m'
              AND rad_type='gamma' AND ABS(energy_keV - 841.63) < 0.1"""
    ).fetchone()[0]
    assert isomer_841 >= 1, "Eu-152m 841.63 keV line missing"

    # The 9.3h isomer should have its own nuclides entry
    n_iso = db.sql(
        f"""SELECT COUNT(*) FROM read_parquet('{data_dir_path}/meta/ensdf/nuclides.parquet')
            WHERE Z=63 AND A=152 AND state='m'"""
    ).fetchone()[0]
    assert n_iso == 1


@pytest.mark.data
def test_tc99m_isomer_labelled(data_dir_path: Path) -> None:
    """Tc-99m — single 140.5 keV gamma, the most common medical isotope line —
    must be labelled with state='m'."""
    db = duckdb.connect()
    rad_glob = data_dir_path / "meta" / "ensdf" / "radiation" / "*.parquet"
    n = db.sql(
        f"""SELECT COUNT(*) FROM read_parquet('{rad_glob}')
            WHERE Z=43 AND A=99 AND state='m'
              AND rad_type='gamma' AND ABS(energy_keV - 140.5) < 0.2"""
    ).fetchone()[0]
    assert n >= 1
