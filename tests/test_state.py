"""Tests for the state/isomer feature — `state` column on `radiation` and the
`gamma_lines()` / `identify_gamma()` helpers.

Two layers:
  * Synthetic unit tests using an in-memory DuckDB over a hand-rolled
    `radiation` / `nuclides` pair covering the full `(Z, A, state)` matrix.
  * `@pytest.mark.data` invariants + structural spot-checks over the real
    shipped parquet files.

Design notes:
  - We assert *structural* facts (row with a given state exists) rather than
    intensities — ENSDF republishes intensities between releases.
  - The legacy IAEA-fetcher rescue tests (`_assign_states` threshold,
    `_surface_diagnostics` ceilings, `state_map` fabrication guards) were
    removed in v0.11.0 when the G4 migration eliminated the bug classes
    those tests pinned. The new v0.11 acceptance gate lives in
    ``tests/test_g4_diff_harness.py``.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from nucl_parquet.loader import gamma_lines, identify_gamma
from nucl_parquet.state_vocabulary import GROUND, ISOMERS

# ---------------------------------------------------------------------------
# Synthetic fixture — one ground-state nuclide plus two isomers (m, m2).
# ---------------------------------------------------------------------------


@pytest.fixture()
def iso_db() -> duckdb.DuckDBPyConnection:
    """DuckDB with `radiation` and `nuclides` tables built via SQL.

    Isotope Z=99, A=199 — fictional. Ground state is `'g'`: this fixture stands
    in for `meta/ensdf/radiation` and `nuclides`, both of which migrated off `''`
    in the 2026.8.5 release, and a fixture on the retired spelling would assert
    against a vocabulary nothing ships. Gamma lines:
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
            (99, 199, 'g',  'gamma', 100.0, 14.0, 'B-', NULL, 0.0,   0.0),
            (99, 199, 'g',  'gamma', 200.0,  8.0, 'B-', NULL, 0.0,   0.0),
            (99, 199, 'm',  'gamma', 200.0,  2.0, 'B-', NULL, 0.0,  45.0),
            (99, 199, 'm',  'gamma', 300.0, 20.0, 'B-', NULL, 0.0,  45.0),
            (99, 199, 'm2', 'gamma', 400.0, 50.0, 'B-', NULL, 0.0, 150.0);

        CREATE TABLE nuclides (
            Z INTEGER, A INTEGER, state VARCHAR,
            symbol VARCHAR, half_life_s DOUBLE
        );
        INSERT INTO nuclides VALUES
            (99, 199, 'g',  'Xx', 1e7),
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
# Invariants + spot-checks over real data (@pytest.mark.data).
# ---------------------------------------------------------------------------


@pytest.mark.data
def test_radiation_state_subset_of_nuclides(data_dir_path: Path) -> None:
    """Invariant: for every (Z, A) emitting radiation, every distinct `state`
    value should also exist in `nuclides.parquet`.

    Bounded budget for v0.11: G4 PhotonEvaporation6.1.2 covers a slightly
    wider nuclide universe than G4ENSDFSTATE3.0 (some short-lived isotopes
    have level schemes but no catalog state entries). v0.10.x's IAEA
    pipeline had ~134 phantom-isomer orphans by *fabrication*; the v0.11
    G4 path has ~12 by *upstream coverage gap*. Allow ≤ 30 such ground-only
    orphans and reject any nonzero count for state != '' (every non-ground
    state in radiation MUST be backed by the catalog).
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
    # Three buckets, not two. `state != ""` used to mean "an isomer"; since
    # #387 it would also catch NULL, which means "the state could not be
    # determined" — the opposite of a claim to be an isomer, and something that
    # cannot join to a catalog keyed on resolved states.
    isomer_orphans = [(z, a, s) for (z, a, s) in bad if s in ISOMERS]
    assert isomer_orphans == [], (
        f"{len(isomer_orphans)} ISOMER (Z,A,state) triples present in radiation "
        f"but missing from nuclides — every isomeric state must be backed "
        f"by the catalog. First 5: {isomer_orphans[:5]}"
    )
    ground_orphans = [(z, a, s) for (z, a, s) in bad if s == GROUND]
    assert len(ground_orphans) <= 30, (
        f"{len(ground_orphans)} ground-state (Z,A) triples present in "
        f"radiation but missing from nuclides — exceeds the 30-row "
        f"G4-coverage-gap budget. First 5: {ground_orphans[:5]}"
    )
    # NULL cannot join `USING (Z, A, state)` at all, so every unresolved
    # radiation row lands here by construction rather than by absence. #387
    # chose NULL over guessing; the budget bounds how much guessing was
    # declined, and 13 of the 13 shipped today are that.
    unresolved = [(z, a, s) for (z, a, s) in bad if s is None]
    assert len(unresolved) <= 30, (
        f"{len(unresolved)} radiation (Z,A) with an unresolved state — exceeds "
        f"the 30-row budget for what #387 declined to guess. First 5: {unresolved[:5]}"
    )


@pytest.mark.data
def test_eu152_ground_and_isomer_distinguished(data_dir_path: Path) -> None:
    """Eu-152 must have both isomers (m, m2) catalogued, and each must
    have at least one radiation emission row.

    Updated for v0.11 (G4-derived): the v0.10.x 121.78 keV "ground-state
    Eu-152 line" was actually a Sm-152 daughter line credited via the
    IAEA fetcher's parent-merging artifact. Under the de-exciting-nucleus
    convention it lives in Sm.parquet (Z=62), not Eu.parquet.

    Eu-152m's 45.6 keV M1+E2 IT transition is heavily IC-converted (no
    gamma row, only Auger + X-ray emissions); Eu-152m2 has a 147.86 keV
    cascade gamma. Assert appropriate emissions for each state.
    """
    db = duckdb.connect()
    rad_glob = data_dir_path / "meta" / "ensdf" / "radiation" / "*.parquet"

    # Eu-152m must emit *some* radiation (auger/xray from IC of the 45.6
    # keV transition; gamma is suppressed by ICC).
    eu152m_rows = db.sql(
        f"""SELECT COUNT(*) FROM read_parquet('{rad_glob}')
            WHERE Z=63 AND A=152 AND state='m'"""
    ).fetchone()[0]
    assert eu152m_rows >= 1, "Eu-152m emissions missing"

    # Eu-152m2 must have at least one cascade gamma (147.86 keV → lower).
    eu152m2_gammas = db.sql(
        f"""SELECT COUNT(*) FROM read_parquet('{rad_glob}')
            WHERE Z=63 AND A=152 AND state='m2' AND rad_type='gamma'"""
    ).fetchone()[0]
    assert eu152m2_gammas >= 1, "Eu-152m2 cascade gammas missing"

    # Both Eu-152m and Eu-152m2 must have nuclides catalog entries
    n_iso = db.sql(
        f"""SELECT COUNT(DISTINCT state) FROM read_parquet(
              '{data_dir_path}/meta/ensdf/nuclides.parquet')
            WHERE Z=63 AND A=152 AND state IN ('m', 'm2')"""
    ).fetchone()[0]
    assert n_iso == 2, "Eu-152 must have both 'm' and 'm2' in nuclides"


@pytest.mark.data
def test_tc99m_isomer_labelled(data_dir_path: Path) -> None:
    """Tc-99m must be in the catalog with state='m' and emit its 142.68 keV
    gamma (often informally quoted as 140.5 keV in older medical-physics
    literature; the precise ENSDF value is 142.6836)."""
    db = duckdb.connect()
    rad_glob = data_dir_path / "meta" / "ensdf" / "radiation" / "*.parquet"
    n = db.sql(
        f"""SELECT COUNT(*) FROM read_parquet('{rad_glob}')
            WHERE Z=43 AND A=99 AND state='m'
              AND rad_type='gamma' AND ABS(energy_keV - 142.68) < 0.2"""
    ).fetchone()[0]
    assert n >= 1
