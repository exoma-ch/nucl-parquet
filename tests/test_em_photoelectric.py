"""Tests for the photon_pe_high_z_params + photon_pe_angular views (issue #96).

The per-shell σ_PE table from strata's EPICS2017 dataset has a suspected
unit-conversion bug upstream (values ~600× too low vs XCOM); deferred to
a follow-up once strata fixes upstream. Existing ``epdl_subshell_pe`` view
provides per-shell PE in the meantime.
"""

from __future__ import annotations

import pytest

from nucl_parquet.loader import connect


@pytest.fixture(scope="module")
def db():
    return connect()


@pytest.mark.data
class TestHighZParams:
    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM photon_pe_high_z_params").fetchone()[0]
        assert n > 0

    def test_one_row_per_z(self, db) -> None:
        # Strata ships analytic params for Z=1..100, one row each.
        n = db.execute("SELECT COUNT(DISTINCT Z) FROM photon_pe_high_z_params").fetchone()[0]
        assert n >= 90

    def test_boundaries_ordered(self, db) -> None:
        # high_boundary_keV > low_boundary_keV per Z (analytic regions don't
        # overlap with reversed limits).
        n_bad = db.execute(
            "SELECT COUNT(*) FROM photon_pe_high_z_params WHERE high_boundary_kev <= low_boundary_kev"
        ).fetchone()[0]
        assert n_bad == 0


@pytest.mark.data
class TestAngular:
    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM photon_pe_angular").fetchone()[0]
        assert n > 0

    def test_two_table_ids(self, db) -> None:
        # G4 ships table_id ∈ {0, 1}: K-shell vs other-shell sampling kernels.
        ids = sorted(row[0] for row in db.execute("SELECT DISTINCT table_id FROM photon_pe_angular").fetchall())
        assert ids == [0, 1]

    def test_energy_range(self, db) -> None:
        e_min, e_max = db.execute("SELECT MIN(energy_MeV), MAX(energy_MeV) FROM photon_pe_angular").fetchone()
        # Photoelectric angular kernel grid covers ~20 keV to ~1 MeV.
        assert e_min < 0.05
        assert e_max >= 0.99
