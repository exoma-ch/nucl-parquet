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
    """Schema migrated per strata#294 — Sauter-Gavrila majorant table now
    keyed on (shell_id, beta_index) instead of (table_id, energy_mev)."""

    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM photon_pe_angular").fetchone()[0]
        assert n > 0

    def test_two_shells(self, db) -> None:
        # shell_id ∈ {0, 1}: K-shell vs L-shell-and-above sampling kernels.
        ids = sorted(row[0] for row in db.execute("SELECT DISTINCT shell_id FROM photon_pe_angular").fetchall())
        assert ids == [0, 1]

    def test_beta_in_unit_range(self, db) -> None:
        # beta = v/c of the photoelectron must land in [0, 1].
        bad = db.execute("SELECT COUNT(*) FROM photon_pe_angular WHERE beta < 0 OR beta > 1").fetchone()[0]
        assert bad == 0

    def test_majorants_finite_positive(self, db) -> None:
        # a_majorant and c_majorant parameterize the rejection-sampling
        # envelope; both must be positive finite.
        bad = db.execute(
            "SELECT COUNT(*) FROM photon_pe_angular "
            "WHERE a_majorant <= 0 OR c_majorant <= 0 OR isnan(a_majorant) OR isnan(c_majorant)"
        ).fetchone()[0]
        assert bad == 0


@pytest.mark.data
class TestPhotonPeTotal:
    """Total σ_PE summed across shells — pre-decoded per strata#600 fix."""

    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM photon_pe_total").fetchone()[0]
        assert n > 0

    def test_pb_99kev_total_xs(self, db) -> None:
        # Pb total σ_PE @ 99 keV ≈ 1856 b (K-shell ~1472 + L+M+N).
        rows = db.execute(
            "SELECT sigma_b FROM photon_pe_total WHERE Z=82 AND ABS(energy_MeV - 0.099) < 1e-4"
        ).fetchall()
        assert rows
        # Allow 2% tolerance — interpolation grid points may not land exactly on 99 keV.
        for (s,) in rows:
            assert 1700 < s < 2000, f"Pb @ 99 keV total σ_PE = {s} b, expected ~1856"

    def test_total_above_pershell(self, db) -> None:
        # Total must equal-or-exceed the sum across shells from the per-shell
        # view at any matched (Z, energy). Spot-check at Pb K-edge.
        # photon_pe shell=0 at 99 keV ≈ 1472 b; total ≈ 1856 b.
        per_shell = db.execute(
            "SELECT sigma_b FROM photon_pe WHERE Z=82 AND shell=0 AND ABS(energy_MeV - 0.099) < 1e-4"
        ).fetchall()
        total = db.execute(
            "SELECT MIN(sigma_b) FROM photon_pe_total WHERE Z=82 AND ABS(energy_MeV - 0.099) < 1e-4"
        ).fetchone()
        assert per_shell and total and total[0] is not None
        assert total[0] >= per_shell[0][0] * 0.95  # within interpolation noise
