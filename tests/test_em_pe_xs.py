"""Tests for the photon_pe view (issue #105 — per-shell σ_PE from EPICS2017).

Validates that the E³·CS decode recovers EPDL97 reference values within
tight tolerance, and that 0-indexed shell labels (0=K, 1=L1, ...) are
preserved as published by strata.
"""

from __future__ import annotations

import pytest

from nucl_parquet.loader import connect


@pytest.fixture(scope="module")
def db():
    return connect()


@pytest.mark.data
class TestPhotonPeView:
    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM photon_pe").fetchone()[0]
        assert n > 0

    def test_no_negative_xs(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM photon_pe WHERE sigma_b < 0").fetchone()[0]
        assert n == 0

    def test_z_range(self, db) -> None:
        z_min, z_max = db.execute("SELECT MIN(Z), MAX(Z) FROM photon_pe").fetchone()
        assert z_min >= 1
        assert z_max >= 90

    def test_zero_indexed_shells(self, db) -> None:
        # Strata uses 0-indexed shells: K=0, L1=1, L2=2, L3=3, M1=4, ...
        # We preserve this convention. Pb has 24 shells (0..23).
        max_shell = db.execute("SELECT MAX(shell) FROM photon_pe WHERE Z = 82").fetchone()[0]
        assert max_shell >= 20
        min_shell = db.execute("SELECT MIN(shell) FROM photon_pe WHERE Z = 82").fetchone()[0]
        assert min_shell == 0


@pytest.mark.data
class TestEpdl97Match:
    """σ_PE decoded from EPICS2017 should match EPDL97 within 1% at most
    energies (both derive ultimately from the same atomic-physics tables;
    EPICS2017 is a re-evaluated version with mostly the same numbers).
    """

    @pytest.mark.parametrize(
        "z, energy_mev, sigma_ref_b",
        [
            (82, 0.0990, 1472.0),  # Pb K-shell at 99 keV (EPDL97 = 1472.076 b)
            (82, 0.0950, 1628.0),  # Pb K-shell at 95 keV (EPDL97 = 1628.09 b)
            (82, 0.08829, 1955.0),  # Pb K-shell just above K-edge (EPDL97 = 1955.82 b)
        ],
    )
    def test_pb_k_shell_matches_epdl(self, db, z: int, energy_mev: float, sigma_ref_b: float) -> None:
        sigma = db.execute(
            "SELECT sigma_b FROM photon_pe WHERE Z = ? AND shell = 0 AND ABS(energy_MeV - ?) < 1e-5 LIMIT 1",
            [z, energy_mev],
        ).fetchone()
        if sigma is None:
            pytest.skip(f"no row for Z={z}, E={energy_mev}, shell=K")
        rel = abs(sigma[0] - sigma_ref_b) / sigma_ref_b
        assert rel < 0.01, f"Z={z}, E={energy_mev}: σ={sigma[0]:.2f} b, EPDL97 ref={sigma_ref_b}"


@pytest.mark.data
class TestPhysicalSanity:
    def test_pb_k_decreases_with_energy(self, db) -> None:
        # Photoelectric falls as ~1/E^3.5 above edges. EPICS2017 covers
        # Pb K-shell only to ~190 keV; compare 100 keV to ~180 keV.
        s_100 = db.execute(
            "SELECT sigma_b FROM photon_pe WHERE Z = 82 AND shell = 0 ORDER BY ABS(energy_MeV - 0.100) ASC LIMIT 1"
        ).fetchone()[0]
        s_180 = db.execute(
            "SELECT sigma_b FROM photon_pe WHERE Z = 82 AND shell = 0 ORDER BY ABS(energy_MeV - 0.180) ASC LIMIT 1"
        ).fetchone()[0]
        assert s_180 < s_100, f"Pb K σ at 180 keV ({s_180}) >= 100 keV ({s_100})"
        # σ should drop by at least 4× over a 1.8× energy increase (1/E^3.5 → 7×).
        assert s_100 / s_180 > 4
