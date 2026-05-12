"""Tests for the photon_compton / compton_scattering_function /
compton_doppler_profiles views (issue #97)."""

from __future__ import annotations

import pytest

from nucl_parquet.loader import connect


@pytest.fixture(scope="module")
def db():
    return connect()


@pytest.mark.data
class TestPhotonComptonView:
    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM photon_compton").fetchone()[0]
        assert n > 0

    def test_no_negative_xs(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM photon_compton WHERE sigma_b < 0").fetchone()[0]
        assert n == 0

    @pytest.mark.parametrize(
        # σ_C(1 MeV) reference values (Klein-Nishina × Z, free-electron approx).
        # Bound-electron corrections shift these by 10-30% — use 50% tolerance.
        "z, sigma_ref, tol_rel",
        [
            (6, 1.27, 0.5),  # C
            (29, 5.6, 0.5),  # Cu
            (82, 14.0, 0.5),  # Pb
        ],
    )
    def test_sigma_at_1_mev(self, db, z: int, sigma_ref: float, tol_rel: float) -> None:
        sigma = db.execute(
            "SELECT sigma_b FROM photon_compton WHERE Z = ? AND energy_MeV = 1.0 LIMIT 1",
            [z],
        ).fetchone()
        if sigma is None:
            pytest.skip(f"no row for Z={z} at 1.0 MeV")
        rel = abs(sigma[0] - sigma_ref) / sigma_ref
        assert rel < tol_rel, f"Z={z}: σ_C(1 MeV) = {sigma[0]:.3f} b, ref={sigma_ref}"


@pytest.mark.data
class TestComptonScatteringFunction:
    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM compton_scattering_function").fetchone()[0]
        assert n > 0

    @pytest.mark.parametrize("z", [1, 6, 29, 82])
    def test_low_x_suppression(self, db, z: int) -> None:
        # S(x_min, Z) should be far below Z (bound electron suppression).
        s_low = db.execute(
            "SELECT S FROM compton_scattering_function WHERE Z = ? ORDER BY x_inv_angstrom ASC LIMIT 1",
            [z],
        ).fetchone()[0]
        assert s_low < z, f"Z={z}: S({z=}, x_min) = {s_low}, expected < Z"

    @pytest.mark.parametrize("z", [1, 6, 29, 82])
    def test_high_x_free_electron_limit(self, db, z: int) -> None:
        # S(x_max, Z) → Z within 1% (free-electron limit, Pauli blocking lifted).
        s_high = db.execute(
            "SELECT S FROM compton_scattering_function WHERE Z = ? ORDER BY x_inv_angstrom DESC LIMIT 1",
            [z],
        ).fetchone()[0]
        rel = abs(s_high - z) / z
        assert rel < 0.01, f"Z={z}: S(x_max) = {s_high}, expected ≈ Z={z}"


@pytest.mark.data
class TestComptonDopplerProfiles:
    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM compton_doppler_profiles").fetchone()[0]
        assert n > 0

    def test_z_range(self, db) -> None:
        z_min, z_max = db.execute("SELECT MIN(Z), MAX(Z) FROM compton_doppler_profiles").fetchone()
        assert z_min >= 1
        assert z_max >= 90

    def test_per_shell_profiles_present(self, db) -> None:
        # Pb has multiple shells (K, L1, L2, L3, M1..., N1..., O1..., P1...).
        n_shells = db.execute("SELECT COUNT(DISTINCT shell) FROM compton_doppler_profiles WHERE Z = 82").fetchone()[0]
        assert n_shells >= 10
