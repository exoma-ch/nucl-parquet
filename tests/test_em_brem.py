"""Tests for the electron_brem view (epic #114, P0)."""

from __future__ import annotations

import pytest

from nucl_parquet.loader import connect


@pytest.fixture(scope="module")
def db():
    return connect()


@pytest.mark.data
class TestElectronBremView:
    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM electron_brem").fetchone()[0]
        assert n > 0

    def test_no_negative_xs(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM electron_brem WHERE sigma_mb < 0").fetchone()[0]
        assert n == 0

    def test_no_sentinel_energy(self, db) -> None:
        # Strata's source has sentinel rows with negative energy_mev; those
        # must be filtered out at import.
        n = db.execute("SELECT COUNT(*) FROM electron_brem WHERE energy_MeV <= 0").fetchone()[0]
        assert n == 0

    def test_z_range(self, db) -> None:
        z_min, z_max = db.execute("SELECT MIN(Z), MAX(Z) FROM electron_brem").fetchone()
        assert z_min >= 1
        assert z_max >= 90

    def test_increases_with_z(self, db) -> None:
        # Bremsstrahlung σ scales roughly as Z² at fixed energy.
        # Compare Cu (Z=29) vs Pb (Z=82) at ~100 MeV: Pb should be much larger.
        cu = db.execute(
            "SELECT sigma_mb FROM electron_brem WHERE Z = 29 ORDER BY ABS(energy_MeV - 100.0) ASC LIMIT 1"
        ).fetchone()[0]
        pb = db.execute(
            "SELECT sigma_mb FROM electron_brem WHERE Z = 82 ORDER BY ABS(energy_MeV - 100.0) ASC LIMIT 1"
        ).fetchone()[0]
        # (82/29)² = 8.0 — actual ratio is closer to ~6× due to screening.
        # Generous tolerance: 4× to 12× works.
        ratio = pb / cu
        assert 4.0 < ratio < 12.0, f"σ_brem(Pb/Cu) at 100 MeV = {ratio:.2f}"
