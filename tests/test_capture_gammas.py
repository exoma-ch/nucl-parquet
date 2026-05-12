"""Tests for capture_gammas / capture_gammas_summary views (NUDEX, epic #115)."""

from __future__ import annotations

import polars as pl
import pytest

from nucl_parquet.g4.nudex_capture_gammas import _expand_za
from nucl_parquet.loader import connect


@pytest.fixture(scope="module")
def db():
    return connect()


# --------------------------------------------------------------------------- unit


class TestZaExpansion:
    def test_packs_back(self) -> None:
        df = pl.DataFrame({"za": [27060, 1001, 92238]})
        out = _expand_za(df)
        assert out["Z"].to_list() == [27, 1, 92]
        assert out["A"].to_list() == [60, 1, 238]


# --------------------------------------------------------------------------- data


@pytest.mark.data
class TestCaptureGammas:
    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM capture_gammas").fetchone()[0]
        assert n > 0

    def test_no_negative_intensity(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM capture_gammas WHERE intensity_pct < 0").fetchone()[0]
        assert n == 0

    def test_no_negative_energy(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM capture_gammas WHERE energy_keV < 0").fetchone()[0]
        assert n == 0

    def test_canonical_isotopes_present(self, db) -> None:
        # H-1(n,γ)D — the simplest capture: 2.223 MeV gamma at 100% intensity.
        rows = db.execute(
            "SELECT energy_keV, intensity_pct FROM capture_gammas WHERE Z = 1 AND A = 2 AND variant = 'default'"
        ).fetchall()
        assert rows, "no D capture gamma"
        # Deuteron binding energy ≈ 2224 keV.
        e = rows[0][0]
        assert 2200 < e < 2240, f"H(n,γ)D primary gamma at {e} keV (expected ~2224)"

    def test_co60_capture_present(self, db) -> None:
        # 59Co(n,γ)60Co — common activation channel.
        n = db.execute("SELECT COUNT(*) FROM capture_gammas WHERE Z = 27 AND A = 60").fetchone()[0]
        assert n >= 1


@pytest.mark.data
class TestCaptureGammasSummary:
    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM capture_gammas_summary").fetchone()[0]
        assert n > 0

    def test_separation_energy_is_positive(self, db) -> None:
        # Neutron separation energy (S_n) of bound nuclei is positive.
        # NUDEX includes some unbound entries with s_n_keV = 0; allow zero.
        n = db.execute("SELECT COUNT(*) FROM capture_gammas_summary WHERE s_n_keV < 0").fetchone()[0]
        assert n == 0

    def test_d_separation_energy(self, db) -> None:
        # Deuteron S_n ≈ 2224 keV.
        s_n = db.execute(
            "SELECT s_n_keV FROM capture_gammas_summary WHERE Z = 1 AND A = 2 AND variant = 'default'"
        ).fetchone()
        assert s_n is not None
        assert 2200 < s_n[0] < 2240, f"D S_n = {s_n[0]} keV (expected ~2224)"
