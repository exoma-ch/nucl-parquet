"""Tests for the photon_rayleigh_cdf + xray_form_factor views (issue #98)."""

from __future__ import annotations

import polars as pl
import pytest

from nucl_parquet.g4.em_rayleigh import (
    _F1_BELOW_THRESHOLD_SENTINEL,
    _assert_cdf_invariants,
)
from nucl_parquet.loader import connect


@pytest.fixture(scope="module")
def db():
    return connect()


# --------------------------------------------------------------------------- unit tests


class TestCdfInvariants:
    def test_non_zero_origin_rejected(self) -> None:
        bad = pl.DataFrame({"Z": [82, 82], "q_inv_angstrom": [0.1, 1.0], "cdf": [0.0, 1.0]})
        with pytest.raises(ValueError, match="does not start at q=0"):
            _assert_cdf_invariants(bad)

    def test_non_monotone_rejected(self) -> None:
        bad = pl.DataFrame(
            {
                "Z": [82, 82, 82],
                "q_inv_angstrom": [0.0, 0.5, 1.0],
                "cdf": [0.0, 0.5, 0.4],  # drops
            }
        )
        with pytest.raises(ValueError, match="monotone"):
            _assert_cdf_invariants(bad)


# --------------------------------------------------------------------------- data tests


@pytest.mark.data
class TestPhotonRayleighCdfView:
    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM photon_rayleigh_cdf").fetchone()[0]
        assert n > 0

    def test_cdf_starts_at_zero_each_z(self, db) -> None:
        rows = db.execute("SELECT Z, MIN(q_inv_angstrom), MIN(cdf) FROM photon_rayleigh_cdf GROUP BY Z").fetchall()
        for z, q_min, cdf_min in rows:
            assert q_min == 0.0, f"Z={z}: min q = {q_min}"
            assert cdf_min == 0.0, f"Z={z}: cdf at q=0 = {cdf_min}"

    def test_cdf_approaches_one(self, db) -> None:
        # Upstream G4 sample CDFs go to 1 within a small numerical tolerance.
        rows = db.execute("SELECT Z, MAX(cdf) FROM photon_rayleigh_cdf GROUP BY Z").fetchall()
        for z, cdf_max in rows:
            assert 0.99 <= cdf_max <= 1.001, f"Z={z}: max cdf = {cdf_max}"


@pytest.mark.data
class TestXrayFormFactor:
    def test_no_sentinel_rows(self, db) -> None:
        n = db.execute(
            "SELECT COUNT(*) FROM xray_form_factor WHERE f1 = ?",
            [_F1_BELOW_THRESHOLD_SENTINEL],
        ).fetchone()[0]
        assert n == 0

    @pytest.mark.parametrize(
        "z, expected_f1",
        [(6, 6.0), (29, 29.0), (47, 47.0), (50, 50.0), (74, 74.0), (82, 82.0)],
    )
    def test_thomson_limit_approached(self, db, z: int, expected_f1: float) -> None:
        # f1 → Z at the highest tabulated energy (Thomson limit). The Henke
        # tables top out at 30 keV; for elements whose K-edge sits near that
        # cutoff (e.g. Sn at 29.2 keV), the anomalous correction is still ~5%
        # there, so the asymptote isn't fully reached. Accept 6% — physics is
        # correct, the tabulation just doesn't extend high enough.
        f1_max = db.execute(
            "SELECT f1 FROM xray_form_factor WHERE Z = ? ORDER BY energy_eV DESC LIMIT 1",
            [z],
        ).fetchone()[0]
        rel = abs(f1_max - expected_f1) / expected_f1
        assert rel < 0.06, f"Z={z}: f1 at top E = {f1_max}, expected ~{expected_f1}"
