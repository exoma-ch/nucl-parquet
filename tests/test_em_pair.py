"""Tests for the photon_pair view (issue #99 — pair + triplet production).

Validates the kinematic thresholds, channel-decomposition invariant, and
order-of-magnitude reference values against published cross-sections.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nucl_parquet.g4.em_pair import (
    NUCLEAR_THRESHOLD_MEV,
    TRIPLET_THRESHOLD_MEV,
    _assert_invariants,
)
from nucl_parquet.loader import connect

_REPO_ROOT = Path(__file__).parent.parent
_PAIR_FILE = _REPO_ROOT / "data" / "em" / "photon_pair.parquet"


@pytest.fixture(scope="module")
def db():
    return connect()


# --------------------------------------------------------------------------- unit tests


class TestInvariants:
    """`_assert_invariants` rejects the exact failure modes documented in #99."""

    def test_negative_xs_rejected(self) -> None:
        bad = pl.DataFrame(
            {
                "Z": [82],
                "energy_MeV": [5.0],
                "sigma_b": [-1.0],
                "channel": ["nuclear"],
            }
        )
        with pytest.raises(ValueError, match="negative cross-section"):
            _assert_invariants(bad)

    def test_nuclear_below_threshold_rejected(self) -> None:
        bad = pl.DataFrame(
            {
                "Z": [82],
                "energy_MeV": [0.5],  # below 1.022 MeV
                "sigma_b": [1.0],
                "channel": ["nuclear"],
            }
        )
        with pytest.raises(ValueError, match="σ_nuclear > 0 below"):
            _assert_invariants(bad)

    def test_triplet_below_threshold_rejected(self) -> None:
        bad = pl.DataFrame(
            {
                "Z": [82],
                "energy_MeV": [1.5],  # below 2.044 MeV
                "sigma_b": [1.0],
                "channel": ["triplet"],
            }
        )
        with pytest.raises(ValueError, match="σ_triplet > 0 below"):
            _assert_invariants(bad)


# --------------------------------------------------------------------------- data tests


@pytest.mark.data
class TestPhotonPairView:
    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM photon_pair").fetchone()[0]
        assert n > 0

    def test_channels(self, db) -> None:
        kinds = {row[0] for row in db.execute("SELECT DISTINCT channel FROM photon_pair").fetchall()}
        assert kinds == {"nuclear", "triplet", "total"}

    def test_no_negative_xs(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM photon_pair WHERE sigma_b < 0").fetchone()[0]
        assert n == 0

    def test_nuclear_threshold(self, db) -> None:
        n = db.execute(
            "SELECT COUNT(*) FROM photon_pair WHERE channel = 'nuclear' AND sigma_b > 0 AND energy_MeV < ?",
            [NUCLEAR_THRESHOLD_MEV],
        ).fetchone()[0]
        assert n == 0

    def test_triplet_threshold(self, db) -> None:
        n = db.execute(
            "SELECT COUNT(*) FROM photon_pair WHERE channel = 'triplet' AND sigma_b > 0 AND energy_MeV < ?",
            [TRIPLET_THRESHOLD_MEV],
        ).fetchone()[0]
        assert n == 0

    @pytest.mark.parametrize("z", [6, 29, 82])
    @pytest.mark.parametrize("e_mev", [5.0, 10.0, 50.0])
    def test_total_equals_nuclear_plus_triplet(self, db, z: int, e_mev: float) -> None:
        # Above 2.5 MeV the channel decomposition is exact within 1%.
        rows = db.execute(
            "SELECT channel, sigma_b FROM photon_pair WHERE Z = ? AND energy_MeV = ?",
            [z, e_mev],
        ).fetchall()
        if not rows:
            pytest.skip(f"no row for Z={z}, E={e_mev} MeV")
        by_channel = {ch: s for ch, s in rows}
        if {"nuclear", "triplet", "total"} - by_channel.keys():
            pytest.skip(f"missing channel for Z={z}, E={e_mev}")
        residual = abs(by_channel["total"] - (by_channel["nuclear"] + by_channel["triplet"]))
        rel = residual / by_channel["total"]
        assert rel < 0.01, f"Z={z}, E={e_mev}: total ≠ nuclear+triplet, rel={rel:.3%}"

    def test_pb_at_5_mev_order_of_magnitude(self, db) -> None:
        # Pb pair production at 5 MeV is of order ~hundreds of barns.
        # XCOM tabulates the pair component for Pb at 5 MeV as ≈ 0.0073 cm²/g
        # × 207.2 g/mol / Nₐ → 731 barns/atom. We accept anything in 500-1000 b.
        sigma = db.execute(
            "SELECT sigma_b FROM photon_pair WHERE Z = 82 AND channel = 'total' AND energy_MeV = 5.0"
        ).fetchone()[0]
        assert 500 <= sigma <= 1000, f"Pb @ 5 MeV total = {sigma:.1f} b"
