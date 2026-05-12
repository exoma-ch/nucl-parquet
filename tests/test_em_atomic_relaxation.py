"""Tests for the atomic_relaxation / fluorescence DuckDB views (issue #100).

Promotes EADL relaxation data from internal-only (consumed by g4/xray_auger.py)
to first-class DuckDB views: ``atomic_relaxation`` (full vacancy cascade) and
``fluorescence`` (radiative subset only).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nucl_parquet.loader import connect

_REPO_ROOT = Path(__file__).parent.parent
_EADL_DIR = _REPO_ROOT / "data" / "meta" / "eadl"

# Hubbell & Trehan (1985) / Krause (1979) tabulation of K-shell fluorescence
# yields ω_K. Used as the reference for the sweep invariant — EADL agrees with
# these to ~2% across the range.
_HUBBELL_OMEGA_K = {
    20: 0.163,  # Ca
    26: 0.347,  # Fe
    29: 0.443,  # Cu
    47: 0.831,  # Ag
    50: 0.859,  # Sn
    74: 0.946,  # W
    82: 0.961,  # Pb
}


@pytest.fixture(scope="module")
def db():
    return connect()


@pytest.mark.data
class TestAtomicRelaxationView:
    def test_atomic_relaxation_view_exists(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM atomic_relaxation").fetchone()[0]
        assert n > 0

    def test_eadl_transitions_alias_matches(self, db) -> None:
        a = db.execute("SELECT COUNT(*) FROM atomic_relaxation").fetchone()[0]
        b = db.execute("SELECT COUNT(*) FROM eadl_transitions").fetchone()[0]
        assert a == b

    def test_transition_types_are_radiative_or_auger(self, db) -> None:
        kinds = {row[0] for row in db.execute("SELECT DISTINCT transition_type FROM atomic_relaxation").fetchall()}
        assert kinds <= {"radiative", "auger"}

    def test_z_range_matches_eadl_coverage(self, db) -> None:
        z_min, z_max = db.execute("SELECT MIN(Z), MAX(Z) FROM atomic_relaxation").fetchone()
        # EADL covers Z=6 through Z=100.
        assert z_min >= 6
        assert z_max >= 92


@pytest.mark.data
class TestFluorescenceView:
    def test_fluorescence_is_radiative_subset(self, db) -> None:
        kinds = {row[0] for row in db.execute("SELECT DISTINCT transition_type FROM fluorescence").fetchall()}
        assert kinds == {"radiative"}

    def test_fluorescence_count_matches_radiative_subset(self, db) -> None:
        a = db.execute("SELECT COUNT(*) FROM atomic_relaxation WHERE transition_type='radiative'").fetchone()[0]
        b = db.execute("SELECT COUNT(*) FROM fluorescence").fetchone()[0]
        assert a == b


@pytest.mark.data
class TestKShellFluorescenceYield:
    """ω_K = sum of probabilities for radiative transitions filling a K-vacancy.

    EADL is normalized so radiative + auger probabilities for a given vacancy
    sum to 1; therefore ω_K = SUM(probability) for radiative K-vacancy rows.
    Validate against Hubbell-Trehan / Krause within 5% across the range
    Z=20..82.
    """

    @pytest.mark.parametrize("z, omega_ref", sorted(_HUBBELL_OMEGA_K.items()))
    def test_omega_k_matches_hubbell(self, db, z: int, omega_ref: float) -> None:
        omega = db.execute(
            "SELECT COALESCE(SUM(probability), 0) FROM fluorescence WHERE Z = ? AND vacancy_shell = 'K'",
            [z],
        ).fetchone()[0]
        # 5% absolute tolerance for low-Z (where ω is small and the relative
        # error noise dominates), 5% relative tolerance for high-Z.
        tol = max(0.05 * omega_ref, 0.02)
        assert abs(omega - omega_ref) < tol, f"Z={z}: ω_K={omega:.4f}, Hubbell-Trehan ref={omega_ref:.4f}"


@pytest.mark.data
class TestVacancyNormalization:
    """Sum of probabilities over all transitions filling a given vacancy ≈ 1.

    EADL ships per-vacancy normalized cascades; this is the sweep invariant
    that downstream consumers rely on. Allow 5% slack — EADL doesn't always
    sum to exactly 1 due to omitted minor branches.
    """

    @pytest.mark.parametrize("z", [26, 50, 82])
    def test_k_vacancy_sums_to_unity(self, db, z: int) -> None:
        total = db.execute(
            "SELECT SUM(probability) FROM atomic_relaxation WHERE Z = ? AND vacancy_shell = 'K'",
            [z],
        ).fetchone()[0]
        assert 0.95 <= total <= 1.05, f"Z={z}: K-vacancy sum = {total:.4f}"
