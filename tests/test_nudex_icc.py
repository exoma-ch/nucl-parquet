"""Tests for icc_factors view (NUDEX per-shell ICC, epic #115 / issue #123)."""

from __future__ import annotations

import pytest

from nucl_parquet.loader import connect


@pytest.fixture(scope="module")
def db():
    return connect()


_SHELLS = {
    "K",
    "L1",
    "L2",
    "L3",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "N1",
    "N2",
    "N3",
    "N4",
    "N5",
    "N6",
    "N7",
    "O1",
    "O2",
    "O3",
    "O4",
    "O5",
    "O6",
    "O7",
    "O8",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "P6",
    "Q1",
    "Q2",
    "Q3",
    "R1",
    "R2",
}
_MULTIPOLARITIES = {"E1", "E2", "E3", "E4", "E5", "M1", "M2", "M3", "M4", "M5"}


@pytest.mark.data
class TestIccFactorsView:
    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM icc_factors").fetchone()[0]
        assert n > 0

    def test_expected_row_count(self, db) -> None:
        # Upstream: 61140 (Z, A, shell, E_γ) wide rows × 10 multipolarities.
        n = db.execute("SELECT COUNT(*) FROM icc_factors").fetchone()[0]
        assert n == 611400, f"expected 611400 long-format rows, got {n}"

    def test_no_negative_alpha(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM icc_factors WHERE alpha < 0").fetchone()[0]
        assert n == 0

    def test_binding_energy_positive(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM icc_factors WHERE binding_energy_eV <= 0").fetchone()[0]
        assert n == 0

    def test_gamma_above_binding(self, db) -> None:
        # Below shell binding, conversion is energetically forbidden — upstream
        # only tabulates above-threshold rows. Validates the upstream invariant
        # so any future schema change that violates it gets caught.
        n = db.execute(
            "SELECT COUNT(*) FROM icc_factors WHERE gamma_energy_keV * 1000.0 <= binding_energy_eV"
        ).fetchone()[0]
        assert n == 0

    def test_all_shells_present(self, db) -> None:
        rows = db.execute("SELECT DISTINCT shell FROM icc_factors").fetchall()
        shells = {r[0] for r in rows}
        assert shells == _SHELLS, f"unexpected shell set: {shells ^ _SHELLS}"

    def test_all_multipolarities_present(self, db) -> None:
        rows = db.execute("SELECT DISTINCT multipolarity FROM icc_factors").fetchall()
        mults = {r[0] for r in rows}
        assert mults == _MULTIPOLARITIES

    def test_z_coverage(self, db) -> None:
        # NUDEX's ICC tables cover roughly Z=10 (Ne) up through Z=110 (Ds).
        zmin, zmax, nz = db.execute("SELECT MIN(Z), MAX(Z), COUNT(DISTINCT Z) FROM icc_factors").fetchone()
        assert zmin <= 12 and zmax >= 92
        assert nz >= 100, f"only {nz} unique Z — coverage shrunk?"


@pytest.mark.data
class TestUpstreamSpotChecks:
    """Faithful pass-through of upstream values — guards against silent
    decode/scale changes during ETL refactors."""

    def test_ne20_k_shell_e1_at_1_mev(self, db) -> None:
        # From G4NUDEXLIB1.0/ICC_factors.dat row 1: Ne-20 K-shell E1 at 1.0 MeV
        # = 3.352E+03. The strata wide table preserves this; long-format must too.
        rows = db.execute(
            """SELECT alpha FROM icc_factors
               WHERE Z = 10 AND shell = 'K'
                 AND multipolarity = 'E1'
                 AND ABS(gamma_energy_keV - 1000.0) < 0.5"""
        ).fetchall()
        assert rows, "Ne-20 K-shell E1 @ 1 MeV missing"
        assert abs(rows[0][0] - 3352.0) < 1.0, f"got {rows[0][0]}, expected 3352"

    def test_ne20_k_binding_energy(self, db) -> None:
        # Ne K-shell binding ≈ 870 eV (matches the upstream header line).
        rows = db.execute("SELECT DISTINCT binding_energy_eV FROM icc_factors WHERE Z = 10 AND shell = 'K'").fetchall()
        assert len(rows) == 1
        assert 860 < rows[0][0] < 880, f"got {rows[0][0]} eV, expected ~870"

    def test_ne20_l1_binding_energy(self, db) -> None:
        # Ne L1-shell binding ≈ 48 eV (header line for L1 block).
        rows = db.execute("SELECT DISTINCT binding_energy_eV FROM icc_factors WHERE Z = 10 AND shell = 'L1'").fetchall()
        assert len(rows) == 1
        assert 40 < rows[0][0] < 60


@pytest.mark.data
class TestDecreasingAlphaWithEnergy:
    """ICC factors strictly decrease as gamma energy rises far above threshold —
    a textbook physics invariant. Lets us catch ordering / pivot bugs."""

    def test_pb_k_e1_decreases(self, db) -> None:
        # Pb (Z=82) K-shell E1: at high energies (well above K binding ~88 keV),
        # alpha must decrease monotonically with E_γ.
        rows = db.execute(
            """SELECT gamma_energy_keV, alpha FROM icc_factors
               WHERE Z = 82 AND shell = 'K' AND multipolarity = 'E1'
                 AND gamma_energy_keV >= 200000
               ORDER BY gamma_energy_keV"""
        ).fetchall()
        assert len(rows) >= 4
        alphas = [r[1] for r in rows]
        for prev, curr in zip(alphas, alphas[1:]):
            assert curr < prev, f"alpha non-monotonic: {prev} -> {curr}"
