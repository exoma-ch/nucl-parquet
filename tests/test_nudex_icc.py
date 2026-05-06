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

    def test_row_count_invariants(self, db) -> None:
        # Long-format row count = unique (Z, shell, E_γ) × 10 multipolarities.
        # Don't pin an exact number — pinning breaks on legitimate upstream
        # regen even when the converter is correct.
        n = db.execute("SELECT COUNT(*) FROM icc_factors").fetchone()[0]
        assert n % 10 == 0, f"row count {n} not a multiple of 10 multipolarities"
        assert n >= 500_000, f"row count {n} unexpectedly small — coverage shrunk?"

    def test_no_duplicate_rows(self, db) -> None:
        # Guards against the strata#610 contamination (totals-block rows
        # mis-tagged with the previous shell). The converter dedupes on
        # (Z, shell, E_γ, multipolarity); if dedup is removed or upstream
        # changes shape, this catches it.
        n = db.execute(
            """SELECT COUNT(*) FROM (
                 SELECT Z, shell, gamma_energy_keV, multipolarity, COUNT(*) AS c
                   FROM icc_factors
                  GROUP BY Z, shell, gamma_energy_keV, multipolarity
                 HAVING c > 1
               )"""
        ).fetchone()[0]
        assert n == 0, f"{n} duplicate (Z, shell, E_γ, multipolarity) tuples — strata#610 regression?"

    def test_no_negative_alpha(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM icc_factors WHERE alpha < 0").fetchone()[0]
        assert n == 0

    def test_alpha_finite(self, db) -> None:
        # NULL/NaN protect: '<' returns False for NaN so a separate check is needed.
        n = db.execute(
            "SELECT COUNT(*) FROM icc_factors WHERE alpha IS NULL OR isnan(alpha) OR isinf(alpha)"
        ).fetchone()[0]
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

    def test_no_total_contamination(self, db) -> None:
        # Strata#610: 'TOT' tag ever leaks through, fail loudly.
        n = db.execute("SELECT COUNT(*) FROM icc_factors WHERE shell = 'TOT'").fetchone()[0]
        assert n == 0

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

    def test_cm247_p4_dedup_kept_real_row(self, db) -> None:
        # Strata#610 specific: the contamination at (Z=96, P4, 129 MeV) had
        # M5=6.182e+04. After dedup we must keep the real per-shell row
        # (M5≈1.802, ~5 orders of magnitude smaller).
        rows = db.execute(
            """SELECT alpha FROM icc_factors
               WHERE Z = 96 AND shell = 'P4' AND multipolarity = 'M5'
                 AND ABS(gamma_energy_keV - 129000.0) < 0.5"""
        ).fetchall()
        assert len(rows) == 1, "Cm-247 P4 M5 @ 129 MeV not unique after dedup"
        assert rows[0][0] < 10.0, f"kept the contaminated totals row (alpha={rows[0][0]})"


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
