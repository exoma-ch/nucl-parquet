"""Tests for nudex_levels/level_gammas/isotopes views (epic #115 / issue #122)."""

from __future__ import annotations

import pytest

from nucl_parquet.loader import connect


@pytest.fixture(scope="module")
def db():
    return connect()


@pytest.mark.data
class TestNudexLevels:
    """Per-level data — energy, spin/parity, half-life, decay modes."""

    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM nudex_levels").fetchone()[0]
        # Upstream 158919; we filter 19 strata#621 orphans → 158900.
        assert n == 158900, f"expected 158900 levels (post strata#621 filter), got {n}"

    def test_no_null_z(self, db) -> None:
        # strata#621 protection — all rows must have valid Z and A.
        n = db.execute("SELECT COUNT(*) FROM nudex_levels WHERE Z IS NULL OR A IS NULL").fetchone()[0]
        assert n == 0

    def test_isotope_coverage(self, db) -> None:
        n_iso = db.execute("SELECT COUNT(DISTINCT (Z, A)) FROM nudex_levels").fetchone()[0]
        assert n_iso == 3331

    def test_no_negative_energy(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM nudex_levels WHERE energy_MeV < 0").fetchone()[0]
        assert n == 0

    def test_level_idx_one_indexed(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM nudex_levels WHERE level_idx < 1").fetchone()[0]
        assert n == 0

    def test_co60_ground_state_half_life(self, db) -> None:
        # Co-60 ground state: 5.2714 yr = 1.663e8 s — textbook calibration source.
        rows = db.execute(
            "SELECT half_life_s, jpi_str FROM nudex_levels WHERE Z=27 AND A=60 AND level_idx=1"
        ).fetchone()
        assert rows is not None
        # 1.663e8 s within 1%.
        assert 1.6e8 < rows[0] < 1.7e8, f"Co-60 ground T_1/2={rows[0]} s, expected ~1.663e8"
        assert rows[1] == "5+", f"Co-60 ground J^π={rows[1]}, expected 5+"

    def test_co60m_isomer(self, db) -> None:
        # Co-60m metastable state at 58.6 keV with 10.467 min half-life (628 s).
        rows = db.execute(
            "SELECT energy_MeV, half_life_s FROM nudex_levels WHERE Z=27 AND A=60 AND level_idx=2"
        ).fetchone()
        assert rows is not None
        assert abs(rows[0] - 0.05859) < 1e-4, f"Co-60m at {rows[0]} MeV, expected 0.0586"
        assert 600 < rows[1] < 650, f"Co-60m T_1/2={rows[1]} s, expected ~628"

    def test_parity_is_signed(self, db) -> None:
        # Parity must be +1 or -1 (or 0 for unknown, depending on convention).
        rows = db.execute("SELECT DISTINCT parity FROM nudex_levels").fetchall()
        parities = {r[0] for r in rows}
        assert parities.issubset({-1, 0, 1}), f"unexpected parity values: {parities}"


@pytest.mark.data
class TestNudexLevelGammas:
    """Per-transition gamma data — source/dest level, energy, intensity."""

    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM nudex_level_gammas").fetchone()[0]
        assert n == 245975

    def test_no_negative_energy(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM nudex_level_gammas WHERE gamma_energy_MeV < 0").fetchone()[0]
        assert n == 0

    def test_no_negative_intensity(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM nudex_level_gammas WHERE intensity < 0").fetchone()[0]
        assert n == 0

    def test_source_above_dest(self, db) -> None:
        # Cascade direction: source level idx must be > destination idx.
        # Allow a small tolerance for upstream weirdness (logged as warning at build).
        n = db.execute("SELECT COUNT(*) FROM nudex_level_gammas WHERE source_level_idx <= dest_level_idx").fetchone()[0]
        assert n < 100, f"{n} transitions with source ≤ dest — cascade-direction regression?"

    def test_co60_canonical_gammas(self, db) -> None:
        # Co-60 textbook decay: 1173 keV + 1332 keV gamma cascade (β⁻ to Ni-60 levels).
        # In this view we'd see gammas BETWEEN levels of Co-60 itself, not the
        # Ni-60 daughter cascade. Just check that some gammas exist.
        n = db.execute("SELECT COUNT(*) FROM nudex_level_gammas WHERE Z=27 AND A=60").fetchone()[0]
        assert n > 0, "no gamma transitions tabulated for Co-60"

    def test_pb208_high_energy_transitions(self, db) -> None:
        # Pb-208 has well-known excited states with high-energy γ transitions.
        # Top transitions should land in physical 1-10 MeV range.
        rows = db.execute("SELECT MAX(gamma_energy_MeV) FROM nudex_level_gammas WHERE Z=82 AND A=208").fetchone()
        assert rows[0] is not None and 1 < rows[0] < 15


@pytest.mark.data
class TestNudexIsotopes:
    """Per-isotope summary — counts, mass excess, neutron separation."""

    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM nudex_isotopes").fetchone()[0]
        assert n == 3331

    def test_unique_z_a(self, db) -> None:
        n_unique = db.execute("SELECT COUNT(DISTINCT (Z, A)) FROM nudex_isotopes").fetchone()[0]
        assert n_unique == 3331

    def test_pb208_summary(self, db) -> None:
        # Pb-208 — doubly magic, well-known spectroscopy.
        # Mass excess ≈ -21.749 MeV per AME2020; but G4NUDEX sources may differ.
        # S_n ≈ 7.367 MeV for Pb-208.
        rows = db.execute(
            "SELECT n_levels_total, n_gammas_total, s_n_MeV FROM nudex_isotopes WHERE Z=82 AND A=208"
        ).fetchone()
        assert rows is not None
        assert rows[0] >= 100, f"Pb-208 has {rows[0]} levels, expected ≥100"
        assert rows[1] >= 50, f"Pb-208 has {rows[1]} gammas, expected ≥50"
        assert 7.0 < rows[2] < 8.5, f"Pb-208 S_n={rows[2]} MeV, expected ~7.37"

    def test_d2_separation_energy(self, db) -> None:
        # Deuteron neutron-separation energy = 2.224 MeV (textbook).
        rows = db.execute("SELECT s_n_MeV FROM nudex_isotopes WHERE Z=1 AND A=2").fetchone()
        assert rows is not None
        assert abs(rows[0] - 2.22457) < 0.001


@pytest.mark.data
class TestCrossTableInvariants:
    """Sanity invariants linking levels / level_gammas / isotopes."""

    def test_isotopes_with_levels_have_level_rows(self, db) -> None:
        # Every (Z, A) row in nudex_isotopes with n_levels_total > 0 should
        # have at least one matching row in nudex_levels.
        # (The exact count semantics of n_levels_in_table vs n_levels_total
        # vs the actual nudex_levels row count are upstream-opaque, so we
        # only check existence.)
        rows = db.execute(
            """SELECT i.Z, i.A
                 FROM nudex_isotopes i
                 LEFT JOIN (SELECT DISTINCT Z, A FROM nudex_levels) l
                   ON i.Z = l.Z AND i.A = l.A
                WHERE i.n_levels_total > 0 AND l.Z IS NULL"""
        ).fetchall()
        assert len(rows) == 0, f"{len(rows)} isotopes claim levels but have none in nudex_levels"

    def test_coexists_with_v011_ensdf_levels(self, db) -> None:
        # Both views must work side-by-side. ensdf_levels (PhotonEvaporation
        # summary) and nudex_levels (full ENSDF) overlap on (Z, A) but
        # differ in level set / count.
        n_nudex = db.execute("SELECT COUNT(DISTINCT (Z, A)) FROM nudex_levels").fetchone()[0]
        n_ensdf_iso = db.execute("SELECT COUNT(DISTINCT (Z, A)) FROM ensdf_levels").fetchone()[0]
        assert n_nudex > 0 and n_ensdf_iso > 0
        # nudex covers more isotopes (3331 vs ~3000-something for ensdf).
        # Allow ensdf >= nudex too — different sources.
