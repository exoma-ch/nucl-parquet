"""Tests for psf_* views (NUDEX photon strength functions, epic #115 / issue #124)."""

from __future__ import annotations

import pytest

from nucl_parquet.loader import connect


@pytest.fixture(scope="module")
def db():
    return connect()


@pytest.mark.data
class TestSmloE1View:
    """IAEA CRP SMLO E1 — modern recommended default."""

    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM psf_e1").fetchone()[0]
        assert n == 8980, f"expected 8980 SMLO E1 rows, got {n}"

    def test_one_row_per_nuclide(self, db) -> None:
        n_unique = db.execute("SELECT COUNT(*) FROM (SELECT DISTINCT Z, A FROM psf_e1)").fetchone()[0]
        n_total = db.execute("SELECT COUNT(*) FROM psf_e1").fetchone()[0]
        assert n_unique == n_total, "psf_e1 should have one row per (Z, A)"

    def test_e1_positive(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM psf_e1 WHERE E1_MeV <= 0").fetchone()[0]
        assert n == 0

    def test_pb208_e1(self, db) -> None:
        # Pb-208 GDR peak sits around 13.4 MeV (well-known experimental value).
        rows = db.execute("SELECT E1_MeV, W1_MeV, S1_mb FROM psf_e1 WHERE Z=82 AND A=208").fetchone()
        assert rows is not None
        assert 13.0 < rows[0] < 14.0, f"Pb-208 E1={rows[0]}, expected ~13.4 MeV"
        assert rows[1] > 0
        assert rows[2] > 0


@pytest.mark.data
class TestExperimentalLorTables:
    """Older Standard Lorentzian (CENPL.GDP-1.1) — multiple measurements per nuclide."""

    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM psf_gdr_lor").fetchone()[0]
        assert n == 145

    def test_unique_on_z_a_ref(self, db) -> None:
        n_unique = db.execute("SELECT COUNT(*) FROM (SELECT DISTINCT Z, A, reference FROM psf_gdr_lor)").fetchone()[0]
        assert n_unique == 145, "(Z, A, reference) should be unique"

    def test_pb208_present_in_lor(self, db) -> None:
        rows = db.execute("SELECT E1_MeV FROM psf_gdr_lor WHERE Z=82 AND A=208").fetchall()
        assert rows, "Pb-208 missing from psf_gdr_lor"


@pytest.mark.data
class TestExperimentalMloSloTables:
    """MLO and SLO share schema; together they let consumers compare
    parameterizations of the same underlying experimental data."""

    def test_mlo_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM psf_gdr_mlo").fetchone()[0]
        assert n == 178

    def test_slo_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM psf_gdr_slo").fetchone()[0]
        assert n == 180

    def test_mlo_slo_share_nuclides(self, db) -> None:
        # MLO covers 120 nuclides; SLO covers 120. The intersection should
        # be substantial (both fit the same upstream measurements).
        n_intersect = db.execute(
            "SELECT COUNT(*) FROM ("
            "  SELECT DISTINCT Z, A FROM psf_gdr_mlo "
            "  INTERSECT "
            "  SELECT DISTINCT Z, A FROM psf_gdr_slo"
            ")"
        ).fetchone()[0]
        assert n_intersect >= 100, f"MLO/SLO intersection only {n_intersect}"

    def test_mlo_slo_differ(self, db) -> None:
        # Different parameterizations: per-row E1 must differ at least
        # somewhere — if they're literally identical, file mix-up.
        n_same = db.execute(
            """SELECT COUNT(*) FROM psf_gdr_mlo m
               JOIN psf_gdr_slo s USING (Z, A, reaction_code, reference)
               WHERE ABS(m.E1_MeV - s.E1_MeV) < 1e-6"""
        ).fetchone()[0]
        n_total = db.execute(
            """SELECT COUNT(*) FROM psf_gdr_mlo m
               JOIN psf_gdr_slo s USING (Z, A, reaction_code, reference)"""
        ).fetchone()[0]
        # Some rows could match by coincidence — but not all of them.
        assert n_same < n_total, "every joined row identical — file mix-up?"

    def test_pb208_slo_peaks(self, db) -> None:
        # Pb-208 SLO has multiple measurements; all should peak around 13.4 MeV.
        rows = db.execute("SELECT E1_MeV FROM psf_gdr_slo WHERE Z=82 AND A=208").fetchall()
        assert len(rows) >= 3
        for (e1,) in rows:
            assert 13.0 < e1 < 14.0, f"Pb-208 SLO peak {e1} outside 13.0-14.0 MeV"


@pytest.mark.data
class TestTheoreticalGdrTable:
    """Goriely systematics — theoretical PSF for nuclides without experiment."""

    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM psf_gdr_theor").fetchone()[0]
        assert n == 5986

    def test_one_row_per_nuclide(self, db) -> None:
        n_unique = db.execute("SELECT COUNT(*) FROM (SELECT DISTINCT Z, A FROM psf_gdr_theor)").fetchone()[0]
        assert n_unique == 5986

    def test_e1_positive(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM psf_gdr_theor WHERE E1_MeV <= 0").fetchone()[0]
        assert n == 0

    def test_deformation_finite(self, db) -> None:
        # Eta ≈ 1 for spherical, > 1 for deformed nuclei. Should be in [0.5, 2].
        rows = db.execute("SELECT MIN(deformation_eta), MAX(deformation_eta) FROM psf_gdr_theor").fetchone()
        assert 0.5 <= rows[0] and rows[1] <= 2.0, f"eta range [{rows[0]}, {rows[1]}] suspicious"


@pytest.mark.data
class TestPhotonuclearTable:
    """Photonuclear (γ,abs)/(γ,sn)/(γ,xn) experimental peaks."""

    def test_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM psf_photonuclear").fetchone()[0]
        assert n == 1912

    def test_resonance_energy_positive(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM psf_photonuclear WHERE Er_MeV <= 0").fetchone()[0]
        assert n == 0

    def test_reaction_codes_known(self, db) -> None:
        rows = db.execute("SELECT DISTINCT reaction FROM psf_photonuclear").fetchall()
        codes = {r[0] for r in rows}
        # Common photonuclear reaction codes: g,abs / g,sn / g,xn / g,n / g,2n
        expected_subset = {"g,abs", "g,sn"}
        assert expected_subset.issubset(codes), f"missing expected reaction codes; got {codes}"

    def test_pb208_photonuclear_primary_peak(self, db) -> None:
        # peak_idx=0 is the primary GDR peak; secondary peaks (peak_idx>=1)
        # are listed separately and can land far from the GDR centroid.
        rows = db.execute(
            "SELECT Er_MeV FROM psf_photonuclear WHERE Z=82 AND A=208 AND reaction='g,sn' AND peak_idx=0"
        ).fetchall()
        assert rows, "Pb-208 (γ,sn) primary peak missing"
        for (er,) in rows:
            assert 12.0 < er < 15.0, f"Pb-208 (γ,sn) primary peak {er} far from GDR ~13.4 MeV"


@pytest.mark.data
class TestCrossTableInvariants:
    """Sanity invariants across the 6 PSF tables."""

    def test_e1_covers_more_than_experimental(self, db) -> None:
        # SMLO E1 covers ~9000 nuclides; experimental tables cover 100-200.
        n_e1 = db.execute("SELECT COUNT(DISTINCT Z*1000+A) FROM psf_e1").fetchone()[0]
        n_lor = db.execute("SELECT COUNT(DISTINCT Z*1000+A) FROM psf_gdr_lor").fetchone()[0]
        assert n_e1 > 10 * n_lor

    def test_theor_covers_more_than_experimental(self, db) -> None:
        # Goriely theoretical covers ~6000 nuclides; experimental ~100-200.
        n_theor = db.execute("SELECT COUNT(DISTINCT Z*1000+A) FROM psf_gdr_theor").fetchone()[0]
        n_lor = db.execute("SELECT COUNT(DISTINCT Z*1000+A) FROM psf_gdr_lor").fetchone()[0]
        assert n_theor > 10 * n_lor

    def test_consistent_pb208_peak(self, db) -> None:
        # All five tables that cover Pb-208 should agree on the GDR peak
        # within ~1 MeV — physical sanity, not just a strata-pin lock.
        e1 = db.execute("SELECT E1_MeV FROM psf_e1 WHERE Z=82 AND A=208").fetchone()[0]
        theor = db.execute("SELECT E1_MeV FROM psf_gdr_theor WHERE Z=82 AND A=208").fetchone()[0]
        lor_avg = db.execute("SELECT AVG(E1_MeV) FROM psf_gdr_lor WHERE Z=82 AND A=208").fetchone()[0]
        slo_avg = db.execute("SELECT AVG(E1_MeV) FROM psf_gdr_slo WHERE Z=82 AND A=208").fetchone()[0]
        for label, val in [("E1", e1), ("theor", theor), ("lor", lor_avg), ("slo", slo_avg)]:
            assert val is not None and 12.5 < val < 14.5, f"Pb-208 {label} peak {val} unexpected"
