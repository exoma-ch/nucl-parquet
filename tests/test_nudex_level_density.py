"""Tests for level_density_bfm/ctm/params views (NUDEX, epic #115 / issue #125)."""

from __future__ import annotations

import pytest

from nucl_parquet.loader import connect


@pytest.fixture(scope="module")
def db():
    return connect()


@pytest.mark.data
class TestBfmCtmTables:
    """Back-shifted Fermi-gas + constant-temperature effective parameters
    share the 289-nuclide subset of well-measured s-wave-resonance nuclides."""

    def test_bfm_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM level_density_bfm").fetchone()[0]
        assert n == 289, f"expected 289 BFM rows (well-measured nuclide subset), got {n}"

    def test_ctm_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM level_density_ctm").fetchone()[0]
        assert n == 289

    def test_bfm_ctm_share_nuclides(self, db) -> None:
        # Both tables must cover the same 289 nuclides — BFM and CTM are
        # alternative parameterizations of the same fitted dataset.
        diff = db.execute(
            "SELECT COUNT(*) FROM (SELECT Z, A FROM level_density_bfm EXCEPT SELECT Z, A FROM level_density_ctm)"
        ).fetchone()[0]
        assert diff == 0

    def test_bfm_neutron_binding_positive(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM level_density_bfm WHERE neutron_binding_MeV <= 0").fetchone()[0]
        assert n == 0

    def test_ctm_neutron_binding_positive(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM level_density_ctm WHERE neutron_binding_MeV <= 0").fetchone()[0]
        assert n == 0

    def test_a_asymptotic_physical_range(self, db) -> None:
        # Asymptotic level-density param a~A/8 MeV^-1 (semi-classical).
        # For the BFM 289 set (mostly A=20-250), a should lie in 1-30.
        rows = db.execute(
            "SELECT MIN(a_asymptotic_per_MeV), MAX(a_asymptotic_per_MeV) FROM level_density_bfm"
        ).fetchone()
        assert 1.0 <= rows[0] and rows[1] <= 35.0, f"a_inf out of range: [{rows[0]}, {rows[1]}]"

    def test_d0_in_eV_not_keV(self, db) -> None:
        # Upstream readme says D0 is in eV. Spot-check: closed-shell Pb-208
        # has unusually large D0 (~38 keV); typical mid-A nuclides have
        # D0 ~10-1000 eV. If upstream silently switched to keV, the
        # MAX(D0_eV) would still be < 1e7, but MIN(D0_eV) would no longer
        # land below 100 (which it should, e.g. U-238 D0=3.5 eV thermal).
        # This is the smoke-alarm version — catches an upstream unit shift.
        mn, mx = db.execute("SELECT MIN(D0_eV), MAX(D0_eV) FROM level_density_bfm").fetchone()
        assert mn > 0, f"D0 must be positive, got {mn}"
        assert mn < 100, f"D0 minimum {mn} eV unusually large — did upstream switch units?"
        assert mx < 1e7, f"D0 maximum {mx} eV unusually large — possible unit drift"


@pytest.mark.data
class TestBfmCtmSpotChecks:
    """Faithful pass-through of upstream values — locks in the strata pin."""

    def test_pb208_bfm(self, db) -> None:
        # Pb-208 doubly-magic — well-measured. BFM a_inf ~21.2, B_n=7.37, pairing=+1.80.
        rows = db.execute(
            "SELECT a_asymptotic_per_MeV, neutron_binding_MeV, pairing_MeV "
            "FROM level_density_bfm WHERE Z = 82 AND A = 208"
        ).fetchone()
        assert rows is not None
        assert abs(rows[0] - 21.21) < 0.02
        assert abs(rows[1] - 7.368) < 0.01
        assert abs(rows[2] - 1.801) < 0.01

    def test_au198_bfm(self, db) -> None:
        # Au-198 — well-measured neutron-capture target.
        rows = db.execute(
            "SELECT a_asymptotic_per_MeV, neutron_binding_MeV FROM level_density_bfm WHERE Z = 79 AND A = 198"
        ).fetchone()
        assert rows is not None
        assert abs(rows[0] - 20.57) < 0.02
        assert abs(rows[1] - 6.512) < 0.01

    def test_pb208_ctm_differs_from_bfm(self, db) -> None:
        # CTM and BFM use different parameterizations — the values shouldn't
        # match. If they do, the converter is reading the wrong file.
        bfm = db.execute("SELECT a_asymptotic_per_MeV FROM level_density_bfm WHERE Z = 82 AND A = 208").fetchone()[0]
        ctm = db.execute("SELECT a_asymptotic_per_MeV FROM level_density_ctm WHERE Z = 82 AND A = 208").fetchone()[0]
        assert abs(bfm - ctm) > 0.01, "BFM and CTM identical — file mix-up?"

    def test_bfm_ctm_pairing_differ(self, db) -> None:
        # Defense-in-depth: also assert pairing differs across the table —
        # catches a column-aliasing bug in the shared `_normalize_eff` that
        # would have both files map to the same numeric source.
        n_same = db.execute(
            """SELECT COUNT(*) FROM level_density_bfm bfm
               JOIN level_density_ctm ctm USING (Z, A)
               WHERE ABS(bfm.pairing_MeV - ctm.pairing_MeV) < 1e-6"""
        ).fetchone()[0]
        # Some nuclides could share pairing by coincidence — but not all 289.
        assert n_same < 289, "every (Z,A) has identical BFM/CTM pairing — column-alias bug?"


@pytest.mark.data
class TestParamsTable:
    """Per-nuclide level-density normalization & cutoff parameters (3353 rows)."""

    def test_params_view_registered(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM level_density_params").fetchone()[0]
        assert n >= 3000, f"expected ≥3000 params rows, got {n}"

    def test_no_negative_levels(self, db) -> None:
        n = db.execute("SELECT COUNT(*) FROM level_density_params WHERE N_levels < 0").fetchone()[0]
        assert n == 0

    def test_temperature_physical_range(self, db) -> None:
        # T=0 means "no fit"; nonzero T should sit in the nuclear-physics
        # range (~0.3-8 MeV; 0.3 for actinides up to ~7-8 for very light
        # nuclei where the CTM fit covers fewer levels).
        n = db.execute(
            "SELECT COUNT(*) FROM level_density_params WHERE T_MeV > 0 AND (T_MeV < 0.1 OR T_MeV > 10.0)"
        ).fetchone()[0]
        assert n == 0

    def test_co59_params(self, db) -> None:
        # Spot-check: Co-59 has 165 known levels and a fitted T ~1 MeV.
        rows = db.execute(
            "SELECT N_levels, T_MeV, U_cutoff_MeV FROM level_density_params WHERE Z = 27 AND A = 59"
        ).fetchone()
        assert rows is not None
        assert rows[0] == 165
        assert abs(rows[1] - 1.00302) < 0.001
        assert rows[2] > 0, f"Co-59 U_cutoff_MeV expected positive, got {rows[2]}"

    def test_params_covers_more_nuclides_than_bfm(self, db) -> None:
        # params has 3353 rows (every nuclide with discrete-level data);
        # BFM/CTM has 289 (only those with adequate s-wave resonances).
        n_params = db.execute("SELECT COUNT(*) FROM level_density_params").fetchone()[0]
        n_bfm = db.execute("SELECT COUNT(*) FROM level_density_bfm").fetchone()[0]
        assert n_params > 10 * n_bfm

    def test_ex_sigma_columns_exist(self, db) -> None:
        # EX and sigma_MeV are 100% NaN in current strata pin (placeholder
        # columns not populated upstream). Schema must preserve them so
        # downstream consumers don't break if NUDEX starts populating them.
        cols = {r[0] for r in db.execute("DESCRIBE level_density_params").fetchall()}
        assert "EX_MeV" in cols
        assert "sigma_MeV" in cols

    def test_ex_sigma_all_nan_strata_pin(self, db) -> None:
        # Pin: in current strata revision, EX/sigma columns are 100% NaN.
        # If upstream populates them, this test starts failing — prompting
        # a unit/scale audit before consumers see real values.
        n_finite_ex = db.execute(
            "SELECT COUNT(*) FROM level_density_params WHERE EX_MeV IS NOT NULL AND NOT isnan(EX_MeV)"
        ).fetchone()[0]
        n_finite_sigma = db.execute(
            "SELECT COUNT(*) FROM level_density_params WHERE sigma_MeV IS NOT NULL AND NOT isnan(sigma_MeV)"
        ).fetchone()[0]
        assert n_finite_ex == 0, f"upstream now populates EX_MeV ({n_finite_ex} rows) — audit units!"
        assert n_finite_sigma == 0, f"upstream now populates sigma_MeV ({n_finite_sigma} rows) — audit units!"
