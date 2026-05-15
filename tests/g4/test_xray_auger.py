"""Tests for the X-ray + Auger synthesis G4 module (issue #74).

Two layers (matches the convention from `tests/g4/test_radioactive_decay.py`):

- Synthetic unit tests on small hand-built DataFrames covering pure helpers
  (subtype labelling, decay-mode → shell mapping, state assignment) and the
  small-input convolution.
- ``@pytest.mark.data`` spot-checks against the real strata + EADL inputs:
  Tc-99m, Co-57, I-125 emission fingerprints; per-vacancy
  energy-conservation invariant.

The synthetic layer must run with no data files present.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nucl_parquet.g4.xray_auger import (
    _SHELL_TOTAL_TO_VACANCY,
    assign_state,
    auger_subtype,
    check_energy_conservation,
    compute_ec_vacancy_rates,
    compute_ic_vacancy_rates,
    map_decay_mode_to_shell,
    synthesize_emissions,
    synthesize_xray_auger,
    xray_subtype,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EADL_DIR = REPO_ROOT / "data" / "meta" / "eadl"
DECAY_DETAILED = REPO_ROOT / "data" / "meta" / "decay_detailed.parquet"
GAMMAS = REPO_ROOT / "data" / "g4_raw" / "strata-nuclear" / "photon_evap_gammas.parquet"
LEVELS = REPO_ROOT / "data" / "g4_raw" / "strata-nuclear" / "photon_evap_levels.parquet"
NUCLIDES = REPO_ROOT / "data" / "meta" / "ensdf" / "nuclides.parquet"

_real_data = pytest.mark.skipif(
    not (EADL_DIR.exists() and DECAY_DETAILED.exists() and GAMMAS.exists() and LEVELS.exists() and NUCLIDES.exists()),
    reason="strata + EADL + decay_detailed inputs not all present",
)


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _make_decay_detailed(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "Z": pl.Int32,
        "A": pl.Int32,
        "parent_ex_kev": pl.Float64,
        "parent_level_flag": pl.Utf8,
        "half_life_s": pl.Float64,
        "decay_mode": pl.Utf8,
        "daughter_Z": pl.Int32,
        "daughter_A": pl.Int32,
        "daughter_ex_kev": pl.Float64,
        "daughter_level_flag": pl.Utf8,
        "branching": pl.Float64,
        "q_value_kev": pl.Float64,
        "forbiddenness": pl.Utf8,
    }
    return pl.DataFrame(rows, schema=schema)


def _make_eadl(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "Z": pl.Int32,
        "vacancy_shell": pl.Utf8,
        "filling_shell": pl.Utf8,
        "transition_type": pl.Utf8,
        "energy_keV": pl.Float64,
        "probability": pl.Float64,
        "edge_keV": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def _make_nuclides(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic unit tests
# ---------------------------------------------------------------------------


class TestPureHelpers:
    """Pure-function tests, no DataFrames touched."""

    def test_xray_subtype_canonical_K(self) -> None:
        assert xray_subtype("K", "L3") == "Kα1"
        assert xray_subtype("K", "L2") == "Kα2"
        assert xray_subtype("K", "M3") == "Kβ1"
        assert xray_subtype("K", "M2") == "Kβ3"

    def test_xray_subtype_canonical_L(self) -> None:
        assert xray_subtype("L3", "M5") == "Lα1"
        assert xray_subtype("L3", "M4") == "Lα2"
        assert xray_subtype("L2", "M4") == "Lβ1"

    def test_xray_subtype_unknown_falls_back(self) -> None:
        # Combo not in the IUPAC table (purely synthetic) must yield
        # "{vacancy}-{filling}" so we never silently drop a transition.
        assert xray_subtype("Q1", "Q2") == "Q1-Q2"

    def test_auger_subtype_KLL(self) -> None:
        assert auger_subtype("K", "L1") == "KLL"
        assert auger_subtype("K", "L2") == "KLL"
        assert auger_subtype("K", "L3") == "KLL"

    def test_auger_subtype_LMM_and_KMM(self) -> None:
        assert auger_subtype("L1", "M1") == "LMM"
        assert auger_subtype("L3", "M5") == "LMM"
        assert auger_subtype("K", "M2") == "KMM"

    def test_decay_mode_to_shell(self) -> None:
        assert map_decay_mode_to_shell("KshellEC") == "K"
        assert map_decay_mode_to_shell("LshellEC") == "L"
        assert map_decay_mode_to_shell("MshellEC") == "M"
        assert map_decay_mode_to_shell("NshellEC") == "N"
        assert map_decay_mode_to_shell("BetaMinus") is None
        assert map_decay_mode_to_shell("IT") is None

    def test_shell_total_to_vacancy_mapping(self) -> None:
        # The L/M/N total-to-L1/M1/N1 approximation must be present and
        # consistent. K maps to K (no sub-shell).
        assert _SHELL_TOTAL_TO_VACANCY == {"K": "K", "L": "L1", "M": "M1", "N": "N1"}


class TestStateAssignment:
    """`assign_state` is the v0.10.x state-synthesis convention."""

    @pytest.fixture()
    def nuclides(self) -> pl.DataFrame:
        return _make_nuclides(
            [
                {"Z": 43, "A": 99, "state": "", "symbol": "Tc", "level_keV": 0.0},
                {"Z": 43, "A": 99, "state": "m", "symbol": "Tc", "level_keV": 142.6836},
                {"Z": 27, "A": 57, "state": "", "symbol": "Co", "level_keV": 0.0},
            ]
        )

    def test_ground_state_zero(self, nuclides: pl.DataFrame) -> None:
        assert assign_state(43, 99, 0.0, nuclides) == ""

    def test_isomer_within_tolerance(self, nuclides: pl.DataFrame) -> None:
        assert assign_state(43, 99, 142.6836, nuclides) == "m"
        # Within ±1 keV
        assert assign_state(43, 99, 142.6, nuclides) == "m"
        assert assign_state(43, 99, 143.5, nuclides) == "m"

    def test_excited_no_match_returns_none(self, nuclides: pl.DataFrame) -> None:
        # 200 keV — not an isomer in our catalog, must be skipped (return None).
        assert assign_state(43, 99, 200.0, nuclides) is None


class TestComputeECVacancyRates:
    def test_aggregates_per_shell(self) -> None:
        # Co-57 EC: K = 0.886+0.0015 = 0.88766, L = 0.0954+0.000179 = 0.09558
        src = _make_decay_detailed(
            [
                {
                    "Z": 27,
                    "A": 57,
                    "parent_ex_kev": 0.0,
                    "parent_level_flag": "-",
                    "half_life_s": 1e7,
                    "decay_mode": "KshellEC",
                    "daughter_Z": 26,
                    "daughter_A": 57,
                    "daughter_ex_kev": 136.0,
                    "daughter_level_flag": "-",
                    "branching": 0.886,
                    "q_value_kev": 700.0,
                    "forbiddenness": "",
                },
                {
                    "Z": 27,
                    "A": 57,
                    "parent_ex_kev": 0.0,
                    "parent_level_flag": "-",
                    "half_life_s": 1e7,
                    "decay_mode": "KshellEC",
                    "daughter_Z": 26,
                    "daughter_A": 57,
                    "daughter_ex_kev": 706.0,
                    "daughter_level_flag": "-",
                    "branching": 0.0015,
                    "q_value_kev": 130.0,
                    "forbiddenness": "",
                },
                {
                    "Z": 27,
                    "A": 57,
                    "parent_ex_kev": 0.0,
                    "parent_level_flag": "-",
                    "half_life_s": 1e7,
                    "decay_mode": "LshellEC",
                    "daughter_Z": 26,
                    "daughter_A": 57,
                    "daughter_ex_kev": 136.0,
                    "daughter_level_flag": "-",
                    "branching": 0.0954,
                    "q_value_kev": 700.0,
                    "forbiddenness": "",
                },
                # Non-EC row should be ignored.
                {
                    "Z": 27,
                    "A": 57,
                    "parent_ex_kev": 0.0,
                    "parent_level_flag": "-",
                    "half_life_s": 1e7,
                    "decay_mode": "BetaMinus",
                    "daughter_Z": 28,
                    "daughter_A": 57,
                    "daughter_ex_kev": 0.0,
                    "daughter_level_flag": "-",
                    "branching": 0.001,
                    "q_value_kev": 100.0,
                    "forbiddenness": "",
                },
            ]
        )
        out = compute_ec_vacancy_rates(src)
        # Two shell rows expected: K (sum = 0.8875), L (= 0.0954).
        assert out.height == 2
        as_dict = {r["shell"]: r["vacancy_rate"] for r in out.iter_rows(named=True)}
        assert as_dict["K"] == pytest.approx(0.886 + 0.0015, rel=1e-6)
        assert as_dict["L"] == pytest.approx(0.0954, rel=1e-6)

    def test_empty_input_returns_empty_with_correct_schema(self) -> None:
        empty = _make_decay_detailed([])
        out = compute_ec_vacancy_rates(empty)
        assert out.is_empty()
        assert {"Z", "A", "shell", "vacancy_rate"}.issubset(set(out.columns))


class TestSynthesizeEmissions:
    """Convolution of vacancy rates × EADL transitions on a synthetic atom."""

    @pytest.fixture()
    def eadl_dir(self, tmp_path: Path) -> Path:
        # Toy element X (Z=99) with a single K vacancy producing one X-ray
        # and one Auger, normalized to sum to 1 per vacancy.
        eadl = _make_eadl(
            [
                {
                    "Z": 99,
                    "vacancy_shell": "K",
                    "filling_shell": "L3",
                    "transition_type": "radiative",
                    "energy_keV": 50.0,
                    "probability": 0.4,
                    "edge_keV": 60.0,
                },
                {
                    "Z": 99,
                    "vacancy_shell": "K",
                    "filling_shell": "L1",
                    "transition_type": "auger",
                    "energy_keV": 45.0,
                    "probability": 0.6,
                    "edge_keV": 60.0,
                },
            ]
        )
        d = tmp_path / "eadl"
        d.mkdir()
        eadl.write_parquet(d / "Xx.parquet")
        return d

    def test_basic_convolution(self, eadl_dir: Path) -> None:
        vacancies = pl.DataFrame(
            {
                "Z": [99],
                "A": [199],
                "parent_ex_kev": [0.0],
                "daughter_Z": [99],
                "shell": ["K"],
                "vacancy_rate": [0.5],  # 50% K-vacancy
            },
            schema={
                "Z": pl.Int32,
                "A": pl.Int32,
                "parent_ex_kev": pl.Float64,
                "daughter_Z": pl.Int32,
                "shell": pl.Utf8,
                "vacancy_rate": pl.Float64,
            },
        )
        out = synthesize_emissions(
            vacancies,
            decay_mode_label="EC",
            eadl_dir=eadl_dir,
            z_to_symbol={99: "Xx"},
        )
        assert out.height == 2
        rows = out.sort("rad_type").to_dicts()
        # Auger row first alphabetically: 0.5 × 0.6 × 100 = 30%
        assert rows[0]["rad_type"] == "auger"
        assert rows[0]["intensity_pct"] == pytest.approx(30.0)
        assert rows[0]["energy_keV"] == pytest.approx(45.0)
        assert rows[0]["rad_subtype"] == "KLL"
        # X-ray row: 0.5 × 0.4 × 100 = 20%
        assert rows[1]["rad_type"] == "xray"
        assert rows[1]["intensity_pct"] == pytest.approx(20.0)
        assert rows[1]["energy_keV"] == pytest.approx(50.0)
        assert rows[1]["rad_subtype"] == "Kα1"  # K → L3

    def test_decay_mode_label_propagates(self, eadl_dir: Path) -> None:
        vacancies = pl.DataFrame(
            {
                "Z": [99],
                "A": [199],
                "parent_ex_kev": [100.0],
                "daughter_Z": [99],
                "shell": ["K"],
                "vacancy_rate": [0.1],
            },
            schema={
                "Z": pl.Int32,
                "A": pl.Int32,
                "parent_ex_kev": pl.Float64,
                "daughter_Z": pl.Int32,
                "shell": pl.Utf8,
                "vacancy_rate": pl.Float64,
            },
        )
        out_it = synthesize_emissions(
            vacancies,
            decay_mode_label="IT",
            eadl_dir=eadl_dir,
            z_to_symbol={99: "Xx"},
        )
        assert set(out_it["decay_mode"].unique().to_list()) == {"IT"}


class TestEnergyConservationCheck:
    """The check_energy_conservation invariant."""

    def test_passes_when_emissions_match_vacancies(self) -> None:
        # Per-vacancy intensity_pct sums must ≈ vacancy_rate × 100.
        emissions = pl.DataFrame(
            {
                "Z": [27, 27],
                "A": [57, 57],
                "state": ["", ""],
                "rad_type": ["xray", "auger"],
                "energy_keV": [6.4, 5.6],
                "intensity_pct": [30.0, 60.0],  # totals 90% of K vacancy
                "decay_mode": ["EC", "EC"],
                "parent_level_keV": [0.0, 0.0],
                "rad_subtype": ["Kα1", "KLL"],
            },
            schema={
                "Z": pl.Int32,
                "A": pl.Int32,
                "state": pl.Utf8,
                "rad_type": pl.Utf8,
                "energy_keV": pl.Float64,
                "intensity_pct": pl.Float64,
                "decay_mode": pl.Utf8,
                "parent_level_keV": pl.Float64,
                "rad_subtype": pl.Utf8,
            },
        )
        vacancies = pl.DataFrame(
            {
                "Z": [27],
                "A": [57],
                "parent_ex_kev": [0.0],
                "daughter_Z": [26],
                "shell": ["K"],
                "vacancy_rate": [0.9],
            },
            schema={
                "Z": pl.Int32,
                "A": pl.Int32,
                "parent_ex_kev": pl.Float64,
                "daughter_Z": pl.Int32,
                "shell": pl.Utf8,
                "vacancy_rate": pl.Float64,
            },
        )
        violations = check_energy_conservation(emissions, vacancies, slack_pct=5.0)
        assert violations.is_empty()

    def test_violation_detected(self) -> None:
        # Emissions only sum to 50% but vacancy rate is 0.9 → 40 pct delta.
        emissions = pl.DataFrame(
            {
                "Z": [27],
                "A": [57],
                "state": [""],
                "rad_type": ["xray"],
                "energy_keV": [6.4],
                "intensity_pct": [50.0],
                "decay_mode": ["EC"],
                "parent_level_keV": [0.0],
                "rad_subtype": ["Kα1"],
            },
            schema={
                "Z": pl.Int32,
                "A": pl.Int32,
                "state": pl.Utf8,
                "rad_type": pl.Utf8,
                "energy_keV": pl.Float64,
                "intensity_pct": pl.Float64,
                "decay_mode": pl.Utf8,
                "parent_level_keV": pl.Float64,
                "rad_subtype": pl.Utf8,
            },
        )
        vacancies = pl.DataFrame(
            {
                "Z": [27],
                "A": [57],
                "parent_ex_kev": [0.0],
                "daughter_Z": [26],
                "shell": ["K"],
                "vacancy_rate": [0.9],
            },
            schema={
                "Z": pl.Int32,
                "A": pl.Int32,
                "parent_ex_kev": pl.Float64,
                "daughter_Z": pl.Int32,
                "shell": pl.Utf8,
                "vacancy_rate": pl.Float64,
            },
        )
        violations = check_energy_conservation(emissions, vacancies, slack_pct=5.0)
        assert violations.height == 1
        assert violations["delta_pct"][0] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# @pytest.mark.data — real strata + EADL inputs.
# ---------------------------------------------------------------------------


@_real_data
class TestSpotChecks:
    """Acceptance criteria from issue #74 against the pinned strata revision."""

    @pytest.fixture(scope="class")
    def emissions(self) -> pl.DataFrame:
        decay_detailed = pl.read_parquet(DECAY_DETAILED)
        gammas = pl.read_parquet(GAMMAS)
        levels = pl.read_parquet(LEVELS)
        nuclides = pl.read_parquet(NUCLIDES)
        return synthesize_xray_auger(decay_detailed, gammas, levels, nuclides, EADL_DIR)

    def test_co57_fe_kalpha_at_64kev(self, emissions: pl.DataFrame) -> None:
        """Co-57 EC produces Fe Kα1 at 6.4 keV (Mössbauer line)."""
        co = emissions.filter(
            (pl.col("Z") == 27)
            & (pl.col("A") == 57)
            & (pl.col("state") == "")
            & (pl.col("rad_subtype") == "Kα1")
            & (pl.col("rad_type") == "xray")
        )
        assert co.height == 1, f"expected 1 Co-57 Fe Kα1 row, got {co.height}"
        energy = co["energy_keV"][0]
        assert 6.3 <= energy <= 6.5, f"Co-57 Kα1 energy {energy} not in [6.3, 6.5]"
        # Intensity: KshellEC × Fe K-fluorescence yield × Kα1 fraction
        # ~ 0.888 × 0.336 × (0.198/(0.198+0.101)) ≈ 0.198 → ~ 19.7%
        assert 15.0 <= co["intensity_pct"][0] <= 25.0

    def test_tc99m_kalpha_around_184kev(self, emissions: pl.DataFrame) -> None:
        """Tc-99m IC produces Tc Kα at ~18.3 keV."""
        tc = emissions.filter(
            (pl.col("Z") == 43)
            & (pl.col("A") == 99)
            & (pl.col("state") == "m")
            & (pl.col("rad_subtype") == "Kα1")
            & (pl.col("rad_type") == "xray")
        )
        assert tc.height == 1
        assert 18.2 <= tc["energy_keV"][0] <= 18.5

    def test_tc99m_kbeta_around_21kev(self, emissions: pl.DataFrame) -> None:
        """Tc-99m IC produces Tc Kβ1 at ~20.6 keV."""
        tc = emissions.filter(
            (pl.col("Z") == 43)
            & (pl.col("A") == 99)
            & (pl.col("state") == "m")
            & (pl.col("rad_subtype") == "Kβ1")
            & (pl.col("rad_type") == "xray")
        )
        assert tc.height == 1
        assert 20.0 <= tc["energy_keV"][0] <= 21.5

    def test_i125_kalpha_te_daughter(self, emissions: pl.DataFrame) -> None:
        """I-125 EC produces Te Kα1 at ~27.5 keV with high yield."""
        i = emissions.filter(
            (pl.col("Z") == 53)
            & (pl.col("A") == 125)
            & (pl.col("state") == "")
            & (pl.col("rad_subtype") == "Kα1")
            & (pl.col("rad_type") == "xray")
        )
        assert i.height == 1
        assert 27.0 <= i["energy_keV"][0] <= 28.0
        # I-125's K-shell EC is dominant; expect Kα1 intensity > 30%
        assert i["intensity_pct"][0] > 30.0

    def test_i125_KLL_auger_present(self, emissions: pl.DataFrame) -> None:
        """High-Z EC isotopes have prominent KLL Auger groups (issue #74)."""
        kll = emissions.filter(
            (pl.col("Z") == 53)
            & (pl.col("A") == 125)
            & (pl.col("rad_type") == "auger")
            & (pl.col("rad_subtype") == "KLL")
        )
        assert kll.height >= 5, "I-125 KLL Auger group should be populated"
        # Sum of KLL intensities should be a few percent at minimum.
        assert kll["intensity_pct"].sum() > 1.0

    def test_ba137m_kalpha_present(self, emissions: pl.DataFrame) -> None:
        """Ba-137m IT (661.66 keV M4) produces Ba Kα at ~32.2 keV.

        Ba-137m is the canonical IT-only isomer (Cs-137 daughter). It has
        no detail rows in strata's radioactive_decay (only is_summary), so
        the parent state must be seeded from photon_evap_levels and the
        nuclides catalog has level_keV=NULL — exercises the NULL-fallback
        path in `assign_state`."""
        ba = emissions.filter(
            (pl.col("Z") == 56)
            & (pl.col("A") == 137)
            & (pl.col("state") == "m")
            & (pl.col("rad_subtype") == "Kα1")
            & (pl.col("rad_type") == "xray")
        )
        assert ba.height == 1, f"expected 1 Ba-137m Kα1 row, got {ba.height}"
        assert 31.5 <= ba["energy_keV"][0] <= 32.5
        # Canonical Ba-137m K-vacancy yield is ~9 %; Kα1 is the dominant
        # X-ray line (≈ 4–6 % per decay).
        assert 3.0 <= ba["intensity_pct"][0] <= 8.0

    def test_au197m_l_xrays_present(self, emissions: pl.DataFrame) -> None:
        """Au-197m 130 keV M4: L-shell IC dominates K-shell. Per-shell ICC
        fractions (icc_frac_K=0, icc_frac_L1..L3 large) must produce L
        X-ray lines. Before #193, all IC was attributed to K-shell and
        L X-rays were completely absent."""
        au = emissions.filter(
            (pl.col("Z") == 79) & (pl.col("A") == 197) & (pl.col("rad_type") == "xray")
        )
        l_xrays = au.filter(pl.col("rad_subtype").str.starts_with("L"))
        k_xrays = au.filter(pl.col("rad_subtype").str.starts_with("K"))
        l_total = l_xrays["intensity_pct"].sum()
        k_total = k_xrays["intensity_pct"].sum()
        # L X-rays must be present and substantial (α_L >> α_K for 130 keV M4).
        # Before #193, L was 0% and K was ~96%. With per-shell ICC, L should
        # be comparable to K (~20-75% depending on state-assignment filtering).
        assert l_total > 0.0, "Au-197m L X-rays absent — per-shell ICC not wired"
        assert l_total > 10.0, (
            f"Au-197m L X-ray total ({l_total:.1f}%) too low — "
            f"per-shell ICC fractions may not be propagating"
        )
        # Lα1 at ~9.7 keV should be one of the top lines.
        la1 = au.filter(pl.col("rad_subtype") == "Lα1")
        assert la1.height >= 1, "Au-197m Lα1 line missing"
        assert 9.0 <= la1["energy_keV"][0] <= 10.5

    def test_co57_energy_conservation_K_shell(self, emissions: pl.DataFrame) -> None:
        """Σ(K X-rays + K Augers) ≈ K-vacancy rate × 100 within 5%."""
        co = emissions.filter((pl.col("Z") == 27) & (pl.col("A") == 57) & (pl.col("state") == ""))
        k_total = co.filter(pl.col("rad_subtype").str.starts_with("K"))["intensity_pct"].sum()
        # Co-57 KshellEC = 0.886 + 0.0015 (per radioactive_decay.parquet)
        # K-shell auger normalization in EADL Fe is ~0.99957 of (radiative
        # + auger) within the K vacancy block, so ≈ 0.886 × 1 × 100 = 88.6%.
        assert 85.0 <= k_total <= 92.0, f"Co-57 K-shell sum {k_total}"


@_real_data
class TestEnergyConservationSweep:
    """Sweep test (per #74 acceptance): the energy-conservation invariant
    must hold across every parent in the synthesized table."""

    def test_no_violations_above_slack(self) -> None:
        """The energy-conservation invariant must hold for every parent that
        the synthesizer actually emits, **per decay_mode**. EC and IT are
        checked separately because their X-rays go to different daughter
        atoms (EC: daughter_Z = Z-1; IT: daughter_Z = Z) and therefore
        different EADL tables — summing them across decay_mode is not a
        physical invariant.

        Parents dropped by state assignment (excited but no isomer catalog
        match) are excluded — they have a legitimate vacancy rate but no
        emission row, by design."""
        from nucl_parquet.g4.xray_auger import load_k_edges

        decay_detailed = pl.read_parquet(DECAY_DETAILED)
        gammas = pl.read_parquet(GAMMAS)
        levels = pl.read_parquet(LEVELS)
        nuclides = pl.read_parquet(NUCLIDES)

        emissions = synthesize_xray_auger(decay_detailed, gammas, levels, nuclides, EADL_DIR)

        emitted_parents = (
            emissions.select(["Z", "A", "parent_level_keV"]).unique().rename({"parent_level_keV": "parent_ex_kev"})
        )

        # ---- EC invariant ----
        # Tightened from <25 absolute to ≤2% of parents (PR #87 review).
        # NaN rows (no emissions or no vacancies — degenerate cases, not
        # actual conservation violations) are filtered out before counting.
        # Prints offenders so a regression surfaces specific (Z,A) for audit.
        ec_vac = compute_ec_vacancy_rates(decay_detailed).join(
            emitted_parents, on=["Z", "A", "parent_ex_kev"], how="inner"
        )
        ec_em = emissions.filter(pl.col("decay_mode") == "EC")
        ec_violations = check_energy_conservation(ec_em, ec_vac, slack_pct=5.0)
        ec_material = ec_violations.filter((pl.col("expected_pct") >= 1.0) & ~pl.col("delta_pct").is_nan())
        ec_population = ec_vac.height
        ec_budget = max(2, int(0.02 * ec_population))
        assert ec_material.height <= ec_budget, (
            f"EC energy-conservation violations exceed 2% budget "
            f"({ec_material.height} > {ec_budget} of {ec_population} parents):\n"
            f"{ec_material.head(20).to_dicts()}"
        )

        # ---- IC invariant ----
        k_edges = load_k_edges(EADL_DIR)
        parent_states = decay_detailed.select(["Z", "A", "parent_ex_kev"]).unique()
        ic_vac = compute_ic_vacancy_rates(gammas, levels, parent_states, k_edges=k_edges).join(
            emitted_parents, on=["Z", "A", "parent_ex_kev"], how="inner"
        )
        ic_em = emissions.filter(pl.col("decay_mode") == "IT")
        ic_violations = check_energy_conservation(ic_em, ic_vac, slack_pct=5.0)
        ic_material = ic_violations.filter((pl.col("expected_pct") >= 1.0) & ~pl.col("delta_pct").is_nan())
        ic_population = ic_vac.height
        ic_budget = max(2, int(0.02 * ic_population))
        assert ic_material.height <= ic_budget, (
            f"IC energy-conservation violations exceed 2% budget "
            f"({ic_material.height} > {ic_budget} of {ic_population} parents):\n"
            f"{ic_material.head(20).to_dicts()}"
        )
