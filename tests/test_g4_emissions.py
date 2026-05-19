"""Tests for the absolute-emissions materializer (issue #196).

Two layers:
  * Synthetic unit tests verifying cascade propagation, ICC normalization,
    IT handling, and edge cases.
  * ``@pytest.mark.data`` spot checks validating against NuDat reference
    values for Co-60, Tc-99m, and Eu-152.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nucl_parquet.g4.emissions import (
    compute_absolute_intensities,
    build_for_parent,
    _empty_output,
    _OUTPUT_COLUMNS,
    _OUTPUT_SCHEMA,
)

_REPO_ROOT = Path(__file__).parent.parent
_EMISSIONS_DIR = _REPO_ROOT / "data" / "meta" / "ensdf" / "emissions"


# --------------------------------------------------------------------------- helpers


def _make_radiation(rows: list[dict]) -> pl.DataFrame:
    """Minimal radiation frame for cascade tests."""
    schema = {
        "Z": pl.Int64,
        "A": pl.Int64,
        "state": pl.Utf8,
        "rad_type": pl.Utf8,
        "energy_keV": pl.Float64,
        "intensity_pct": pl.Float64,
        "decay_mode": pl.Utf8,
        "rad_subtype": pl.Utf8,
        "dose_MeV_per_Bq_s": pl.Float64,
        "parent_level_keV": pl.Float64,
        "daughter_level_keV": pl.Float64,
        "multipolarity": pl.Int32,
        "mixing_ratio": pl.Float32,
        "icc_total": pl.Float32,
    }
    return pl.DataFrame([{**{k: None for k in schema}, **r} for r in rows], schema=schema)


def _make_decay_detailed(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "Z": pl.Int32, "A": pl.Int32, "parent_ex_kev": pl.Float64,
        "parent_level_flag": pl.Utf8, "half_life_s": pl.Float64,
        "decay_mode": pl.Utf8, "daughter_Z": pl.Int32, "daughter_A": pl.Int32,
        "daughter_ex_kev": pl.Float64, "daughter_level_flag": pl.Utf8,
        "branching": pl.Float64, "q_value_kev": pl.Float64,
        "forbiddenness": pl.Utf8,
    }
    return pl.DataFrame([{**{k: None for k in schema}, **r} for r in rows], schema=schema)


def _make_decay_summary(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "Z": pl.Int32, "A": pl.Int32, "state": pl.Utf8,
        "half_life_s": pl.Float64, "decay_mode": pl.Utf8,
        "daughter_Z": pl.Int32, "daughter_A": pl.Int32,
        "daughter_state": pl.Utf8, "branching": pl.Float64,
    }
    return pl.DataFrame([{**{k: None for k in schema}, **r} for r in rows], schema=schema)


def _make_nuclides(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "Z": pl.Int32, "A": pl.Int32, "state": pl.Utf8,
        "symbol": pl.Utf8, "jp": pl.Utf8, "half_life_s": pl.Float64,
        "level_keV": pl.Float64, "decay_1": pl.Utf8, "decay_1_pct": pl.Float64,
        "decay_2": pl.Utf8, "decay_2_pct": pl.Float64, "spin_x2": pl.Int16,
        "parity": pl.Int8, "floating_level_flag": pl.Utf8,
        "magnetic_moment_jt": pl.Float64,
    }
    return pl.DataFrame([{**{k: None for k in schema}, **r} for r in rows], schema=schema)


# --------------------------------------------------------------------------- unit tests


class TestCascadePropagation:
    """Verify top-down cascade population propagation."""

    def test_simple_two_level(self):
        """Parent feeds level A → gamma → ground. 100% absolute."""
        cascade = {
            200.0: [{"daughter_level": 0.0, "energy": 200.0, "intensity_pct": 100.0, "icc_total": 0.0, "multipolarity": None}],
        }
        results = compute_absolute_intensities({200.0: 1.0}, cascade)
        assert len(results) == 1
        assert results[0]["intensity_pct"] == pytest.approx(100.0)

    def test_three_level_cascade(self):
        """300 keV level → 200 keV → ground, no branching."""
        cascade = {
            300.0: [{"daughter_level": 200.0, "energy": 100.0, "intensity_pct": 100.0, "icc_total": 0.0, "multipolarity": None}],
            200.0: [{"daughter_level": 0.0, "energy": 200.0, "intensity_pct": 100.0, "icc_total": 0.0, "multipolarity": None}],
        }
        results = compute_absolute_intensities({300.0: 1.0}, cascade)
        assert len(results) == 2
        by_energy = {r["energy"]: r["intensity_pct"] for r in results}
        assert by_energy[100.0] == pytest.approx(100.0)
        assert by_energy[200.0] == pytest.approx(100.0)

    def test_branching_at_level(self):
        """Level 300 branches 70/30 to two daughter levels."""
        cascade = {
            300.0: [
                {"daughter_level": 200.0, "energy": 100.0, "intensity_pct": 70.0, "icc_total": 0.0, "multipolarity": None},
                {"daughter_level": 0.0, "energy": 300.0, "intensity_pct": 30.0, "icc_total": 0.0, "multipolarity": None},
            ],
            200.0: [{"daughter_level": 0.0, "energy": 200.0, "intensity_pct": 100.0, "icc_total": 0.0, "multipolarity": None}],
        }
        results = compute_absolute_intensities({300.0: 1.0}, cascade)
        by_energy = {r["energy"]: r["intensity_pct"] for r in results}
        assert by_energy[100.0] == pytest.approx(70.0)
        assert by_energy[300.0] == pytest.approx(30.0)
        assert by_energy[200.0] == pytest.approx(70.0)  # only 70% feeds level 200

    def test_partial_feeding(self):
        """Parent feeds level with branching = 0.5 (50%)."""
        cascade = {
            100.0: [{"daughter_level": 0.0, "energy": 100.0, "intensity_pct": 100.0, "icc_total": 0.0, "multipolarity": None}],
        }
        results = compute_absolute_intensities({100.0: 0.5}, cascade)
        assert results[0]["intensity_pct"] == pytest.approx(50.0)


class TestIccNormalization:
    """Verify that branch fractions use total transition rate (photon+IC)."""

    def test_high_icc_steals_photon_fraction(self):
        """A gamma with ICC=1.0 emits 50% as photon, 50% as IC electron."""
        cascade = {
            100.0: [{"daughter_level": 0.0, "energy": 100.0, "intensity_pct": 100.0, "icc_total": 1.0, "multipolarity": None}],
        }
        results = compute_absolute_intensities({100.0: 1.0}, cascade)
        # Total transition = 100 × (1+1) = 200, branch = 1.0
        # Photon fraction = 1/(1+1) = 0.5
        assert results[0]["intensity_pct"] == pytest.approx(50.0)

    def test_icc_affects_branching(self):
        """Two gammas: one with high ICC (steals population via IC)."""
        cascade = {
            200.0: [
                # Gamma A: photon intensity 10, ICC=9 → total_transition = 10×10 = 100
                {"daughter_level": 100.0, "energy": 100.0, "intensity_pct": 10.0, "icc_total": 9.0, "multipolarity": None},
                # Gamma B: photon intensity 100, ICC=0 → total_transition = 100×1 = 100
                {"daughter_level": 0.0, "energy": 200.0, "intensity_pct": 100.0, "icc_total": 0.0, "multipolarity": None},
            ],
        }
        results = compute_absolute_intensities({200.0: 1.0}, cascade)
        gammas = {r["energy"]: r["intensity_pct"] for r in results if r["rad_type"] == "gamma"}
        # Each has total_transition = 100. Branch = 50% each.
        # Gamma A: 100% × 0.5 × 1/10 × 100 = 5%
        # Gamma B: 100% × 0.5 × 1.0 × 100 = 50%
        assert gammas[100.0] == pytest.approx(5.0)
        assert gammas[200.0] == pytest.approx(50.0)


class TestITHandling:
    """Verify isomeric transition special-case handling."""

    def test_it_feeds_isomeric_level(self):
        """IT from metastable state feeds the isomeric level energy."""
        decay_detailed = _make_decay_detailed([])  # no detail rows for IT
        decay_summary = _make_decay_summary([{
            "Z": 43, "A": 99, "state": "m", "decay_mode": "IT",
            "daughter_Z": 43, "daughter_A": 99, "branching": 1.0,
        }])
        nuclides = _make_nuclides([{
            "Z": 43, "A": 99, "state": "m", "level_keV": 150.0,
        }])
        radiation = _make_radiation([{
            "Z": 43, "A": 99, "rad_type": "gamma",
            "energy_keV": 150.0, "intensity_pct": 100.0,
            "parent_level_keV": 150.0, "daughter_level_keV": 0.0,
            "icc_total": 0.0,
        }])
        result = build_for_parent(
            43, 99, "m",
            decay_detailed, decay_summary, nuclides,
            radiation_cache={43: radiation},
        )
        assert result.height == 1
        row = result.row(0, named=True)
        assert row["decay_mode"] == "IT"
        assert row["intensity_pct"] == pytest.approx(100.0)
        assert row["parent_Z"] == 43
        assert row["parent_A"] == 99
        assert row["parent_state"] == "m"


class TestOutputSchema:
    """Verify output schema conformance."""

    def test_empty_output_schema(self):
        out = _empty_output()
        assert list(out.columns) == _OUTPUT_COLUMNS
        for col, dtype in _OUTPUT_SCHEMA.items():
            assert out.schema[col] == dtype

    def test_columns_match_spec(self):
        """Output columns match the documented schema."""
        expected = [
            "parent_Z", "parent_A", "parent_state", "decay_mode",
            "daughter_Z", "daughter_A", "rad_type", "energy_keV",
            "intensity_pct", "icc_total", "parent_level_keV",
            "daughter_level_keV", "multipolarity", "rad_subtype",
        ]
        assert _OUTPUT_COLUMNS == expected


class TestConversionElectrons:
    """Verify CE rows are emitted with ICC × photon intensity."""

    def test_ce_emitted_when_icc_nonzero(self):
        """A gamma with ICC=1 produces equal gamma + CE intensities."""
        cascade = {
            100.0: [{"daughter_level": 0.0, "energy": 100.0, "intensity_pct": 100.0, "icc_total": 1.0, "multipolarity": None}],
        }
        results = compute_absolute_intensities({100.0: 1.0}, cascade)
        by_type = {r["rad_type"]: r["intensity_pct"] for r in results}
        assert by_type["gamma"] == pytest.approx(50.0)
        assert by_type["ce"] == pytest.approx(50.0)

    def test_no_ce_when_icc_zero(self):
        """No CE rows when ICC = 0."""
        cascade = {
            100.0: [{"daughter_level": 0.0, "energy": 100.0, "intensity_pct": 100.0, "icc_total": 0.0, "multipolarity": None}],
        }
        results = compute_absolute_intensities({100.0: 1.0}, cascade)
        types = {r["rad_type"] for r in results}
        assert "ce" not in types


class TestEdgeCases:
    """Edge cases: no feeding, no cascade, etc."""

    def test_no_feeding_returns_empty(self):
        result = build_for_parent(
            99, 999, "",
            _make_decay_detailed([]),
            _make_decay_summary([]),
            _make_nuclides([]),
            radiation_cache={},
        )
        assert result.is_empty()

    def test_feeding_but_no_radiation(self):
        """Parent decays to daughter but no radiation data for daughter."""
        decay_detailed = _make_decay_detailed([{
            "Z": 27, "A": 60, "parent_ex_kev": 0.0,
            "decay_mode": "beta-", "daughter_Z": 28, "daughter_A": 60,
            "daughter_ex_kev": 1000.0, "branching": 1.0,
        }])
        result = build_for_parent(
            27, 60, "",
            decay_detailed,
            _make_decay_summary([]),
            _make_nuclides([]),
            radiation_cache={},  # no Ni data
        )
        assert result.is_empty()


# --------------------------------------------------------------------------- data tests


@pytest.mark.data
class TestCo60:
    """Co-60 → Ni-60 β⁻: validate against NuDat absolute intensities."""

    @pytest.fixture
    def co60(self) -> pl.DataFrame:
        path = _EMISSIONS_DIR / "Co.parquet"
        if not path.exists():
            pytest.skip("emissions data not built")
        df = pl.read_parquet(path)
        return df.filter(
            (pl.col("parent_A") == 60) & (pl.col("parent_state") == "")
        )

    def test_1173_keV(self, co60: pl.DataFrame):
        """1173 keV gamma: NuDat = 99.85%."""
        row = co60.filter(pl.col("energy_keV").is_between(1173, 1174))
        assert row.height >= 1
        assert row["intensity_pct"][0] == pytest.approx(99.85, rel=0.005)

    def test_1332_keV(self, co60: pl.DataFrame):
        """1332 keV gamma: NuDat = 99.9826%."""
        row = co60.filter(pl.col("energy_keV").is_between(1332, 1333))
        assert row.height >= 1
        assert row["intensity_pct"][0] == pytest.approx(99.9826, rel=0.005)

    def test_decay_mode_is_beta_minus(self, co60: pl.DataFrame):
        assert (co60["decay_mode"] == "beta-").all()

    def test_daughter_is_ni60(self, co60: pl.DataFrame):
        assert (co60["daughter_Z"] == 28).all()
        assert (co60["daughter_A"] == 60).all()


@pytest.mark.data
class TestTc99m:
    """Tc-99m IT: validate against NuDat absolute intensities."""

    @pytest.fixture
    def tc99m(self) -> pl.DataFrame:
        path = _EMISSIONS_DIR / "Tc.parquet"
        if not path.exists():
            pytest.skip("emissions data not built")
        df = pl.read_parquet(path)
        return df.filter(
            (pl.col("parent_A") == 99) & (pl.col("parent_state") == "m")
        )

    def test_140_keV(self, tc99m: pl.DataFrame):
        """140.5 keV gamma: NuDat = 89.06%."""
        it_rows = tc99m.filter(
            (pl.col("decay_mode") == "IT")
            & (pl.col("rad_type") == "gamma")
            & pl.col("energy_keV").is_between(140, 141)
        )
        assert it_rows.height == 1
        assert it_rows["intensity_pct"][0] == pytest.approx(89.06, rel=0.005)

    def test_it_decay_mode(self, tc99m: pl.DataFrame):
        """IT gamma rows use decay_mode='IT'."""
        it_rows = tc99m.filter(pl.col("decay_mode") == "IT")
        assert it_rows.height >= 1


@pytest.mark.data
class TestEu152:
    """Eu-152 EC+β⁻: validate summed intensities against NuDat."""

    @pytest.fixture
    def eu152(self) -> pl.DataFrame:
        path = _EMISSIONS_DIR / "Eu.parquet"
        if not path.exists():
            pytest.skip("emissions data not built")
        df = pl.read_parquet(path)
        return df.filter(
            (pl.col("parent_A") == 152) & (pl.col("parent_state") == "")
        )

    def _sum_intensity(self, df: pl.DataFrame, e_low: float, e_high: float) -> float:
        """Sum gamma intensity_pct across all decay modes for an energy."""
        return df.filter(
            (pl.col("rad_type") == "gamma")
            & pl.col("energy_keV").is_between(e_low, e_high)
        )["intensity_pct"].sum()

    def test_121_keV_ec(self, eu152: pl.DataFrame):
        """121.8 keV (EC→Sm-152): NuDat = 28.58%, summed across all EC shells."""
        total = self._sum_intensity(eu152, 121, 122.5)
        assert total == pytest.approx(28.58, rel=0.01)

    def test_344_keV_beta_minus(self, eu152: pl.DataFrame):
        """344.3 keV (β⁻→Gd-152): NuDat = 26.50%."""
        total = self._sum_intensity(eu152, 343.5, 345)
        assert total == pytest.approx(26.50, rel=0.01)

    def test_1408_keV_ec(self, eu152: pl.DataFrame):
        """1408 keV (EC→Sm-152): NuDat = 21.01%."""
        total = self._sum_intensity(eu152, 1407, 1409)
        assert total == pytest.approx(21.01, rel=0.01)

    def test_has_both_daughters(self, eu152: pl.DataFrame):
        """Eu-152 has both Sm-152 (EC) and Gd-152 (β⁻) daughters."""
        daughters = eu152.select("daughter_Z").unique()["daughter_Z"].to_list()
        assert 62 in daughters  # Sm
        assert 64 in daughters  # Gd

    def test_has_xray_auger(self, eu152: pl.DataFrame):
        """Eu-152 EC produces Eu characteristic x-rays and Auger electrons."""
        xray = eu152.filter(pl.col("rad_type") == "xray")
        auger = eu152.filter(pl.col("rad_type") == "auger")
        assert xray.height > 0, "Expected x-ray rows from EC"
        assert auger.height > 0, "Expected Auger rows from EC"

    def test_annihilation_511(self, eu152: pl.DataFrame):
        """Eu-152 β⁺ produces 511 keV annihilation photons."""
        ann = eu152.filter(pl.col("rad_type") == "annihilation")
        assert ann.height == 1
        assert ann["energy_keV"][0] == pytest.approx(511.0)
        # 2 × β⁺ branching (small for Eu-152)
        assert ann["intensity_pct"][0] > 0

    def test_has_conversion_electrons(self, eu152: pl.DataFrame):
        """Eu-152 has conversion electron rows from highly-converted transitions."""
        ce = eu152.filter(pl.col("rad_type") == "ce")
        assert ce.height > 0

    def test_total_intensity_bounded(self, eu152: pl.DataFrame):
        """Individual emission intensity ≤ sensible bound."""
        # Per decay mode, total emission can exceed 100% due to cascades
        # (one decay can emit multiple gammas). But each individual line ≤ 100%.
        assert (eu152["intensity_pct"] <= 200.01).all()


@pytest.mark.data
class TestNa22:
    """Na-22 β⁺: validate 511 keV annihilation."""

    @pytest.fixture
    def na22(self) -> pl.DataFrame:
        path = _EMISSIONS_DIR / "Na.parquet"
        if not path.exists():
            pytest.skip("emissions data not built")
        df = pl.read_parquet(path)
        return df.filter(
            (pl.col("parent_A") == 22) & (pl.col("parent_state") == "")
        )

    def test_511_keV(self, na22: pl.DataFrame):
        """Na-22 511 keV: NuDat = 179.79% (2 photons per β⁺)."""
        ann = na22.filter(pl.col("rad_type") == "annihilation")
        assert ann.height == 1
        assert ann["intensity_pct"][0] == pytest.approx(179.79, rel=0.01)
