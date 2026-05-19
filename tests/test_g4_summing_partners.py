"""Tests for the summing_partners materializer (issue #177).

Two layers:
  * Synthetic unit tests verifying ICC correction math, canonicalization,
    mixed-pair handling, and NULL coalescing.
  * ``@pytest.mark.data`` spot checks confirming the acceptance criteria:
    - Eu-152 344 keV: ≥10 γ-γ partners
    - Co-60 1173/1333 keV: canonical symmetric pair, icc_correction ≈ 1
    - Tc-99m 140 keV: significant ICC correction
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nucl_parquet.g4.summing_partners import _OUTPUT_COLUMNS, build_for_element

_REPO_ROOT = Path(__file__).parent.parent
_SP_DIR = _REPO_ROOT / "data" / "meta" / "ensdf" / "summing_partners"


# --------------------------------------------------------------------------- helpers


def _make_coinc_gg(
    rows: list[dict],
) -> pl.DataFrame:
    """Minimal coincidences frame with γ-γ rows + emission-agnostic columns."""
    schema = {
        "Z": pl.Int64,
        "A": pl.Int64,
        "dataset": pl.Int64,
        "gamma_energy_keV": pl.Float64,
        "coinc_energy_keV": pl.Float64,
        "gamma1_energy_keV": pl.Float64,
        "gamma1_intensity": pl.Float32,
        "gamma2_energy_keV": pl.Float64,
        "gamma2_intensity": pl.Float32,
        "parent_level_keV": pl.Float64,
        "intermediate_level_keV": pl.Float64,
        "final_level_keV": pl.Float64,
        "pair_intensity": pl.Float32,
        "gamma1_icc_total": pl.Float32,
        "gamma2_icc_total": pl.Float32,
        "parent_state": pl.Utf8,
        "parent_decay_mode": pl.Utf8,
        "daughter_ex_keV": pl.Float64,
        "emission1_rad_type": pl.Utf8,
        "emission1_energy_keV": pl.Float64,
        "emission1_intensity": pl.Float32,
        "emission1_shell": pl.Utf8,
        "emission2_rad_type": pl.Utf8,
        "emission2_energy_keV": pl.Float64,
        "emission2_intensity": pl.Float32,
        "emission2_shell": pl.Utf8,
    }
    return pl.DataFrame([{**{k: None for k in schema}, **r} for r in rows], schema=schema)


# --------------------------------------------------------------------------- unit tests


class TestIccCorrection:
    """Verify ICC math: 1/((1+α₁)×(1+α₂))."""

    def test_both_zero(self):
        """No ICC → correction factor = 1.0."""
        coinc = _make_coinc_gg(
            [
                {
                    "Z": 28,
                    "A": 60,
                    "gamma1_energy_keV": 100.0,
                    "gamma2_energy_keV": 200.0,
                    "gamma1_intensity": 100.0,
                    "gamma2_intensity": 100.0,
                    "parent_level_keV": 300.0,
                    "intermediate_level_keV": 200.0,
                    "final_level_keV": 0.0,
                    "pair_intensity": 100.0,
                    "gamma1_icc_total": 0.0,
                    "gamma2_icc_total": 0.0,
                    "parent_state": "",
                    "emission1_rad_type": "gamma",
                    "emission2_rad_type": "gamma",
                    "emission1_energy_keV": 100.0,
                    "emission2_energy_keV": 200.0,
                }
            ]
        )
        result = build_for_element(coinc)
        assert result.height == 1
        row = result.row(0, named=True)
        assert abs(row["icc_correction_factor"] - 1.0) < 1e-5

    def test_significant_icc(self):
        """α₁=0.1, α₂=0.2 → correction = 1/((1.1)×(1.2)) ≈ 0.7576."""
        coinc = _make_coinc_gg(
            [
                {
                    "Z": 28,
                    "A": 60,
                    "gamma1_energy_keV": 100.0,
                    "gamma2_energy_keV": 200.0,
                    "gamma1_intensity": 100.0,
                    "gamma2_intensity": 100.0,
                    "parent_level_keV": 300.0,
                    "intermediate_level_keV": 200.0,
                    "final_level_keV": 0.0,
                    "pair_intensity": 100.0,
                    "gamma1_icc_total": 0.1,
                    "gamma2_icc_total": 0.2,
                    "parent_state": "",
                    "emission1_rad_type": "gamma",
                    "emission2_rad_type": "gamma",
                    "emission1_energy_keV": 100.0,
                    "emission2_energy_keV": 200.0,
                }
            ]
        )
        result = build_for_element(coinc)
        row = result.row(0, named=True)
        expected = 1.0 / (1.1 * 1.2)
        assert abs(row["icc_correction_factor"] - expected) < 1e-4
        assert abs(row["pure_emission_joint_intensity"] - 100.0 * expected) < 0.1

    def test_null_icc_treated_as_zero(self):
        """NULL ICC → coalesced to 0 → correction = 1.0."""
        coinc = _make_coinc_gg(
            [
                {
                    "Z": 28,
                    "A": 60,
                    "gamma1_energy_keV": 100.0,
                    "gamma2_energy_keV": 200.0,
                    "gamma1_intensity": 100.0,
                    "gamma2_intensity": 100.0,
                    "parent_level_keV": 300.0,
                    "intermediate_level_keV": 200.0,
                    "final_level_keV": 0.0,
                    "pair_intensity": 100.0,
                    "gamma1_icc_total": None,
                    "gamma2_icc_total": None,
                    "parent_state": "",
                    "emission1_rad_type": "gamma",
                    "emission2_rad_type": "gamma",
                    "emission1_energy_keV": 100.0,
                    "emission2_energy_keV": 200.0,
                }
            ]
        )
        result = build_for_element(coinc)
        row = result.row(0, named=True)
        assert abs(row["icc_correction_factor"] - 1.0) < 1e-5


class TestCanonicalization:
    """γ-γ pairs should be canonicalized with E1 ≤ E2."""

    def test_symmetric_pair_deduped(self):
        """Two rows (E1=100,E2=200) and (E1=200,E2=100) → one output row."""
        coinc = _make_coinc_gg(
            [
                {
                    "Z": 28,
                    "A": 60,
                    "gamma1_energy_keV": 100.0,
                    "gamma2_energy_keV": 200.0,
                    "gamma1_intensity": 100.0,
                    "gamma2_intensity": 50.0,
                    "parent_level_keV": 300.0,
                    "intermediate_level_keV": 200.0,
                    "final_level_keV": 0.0,
                    "pair_intensity": 50.0,
                    "gamma1_icc_total": 0.0,
                    "gamma2_icc_total": 0.0,
                    "parent_state": "",
                    "emission1_rad_type": "gamma",
                    "emission2_rad_type": "gamma",
                    "emission1_energy_keV": 100.0,
                    "emission2_energy_keV": 200.0,
                },
                {
                    "Z": 28,
                    "A": 60,
                    "gamma1_energy_keV": 200.0,
                    "gamma2_energy_keV": 100.0,
                    "gamma1_intensity": 50.0,
                    "gamma2_intensity": 100.0,
                    "parent_level_keV": 300.0,
                    "intermediate_level_keV": 200.0,
                    "final_level_keV": 0.0,
                    "pair_intensity": 50.0,
                    "gamma1_icc_total": 0.0,
                    "gamma2_icc_total": 0.0,
                    "parent_state": "",
                    "emission1_rad_type": "gamma",
                    "emission2_rad_type": "gamma",
                    "emission1_energy_keV": 200.0,
                    "emission2_energy_keV": 100.0,
                },
            ]
        )
        result = build_for_element(coinc)
        assert result.height == 1
        row = result.row(0, named=True)
        assert row["emission1_energy_keV"] <= row["emission2_energy_keV"]


class TestMixedPairs:
    """X-ray/Auger ⊗ γ pairs: ICC only on gamma side."""

    def test_xray_gamma_icc(self):
        """X-ray side has no ICC; gamma side α=0.1 → correction = 1/1.1."""
        coinc = _make_coinc_gg(
            [
                {
                    "Z": 62,
                    "A": 152,
                    "gamma1_energy_keV": None,
                    "gamma2_energy_keV": None,
                    "gamma1_intensity": None,
                    "gamma2_intensity": None,
                    "parent_level_keV": 344.0,
                    "intermediate_level_keV": None,
                    "final_level_keV": None,
                    "pair_intensity": 10.0,
                    "gamma1_icc_total": None,
                    "gamma2_icc_total": 0.1,
                    "parent_state": "",
                    "parent_decay_mode": "KshellEC",
                    "emission1_rad_type": "xray",
                    "emission2_rad_type": "gamma",
                    "emission1_energy_keV": 40.0,
                    "emission1_intensity": 5.0,
                    "emission1_shell": "K",
                    "emission2_energy_keV": 344.0,
                    "emission2_intensity": 100.0,
                }
            ]
        )
        result = build_for_element(coinc)
        assert result.height == 1
        row = result.row(0, named=True)
        expected = 1.0 / 1.1
        assert abs(row["icc_correction_factor"] - expected) < 1e-4
        assert row["emission1_icc_total"] is None  # X-ray: no ICC
        assert abs(row["emission2_icc_total"] - 0.1) < 1e-5


class TestOutputSchema:
    """Output has all required columns in the right order."""

    def test_column_names(self):
        coinc = _make_coinc_gg(
            [
                {
                    "Z": 28,
                    "A": 60,
                    "gamma1_energy_keV": 100.0,
                    "gamma2_energy_keV": 200.0,
                    "gamma1_intensity": 100.0,
                    "gamma2_intensity": 100.0,
                    "parent_level_keV": 300.0,
                    "intermediate_level_keV": 200.0,
                    "final_level_keV": 0.0,
                    "pair_intensity": 100.0,
                    "gamma1_icc_total": 0.0,
                    "gamma2_icc_total": 0.0,
                    "parent_state": "",
                    "emission1_rad_type": "gamma",
                    "emission2_rad_type": "gamma",
                    "emission1_energy_keV": 100.0,
                    "emission2_energy_keV": 200.0,
                }
            ]
        )
        result = build_for_element(coinc)
        assert result.columns == _OUTPUT_COLUMNS


# --------------------------------------------------------------------------- data tests


@pytest.mark.data
class TestCo60:
    """Co-60 → Ni-60: 1173/1333 keV canonical pair."""

    def test_pair_exists(self):
        ni = pl.read_parquet(_SP_DIR / "Ni.parquet")
        co60 = ni.filter((pl.col("A") == 60) & (pl.col("emission1_rad_type") == "gamma"))
        pair = co60.filter(
            ((pl.col("emission1_energy_keV") - 1173.2).abs() < 1.0)
            & ((pl.col("emission2_energy_keV") - 1332.5).abs() < 1.0)
        )
        assert pair.height >= 1, "Co-60 1173/1333 keV pair missing"

    def test_icc_correction_near_unity(self):
        """α for 1173 and 1333 keV are < 0.001 → correction ≈ 1."""
        ni = pl.read_parquet(_SP_DIR / "Ni.parquet")
        pair = ni.filter(
            (pl.col("A") == 60)
            & (pl.col("emission1_rad_type") == "gamma")
            & ((pl.col("emission1_energy_keV") - 1173.2).abs() < 1.0)
            & ((pl.col("emission2_energy_keV") - 1332.5).abs() < 1.0)
        )
        row = pair.row(0, named=True)
        assert row["icc_correction_factor"] > 0.99, f"Expected ~1.0, got {row['icc_correction_factor']}"

    def test_canonicalized(self):
        """1173 < 1333, so emission1 should be the lower energy."""
        ni = pl.read_parquet(_SP_DIR / "Ni.parquet")
        pair = ni.filter(
            (pl.col("A") == 60)
            & (pl.col("emission1_rad_type") == "gamma")
            & ((pl.col("emission1_energy_keV") - 1173.2).abs() < 1.0)
            & ((pl.col("emission2_energy_keV") - 1332.5).abs() < 1.0)
        )
        row = pair.row(0, named=True)
        assert row["emission1_energy_keV"] < row["emission2_energy_keV"]


@pytest.mark.data
class TestEu152:
    """Eu-152 → Gd-152 (β-): 344 keV γ should have ≥10 summing partners."""

    def test_partner_count(self):
        gd = pl.read_parquet(_SP_DIR / "Gd.parquet")
        partners = gd.filter(
            (pl.col("A") == 152)
            & (pl.col("emission1_rad_type") == "gamma")
            & (
                ((pl.col("emission1_energy_keV") - 344.28).abs() < 1.0)
                | ((pl.col("emission2_energy_keV") - 344.28).abs() < 1.0)
            )
        )
        assert partners.height >= 10, f"Expected ≥10 partners, got {partners.height}"

    def test_mixed_pairs_exist(self):
        """Eu-152 EC decay → Sm-152: X-ray ⊗ γ pairs should exist."""
        sm = pl.read_parquet(_SP_DIR / "Sm.parquet")
        mixed = sm.filter((pl.col("A") == 152) & (pl.col("emission1_rad_type") != "gamma"))
        assert mixed.height > 0, "Eu-152 → Sm-152 should have X-ray/Auger ⊗ γ pairs"


@pytest.mark.data
class TestTc99m:
    """Tc-99m → Tc-99: 140 keV M1+E2 (α_K ≈ 0.107) → significant ICC correction."""

    def test_icc_significant(self):
        tc = pl.read_parquet(_SP_DIR / "Tc.parquet")
        tc99 = tc.filter(
            (pl.col("A") == 99)
            & (
                ((pl.col("emission1_energy_keV") - 140.5).abs() < 1.0)
                | ((pl.col("emission2_energy_keV") - 140.5).abs() < 1.0)
            )
        )
        assert tc99.height > 0, "Tc-99m 140 keV summing partners missing"
        # At least one pair should show significant ICC correction (< 0.95)
        min_corr = tc99["icc_correction_factor"].min()
        assert min_corr < 0.95, f"Expected significant ICC correction, min factor = {min_corr}"
