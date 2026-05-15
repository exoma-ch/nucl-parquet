"""Tests for the mixed-emission coincidence build (issue #170).

Two layers, mirroring ``test_g4_coincidences.py``:

* Synthetic unit tests over hand-rolled inputs covering parent emission
  synthesis (β endpoint, 511 keV annihilation, EC shell X-rays) and pair
  generation.
* ``@pytest.mark.data`` spot checks on the per-element output for canonical
  cases — Co-60 β⁻ ⊗ 1173 keV γ, Y-86 K X-ray + 511 keV ⊗ 1077 keV γ, the
  Sr-90 → Y-90 pure-β⁻ negative case, and the v0.11 γ-γ regression on the
  Co-60 1173/1333 cascade pair.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from nucl_parquet.g4.mixed_coincidences import (
    _LEVEL_KEV_TOL,
    annotate_gamma_gamma,
    build_mixed_pairs,
    load_daughter_cascade,
    match_fed_levels,
    synthesize_parent_emissions,
)

_REPO_ROOT = Path(__file__).parent.parent
_COINC_DIR = _REPO_ROOT / "data" / "meta" / "ensdf" / "coincidences"


# --------------------------------------------------------------------------- helpers


def _decay_detailed_rows(rows: list[dict]) -> pl.DataFrame:
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


def _nudex_levels(rows: list[tuple[int, int, int, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Z": [r[0] for r in rows],
            "A": [r[1] for r in rows],
            "level_idx": [r[2] for r in rows],
            "energy_MeV": [r[3] for r in rows],
        },
        schema={"Z": pl.Int32, "A": pl.Int32, "level_idx": pl.Int32, "energy_MeV": pl.Float64},
    )


def _nudex_gammas(rows: list[tuple[int, int, int, int, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Z": [r[0] for r in rows],
            "A": [r[1] for r in rows],
            "source_level_idx": [r[2] for r in rows],
            "dest_level_idx": [r[3] for r in rows],
            "gamma_energy_MeV": [r[4] for r in rows],
            "intensity": [r[5] for r in rows],
        },
        schema={
            "Z": pl.Int32,
            "A": pl.Int32,
            "source_level_idx": pl.Int32,
            "dest_level_idx": pl.Int32,
            "gamma_energy_MeV": pl.Float64,
            "intensity": pl.Float64,
        },
    )


# --------------------------------------------------------------------------- unit tests


def test_match_fed_levels_picks_closest_within_tolerance():
    # Two candidate levels straddling the requested daughter_ex_kev; the closest
    # one wins.
    dd = _decay_detailed_rows(
        [
            {
                "Z": 27,
                "A": 60,
                "parent_ex_kev": 0.0,
                "parent_level_flag": "-",
                "half_life_s": 1.0,
                "decay_mode": "beta-",
                "daughter_Z": 28,
                "daughter_A": 60,
                "daughter_ex_kev": 1332.5,
                "daughter_level_flag": "-",
                "branching": 0.99,
                "q_value_kev": 317.0,
                "forbiddenness": "",
            },
        ]
    )
    levels = _nudex_levels(
        [
            (28, 60, 1, 0.0),
            (28, 60, 2, 1.33252),  # 0.02 keV away (best match)
            (28, 60, 3, 1.336),  # 3.5 keV away (outside tolerance)
        ]
    )
    out = match_fed_levels(dd, levels)
    assert out.height == 1
    assert out["level_idx"][0] == 2
    assert abs(out["level_energy_keV"][0] - 1332.52) < 1e-3


def test_synthesize_parent_emissions_beta_minus_uses_branching_as_intensity():
    dd = _decay_detailed_rows(
        [
            {
                "Z": 27,
                "A": 60,
                "parent_ex_kev": 0.0,
                "parent_level_flag": "-",
                "half_life_s": 1.0,
                "decay_mode": "beta-",
                "daughter_Z": 28,
                "daughter_A": 60,
                "daughter_ex_kev": 2505.75,
                "daughter_level_flag": "-",
                "branching": 0.9988,
                "q_value_kev": 317.057,
                "forbiddenness": "",
            },
        ]
    )
    levels = _nudex_levels([(28, 60, 5, 2.50575)])
    matched = match_fed_levels(dd, levels)
    out = synthesize_parent_emissions(matched, radiation=None)
    beta = out.filter(pl.col("emission1_rad_type") == "beta")
    assert beta.height == 1
    assert beta["emission1_energy_keV"][0] == pytest.approx(317.057)
    assert beta["emission1_intensity_per_decay"][0] == pytest.approx(0.9988, abs=1e-4)


def test_synthesize_parent_emissions_beta_plus_emits_annihilation():
    dd = _decay_detailed_rows(
        [
            {
                "Z": 39,
                "A": 86,
                "parent_ex_kev": 0.0,
                "parent_level_flag": "-",
                "half_life_s": 1.0,
                "decay_mode": "beta+",
                "daughter_Z": 38,
                "daughter_A": 86,
                "daughter_ex_kev": 1076.68,
                "daughter_level_flag": "-",
                "branching": 0.02,
                "q_value_kev": 3000.0,
                "forbiddenness": "",
            },
        ]
    )
    levels = _nudex_levels([(38, 86, 2, 1.07668)])
    matched = match_fed_levels(dd, levels)
    out = synthesize_parent_emissions(matched, radiation=None)
    types = sorted(out["emission1_rad_type"].unique().to_list())
    assert types == ["annihilation_511", "beta"]
    annih = out.filter(pl.col("emission1_rad_type") == "annihilation_511")
    assert annih.height == 1
    assert annih["emission1_energy_keV"][0] == pytest.approx(511.0)
    # Two 511 keV photons per β⁺ decay through this channel.
    assert annih["emission1_intensity_per_decay"][0] == pytest.approx(0.04, abs=1e-4)


def test_build_mixed_pairs_co60_textbook():
    # Co-60 β⁻ feeds Ni-60 level 5 at 2505.75 keV; 1173 keV γ depopulates
    # that level with intensity 0.9998 → pair_intensity ≈ 0.9986.
    dd = _decay_detailed_rows(
        [
            {
                "Z": 27,
                "A": 60,
                "parent_ex_kev": 0.0,
                "parent_level_flag": "-",
                "half_life_s": 1.0,
                "decay_mode": "beta-",
                "daughter_Z": 28,
                "daughter_A": 60,
                "daughter_ex_kev": 2505.75,
                "daughter_level_flag": "-",
                "branching": 0.9988,
                "q_value_kev": 317.057,
                "forbiddenness": "",
            },
        ]
    )
    levels = _nudex_levels([(28, 60, 1, 0.0), (28, 60, 2, 1.33252), (28, 60, 5, 2.50575)])
    gammas = _nudex_gammas(
        [
            (28, 60, 5, 2, 1.17324, 0.9998),
            (28, 60, 2, 1, 1.33251, 0.9998),
        ]
    )
    matched = match_fed_levels(dd, levels)
    pe = synthesize_parent_emissions(matched, radiation=None)
    cascade = load_daughter_cascade(levels, gammas)
    pairs = build_mixed_pairs(pe, cascade)
    co60 = pairs.filter((pl.col("emission1_rad_type") == "beta") & ((pl.col("emission2_energy_keV") - 1173).abs() < 2))
    assert co60.height == 1
    assert co60["pair_intensity"][0] == pytest.approx(0.9986, abs=1e-3)


def test_build_mixed_pairs_pure_beta_no_gamma_yields_empty():
    # Sr-90 → Y-90 β⁻ with no daughter γ should produce zero mixed pairs.
    dd = _decay_detailed_rows(
        [
            {
                "Z": 38,
                "A": 90,
                "parent_ex_kev": 0.0,
                "parent_level_flag": "-",
                "half_life_s": 1.0,
                "decay_mode": "beta-",
                "daughter_Z": 39,
                "daughter_A": 90,
                "daughter_ex_kev": 0.0,  # ground-state feeding
                "daughter_level_flag": "-",
                "branching": 1.0,
                "q_value_kev": 546.0,
                "forbiddenness": "",
            },
        ]
    )
    levels = _nudex_levels([(39, 90, 1, 0.0)])
    gammas = _nudex_gammas([])
    matched = match_fed_levels(dd, levels)
    pe = synthesize_parent_emissions(matched, radiation=None)
    cascade = load_daughter_cascade(levels, gammas)
    pairs = build_mixed_pairs(pe, cascade)
    assert pairs.height == 0


def test_annotate_gamma_gamma_preserves_row_count_when_no_match():
    # Existing γ-γ rows for which no parent decay matches must survive
    # with parent_decay_mode = NULL (not be dropped).
    gg = pl.DataFrame(
        {
            "Z": [82, 82],
            "A": [208, 208],
            "dataset": [1, 1],
            "gamma_energy_keV": [583.0, 2614.0],
            "coinc_energy_keV": [2614.0, 583.0],
            "gamma1_energy_keV": [583.0, 2614.0],
            "gamma1_intensity": [0.85, 1.0],
            "gamma2_energy_keV": [2614.0, 583.0],
            "gamma2_intensity": [1.0, 0.85],
            "parent_level_keV": [3197.0, 3197.0],
            "intermediate_level_keV": [2614.0, 2614.0],
            "final_level_keV": [0.0, 0.0],
            "pair_intensity": [85.0, 85.0],
            "gamma1_icc_total": [0.0, 0.0],
            "gamma2_icc_total": [0.0, 0.0],
        }
    )
    # No matching parent decay (empty parent_lookup input).
    parent_channels = pl.DataFrame(
        schema={
            "Z_parent": pl.Int32,
            "A_parent": pl.Int32,
            "parent_ex_kev": pl.Float64,
            "decay_mode": pl.Utf8,
            "daughter_Z": pl.Int32,
            "daughter_A": pl.Int32,
            "daughter_ex_kev": pl.Float64,
            "branching": pl.Float64,
            "level_idx": pl.Int32,
            "level_energy_keV": pl.Float64,
            "q_value_kev": pl.Float64,
        }
    )
    out = annotate_gamma_gamma(gg, parent_channels)
    assert out.height == 2  # both γ-γ rows preserved
    assert out["parent_decay_mode"].null_count() == 2


# --------------------------------------------------------------------------- data tests


_pytest_data = pytest.mark.data
_skipif_no_data = pytest.mark.skipif(
    not (_COINC_DIR / "Ni.parquet").exists(),
    reason="coincidences data not present (run mixed_coincidences build first)",
)


@_pytest_data
@_skipif_no_data
def test_co60_beta_gamma_pair_ships():
    """β⁻ (317 keV endpoint) ⊗ 1173 keV γ for Co-60 → Ni-60."""
    ni = pl.read_parquet(_COINC_DIR / "Ni.parquet")
    co_beta = ni.filter(
        (pl.col("A") == 60)
        & (pl.col("emission1_rad_type") == "beta")
        & (pl.col("parent_decay_mode") == "beta-")
        & ((pl.col("emission2_energy_keV") - 1173).abs() < 2)
    )
    assert co_beta.height >= 1
    # Pick the canonical 317 keV β endpoint channel.
    canonical = co_beta.filter((pl.col("emission1_energy_keV") - 317).abs() < 1)
    assert canonical.height == 1
    assert canonical["pair_intensity"][0] == pytest.approx(0.9986, abs=2e-3)


@_pytest_data
@_skipif_no_data
def test_y86_k_xray_gamma_pair_ships():
    """KshellEC K X-ray ⊗ 1077 keV γ for Y-86 → Sr-86 (prompt-γ PET)."""
    sr = pl.read_parquet(_COINC_DIR / "Sr.parquet")
    pair = sr.filter(
        (pl.col("A") == 86)
        & (pl.col("emission1_rad_type") == "xray")
        & (pl.col("emission1_shell") == "K")
        & (pl.col("parent_decay_mode") == "KshellEC")
        & ((pl.col("emission2_energy_keV") - 1077).abs() < 2)
    )
    assert pair.height >= 1


@_pytest_data
@_skipif_no_data
def test_y86_annihilation_gamma_pair_ships():
    """511 keV annihilation ⊗ 1077 keV γ for Y-86 → Sr-86."""
    sr = pl.read_parquet(_COINC_DIR / "Sr.parquet")
    pair = sr.filter(
        (pl.col("A") == 86)
        & (pl.col("emission1_rad_type") == "annihilation_511")
        & ((pl.col("emission2_energy_keV") - 1077).abs() < 2)
    )
    assert pair.height >= 1


@_pytest_data
@_skipif_no_data
def test_sr90_y90_pure_beta_has_no_mixed_pairs():
    """Sr-90 β⁻ → Y-90 ground state has no daughter γ cascade — zero mixed pairs."""
    y90 = pl.read_parquet(_COINC_DIR / "Y.parquet").filter(pl.col("A") == 90)
    mixed = y90.filter((pl.col("parent_decay_mode") == "beta-") & (pl.col("emission1_rad_type") != "gamma"))
    assert mixed.height == 0


@_pytest_data
@_skipif_no_data
def test_co60_gamma_gamma_regression_preserved():
    """v0.11 γ-γ Co-60 1173/1333 cascade pair is unchanged after mixed-emission augment."""
    ni = pl.read_parquet(_COINC_DIR / "Ni.parquet")
    pair = ni.filter(
        (pl.col("A") == 60)
        & (pl.col("emission1_rad_type") == "gamma")
        & (pl.col("emission2_rad_type") == "gamma")
        & (
            ((pl.col("gamma1_energy_keV") - 1173).abs() < 2) & ((pl.col("gamma2_energy_keV") - 1333).abs() < 2)
            | ((pl.col("gamma1_energy_keV") - 1333).abs() < 2) & ((pl.col("gamma2_energy_keV") - 1173).abs() < 2)
        )
    )
    assert pair.height >= 1
    # parent_decay_mode was back-filled by mixed_coincidences.annotate_gamma_gamma.
    assert "beta-" in pair["parent_decay_mode"].to_list()


@_pytest_data
@_skipif_no_data
def test_level_energy_tolerance_documented():
    """Build script's fuzzy-match tolerance constant is documented."""
    assert _LEVEL_KEV_TOL == 1.0


@_pytest_data
@_skipif_no_data
def test_coincidences_helper_filters_by_emission_type():
    """nucl_parquet.coincidences() helper filters γ-γ vs mixed correctly."""
    import nucl_parquet

    db = nucl_parquet.connect()
    gg = nucl_parquet.coincidences(db, 28, 60, emission1_rad_type="gamma", emission2_rad_type="gamma").fetchall()
    bg = nucl_parquet.coincidences(db, 28, 60, emission1_rad_type="beta").fetchall()
    assert len(gg) > 0
    assert len(bg) > 0
    # emission1_rad_type is the 6th column in the SELECT — see loader.coincidences().
    assert all(row[5] == "gamma" for row in gg)
    assert all(row[5] == "beta" for row in bg)
