"""NUDEX photon strength functions (issue #124).

Imports the six G4NUDEXLIB1.0 PSF tables — Lorentzian/Modified-Lorentzian
parameterizations of giant-dipole-resonance (GDR) and photonuclear data,
needed for statistical-model gamma-cascade modeling above the discrete-level
cutoff.

Sources (strata, gate-class 1 raw-file copies of G4NUDEXLIB1.0/PSF/):

| Output view | Source parquet | Description |
|---|---|---|
| ``psf_e1`` | ``nudex_psf_crp_iaea_smlo_e1.parquet`` | **IAEA CRP SMLO E1 — modern recommended default**. Two-resonance Simple Modified Lorentzian for E1, plus deformation β. Covers 8980 nuclides. |
| ``psf_gdr_lor`` | ``nudex_psf_gdr_exp_lor.parquet`` | Experimental two-resonance Standard Lorentzian (CENPL.GDP-1.1, 1995). 145 measurements over 112 nuclides. |
| ``psf_gdr_mlo`` | ``nudex_psf_gdr_exp_mlo.parquet`` | Experimental two-resonance Modified Lorentzian with uncertainties + energy bounds. 178 measurements over 120 nuclides. |
| ``psf_gdr_slo`` | ``nudex_psf_gdr_exp_slo.parquet`` | Experimental two-resonance Standard Lorentzian (re-fit) with uncertainties + bounds. 180 measurements over 120 nuclides. |
| ``psf_gdr_theor`` | ``nudex_psf_gdr_theor.parquet`` | Theoretical two-resonance Goriely systematics, deformation-aware. 5986 nuclides. |
| ``psf_photonuclear`` | ``nudex_psf_photonuclear_exp.parquet`` | Photonuclear (γ,abs)/(γ,sn)/(γ,xn) experimental peaks. 1912 measurements over 224 nuclides. |

**Default recommendation**: for modern statistical-model decay calculations,
use ``psf_e1`` (IAEA CRP SMLO E1, the recommended default per RIPL-3
guidance). Use the experimental tables (``psf_gdr_lor``/``mlo``/``slo``)
for systematic-uncertainty studies or when comparing alternate
parameterizations. ``psf_gdr_theor`` covers nuclides without measurements
via Goriely systematics.

**Multi-row-per-nuclide convention**: experimental tables (``lor``,
``mlo``, ``slo``, ``photonuclear``) carry multiple rows per ``(Z, A)`` —
one per measurement / reference. The ``Ref`` and ``EXFOR`` columns
identify the unique row. Modeled / theoretical tables (``e1``, ``theor``)
have one row per ``(Z, A)``.

**NaN convention**: tables with optional second-resonance parameters
(``E2_MeV`` / ``CS2_mb`` / ``W2_MeV``) leave them NaN when the upstream
fit used only one resonance. Uncertainties (``dE1`` / ``dCS1`` / ...)
are similarly NaN when not available. These are legitimate "no data"
sentinels, not bugs.

**Upstream-format note** (``photonuclear`` only): 17 rows are duplicated
on ``(Z, A, Reac, peak_idx, Ref, EXFOR)`` with one copy carrying a filled
``sint_mev_mb`` integral and the other NaN. These appear in the raw
upstream ``.dat`` and are preserved as-is for parity. Consumers should
de-duplicate per use case.

Use cases:
- Hauser-Feshbach gamma cascades from compound-nucleus decay.
- Capture-gamma spectrum modeling above the known-level cutoff.
- Sensitivity-vs-systematic studies via SLO/MLO/SMLO comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def build_e1(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """IAEA CRP SMLO E1 — modern recommended default. One row per (Z, A)."""
    raw = pl.read_parquet(strata_path)
    df = raw.select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("a").cast(pl.Int32).alias("A"),
        pl.col("er1_mev").alias("E1_MeV"),
        pl.col("wr1_mev").alias("W1_MeV"),
        pl.col("s1").alias("S1_mb"),
        pl.col("er2_mev").alias("E2_MeV"),
        pl.col("wr2_mev").alias("W2_MeV"),
        pl.col("s2").alias("S2_mb"),
        pl.col("beta").alias("deformation_beta"),
    ).sort("Z", "A")
    if (df["E1_MeV"] <= 0).any():
        raise ValueError("non-positive E1 in SMLO E1 table")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d SMLO E1 rows to %s", df.height, out_path)
    return df


def build_gdr_lor(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Experimental two-resonance Standard Lorentzian (older CENPL.GDP-1.1)."""
    raw = pl.read_parquet(strata_path)
    df = raw.select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("a").cast(pl.Int32).alias("A"),
        pl.col("El").alias("symbol"),
        pl.col("N").alias("n_measurements"),
        pl.col("M").alias("n_resonances"),
        pl.col("E1_mev").alias("E1_MeV"),
        pl.col("CS1_mb"),
        pl.col("W1_mev").alias("W1_MeV"),
        pl.col("E2_mev").alias("E2_MeV"),
        pl.col("CS2_mb"),
        pl.col("W2_mev").alias("W2_MeV"),
        pl.col("Ref").alias("reference"),
    ).sort("Z", "A", "reference")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d GDR LOR rows to %s", df.height, out_path)
    return df


def _build_mlo_slo(strata_path: Path, out_path: Path, label: str) -> pl.DataFrame:
    """MLO and SLO share the same upstream schema — fit + uncertainties + bounds."""
    raw = pl.read_parquet(strata_path)
    df = raw.select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("a").cast(pl.Int32).alias("A"),
        pl.col("El").alias("symbol"),
        pl.col("Reac").alias("reaction_code"),
        pl.col("E1_mev").alias("E1_MeV"),
        pl.col("CS1_mb"),
        pl.col("W1_mev").alias("W1_MeV"),
        pl.col("E2_mev").alias("E2_MeV"),
        pl.col("CS2_mb"),
        pl.col("W2_mev").alias("W2_MeV"),
        pl.col("dE1_mev").alias("dE1_MeV"),
        pl.col("dCS1_mb"),
        pl.col("dW1_mev").alias("dW1_MeV"),
        pl.col("dE2_mev").alias("dE2_MeV"),
        pl.col("dCS2_mb"),
        pl.col("dW2_mev").alias("dW2_MeV"),
        pl.col("Emin_mev").alias("E_min_MeV"),
        pl.col("Emax_mev").alias("E_max_MeV"),
        pl.col("Ref").alias("reference"),
    ).sort("Z", "A", "reaction_code", "reference")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d GDR %s rows to %s", df.height, label, out_path)
    return df


def build_gdr_mlo(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Experimental Modified Lorentzian — temperature-dependent damping."""
    return _build_mlo_slo(strata_path, out_path, "MLO")


def build_gdr_slo(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Experimental Standard Lorentzian (re-fit with uncertainties)."""
    return _build_mlo_slo(strata_path, out_path, "SLO")


def build_gdr_theor(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Theoretical Goriely systematics — deformation-aware. One row per (Z, A)."""
    raw = pl.read_parquet(strata_path)
    df = raw.select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("a").cast(pl.Int32).alias("A"),
        pl.col("El").alias("symbol"),
        pl.col("Eta").alias("deformation_eta"),
        pl.col("E1_mev").alias("E1_MeV"),
        pl.col("W1_mev").alias("W1_MeV"),
        pl.col("E2_mev").alias("E2_MeV"),
        pl.col("W2_mev").alias("W2_MeV"),
    ).sort("Z", "A")
    if (df["E1_MeV"] <= 0).any():
        raise ValueError("non-positive E1 in theoretical GDR table")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d GDR theor rows to %s", df.height, out_path)
    return df


def build_photonuclear(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Photonuclear experimental peaks — (γ,abs), (γ,sn), (γ,xn) reactions."""
    raw = pl.read_parquet(strata_path)
    df = raw.select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("a").cast(pl.Int32).alias("A"),
        pl.col("El").alias("symbol"),
        pl.col("Reac").alias("reaction"),
        pl.col("Nep").alias("n_peaks"),
        pl.col("peak_idx").alias("peak_idx"),
        pl.col("Er_mev").alias("Er_MeV"),
        pl.col("Cs_mb"),
        pl.col("W_mev").alias("W_MeV"),
        pl.col("Egmax_mev").alias("Eg_max_MeV"),
        pl.col("sint_mev_mb").alias("sigma_int_MeV_mb"),
        pl.col("s1int_mb").alias("sigma1_int_mb"),
        pl.col("Ref").alias("reference"),
        pl.col("EXFOR").alias("exfor_author"),
    ).sort("Z", "A", "reaction", "peak_idx", "reference")
    if (df["Er_MeV"] <= 0).any():
        raise ValueError("non-positive resonance energy in photonuclear table")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d photonuclear rows to %s", df.height, out_path)
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--strata-nuclear-dir", required=True, type=Path)
    p.add_argument("--out-dir", default=Path("data/meta"), type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sd = args.strata_nuclear_dir
    od = args.out_dir
    build_e1(sd / "nudex_psf_crp_iaea_smlo_e1.parquet", od / "psf_e1.parquet")
    build_gdr_lor(sd / "nudex_psf_gdr_exp_lor.parquet", od / "psf_gdr_lor.parquet")
    build_gdr_mlo(sd / "nudex_psf_gdr_exp_mlo.parquet", od / "psf_gdr_mlo.parquet")
    build_gdr_slo(sd / "nudex_psf_gdr_exp_slo.parquet", od / "psf_gdr_slo.parquet")
    build_gdr_theor(sd / "nudex_psf_gdr_theor.parquet", od / "psf_gdr_theor.parquet")
    build_photonuclear(sd / "nudex_psf_photonuclear_exp.parquet", od / "psf_photonuclear.parquet")
