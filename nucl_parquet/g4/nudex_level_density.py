"""NUDEX level-density parameters (issue #125).

Imports three NUDEX tables that drive statistical-model decay calculations
above the discrete-level cutoff — needed for high-fidelity capture-gamma
cascade modeling, evaporation-residue deexcitation, and Hauser-Feshbach
statistical decay.

Sources (strata, gate-class 1 raw-file copies of G4NUDEXLIB1.0):

1. ``nuclear/nudex_level_density_bfmeff.parquet`` →
   ``data/meta/level_density_bfm.parquet`` — back-shifted Fermi-gas (BFM)
   *effective* parameters (fitted to discrete levels). 289 rows, one per
   nuclide with adequate s-wave neutron-resonance data.

2. ``nuclear/nudex_level_density_ctmeff.parquet`` →
   ``data/meta/level_density_ctm.parquet`` — constant-temperature (CTM)
   *effective* parameters. Same 289-nuclide subset; complementary
   parameterization to BFM.

3. ``nuclear/nudex_levels_param.parquet`` →
   ``data/meta/level_density_params.parquet`` — per-nuclide level-density
   normalization & cutoff parameters (3353 rows: every nuclide with any
   discrete-level data, NOT the 289-subset with full BFM/CTM fits).

**BFM vs CTM**: not interchangeable. BFM (back-shifted Fermi gas,
``a × exp(2√(aU))`` with shift) is appropriate at high excitation
(typically U ≳ neutron-separation energy); CTM (constant-temperature,
``exp((U-U₀)/T)``) is appropriate at low excitation (U ≲ several MeV).
Each captures the data the other misses; consumers running through both
regimes typically use a hybrid.

**Effective parameters caveat**: both BFM and CTM "eff" tables are
*fitted to the discrete-level region* — they reproduce known levels
within their fit range. Pure-theory calculations (microscopic combinatorial,
HFB-based) use different conventions and won't match these.

**Strata#611 — missing CTM temperature parameter**: the upstream
``level-densities-ctmeff.dat`` ships 19 columns, but strata's parquet
truncates to the 16 columns shared with BFM, dropping ``Ematch`` (matching
energy), ``E0`` (low-energy shift), and ``T`` (the actual CTM
**temperature** in MeV — the parameter that makes "CTM" CTM). Without
``T``, this view is half-useful for the canonical CTM use case
``ρ(U) ~ exp((U−E0)/T)/T``; consumers should fall back to
``level_density_params.T_MeV`` for the per-nuclide temperature, which
covers a different (broader) nuclide set. Filed strata#611; once fixed
and the catalog SHA bumps, extend ``_normalize_eff`` additively to
expose the three CTM-specific columns without breaking existing queries.

**NaN columns in `levels_param`**: the upstream `EX` (excitation energy)
and `sigma_mev` (spin-cutoff parameter) columns are NaN for all 3353
rows in the current strata pin. These appear to be placeholder columns
not populated by NUDEX itself; preserved as-is for upstream parity but
not asserted finite.

Use cases:
- Hauser-Feshbach statistical-decay calculations (needs BFM at high U).
- Discrete-level extrapolation (needs CTM at low U).
- Cumulative-level-count modeling for radiation-transport tools.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def _normalize_eff(df: pl.DataFrame) -> pl.DataFrame:
    """Shared schema for BFM and CTM effective tables.

    Both upstream files have identical column sets (16 columns each):
    nuclide id (z, a, El), spin parameter (I0), neutron-binding (Bn),
    s-wave spacing (D0/Derr), low/high level-fit anchors (Nlow/Ulow,
    Ntop/Utop), shell correction (dW), damping (gamma), asymptotic
    level-density parameter (ainf/aerr), and pairing (pairing).
    """
    return df.select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("a").cast(pl.Int32).alias("A"),
        pl.col("El").alias("symbol"),
        pl.col("I0").alias("target_spin"),
        pl.col("Bn").alias("neutron_binding_MeV"),
        pl.col("D0").alias("D0_eV"),
        pl.col("Derr").alias("D0_err_eV"),
        pl.col("Nlow").alias("N_low"),
        pl.col("Ulow").alias("U_low_MeV"),
        pl.col("Ntop").alias("N_top"),
        pl.col("Utop").alias("U_top_MeV"),
        pl.col("dW").alias("shell_correction_MeV"),
        pl.col("gamma").alias("damping_per_MeV"),
        pl.col("ainf").alias("a_asymptotic_per_MeV"),
        pl.col("aerr").alias("a_asymptotic_err"),
        pl.col("pairing").alias("pairing_MeV"),
    ).sort("Z", "A")


def _assert_eff_invariants(df: pl.DataFrame, name: str) -> None:
    """Shared positivity / finite-ness guards. Smoke-alarms an upstream
    column-shift or unit regression — cheap insurance against silent breakage
    of the kind that bit us in strata#600 / strata#610 before."""
    if (df["neutron_binding_MeV"] <= 0).any():
        raise ValueError(f"non-positive neutron-binding energy in {name} table")
    if (df["D0_eV"] <= 0).any():
        raise ValueError(f"non-positive s-wave resonance spacing in {name} table")
    if (df["a_asymptotic_per_MeV"] <= 0).any():
        raise ValueError(f"non-positive asymptotic level-density parameter in {name} table")
    for col in ("neutron_binding_MeV", "D0_eV", "a_asymptotic_per_MeV", "pairing_MeV", "shell_correction_MeV"):
        if not df[col].is_finite().all():
            raise ValueError(f"non-finite values in {name}.{col}")


def build_bfm(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Back-shifted Fermi-gas effective parameters."""
    df = _normalize_eff(pl.read_parquet(strata_path))
    _assert_eff_invariants(df, "BFM")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d BFM rows to %s", df.height, out_path)
    return df


def build_ctm(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Constant-temperature effective parameters."""
    df = _normalize_eff(pl.read_parquet(strata_path))
    _assert_eff_invariants(df, "CTM")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d CTM rows to %s", df.height, out_path)
    return df


def build_params(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Per-nuclide level-density normalization + cutoff parameters."""
    raw = pl.read_parquet(strata_path)
    df = raw.select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("a").cast(pl.Int32).alias("A"),
        pl.col("el").alias("symbol"),
        pl.col("T_mev").alias("T_MeV"),
        pl.col("dT_mev").alias("T_err_MeV"),
        pl.col("U0_mev").alias("U0_MeV"),
        pl.col("dU0_mev").alias("U0_err_MeV"),
        pl.col("Nlev").alias("N_levels"),
        pl.col("Nmax").alias("N_max"),
        pl.col("N0").alias("N0"),
        pl.col("Nc").alias("N_cutoff"),
        pl.col("Umax_mev").alias("U_max_MeV"),
        pl.col("Uc_mev").alias("U_cutoff_MeV"),
        pl.col("Chi").alias("chi_squared"),
        pl.col("Fit_Flag").alias("fit_flag"),
        pl.col("NoX").alias("n_x_points"),
        pl.col("Xm").alias("Xm"),
        # Preserve EX, sigma_mev as-is (all NaN upstream — see module docstring).
        pl.col("EX").alias("EX_MeV"),
        pl.col("sigma_mev").alias("sigma_MeV"),
    ).sort("Z", "A")

    if (df["N_levels"] < 0).any():
        raise ValueError("negative level count in NUDEX levels_param")
    if (df["T_MeV"] < 0).any():
        raise ValueError("negative temperature in NUDEX levels_param (T=0 means unfit; <0 is a parse error)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d level-density-params rows to %s", df.height, out_path)
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--strata-nuclear-dir", required=True, type=Path)
    p.add_argument("--out-dir", default=Path("data/meta"), type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_bfm(
        args.strata_nuclear_dir / "nudex_level_density_bfmeff.parquet",
        args.out_dir / "level_density_bfm.parquet",
    )
    build_ctm(
        args.strata_nuclear_dir / "nudex_level_density_ctmeff.parquet",
        args.out_dir / "level_density_ctm.parquet",
    )
    build_params(
        args.strata_nuclear_dir / "nudex_levels_param.parquet",
        args.out_dir / "level_density_params.parquet",
    )
