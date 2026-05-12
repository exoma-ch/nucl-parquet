"""NUDEX neutron-capture primary gamma spectra (issue #115 — first deliverable).

Imports two strata `nuclear/` files that ship the most-detailed available
neutron-capture gamma data — currently missing entirely from nucl-parquet:

1. ``data/meta/capture_gammas.parquet`` — per-transition primary gamma
   energies and intensities for `(target → daughter)` thermal capture
   reactions. Source: ``nuclear/nudex_primary_capture_gammas.parquet``.

2. ``data/meta/capture_gammas_summary.parquet`` — one row per
   `(daughter, variant)` with reaction string, separation energy, and
   gamma-multiplicity. Source: ``nuclear/nudex_primary_capture_gammas_summary.parquet``.

The strata files key on ``za = z*1000 + a`` (packed integer); we expand
to separate ``Z`` / ``A`` columns at import for join-friendliness.

Use cases:
- Activation analysis: predict prompt-gamma signature of neutron-irradiated
  samples (NAA, PGAA).
- Reactor instrumentation: correlate detector lines with capture reactions.
- Dosimetry of neutron-irradiated sources: account for capture-gamma
  contribution alongside the existing decay-gamma `radiation` view.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def _expand_za(df: pl.DataFrame) -> pl.DataFrame:
    """Split the strata-native packed `za = z*1000 + a` into Z, A columns."""
    return df.with_columns(
        (pl.col("za") // 1000).cast(pl.Int32).alias("Z"),
        (pl.col("za") % 1000).cast(pl.Int32).alias("A"),
    )


def build_gammas(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Per-transition primary gamma spectra (one row per gamma)."""
    raw = pl.read_parquet(strata_path)
    df = _expand_za(raw).select(
        pl.col("variant"),
        pl.col("Z"),
        pl.col("A"),
        pl.col("za"),
        pl.col("gamma_idx").cast(pl.Int32).alias("gamma_idx"),
        pl.col("eg_kev").alias("energy_keV"),
        pl.col("intensity_pct"),
    )
    if (df["intensity_pct"] < 0).any():
        raise ValueError("negative intensity in NUDEX capture gammas")
    if (df["energy_keV"] < 0).any():
        raise ValueError("negative gamma energy in NUDEX capture gammas")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d capture-gamma rows to %s", df.height, out_path)
    return df


def build_summary(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """One row per (daughter, variant) with the reaction string + S_n."""
    raw = pl.read_parquet(strata_path)
    df = _expand_za(raw).select(
        pl.col("variant"),
        pl.col("Z"),
        pl.col("A"),
        pl.col("za"),
        pl.col("target_symbol"),
        pl.col("reaction"),
        pl.col("s_kev").alias("s_n_keV"),
        pl.col("n_gammas").cast(pl.Int32).alias("n_gammas"),
        pl.col("branching_pct"),
        pl.col("param_a"),
        pl.col("param_b"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d summary rows to %s", df.height, out_path)
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--strata-nuclear-dir", required=True, type=Path)
    p.add_argument("--out-dir", default=Path("data/meta"), type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_gammas(
        args.strata_nuclear_dir / "nudex_primary_capture_gammas.parquet",
        args.out_dir / "capture_gammas.parquet",
    )
    build_summary(
        args.strata_nuclear_dir / "nudex_primary_capture_gammas_summary.parquet",
        args.out_dir / "capture_gammas_summary.parquet",
    )
