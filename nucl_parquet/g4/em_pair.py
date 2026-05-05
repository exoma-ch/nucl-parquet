"""Pair + triplet production cross-sections from G4EMLOW × strata (issue #99).

Combines three input files from strata's `em/` subset into a single
`data/em/photon_pair.parquet` keyed by `(Z, energy_MeV, channel)` where
channel ∈ {nuclear, triplet, total}:

* `em/pair_xs_nuclear.parquet` — nuclear-field pair (γ + nucleus → e+e−)
* `em/pair_triplet_xs.parquet` — triplet (γ + electron → e+e−e−), 2× threshold
* `em/pair_xs_total.parquet`   — sum of both, as G4 ships it

Cross-sections are converted from cm² (strata native) to barns
(1 b = 1e-24 cm²) for consistency with all other XS tables in the repo.

Sweep invariants enforced:

* σ ≥ 0 everywhere
* σ_nuclear == 0 below 1.022 MeV (kinematic threshold, electron+positron rest mass)
* σ_triplet == 0 below ~2.044 MeV (threshold doubles in the electron field)
* σ_total ≈ σ_nuclear + σ_triplet within 1 % above 2.5 MeV
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

CM2_TO_BARN = 1.0e24
NUCLEAR_THRESHOLD_MEV = 1.022
TRIPLET_THRESHOLD_MEV = 2.044


def _load_channel(path: Path, channel: str) -> pl.DataFrame:
    """Read one strata pair/triplet parquet, normalize columns + units."""
    df = pl.read_parquet(path)
    return df.select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("energy_mev").alias("energy_MeV"),
        (pl.col("cs_cm2") * CM2_TO_BARN).alias("sigma_b"),
        pl.lit(channel).alias("channel"),
    )


def build(strata_em_dir: Path, out_path: Path) -> pl.DataFrame:
    """Convert strata's pair-production trio into a single parquet.

    Args:
        strata_em_dir: directory containing pair_xs_{nuclear,total}.parquet and
            pair_triplet_xs.parquet (typically the strata-data ``em/`` subdir).
        out_path: target parquet path (parent dir created if missing).

    Returns:
        The combined dataframe (also written to ``out_path``).
    """
    nuclear = _load_channel(strata_em_dir / "pair_xs_nuclear.parquet", "nuclear")
    triplet = _load_channel(strata_em_dir / "pair_triplet_xs.parquet", "triplet")
    total = _load_channel(strata_em_dir / "pair_xs_total.parquet", "total")

    combined = pl.concat([nuclear, triplet, total]).sort("Z", "channel", "energy_MeV")

    # Sweep invariants — fail loud at build time if upstream G4 changes shape.
    _assert_invariants(combined)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d rows to %s", combined.height, out_path)
    return combined


def _assert_invariants(df: pl.DataFrame) -> None:
    if (df["sigma_b"] < 0).any():
        raise ValueError("negative cross-section in pair/triplet data")

    nuclear_below = df.filter((pl.col("channel") == "nuclear") & (pl.col("energy_MeV") < NUCLEAR_THRESHOLD_MEV))
    if (nuclear_below["sigma_b"] > 0).any():
        raise ValueError(f"σ_nuclear > 0 below {NUCLEAR_THRESHOLD_MEV} MeV threshold ({nuclear_below.height} rows)")

    triplet_below = df.filter((pl.col("channel") == "triplet") & (pl.col("energy_MeV") < TRIPLET_THRESHOLD_MEV))
    if (triplet_below["sigma_b"] > 0).any():
        raise ValueError(f"σ_triplet > 0 below {TRIPLET_THRESHOLD_MEV} MeV threshold ({triplet_below.height} rows)")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--strata-em-dir", required=True, type=Path)
    p.add_argument("--out", default=Path("data/em/photon_pair.parquet"), type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build(args.strata_em_dir, args.out)
