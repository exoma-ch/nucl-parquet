"""Bremsstrahlung total cross-section from G4EMLOW × strata (epic #114, P0).

Outputs:

1. ``data/em/electron_brem.parquet`` — total radiative cross-section
   σ_brem(E, Z) [mb/atom] for electrons. Source: ``em/brem_cs.parquet``.

Future sub-issues will add the differential Seltzer-Berger DCS
(``brem_sb_dcs``) and the Livermore alternative (``livermore_brem``)
once the use case for those crystallizes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def build(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Convert strata's brem_cs to data/em/electron_brem.parquet.

    Strata's file ships ~200 sentinel rows with negative ``energy_mev``
    (per-Z metadata markers, not data). Filter them out at import so
    consumers get clean physical XS.
    """
    raw = pl.read_parquet(strata_path)
    df = (
        raw.filter(pl.col("energy_mev") > 0)
        .select(
            pl.col("z").cast(pl.Int32).alias("Z"),
            pl.col("energy_mev").alias("energy_MeV"),
            pl.col("cs_mb").alias("sigma_mb"),
        )
        .sort("Z", "energy_MeV")
    )
    if (df["sigma_mb"] < 0).any():
        raise ValueError("negative bremsstrahlung cross-section in upstream data")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info(
        "wrote %d brem rows to %s (dropped %d sentinel rows)",
        df.height,
        raw.height - df.height,
        out_path,
    )
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--strata-em-dir", required=True, type=Path)
    p.add_argument("--out", default=Path("data/em/electron_brem.parquet"), type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build(args.strata_em_dir / "brem_cs.parquet", args.out)
