"""Photoelectric data from G4EMLOW × strata (issue #96).

Two outputs from the strata `em/` subset:

1. ``data/em/photon_pe_high_z_params.parquet`` — analytic-refinement
   coefficients for high-Z elements where pure interpolation against the
   tabulated EPICS2017 cross-section diverges near absorption edges.
   Each row is one Z; columns are the boundary energies and 6
   high/low-energy regression coefficients (ha0..ha5 / la0..la5).
   Source: ``em/pe_high_z_params.parquet``.

2. ``data/em/photon_pe_angular.parquet`` — analytic table for sampling the
   photoelectron emission angle. ``table_id`` 0/1 distinguishes K-shell
   from outer-shell sampling. Source: ``em/pe_angular.parquet``.

**Not yet imported**: ``em/pe_shell_cs_epics2017.parquet`` — strata's
EPICS2017 per-shell σ_PE(E, Z, shell) tabulation reads as ~600× too low
relative to XCOM (e.g. Pb K-shell at 100 keV gives 0.15 b vs XCOM's
~1500 b/atom). Suspected unit-conversion bug in strata's E³·CS decoder.
Filed upstream; once resolved this module will gain a ``build_pe_xs``
function and a ``photon_pe`` view. In the meantime, callers needing
per-shell PE cross-sections should use the existing ``epdl_subshell_pe``
view (EPDL97-derived) or compute from the high-Z params analytic form.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def build_high_z_params(strata_path: Path, out_path: Path) -> pl.DataFrame:
    df = pl.read_parquet(strata_path).rename({"z": "Z"})
    df = df.with_columns(pl.col("Z").cast(pl.Int32))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d high-Z param rows to %s", df.height, out_path)
    return df


def build_angular(strata_path: Path, out_path: Path) -> pl.DataFrame:
    df = pl.read_parquet(strata_path).select(
        pl.col("table_id").cast(pl.Int32).alias("table_id"),
        pl.col("energy_mev").alias("energy_MeV"),
        pl.col("f_value"),
        pl.col("e_upper_mev").alias("e_upper_MeV"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d pe-angular rows to %s", df.height, out_path)
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--strata-em-dir", required=True, type=Path)
    p.add_argument("--out-dir", default=Path("data/em"), type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_high_z_params(
        args.strata_em_dir / "pe_high_z_params.parquet",
        args.out_dir / "photon_pe_high_z_params.parquet",
    )
    build_angular(
        args.strata_em_dir / "pe_angular.parquet",
        args.out_dir / "photon_pe_angular.parquet",
    )
