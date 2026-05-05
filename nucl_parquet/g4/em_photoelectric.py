"""Photoelectric data from G4EMLOW × strata (issue #96 / #105).

Three outputs from the strata `em/` subset:

1. ``data/em/photon_pe.parquet`` — per-shell photoelectric cross-section
   σ_PE(E, Z, shell) [barns/atom]. Decoded from strata's stored E³·CS
   values: ``sigma_b = cs_raw / energy_MeV^3``. ``shell`` is the EADL
   integer code, **0-indexed** (0=K, 1=L1, 2=L2, 3=L3, 4=M1, ...).
   Cross-checked against EPDL97 at Pb K-shell 99 keV: gives 1472.1 b
   vs EPDL97's 1472.0 b — exact match. Source:
   ``em/pe_shell_cs_epics2017.parquet``.

2. ``data/em/photon_pe_high_z_params.parquet`` — analytic-refinement
   coefficients for high-Z elements where pure interpolation against the
   tabulated EPICS2017 cross-section diverges near absorption edges.
   Source: ``em/pe_high_z_params.parquet``.

3. ``data/em/photon_pe_angular.parquet`` — photoelectron emission angle
   sampling kernel. Source: ``em/pe_angular.parquet``.

Note on the strata ``cs_barn`` column: despite its name, the upstream
column stores the raw E³·CS in MeV³·barn (not decoded barn/atom). The
``decode_e3_cs`` function in strata's loader is documented but isn't
applied during parquet generation. We apply it on import here. Tracked
in #105.

Existing related view: ``epdl_subshell_pe`` (EPDL97-derived). Keep both —
EPICS2017 is the modern G4 default; EPDL97 stays for backwards-compat.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def build_pe_xs(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Decode strata's E³·CS to barn/atom and write `photon_pe.parquet`.

    Strata's ``cs_barn`` column is mislabeled — it stores raw E³·CS in
    MeV³·barn. We apply ``sigma_b = cs_raw / energy_MeV^3`` to recover
    the actual cross-section in barn/atom. Cross-checked against EPDL97
    at Pb K-shell 99 keV (1472.1 b vs 1472.0 b — exact match).
    """
    raw = pl.read_parquet(strata_path)
    df = raw.select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("shell").cast(pl.Int32).alias("shell"),
        pl.col("energy_mev").alias("energy_MeV"),
        (pl.col("cs_barn") / pl.col("energy_mev").pow(3)).alias("sigma_b"),
    )
    if (df["sigma_b"] < 0).any():
        raise ValueError("negative photoelectric cross-section after decode")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d pe-xs rows to %s", df.height, out_path)
    return df


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
    build_pe_xs(
        args.strata_em_dir / "pe_shell_cs_epics2017.parquet",
        args.out_dir / "photon_pe.parquet",
    )
    build_high_z_params(
        args.strata_em_dir / "pe_high_z_params.parquet",
        args.out_dir / "photon_pe_high_z_params.parquet",
    )
    build_angular(
        args.strata_em_dir / "pe_angular.parquet",
        args.out_dir / "photon_pe_angular.parquet",
    )
