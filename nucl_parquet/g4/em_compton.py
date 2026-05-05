"""Compton (incoherent) scattering data from G4EMLOW × strata (issue #97).

Three outputs from the strata `em/` subset:

1. ``data/em/photon_compton.parquet`` — integrated Compton cross-section
   σ_C(E, Z) [barns/atom]. Source: ``em/compton_cs.parquet``.

2. ``data/em/compton_scattering_function.parquet`` — incoherent scattering
   function S(x, Z) where x = sin(θ/2)/λ in Å⁻¹. At high x → Z (free-electron
   limit), at low x → 0 (bound-electron suppression). Source:
   ``em/compton_sf.parquet``.

3. ``data/em/compton_doppler_profiles.parquet`` — per-shell electron
   momentum profile f(p) for Doppler broadening of the Compton-scattered
   energy. Profile is normalized so ∫ f(p) dp ≈ 1 per shell. Source:
   ``em/doppler_profiles.parquet``.

Note: ``shell`` in doppler_profiles is the EADL integer code (1=K, 2=L1,
3=L2, 4=L3, 5=M1, ...). Decoding to letter labels is left to consumers
since the same convention is used inconsistently across G4 EM tables.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def build_cs(strata_path: Path, out_path: Path) -> pl.DataFrame:
    df = pl.read_parquet(strata_path).select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("energy_mev").alias("energy_MeV"),
        pl.col("cross_section_barn").alias("sigma_b"),
    )
    if (df["sigma_b"] < 0).any():
        raise ValueError("negative Compton cross-section in upstream data")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d compton-cs rows to %s", df.height, out_path)
    return df


def build_sf(strata_path: Path, out_path: Path) -> pl.DataFrame:
    df = pl.read_parquet(strata_path).select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("x_inv_angstrom").alias("x_inv_angstrom"),
        pl.col("scattering_function").alias("S"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d compton-sf rows to %s", df.height, out_path)
    return df


def build_doppler(strata_path: Path, out_path: Path) -> pl.DataFrame:
    df = pl.read_parquet(strata_path).select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("shell").cast(pl.Int32).alias("shell"),
        pl.col("momentum_au").alias("momentum_au"),
        pl.col("profile").alias("profile"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d doppler rows to %s", df.height, out_path)
    return df


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--strata-em-dir", required=True, type=Path)
    p.add_argument("--out-dir", default=Path("data/em"), type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_cs(
        args.strata_em_dir / "compton_cs.parquet",
        args.out_dir / "photon_compton.parquet",
    )
    build_sf(
        args.strata_em_dir / "compton_sf.parquet",
        args.out_dir / "compton_scattering_function.parquet",
    )
    build_doppler(
        args.strata_em_dir / "doppler_profiles.parquet",
        args.out_dir / "compton_doppler_profiles.parquet",
    )
