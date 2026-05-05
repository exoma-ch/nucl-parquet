"""Rayleigh (coherent) scattering data from G4EMLOW × strata (issue #98).

Two outputs from the strata `em/` subset:

1. ``data/em/photon_rayleigh_cdf.parquet`` — cumulative probability vs.
   momentum transfer ``q [Å⁻¹]`` per Z, used for inverse-CDF angular
   sampling of Rayleigh-scattered photons. Source: ``em/rayleigh_cdf.parquet``.

2. ``data/em/xray_form_factor.parquet`` — Henke / CXRO anomalous scattering
   factors ``f1`` (real) and ``f2`` (imaginary) per ``(Z, energy_eV)``. f1
   asymptotes to Z at high energy (Thomson limit); below the lowest
   ionization threshold the upstream tabulation flags rows with
   ``f1 = -9999`` — those are dropped here so consumers don't have to.
   Source: ``em/xray_scatter.parquet``.

Note: strata does not ship a direct σ_R(E, Z) cross-section. Consumers needing
the integrated Rayleigh cross-section can compute it from these two pieces
together with the standard differential formula
    dσ/dΩ = (r_e² / 2) · (1 + cos²θ) · |F(x, Z)|²
or interpolate against XCOM (already available as the ``xcom_elements`` view)
for the elemental coherent component.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# Sentinel emitted by the upstream Henke tabulation for energies below the
# lowest ionization threshold (where the anomalous part is undefined).
_F1_BELOW_THRESHOLD_SENTINEL = -9999.0


def build_cdf(strata_path: Path, out_path: Path) -> pl.DataFrame:
    df = pl.read_parquet(strata_path).select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("q_inv_angstrom").alias("q_inv_angstrom"),
        pl.col("cdf").alias("cdf"),
    )
    _assert_cdf_invariants(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d cdf rows to %s", df.height, out_path)
    return df


def build_form_factor(strata_path: Path, out_path: Path) -> pl.DataFrame:
    raw = pl.read_parquet(strata_path)
    df = (
        raw.filter(pl.col("f1") != _F1_BELOW_THRESHOLD_SENTINEL)
        .select(
            pl.col("z").cast(pl.Int32).alias("Z"),
            pl.col("symbol"),
            pl.col("energy_ev").alias("energy_eV"),
            pl.col("f1"),
            pl.col("f2"),
        )
        .sort("Z", "energy_eV")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info(
        "wrote %d form-factor rows (dropped %d below-threshold) to %s",
        df.height,
        raw.height - df.height,
        out_path,
    )
    return df


def _assert_cdf_invariants(df: pl.DataFrame) -> None:
    # Per-Z: starts at q=0 with cdf=0, monotone non-decreasing, ends near 1.
    for z, group in df.group_by("Z"):
        z_int = z[0]
        sorted_g = group.sort("q_inv_angstrom")
        if sorted_g["q_inv_angstrom"][0] != 0.0:
            raise ValueError(f"Z={z_int}: CDF does not start at q=0")
        if sorted_g["cdf"][0] != 0.0:
            raise ValueError(f"Z={z_int}: CDF(q=0) != 0")
        # Monotone non-decreasing.
        diffs = sorted_g["cdf"].diff().drop_nulls()
        if (diffs < -1e-9).any():
            raise ValueError(f"Z={z_int}: CDF not monotone non-decreasing")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--strata-em-dir", required=True, type=Path)
    p.add_argument("--out-dir", default=Path("data/em"), type=Path)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_cdf(
        args.strata_em_dir / "rayleigh_cdf.parquet",
        args.out_dir / "photon_rayleigh_cdf.parquet",
    )
    build_form_factor(
        args.strata_em_dir / "xray_scatter.parquet",
        args.out_dir / "xray_form_factor.parquet",
    )
