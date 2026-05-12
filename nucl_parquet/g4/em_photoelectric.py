"""Photoelectric data from G4EMLOW × strata (issue #96 / #105).

Four outputs from the strata `em/` subset:

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

4. ``data/em/photon_pe_total.parquet`` — **total** photoelectric
   cross-section σ_PE(E, Z) summed across shells, in barn/atom. Pre-decoded
   upstream (per strata#600 fix). Split into two energy regions
   (``region`` ∈ {0, 1}) for lin-lin interpolation runtime; each region
   covers a contiguous energy range delimited by the K-edge for the
   element. Consumers running attenuation queries should prefer this
   over ``photon_pe`` (which is per-shell). Source:
   ``em/epics2017/pe_cs.parquet``.

Note on the strata ``cs_barn`` column in the per-shell file
(``pe_shell_cs_epics2017.parquet``): the column stores **E³·σ in
MeV³·barn**, *not* decoded barn/atom — this is the pre-conditioned
ordinate that strata's ``LinLinEpsilonE3PerRegion`` runtime path expects
(per strata ADR-054 §1+§5). The encoding is required for log-log
extrapolation correctness above the per-shell grid endpoints; storing
decoded σ instead breaks ``livermore_pe::test_interpolation_continuity``
with 226× discontinuities at high energies (strata#645 closure).

The decode applied in :func:`build_pe_xs` (``sigma_b = cs_raw / E³``) is
the correct ETL transform from stored E³·σ to consumer-facing barn/atom,
**not a workaround** — verified against EPDL97 at Pb K-shell 99 keV
(1472.1 b vs 1472.0 b, exact match).

Strata is planning two follow-up PRs (per strata#645 closure):
- PR-A (small): add ``strata.pe.encoding`` parquet metadata key to the
  per-shell file (the total file ``epics2017/pe_cs.parquet`` already has
  it). Will let consumers detect the encoding programmatically.
- PR-B (larger): implement loader-side decode path for per-shell tables
  (analog of strata #459 for the total file). Optional for consumers
  to switch; doesn't obsolete the decode here.

Existing related view: ``epdl_subshell_pe`` (EPDL97-derived). Keep both —
EPICS2017 is the modern G4 default; EPDL97 stays for backwards-compat.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def build_pe_xs(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Decode strata's E³·σ encoding to barn/atom and write `photon_pe.parquet`.

    Strata's per-shell ``cs_barn`` column stores E³·σ in MeV³·barn — the
    pre-conditioned ordinate the ``LinLinEpsilonE3PerRegion`` runtime
    path expects (strata ADR-054 §1+§5). We apply ``sigma_b = cs_raw /
    energy_MeV³`` to recover the consumer-facing barn/atom value.
    Cross-checked against EPDL97 at Pb K-shell 99 keV: 1472.1 b vs
    1472.0 b — exact match.
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


def build_total_xs(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Total photoelectric XS per (Z, energy) — already decoded upstream.

    Source file is ``em/epics2017/pe_cs.parquet`` (strata#600 fix). The
    ``region`` column splits the lookup grid by lin-lin interpolation
    region (0 = low-energy, 1 = above the K-edge for the element). For
    a continuous σ_PE(E) curve, ``ORDER BY z, energy_MeV`` and ignore
    region; for documented region semantics see the strata fix discussion.
    """
    raw = pl.read_parquet(strata_path)
    df = raw.select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("region").cast(pl.Int32).alias("region"),
        pl.col("energy_mev").alias("energy_MeV"),
        pl.col("cs_barn").alias("sigma_b"),
    ).sort("Z", "energy_MeV")
    if (df["sigma_b"] < 0).any():
        raise ValueError("negative photoelectric total cross-section in upstream data")
    if not df["sigma_b"].is_finite().all():
        raise ValueError("non-finite photoelectric total cross-section in upstream data")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d pe-total rows to %s", df.height, out_path)
    return df


def build_angular(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Photoelectron emission angle sampling kernel (strata#294 fix).

    New schema (after strata#294 fixed the misread ftab format):
    ``(shell_id, beta_index, beta, a_majorant, c_majorant)``. ``shell_id``
    is 0 (K-shell) or 1 (L-shell catch-all per Sauter-Gavrila parameterization).
    ``beta`` = v/c of the photoelectron; ``beta_index`` is the lookup index
    into the per-shell rejection-sampling table; ``a_majorant`` /
    ``c_majorant`` parameterize the Sauter-Gavrila majorant for rejection
    sampling.
    """
    df = (
        pl.read_parquet(strata_path)
        .select(
            pl.col("shell_id").cast(pl.Int32).alias("shell_id"),
            pl.col("beta_index").cast(pl.Int32).alias("beta_index"),
            pl.col("beta"),
            pl.col("a_majorant"),
            pl.col("c_majorant"),
        )
        .sort("shell_id", "beta_index")
    )
    if (df["beta"] < 0).any() or (df["beta"] > 1).any():
        raise ValueError("beta out of [0, 1] range in pe_angular table")
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
    build_total_xs(
        args.strata_em_dir / "epics2017" / "pe_cs.parquet",
        args.out_dir / "photon_pe_total.parquet",
    )
