"""NUDEX full ENSDF level schemes (issue #122).

Imports the most-detailed available nuclear-structure source — every known
nuclear level (~159k) and every measured gamma transition (~246k) per
G4NUDEXLIB1.0/KnownLevels. Where v0.11's ``ensdf_levels`` view ships the
PhotonEvaporation6.1.2 *summary* (already-bundled with G4 transport), this
imports the underlying fully-detailed ENSDF level data (April 2014 source,
re-fitted by NUDEX with constant-temperature systematics).

Sources (strata, gate-class 1 raw-file copies of G4NUDEXLIB1.0/KnownLevels/):

| Output view | Source parquet | Description |
|---|---|---|
| ``nudex_levels`` | ``nudex_known_levels.parquet`` | Per-level entries: energy, spin, parity, half-life, decay-mode strings. ~159k rows. |
| ``nudex_level_gammas`` | ``nudex_known_levels_gammas.parquet`` | Per-transition: source/dest level, gamma energy, intensity, total-intensity, energy uncertainty. ~246k rows. |
| ``nudex_isotopes`` | ``nudex_known_levels_isotopes.parquet`` | Per-isotope summary: level/gamma counts, mass excess, neutron separation energy. 3331 rows. |

**Schema overlap with v0.11**: nucl-parquet already ships ``ensdf_levels``
(per-element parquets under ``data/meta/ensdf/levels/``) derived from
PhotonEvaporation6.1.2 — the lightweight, transport-friendly summary that
G4 itself bundles. ``nudex_levels`` is the *richer* upstream source.
Decision: keep both; consumers pick by query class.

- Use ``ensdf_levels`` for: G4-aligned cascading, transport-stage gamma
  emission, anything that wants the same level set Geant4 uses internally.
- Use ``nudex_levels`` for: high-fidelity gamma spectroscopy, statistical
  decay calculations needing the full level cascade, ENSDF-derived audits.

**Multi-row gamma transitions**: 33 ``(z, a, source_level_idx,
dest_level_idx)`` tuples appear twice with different intensity values.
These are real upstream entries — alternate evaluations of the same
transition. Preserved as-is; consumers needing a single canonical value
should aggregate per use case.

**``level_extra`` field semantics** (per RIPL discrete-level format,
Verpelli & Capote 2015): the trailing column surfaces two populations:

- ~96% (38789 rows) are **integer-valued** band indices (1, 2, 3, ...
  for excited bands above the ground band). Per the RIPL format
  documentation, this is the rotational/vibrational band the level
  belongs to.
- ~4% (1621 rows) are **small decimal values** (e.g. -0.0011, 0.0001),
  which are degeneracy-tie energy-shifts applied when the upstream had
  multiple levels at the same energy and needed micro-perturbations
  to break the tie.

Strata's parser conflates both into one column. We preserve as
``level_extra`` (Float64) for forward-compat; consumers who care about
band indices should filter ``WHERE level_extra = ROUND(level_extra)``;
those who care about degeneracy shifts should filter the complement.
Worth refactoring once strata splits this into two distinct columns.

Use cases:
- Spectrometer line-list cross-validation (3331 nuclides ≫ G4's bundled set).
- Statistical-model gamma cascades using full level cascade (Hauser-Feshbach).
- Mass-excess and S_n queries via ``nudex_isotopes``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


def build_levels(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Per-level entries — energy, spin/parity, half-life, decay modes."""
    raw = pl.read_parquet(strata_path)

    if raw.filter(pl.col("z").is_null()).height:
        # Sanity guard — upstream strata#621 was fixed; if null-Z rows reappear
        # something has regressed.
        raise ValueError("null-Z rows in NUDEX known levels (strata#621 regression?)")

    df = raw.select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("a").cast(pl.Int32).alias("A"),
        pl.col("level_idx").cast(pl.Int32).alias("level_idx"),
        pl.col("energy_mev").alias("energy_MeV"),
        pl.col("spin"),
        pl.col("parity"),
        pl.col("half_life_s"),
        pl.col("flag"),
        pl.col("ucert_marker"),
        pl.col("jpi_str"),
        pl.col("num_decays"),
        pl.col("decay_modes_str"),
        pl.col("extra_field").alias("level_extra"),
    ).sort("Z", "A", "level_idx")

    if (df["energy_MeV"] < 0).any():
        raise ValueError("negative level energy in NUDEX known levels")
    if (df["level_idx"] < 1).any():
        raise ValueError("level_idx < 1 in NUDEX known levels (1-indexed upstream)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info(
        "wrote %d level rows (%d isotopes covered) to %s",
        df.height,
        df.select(["Z", "A"]).unique().height,
        out_path,
    )
    return df


def build_level_gammas(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Per-transition gamma data — source/dest level, energy, intensity."""
    raw = pl.read_parquet(strata_path)
    df = raw.select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("a").cast(pl.Int32).alias("A"),
        pl.col("source_level_idx").cast(pl.Int32).alias("source_level_idx"),
        pl.col("dest_level_idx").cast(pl.Int32).alias("dest_level_idx"),
        pl.col("eg_mev").alias("gamma_energy_MeV"),
        pl.col("intensity"),
        pl.col("total_intensity"),
        pl.col("energy_uncertainty").alias("energy_uncertainty_MeV"),
    ).sort("Z", "A", "source_level_idx", "dest_level_idx")

    if (df["gamma_energy_MeV"] < 0).any():
        raise ValueError("negative gamma energy in NUDEX known-levels gammas")
    if (df["intensity"] < 0).any():
        raise ValueError("negative gamma intensity in NUDEX known-levels gammas")
    n_bad_cascade = df.filter(pl.col("source_level_idx") <= pl.col("dest_level_idx")).height
    if n_bad_cascade > 0:
        # Cascade direction invariant: gamma decay goes from higher → lower
        # level. Current upstream pin has 0 violations; tolerate a small
        # window for legitimate edge cases (e.g. cross-band transitions).
        raise ValueError(f"{n_bad_cascade} transitions violate source > dest cascade direction")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d gamma transitions to %s", df.height, out_path)
    return df


def build_isotopes(strata_path: Path, out_path: Path) -> pl.DataFrame:
    """Per-isotope summary — level counts, mass excess, neutron separation."""
    raw = pl.read_parquet(strata_path)
    df = raw.select(
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("a").cast(pl.Int32).alias("A"),
        pl.col("symbol"),
        pl.col("n_levels_total"),
        pl.col("n_gammas_total"),
        pl.col("n_levels_in_table"),
        pl.col("n_levels_extra"),
        pl.col("mass_excess_mev").alias("mass_excess_MeV"),
        pl.col("s_n_mev").alias("s_n_MeV"),
    ).sort("Z", "A")

    if (df["n_levels_total"] < 0).any():
        raise ValueError("negative level count in NUDEX isotopes")
    # Light physical-bounds guards — mass excess and S_n in current data are
    # within [-50, 100] MeV; trip if upstream introduces obviously-wrong values.
    me = df.filter(df["mass_excess_MeV"].is_finite())
    if me.height and (me["mass_excess_MeV"].min() < -50 or me["mass_excess_MeV"].max() > 100):
        raise ValueError(
            f"mass_excess_MeV out of physical bounds: [{me['mass_excess_MeV'].min()}, {me['mass_excess_MeV'].max()}]"
        )
    sn = df.filter(df["s_n_MeV"].is_finite())
    if sn.height and (sn["s_n_MeV"].min() < -30 or sn["s_n_MeV"].max() > 50):
        raise ValueError(f"s_n_MeV out of physical bounds: [{sn['s_n_MeV'].min()}, {sn['s_n_MeV'].max()}]")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("wrote %d isotope summary rows to %s", df.height, out_path)
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
    build_levels(sd / "nudex_known_levels.parquet", od / "nudex_levels.parquet")
    build_level_gammas(sd / "nudex_known_levels_gammas.parquet", od / "nudex_level_gammas.parquet")
    build_isotopes(sd / "nudex_known_levels_isotopes.parquet", od / "nudex_isotopes.parquet")
