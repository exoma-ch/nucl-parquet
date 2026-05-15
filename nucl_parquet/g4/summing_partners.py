"""Derive ``meta/ensdf/summing_partners/{Symbol}.parquet`` — ICC-corrected
coincidence pairs for HPGe true-coincidence-summing (TCS) corrections.

Per issue #177 / epic #173 Sub-C.

Each output row is one **summing partner pair** — two emissions from the same
parent decay event that can arrive simultaneously at an HPGe detector:

- **γ-γ pairs**: two nuclear gammas from a cascade, with ``icc_correction_factor``
  folding both sides' internal conversion coefficients.
- **X-ray/Auger ⊗ γ pairs**: atomic emission (from EC/IC vacancy) coincident with
  a nuclear gamma.  ICC correction applies only to the gamma side; atomic
  transitions have no ICC.

The table is materialized from ``coincidences/{Symbol}.parquet`` (already built
by :mod:`nucl_parquet.g4.coincidences` + :mod:`nucl_parquet.g4.mixed_coincidences`)
with two derived columns:

- ``icc_correction_factor``: product of ``1 / (1 + α)`` per gamma-typed emission
  side (NULL α treated as 0 = no internal conversion).
- ``pure_emission_joint_intensity``: ``pair_intensity × icc_correction_factor``.

γ-γ pairs are **canonicalized**: ``emission1_energy_keV ≤ emission2_energy_keV``
to eliminate symmetric duplicates (Co-60 1173/1333 appears once, not twice).
Mixed pairs are inherently asymmetric (X-ray/Auger on side 1, gamma on side 2)
so no dedup is needed.

Schema (output, per element ``meta/ensdf/summing_partners/{Symbol}.parquet``)
-----------------------------------------------------------------------------

    Z                            Int64    — daughter nucleus (filing convention)
    A                            Int64
    parent_state                 Utf8     — '' | 'm' | 'm2'
    parent_decay_mode            Utf8     — 'beta-' | 'beta+' | 'KshellEC' | ...
    emission1_rad_type           Utf8     — 'gamma' | 'xray' | 'auger'
    emission1_energy_keV         Float64
    emission1_intensity          Float32
    emission1_icc_total          Float32  — NULL for X-ray/Auger (no ICC)
    emission1_shell              Utf8     — 'K'|'L'|'M'|'N' for X-ray/Auger; NULL for gamma
    emission2_rad_type           Utf8     — always 'gamma' in current build
    emission2_energy_keV         Float64
    emission2_intensity          Float32
    emission2_icc_total          Float32
    emission2_shell              Utf8     — NULL (always for gamma)
    parent_level_keV             Float64
    intermediate_level_keV       Float64
    final_level_keV              Float64
    pair_intensity               Float32  — raw from coincidences
    icc_correction_factor        Float32  — 1/((1+α₁)(1+α₂)) for γ-γ; 1/(1+α₂) for xray⊗γ
    pure_emission_joint_intensity Float32 — pair_intensity × icc_correction_factor

Usage::

    uv run python -m nucl_parquet.g4.summing_partners
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from nucl_parquet.g4.photon_evap_levels import Z_TO_SYMBOL

logger = logging.getLogger(__name__)

# Output column order — stable across all elements.
_OUTPUT_COLUMNS = [
    "Z",
    "A",
    "parent_state",
    "parent_decay_mode",
    "emission1_rad_type",
    "emission1_energy_keV",
    "emission1_intensity",
    "emission1_icc_total",
    "emission1_shell",
    "emission2_rad_type",
    "emission2_energy_keV",
    "emission2_intensity",
    "emission2_icc_total",
    "emission2_shell",
    "parent_level_keV",
    "intermediate_level_keV",
    "final_level_keV",
    "pair_intensity",
    "icc_correction_factor",
    "pure_emission_joint_intensity",
]


def _build_gamma_gamma(coinc: pl.DataFrame) -> pl.DataFrame:
    """Extract γ-γ pairs, canonicalize (E1 ≤ E2), compute ICC correction."""
    gg = coinc.filter(
        (pl.col("emission1_rad_type") == "gamma")
        & (pl.col("emission2_rad_type") == "gamma")
    )
    if gg.is_empty():
        return _empty_output()

    # Canonicalize: enforce emission1_energy ≤ emission2_energy.
    # Swap sides where γ1 > γ2 so each pair appears exactly once.
    needs_swap = pl.col("gamma1_energy_keV") > pl.col("gamma2_energy_keV")

    gg = gg.with_columns(
        # After swap: e1 is the lower-energy gamma, e2 the higher.
        pl.when(needs_swap)
        .then(pl.col("gamma2_energy_keV"))
        .otherwise(pl.col("gamma1_energy_keV"))
        .alias("emission1_energy_keV"),
        pl.when(needs_swap)
        .then(pl.col("gamma2_intensity"))
        .otherwise(pl.col("gamma1_intensity"))
        .alias("emission1_intensity"),
        pl.when(needs_swap)
        .then(pl.col("gamma2_icc_total"))
        .otherwise(pl.col("gamma1_icc_total"))
        .alias("emission1_icc_total"),
        pl.when(needs_swap)
        .then(pl.col("gamma1_energy_keV"))
        .otherwise(pl.col("gamma2_energy_keV"))
        .alias("emission2_energy_keV"),
        pl.when(needs_swap)
        .then(pl.col("gamma1_intensity"))
        .otherwise(pl.col("gamma2_intensity"))
        .alias("emission2_intensity"),
        pl.when(needs_swap)
        .then(pl.col("gamma1_icc_total"))
        .otherwise(pl.col("gamma2_icc_total"))
        .alias("emission2_icc_total"),
    )

    # Deduplicate after canonicalization — same (Z, A, state, E1, E2, parent_level)
    # should appear once. Include parent_state so ground-state and metastable
    # decay channels feeding the same cascade aren't collapsed.
    gg = gg.unique(
        subset=["Z", "A", "parent_state", "emission1_energy_keV", "emission2_energy_keV", "parent_level_keV"],
        keep="first",
    )

    # ICC correction: 1 / ((1 + α₁) × (1 + α₂)), NULL → 0.
    alpha1 = pl.col("emission1_icc_total").fill_null(0.0).cast(pl.Float64)
    alpha2 = pl.col("emission2_icc_total").fill_null(0.0).cast(pl.Float64)
    icc_corr = (1.0 / ((1.0 + alpha1) * (1.0 + alpha2))).cast(pl.Float32)

    return gg.select(
        pl.col("Z"),
        pl.col("A"),
        pl.col("parent_state").fill_null(""),
        pl.col("parent_decay_mode"),
        pl.lit("gamma").alias("emission1_rad_type"),
        pl.col("emission1_energy_keV"),
        pl.col("emission1_intensity"),
        pl.col("emission1_icc_total"),
        pl.lit(None, dtype=pl.Utf8).alias("emission1_shell"),
        pl.lit("gamma").alias("emission2_rad_type"),
        pl.col("emission2_energy_keV"),
        pl.col("emission2_intensity"),
        pl.col("emission2_icc_total"),
        pl.lit(None, dtype=pl.Utf8).alias("emission2_shell"),
        pl.col("parent_level_keV"),
        pl.col("intermediate_level_keV"),
        pl.col("final_level_keV"),
        pl.col("pair_intensity"),
        icc_corr.alias("icc_correction_factor"),
        (pl.col("pair_intensity").cast(pl.Float64) * icc_corr).cast(pl.Float32).alias("pure_emission_joint_intensity"),
    )


def _build_mixed(coinc: pl.DataFrame) -> pl.DataFrame:
    """Extract X-ray/Auger ⊗ γ pairs, compute ICC correction (gamma side only)."""
    mixed = coinc.filter(
        pl.col("emission1_rad_type").is_in(["xray", "auger"])
        & (pl.col("emission2_rad_type") == "gamma")
    )
    if mixed.is_empty():
        return _empty_output()

    # ICC correction: only on the gamma side (emission2). X-ray/Auger has no ICC.
    alpha2 = pl.col("gamma2_icc_total").fill_null(0.0).cast(pl.Float64)
    icc_corr = (1.0 / (1.0 + alpha2)).cast(pl.Float32)

    return mixed.select(
        pl.col("Z"),
        pl.col("A"),
        pl.col("parent_state").fill_null(""),
        pl.col("parent_decay_mode"),
        pl.col("emission1_rad_type"),
        pl.col("emission1_energy_keV"),
        pl.col("emission1_intensity"),
        pl.lit(None, dtype=pl.Float32).alias("emission1_icc_total"),
        pl.col("emission1_shell"),
        pl.col("emission2_rad_type"),
        pl.col("emission2_energy_keV"),
        pl.col("emission2_intensity"),
        pl.col("gamma2_icc_total").alias("emission2_icc_total"),
        pl.col("emission2_shell"),
        pl.col("parent_level_keV"),
        pl.col("intermediate_level_keV"),
        pl.col("final_level_keV"),
        pl.col("pair_intensity"),
        icc_corr.alias("icc_correction_factor"),
        (pl.col("pair_intensity").cast(pl.Float64) * icc_corr).cast(pl.Float32).alias("pure_emission_joint_intensity"),
    )


def _empty_output() -> pl.DataFrame:
    """Empty DataFrame with the full output schema."""
    return pl.DataFrame(
        schema={
            "Z": pl.Int64,
            "A": pl.Int64,
            "parent_state": pl.Utf8,
            "parent_decay_mode": pl.Utf8,
            "emission1_rad_type": pl.Utf8,
            "emission1_energy_keV": pl.Float64,
            "emission1_intensity": pl.Float32,
            "emission1_icc_total": pl.Float32,
            "emission1_shell": pl.Utf8,
            "emission2_rad_type": pl.Utf8,
            "emission2_energy_keV": pl.Float64,
            "emission2_intensity": pl.Float32,
            "emission2_icc_total": pl.Float32,
            "emission2_shell": pl.Utf8,
            "parent_level_keV": pl.Float64,
            "intermediate_level_keV": pl.Float64,
            "final_level_keV": pl.Float64,
            "pair_intensity": pl.Float32,
            "icc_correction_factor": pl.Float32,
            "pure_emission_joint_intensity": pl.Float32,
        }
    )


def build_for_element(coinc: pl.DataFrame) -> pl.DataFrame:
    """Build summing partner rows for a single element's coincidences.

    Parameters
    ----------
    coinc
        All rows from ``coincidences/{Symbol}.parquet`` for one element.

    Returns
    -------
    DataFrame in the output schema, sorted for stable output.
    """
    gg = _build_gamma_gamma(coinc)
    mixed = _build_mixed(coinc)

    if gg.is_empty() and mixed.is_empty():
        return _empty_output()

    frames = [f for f in [gg, mixed] if not f.is_empty()]
    result = pl.concat(frames, how="vertical_relaxed") if len(frames) > 1 else frames[0]

    return result.select(_OUTPUT_COLUMNS).sort(
        ["Z", "A", "parent_state", "emission1_rad_type", "emission1_energy_keV", "emission2_energy_keV"],
        nulls_last=True,
    )


def build(coincidences_dir: Path, out_dir: Path) -> list[Path]:
    """End-to-end: read coincidences, build summing partners, write per-element.

    Parameters
    ----------
    coincidences_dir
        Directory containing ``{Symbol}.parquet`` coincidence files (output of
        :mod:`nucl_parquet.g4.coincidences` + :mod:`nucl_parquet.g4.mixed_coincidences`).
    out_dir
        Directory to write ``{Symbol}.parquet`` summing partner files into.
    """
    if not coincidences_dir.exists():
        raise FileNotFoundError(f"coincidences directory not found at {coincidences_dir}")

    coinc_paths = sorted(coincidences_dir.glob("*.parquet"))
    if not coinc_paths:
        raise FileNotFoundError(f"No coincidence parquet files in {coincidences_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    symbol_to_z = {sym: z for z, sym in Z_TO_SYMBOL.items()}

    written: list[Path] = []
    total_rows = 0
    for p in coinc_paths:
        symbol = p.stem
        z = symbol_to_z.get(symbol)
        if z is None:
            logger.warning("skipping %s: no Z mapping for symbol %s", p.name, symbol)
            continue

        coinc = pl.read_parquet(p)
        result = build_for_element(coinc)

        if result.is_empty():
            logger.debug("[%s] Z=%d: no summing partners", symbol, z)
            continue

        out_path = out_dir / f"{symbol}.parquet"
        out_path.unlink(missing_ok=True)
        result.write_parquet(out_path, compression="zstd")
        written.append(out_path)
        total_rows += result.height
        logger.info("[%s] Z=%d: %d summing partners", symbol, z, result.height)

    logger.info(
        "Wrote %d per-element summing-partner files (%d rows total) to %s",
        len(written),
        total_rows,
        out_dir,
    )
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    repo_root = Path(__file__).resolve().parents[2]
    meta = repo_root / "data" / "meta" / "ensdf"
    parser.add_argument(
        "--coincidences-dir",
        type=Path,
        default=meta / "coincidences",
        help="Directory with coincidence parquet files (input).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=meta / "summing_partners",
        help="Directory to write summing partner parquet files into.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    build(args.coincidences_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
