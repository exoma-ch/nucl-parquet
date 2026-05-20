"""End-to-end build orchestrator for the v0.11 G4-derived nuclear-data pipeline.

Runs the six converters in dependency order, performs the
``radiation/`` ↔ ``radiation_atomic/`` union merge per
:doc:`docs/g4-xray-auger-design.md` (Integration contract), then asserts
the v0.11 sweep invariants on the rebuilt dataset.

Order matters:

1. ``ensdfstate``      — reads strata + AME2020 + IUPAC, writes
                          ``meta/ensdf/nuclides.parquet`` and
                          ``meta/ensdf/ground_states.parquet``.
2. ``photon_evap_levels`` — writes ``meta/ensdf/levels/{Symbol}.parquet``.
3. ``radioactive_decay``  — writes ``meta/decay.parquet`` and
                             ``meta/decay_detailed.parquet`` (the latter is
                             consumed by ``xray_auger`` below).
4. ``photon_evap_gammas`` — writes ``meta/ensdf/radiation/{Symbol}.parquet``
                             (rad_type='gamma' rows only). Reads
                             ``nuclides.parquet`` for state assignment, so
                             ensdfstate must run first.
5. ``coincidences``       — writes ``meta/ensdf/coincidences/{Symbol}.parquet``.
6. ``xray_auger``         — writes ``meta/ensdf/radiation_atomic/{Symbol}.parquet``
                             (rad_type='xray'/'auger' rows only). Reads
                             ``decay_detailed.parquet`` and gammas; depends on
                             #71 + #72 having run.
7. **Merge** ``radiation_atomic/`` rows into ``radiation/`` per the design
   memo's Integration contract — union by row, ``rad_type`` as discriminator,
   bonus columns NULL where source-specific, uniqueness on
   ``(Z, A, state, rad_type, energy_keV, rad_subtype)``. After merge,
   ``radiation_atomic/`` is removed (the canonical layout has only
   ``radiation/``).

Idempotent: re-running the merge step when ``radiation_atomic/`` is empty
or absent is a no-op.

Usage::

    uv run python -m nucl_parquet.g4.build_all
    uv run python -m nucl_parquet.g4.build_all --skip-converters --merge-only
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import polars as pl

from nucl_parquet.download import data_dir as _resolve_data_dir
from nucl_parquet.g4 import (
    coincidences,
    ensdfstate,
    mixed_coincidences,
    photon_evap_gammas,
    photon_evap_levels,
    radioactive_decay,
    summing_partners,
    xray_auger,
)

logger = logging.getLogger(__name__)


# Canonical column ordering for the post-merge radiation/ schema. Includes
# v0.10.x compat columns + bonus columns from both sources. NULL where the
# source-of-row didn't populate them.
_RADIATION_COLUMNS = [
    "Z",
    "A",
    "state",
    "rad_type",
    "energy_keV",
    "intensity_pct",
    "decay_mode",
    "rad_subtype",
    "dose_MeV_per_Bq_s",
    "parent_level_keV",
    # Bonus from photon_evap_gammas (#72)
    "daughter_level_keV",
    "multipolarity",
    "mixing_ratio",
    "icc_total",
    # Bonus from xray_auger (#74) — vacancy provenance
    "vacancy_shell",
]


def _conform_to_radiation_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Add NULL columns for any post-merge column the input lacks.

    Each source frame has different bonus columns; the union must include
    every column either source ships, with NULL where the row's source
    didn't populate it. Cast to a stable dtype so concat doesn't fight
    UInt vs Int promotions.
    """
    target_dtypes = {
        "Z": pl.Int64,
        "A": pl.Int64,
        "state": pl.String,
        "rad_type": pl.String,
        "energy_keV": pl.Float64,
        "intensity_pct": pl.Float64,
        "decay_mode": pl.String,
        "rad_subtype": pl.String,
        "dose_MeV_per_Bq_s": pl.Float64,
        "parent_level_keV": pl.Float64,
        "daughter_level_keV": pl.Float64,
        "multipolarity": pl.Int32,
        "mixing_ratio": pl.Float32,
        "icc_total": pl.Float32,
        "vacancy_shell": pl.String,
    }
    expressions: list[pl.Expr] = []
    for col in _RADIATION_COLUMNS:
        target = target_dtypes[col]
        if col in df.columns:
            expressions.append(pl.col(col).cast(target))
        else:
            expressions.append(pl.lit(None, dtype=target).alias(col))
    return df.select(expressions)


def merge_radiation_atomic(data_dir: Path) -> tuple[int, int]:
    """Union ``radiation_atomic/{Symbol}.parquet`` rows into the canonical
    ``radiation/{Symbol}.parquet`` files, then remove ``radiation_atomic/``.

    Returns ``(n_files_processed, n_rows_added)``. Idempotent: if
    ``radiation_atomic/`` is missing or empty, returns ``(0, 0)`` without
    touching anything.
    """
    rad_dir = data_dir / "meta" / "ensdf" / "radiation"
    atomic_dir = data_dir / "meta" / "ensdf" / "radiation_atomic"
    if not atomic_dir.exists():
        logger.info("radiation_atomic/ not present — merge is a no-op.")
        return 0, 0

    n_files = 0
    n_added = 0
    rad_dir.mkdir(parents=True, exist_ok=True)
    for atomic_path in sorted(atomic_dir.glob("*.parquet")):
        symbol_path = rad_dir / atomic_path.name
        atomic = _conform_to_radiation_schema(pl.read_parquet(atomic_path))
        if symbol_path.exists():
            existing = _conform_to_radiation_schema(pl.read_parquet(symbol_path))
            merged = pl.concat([existing, atomic], how="vertical_relaxed")
        else:
            merged = atomic

        # Uniqueness invariant on (Z, A, state, rad_type, energy_keV,
        # rad_subtype, parent_level_keV). parent_level_keV must be part of
        # the key — distinct cascade parents can legitimately emit the same
        # gamma energy via different transitions; the loose key collapses
        # real physics. The merge can still produce real duplicates if a
        # gamma row from #72 and an X-ray row from #74 happen to share
        # energy AND parent level — the rad_type discriminator handles that
        # case correctly.
        before = merged.height
        merged = merged.unique(
            subset=["Z", "A", "state", "rad_type", "energy_keV", "rad_subtype", "parent_level_keV"],
            keep="first",
        )
        n_dropped = before - merged.height
        if n_dropped:
            logger.warning("%s: dropped %d duplicate rows on uniqueness key", atomic_path.name, n_dropped)

        merged = merged.sort(["Z", "A", "state", "rad_type", "energy_keV"])
        merged.write_parquet(symbol_path, compression="zstd")
        n_files += 1
        n_added += atomic.height
        logger.info(
            "merged %s: +%d rows from radiation_atomic/ (total now %d)",
            atomic_path.name,
            atomic.height,
            merged.height,
        )

    # Wipe radiation_atomic/ — canonical layout has only radiation/.
    shutil.rmtree(atomic_dir)
    logger.info("removed radiation_atomic/ (merged into radiation/)")

    # Conform schema across ALL radiation/ files. Some elements have
    # radiation_atomic/ counterparts (merged above; full union schema), but
    # gamma-only elements (no x-ray/Auger emissions, e.g. very light or
    # superheavy nuclei without EC/IC pathways) retain the gammas-only
    # schema. Normalize everything to the union so downstream globs work.
    n_normalized = 0
    for rad_path in sorted(rad_dir.glob("*.parquet")):
        df = pl.read_parquet(rad_path)
        missing = [c for c in _RADIATION_COLUMNS if c not in df.columns]
        if not missing:
            continue
        conformed = _conform_to_radiation_schema(df).sort(["Z", "A", "state", "rad_type", "energy_keV"])
        conformed.write_parquet(rad_path, compression="zstd")
        n_normalized += 1
    if n_normalized:
        logger.info("normalized %d gamma-only radiation files to the union schema", n_normalized)

    return n_files, n_added


def _check_invariants(data_dir: Path) -> None:
    """Sweep test the rebuilt dataset for the v0.11 invariants.

    Raises ``AssertionError`` on any violation so the caller (CI / build
    pipeline) sees a clear failure with offending rows.
    """
    rad_glob = str(data_dir / "meta" / "ensdf" / "radiation" / "*.parquet")
    nuc_path = data_dir / "meta" / "ensdf" / "nuclides.parquet"
    decay_path = data_dir / "meta" / "decay.parquet"

    radiation = pl.read_parquet(rad_glob)
    nuclides = pl.read_parquet(nuc_path)
    decay = pl.read_parquet(decay_path)

    # 1. No corrupted parent_level_keV (the v0.10.x 2e+215 case).
    max_pl = radiation.select(pl.col("parent_level_keV").max()).item()
    assert max_pl is None or max_pl < 1e8, f"radiation parent_level_keV ceiling breach: {max_pl}"

    # 2. No IT-on-ground rows in decay.
    it_orphans = decay.filter((pl.col("state") == "") & (pl.col("decay_mode").str.to_uppercase() == "IT"))
    assert it_orphans.height == 0, (
        f"IT-on-ground regression: {it_orphans.height} rows; sample: {it_orphans.head(5).to_dicts()}"
    )

    # 3. No phantom isomers — every nuclides row must have a level (NULL OK
    # for ground; isomers must have level_keV).
    phantom = nuclides.filter((pl.col("state") != "") & pl.col("level_keV").is_null())
    assert phantom.height == 0, f"phantom isomer rows (state != '' AND level_keV IS NULL): {phantom.head(5).to_dicts()}"

    # 4. Bounded duplicate (Z, A, state, rad_type, energy_keV, rad_subtype,
    # parent_level_keV) in radiation. Some G4 PhotonEvaporation entries
    # genuinely list the same transition twice with different intensities
    # (alternate ENSDF evaluations merged upstream — Th-233 4786 keV level
    # is the canonical case with ~30 such transitions). These aren't bugs
    # of *our* pipeline; we accept up to 100 such upstream-tagged duplicates
    # and warn loudly so a future ENSDF refresh that introduces a real
    # nucl-parquet duplication still trips the bound.
    before = radiation.height
    deduped = radiation.unique(
        subset=["Z", "A", "state", "rad_type", "energy_keV", "rad_subtype", "parent_level_keV"],
        keep="first",
    )
    n_dup = before - deduped.height
    assert n_dup < 100, (
        f"radiation has {n_dup} duplicate "
        "(Z,A,state,rad_type,E,subtype,parent_level_keV) tuples — exceeds the "
        "100-row upstream-quirk budget; investigate."
    )
    if n_dup > 0:
        logger.warning(
            "radiation has %d duplicate transitions (alternate ENSDF evaluations); within budget but worth noting",
            n_dup,
        )

    # 5. intensity_pct >= 0 throughout.
    neg = radiation.filter(pl.col("intensity_pct") < 0)
    assert neg.height == 0, f"radiation intensity_pct < 0: {neg.head(5).to_dicts()}"

    # 6. decay.parquet half-lives must agree with nuclides catalog (#203).
    # G4 RadioactiveDecay carries per-transition half-lives that are sometimes
    # wrong (e.g. Sc-44 ground/metastable swapped). After the catalog override
    # in build_decay_table, they should match. Flag any >10% disagreement.
    decay = pl.read_parquet(data_dir / "meta" / "decay.parquet")
    decay_hl = (
        decay.select("Z", "A", "state", "half_life_s").unique(["Z", "A", "state"]).rename({"half_life_s": "dec_hl"})
    )
    nuc_hl = (
        nuclides.select("Z", "A", "state", "half_life_s")
        .rename({"half_life_s": "nuc_hl"})
        .with_columns(
            pl.col("Z").cast(pl.Int32),
            pl.col("A").cast(pl.Int32),
        )
    )
    joined = decay_hl.join(nuc_hl, on=["Z", "A", "state"], how="inner").filter(
        (pl.col("dec_hl").is_finite())
        & (pl.col("nuc_hl").is_finite())
        & (pl.col("dec_hl") > 0)
        & (pl.col("nuc_hl") > 0)
    )
    mismatches = joined.filter(((pl.col("dec_hl") - pl.col("nuc_hl")).abs() / pl.col("nuc_hl")) > 0.10)
    assert mismatches.height == 0, (
        f"decay.parquet half-lives disagree with nuclides catalog for {mismatches.height} "
        f"(Z,A,state) entries (>10%): {mismatches.head(5).to_dicts()}"
    )

    logger.info(
        "Sweep invariants pass: parent_level_keV, IT-on-ground, phantom isomer, dedup, non-negative intensity, half-life agreement"
    )


def build_all(data_dir: Path | None = None, *, skip_converters: bool = False, merge_only: bool = False) -> None:
    """Run the full v0.11 G4 build pipeline.

    Parameters
    ----------
    data_dir
        Repo data directory; defaults to the auto-resolved one.
    skip_converters
        Run only the merge + invariant checks. Useful for CI runs that
        consume already-built outputs.
    merge_only
        Alias for ``skip_converters``; clearer at the CLI.
    """
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)
    skip = skip_converters or merge_only

    if not skip:
        strata_dir = data_dir / "g4_raw" / "strata-nuclear"

        logger.info("[1/6] ensdfstate → nuclides + ground_states")
        ensdfstate.build(data_dir=data_dir)

        logger.info("[2/6] photon_evap_levels → levels/{Symbol}")
        photon_evap_levels.build(
            strata_dir / "photon_evap_levels.parquet",
            data_dir / "meta" / "ensdf" / "levels",
        )

        logger.info("[3/6] radioactive_decay → decay + decay_detailed")
        radioactive_decay.write_decay_parquets(
            strata_dir / "radioactive_decay.parquet",
            data_dir / "meta" / "ensdf" / "nuclides.parquet",
            data_dir / "meta",
        )

        logger.info("[4/6] photon_evap_gammas → radiation/{Symbol} (rad_type=gamma)")
        photon_evap_gammas.build(data_dir=data_dir)

        logger.info("[5/6] coincidences → coincidences/{Symbol} (γ-γ cascade pairs)")
        coincidences.build(
            strata_dir / "photon_evap_gammas.parquet",
            strata_dir / "photon_evap_levels.parquet",
            data_dir / "meta" / "ensdf" / "coincidences",
        )

        logger.info("[6/6] xray_auger → radiation_atomic/{Symbol} (rad_type=xray/auger)")
        decay_detailed = pl.read_parquet(data_dir / "meta" / "decay_detailed.parquet")
        gammas_df = pl.read_parquet(strata_dir / "photon_evap_gammas.parquet")
        levels_df = pl.read_parquet(strata_dir / "photon_evap_levels.parquet")
        nuclides_df = pl.read_parquet(data_dir / "meta" / "ensdf" / "nuclides.parquet")
        eadl_dir = data_dir / "meta" / "eadl"
        emissions = xray_auger.synthesize_xray_auger(decay_detailed, gammas_df, levels_df, nuclides_df, eadl_dir)
        xray_auger.write_radiation_atomic(emissions, nuclides_df, data_dir / "meta" / "ensdf" / "radiation_atomic")

    logger.info("[merge] radiation_atomic/ ⇨ radiation/ (per #74 design memo Integration contract)")
    n_files, n_added = merge_radiation_atomic(data_dir)
    logger.info("merged %d files, %d xray/auger rows added", n_files, n_added)

    # Augment coincidences/ with mixed-emission pairs (β/EC/annihilation × γ).
    # Runs *after* the radiation merge so EC X-ray + Auger lines exist in
    # radiation/{Symbol}.parquet for the join. Per issue #170 / ADR-0002.
    logger.info("[coinc-mix] augment coincidences/ with mixed-emission pairs (#170)")
    mixed_coincidences.build(
        data_dir / "meta" / "decay_detailed.parquet",
        data_dir / "meta" / "nudex_levels.parquet",
        data_dir / "meta" / "nudex_level_gammas.parquet",
        data_dir / "meta" / "ensdf" / "radiation",
        data_dir / "meta" / "ensdf" / "coincidences",
    )

    # Materialize summing_partners/ from augmented coincidences/ — ICC-corrected
    # pairs for HPGe TCS corrections. Runs after mixed_coincidences so both γ-γ
    # and X-ray/Auger ⊗ γ pairs are present. Per issue #177.
    logger.info("[summing] materialize summing_partners/ from coincidences/ (#177)")
    sp_paths = summing_partners.build(
        data_dir / "meta" / "ensdf" / "coincidences",
        data_dir / "meta" / "ensdf" / "summing_partners",
    )
    logger.info("wrote %d summing-partner files", len(sp_paths))

    # Materialize emissions/ — absolute per-decay photon emission intensities
    # (NuDat-equivalent parent-keyed table). Runs after radiation merge since it
    # reads the full radiation/ files. Per issue #196.
    logger.info("[emissions] materialize emissions/ — absolute per-decay intensities (#196)")
    em_paths = emissions.build(
        data_dir / "meta" / "decay_detailed.parquet",
        data_dir / "meta" / "decay.parquet",
        data_dir / "meta" / "ensdf" / "nuclides.parquet",
        data_dir / "meta" / "ensdf" / "radiation",
        data_dir / "meta" / "ensdf" / "emissions",
    )
    logger.info("wrote %d emission files", len(em_paths))

    logger.info("[invariants] sweeping the rebuilt dataset")
    _check_invariants(data_dir)
    logger.info("Build pipeline complete and invariants green.")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-converters",
        action="store_true",
        help="Skip the converter stage; only run merge + invariants on existing outputs.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Alias for --skip-converters with clearer intent at the CLI.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    build_all(
        data_dir=args.data_dir,
        skip_converters=args.skip_converters,
        merge_only=args.merge_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
