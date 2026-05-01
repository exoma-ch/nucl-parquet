"""Convert strata's ``photon_evap_gammas.parquet`` → per-element
``meta/ensdf/radiation/{Symbol}.parquet`` rows with ``rad_type='gamma'``.

Per epic #66 / issue #72 / ADR-0002. This is a pure schema/encoding transform
on strata's parsed Geant4 PhotonEvaporation6.1.2 gamma-transition data; we do
not reach into the G4 ASCII files ourselves.

Input schema (strata)
---------------------
::

    z UInt8, a UInt16, parent_level UInt16, daughter_level UInt16,
    gamma_energy_kev Float64, intensity Float32, multipolarity UInt16,
    mixing_ratio Float32, icc_total Float32

Output schema (per element ``meta/ensdf/radiation/{Symbol}.parquet``)
---------------------------------------------------------------------

The columns required for v0.10.x consumer compatibility (Rust crate, MCP,
DuckDB views) are preserved; bonus columns are appended as nullable.

    Z Int64, A Int64, state Utf8, rad_type Utf8 (constant 'gamma'),
    energy_keV Float64, intensity_pct Float64,
    decay_mode Utf8 (NULL — populated by #71), rad_subtype Utf8 (NULL),
    dose_MeV_per_Bq_s Float64 (NULL on this branch — v0.10.x dosimetry slot
    that the loader's GAMMA_LINES_SQL projects; PhotonEvaporation doesn't
    ship dose data, leave NULL),
    parent_level_keV Float64, daughter_level_keV Float64,
    multipolarity Int32, mixing_ratio Float32, icc_total Float32

Transform rules (ADR-0002 §"Transform spec")
--------------------------------------------

- ``energy_keV`` ← ``gamma_energy_kev`` (rename only).
- ``intensity_pct`` ← ``intensity`` (rename + cast to Float64; PhotonEvaporation
  emits intensities normalized to 100 per parent level — same semantics as the
  v0.10.x ``intensity_pct``, which is a per-parent emission probability).
- ``parent_level_keV`` ← LEFT JOIN strata's ``photon_evap_levels.parquet`` on
  (z, a, parent_level → level_idx), pulling ``excitation_kev``. Strata's data
  is consistent by construction (same revision SHA), so coverage is 100%.
  Join failures here would indicate upstream level/transition mismatch and
  raise.
- ``daughter_level_keV`` ← same lookup keyed on (z, a, daughter_level).
- ``state`` ← derived per (Z, A, parent_level_keV) by exact match against
  long-lived isomers (T½ > 1 ms ⇒ ``half_life_s > 1e-3`` or G4 stable
  sentinel ``-1``). Tolerance is **0.1 keV** (G4 source has exact level
  energies; the 0.5 keV hedge from v0.10.x's IAEA-fetcher era is obsolete).
  Rows whose parent level matches no long-lived isomer get ``state=""``.
- ``decay_mode`` is NULL on this branch — the radioactive_decay converter
  (#71) is responsible for stamping decay_mode on radiation rows for clinical
  parent-keyed line lists. Photon_evap rows are keyed by the *excited
  nucleus* (the level structure), not by the parent of any decay, so
  ``decay_mode`` cannot be inferred locally.
- ``rad_subtype`` is NULL (the v0.10.x slot for X-ray subshell labels and
  Auger transitions; not applicable to gamma transitions).

Bonus columns
-------------

- ``multipolarity`` (Int32 nullable, cast from UInt16) — encoded integer per
  G4 PhotonEvaporation README-LevelGammaData, e.g. 1=E0, 2=E1, 3=M1, 4=E2,
  304=M1+E2 mixing. Preserved as-is for #74 (X-ray + Auger synthesis) which
  needs the multipolarity to compute conversion-electron sub-shell fractions.
- ``mixing_ratio`` (Float32) — preserved.
- ``icc_total`` (Float32) — total internal conversion coefficient. Feeds the
  CE synthesis step (#74).
- ``daughter_level_keV`` (Float64) — destination level energy of the
  transition; useful for cascade analysis (#73 coincidences).

Sweep test
----------

A defensive ``parent_level_keV.max() < 1e8`` assert pins the absence of the
v0.10.x IAEA-fetcher 2e+215 corruption. G4 source values stay well under
50 MeV.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from nucl_parquet.download import data_dir as _resolve_data_dir

logger = logging.getLogger(__name__)

# --- Constants --------------------------------------------------------------

# State assignment threshold: G4 source has exact level energies (no
# IAEA-fetcher drift), so the v0.10.x 0.5 keV hedge is obsolete.
LEVEL_MATCH_TOLERANCE_KEV: float = 0.1

# ENSDF "long-lived" isomer threshold = T½ > 1 ms. PhotonEvaporation stores
# half-life directly in seconds.
LONG_LIVED_HALF_LIFE_S_THRESHOLD: float = 1e-3

# G4 stable sentinel in PhotonEvaporation/RadioactiveDecay (half-life column).
HALF_LIFE_STABLE_SENTINEL: float = -1.0

# Defensive cap; G4 source values stay well under 50 MeV.
PARENT_LEVEL_KEV_CEILING: float = 1.0e8

# Default relative input/output paths under the data root.
_STRATA_GAMMAS = Path("g4_raw/strata-nuclear/photon_evap_gammas.parquet")
_STRATA_LEVELS = Path("g4_raw/strata-nuclear/photon_evap_levels.parquet")
_ELEMENTS = Path("meta/elements.parquet")
_OUT_DIR = Path("meta/ensdf/radiation")


# --- Pure transforms (unit-tested) ------------------------------------------


def _is_long_lived_expr(col: str = "half_life_s") -> pl.Expr:
    """Polars expression: row qualifies as a long-lived isomer per ENSDF.

    Matches the convention used in :mod:`nucl_parquet.g4.ensdfstate` and
    :mod:`nucl_parquet.g4.photon_evap_levels` — *single* source-of-truth here
    so a future threshold change updates one place.

    Long-lived := T½ > 1 ms OR G4 stable sentinel (-1).
    """
    return (pl.col(col) > LONG_LIVED_HALF_LIFE_S_THRESHOLD) | (pl.col(col) == HALF_LIFE_STABLE_SENTINEL)


def build_isomer_state_map(levels: pl.DataFrame) -> pl.DataFrame:
    """Compute the (Z, A, level_keV) → state label table from strata levels.

    Mirrors the ENSDFSTATE convention from :func:`assign_state_labels`:
    ground state (lowest excitation per (Z, A)) gets ``state=""``;
    long-lived excited isomers get ``"m"``, ``"m2"``, … in ascending
    excitation order. Short-lived excited levels are excluded from the map
    (a parent gamma originating there will get ``state=""``, which is the
    correct v0.10.x semantics: ``state=""`` means "from the ground-band
    decay chain", not "from the literal ground state").

    Parameters
    ----------
    levels
        Strata's ``photon_evap_levels.parquet`` (raw schema: z, a, level_idx,
        excitation_kev, half_life_s, …).

    Returns
    -------
    DataFrame with columns ``Z`` (Int64), ``A`` (Int64), ``level_keV``
    (Float64), ``state`` (Utf8). Sorted by (Z, A, level_keV).
    """
    # Ground state per (Z, A): lowest excitation_kev. Strata always carries
    # level_idx=0 at excitation_kev=0 for ground; we use min() defensively.
    ground = (
        levels.group_by(["z", "a"])
        .agg(pl.col("excitation_kev").min().alias("level_keV"))
        .with_columns(pl.lit("").alias("state"))
    )

    # Long-lived isomers above ground.
    isomers = (
        levels.filter(_is_long_lived_expr("half_life_s"))
        .filter(pl.col("excitation_kev") > 0.0)
        .sort(["z", "a", "excitation_kev"])
        .with_columns(
            (pl.int_range(0, pl.len()).over(["z", "a"])).alias("_idx"),
        )
        .with_columns(
            pl.when(pl.col("_idx") == 0)
            .then(pl.lit("m"))
            .otherwise(pl.lit("m") + (pl.col("_idx") + 1).cast(pl.String))
            .alias("state"),
        )
        .select(
            pl.col("z"),
            pl.col("a"),
            pl.col("excitation_kev").alias("level_keV"),
            pl.col("state"),
        )
    )

    combined = pl.concat([ground, isomers], how="vertical_relaxed").sort(["z", "a", "level_keV"])
    return combined.rename({"z": "Z", "a": "A"}).with_columns(
        pl.col("Z").cast(pl.Int64),
        pl.col("A").cast(pl.Int64),
    )


def transform(
    gammas: pl.DataFrame,
    levels: pl.DataFrame,
) -> pl.DataFrame:
    """Apply ADR-0002 transforms to produce the full per-row radiation frame.

    Performs:
      1. LEFT JOIN gammas + levels on (z, a, parent_level/daughter_level →
         level_idx) to attach parent/daughter level energies.
      2. State assignment via :func:`assign_state`.
      3. Schema reshape to v0.10.x radiation/*.parquet column set + bonus
         columns from ADR-0002.

    Returns the unioned frame across all elements; partitioning to per-element
    files happens in :func:`write_per_element`.
    """
    # 1. Attach parent_level_keV.
    levels_proj = levels.select(
        pl.col("z"),
        pl.col("a"),
        pl.col("level_idx"),
        pl.col("excitation_kev"),
    )
    enriched = gammas.join(
        levels_proj.rename({"level_idx": "parent_level", "excitation_kev": "parent_level_keV"}),
        on=["z", "a", "parent_level"],
        how="left",
    ).join(
        levels_proj.rename({"level_idx": "daughter_level", "excitation_kev": "daughter_level_keV"}),
        on=["z", "a", "daughter_level"],
        how="left",
    )

    # Sanity: 100% join coverage expected (same revision SHA upstream).
    null_parents = enriched.filter(pl.col("parent_level_keV").is_null()).height
    if null_parents > 0:
        sample = (
            enriched.filter(pl.col("parent_level_keV").is_null()).head(5).select(["z", "a", "parent_level"]).to_dicts()
        )
        raise ValueError(
            f"{null_parents} gamma rows failed parent_level_keV join — "
            f"upstream level/transition mismatch. Sample: {sample}"
        )

    # 2. State assignment.
    state_map = build_isomer_state_map(levels)
    enriched_keyed = enriched.with_columns(
        pl.col("z").cast(pl.Int64).alias("Z"),
        pl.col("a").cast(pl.Int64).alias("A"),
    )
    with_state = _attach_state(enriched_keyed, state_map)

    # 3. Schema reshape. v0.10.x consumers project specific column names; we
    # keep their dtypes (Int64 for Z/A, Float64 for energies).
    return with_state.select(
        pl.col("Z").cast(pl.Int64),
        pl.col("A").cast(pl.Int64),
        pl.col("state").cast(pl.String),
        pl.lit("gamma").alias("rad_type").cast(pl.String),
        pl.col("gamma_energy_kev").cast(pl.Float64).alias("energy_keV"),
        pl.col("intensity").cast(pl.Float64).alias("intensity_pct"),
        pl.lit(None, dtype=pl.String).alias("decay_mode"),
        pl.lit(None, dtype=pl.String).alias("rad_subtype"),
        # v0.10.x loader (GAMMA_LINES_SQL) projects dose_MeV_per_Bq_s. G4
        # PhotonEvaporation doesn't ship dose data; leave NULL — consumers
        # already handle NULL via DuckDB's lenient projection.
        pl.lit(None, dtype=pl.Float64).alias("dose_MeV_per_Bq_s"),
        pl.col("parent_level_keV").cast(pl.Float64),
        pl.col("daughter_level_keV").cast(pl.Float64),
        pl.col("multipolarity").cast(pl.Int32),
        pl.col("mixing_ratio").cast(pl.Float32),
        pl.col("icc_total").cast(pl.Float32),
    ).sort(["Z", "A", "parent_level_keV", "energy_keV"])


def _attach_state(enriched: pl.DataFrame, state_map: pl.DataFrame) -> pl.DataFrame:
    """Inner helper: per-row state assignment by tolerance-match against
    ``state_map``. Avoids the brittle group_by/first dance — uses an
    ordinary LEFT JOIN with an explicit tolerance filter, then collapses
    duplicates by preferring the smaller-magnitude offset.

    Each gamma row may match 0 or 1 state_map row in practice (long-lived
    isomers per (Z, A) are spaced > 0.2 keV apart in G4). Defensively we
    keep the closest match if multiple candidates appear within tolerance.
    """
    candidates = enriched.join(state_map, on=["Z", "A"], how="left")

    # NULL level_keV → no isomer cataloged for this (Z, A) → state=''
    # |Δ| > tolerance → not this isomer → state=''
    # else → carry the matched state.
    tol = LEVEL_MATCH_TOLERANCE_KEV
    candidates = candidates.with_columns(
        (pl.col("parent_level_keV") - pl.col("level_keV")).abs().alias("_delta")
    ).with_columns(
        pl.when(pl.col("level_keV").is_not_null() & (pl.col("_delta") <= tol))
        .then(pl.col("state"))
        .otherwise(None)
        .alias("_state_match")
    )

    # Pick the closest match per source row. Source row identity is given
    # by all of (Z, A, parent_level, daughter_level, energy_keV) — those
    # are unique per upstream gamma row.
    deduped = (
        candidates.sort("_delta", nulls_last=True)
        .group_by(
            ["Z", "A", "parent_level", "daughter_level", "gamma_energy_kev"],
            maintain_order=True,
        )
        .first()
        .with_columns(pl.col("_state_match").fill_null("").alias("state"))
        .drop(["level_keV", "_delta", "_state_match"])
    )
    return deduped


# --- Build entry-point ------------------------------------------------------


def _load_inputs(
    data_dir: Path,
    *,
    gammas_path: Path | None = None,
    levels_path: Path | None = None,
    elements_path: Path | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    paths = {
        "gammas": gammas_path or (data_dir / _STRATA_GAMMAS),
        "levels": levels_path or (data_dir / _STRATA_LEVELS),
        "elements": elements_path or (data_dir / _ELEMENTS),
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "photon_evap_gammas converter requires the following inputs:\n  "
            + "\n  ".join(missing)
            + "\nRun `uv run python scripts/fetch_strata_nuclear.py` first."
        )
    return (
        pl.read_parquet(paths["gammas"]),
        pl.read_parquet(paths["levels"]),
        pl.read_parquet(paths["elements"]),
    )


def write_per_element(df: pl.DataFrame, out_dir: Path, elements: pl.DataFrame) -> list[Path]:
    """Partition by Z → ``meta/ensdf/radiation/{Symbol}.parquet``.

    Uses ``meta/elements.parquet`` (Z → IUPAC symbol) for the lookup —
    same source as :mod:`nucl_parquet.g4.ensdfstate`. The unlink dance
    handles case-insensitive filesystems (APFS): a stray lowercase
    ``n.parquet`` from an older build would otherwise silently capture
    fresh writes for nitrogen.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    z_to_symbol: dict[int, str] = dict(zip(elements["Z"].to_list(), elements["symbol"].to_list(), strict=True))
    written: list[Path] = []
    for z, group in sorted(df.group_by("Z"), key=lambda kg: kg[0][0]):
        z_int = int(z[0])
        symbol = z_to_symbol.get(z_int)
        if symbol is None:
            logger.warning(
                "Skipping Z=%d: no symbol mapping in meta/elements.parquet",
                z_int,
            )
            continue
        path = out_dir / f"{symbol}.parquet"
        path.unlink(missing_ok=True)
        group.sort(["A", "parent_level_keV", "energy_keV"]).write_parquet(path, compression="zstd")
        written.append(path)
    return written


def build(
    data_dir: Path | None = None,
    *,
    gammas_path: Path | None = None,
    levels_path: Path | None = None,
    elements_path: Path | None = None,
    out_dir: Path | None = None,
) -> list[Path]:
    """End-to-end: read strata inputs, transform, write per-element parquet
    files. Returns the list of paths written.

    Notes
    -----
    The output directory is **not** wiped before writing — only files we
    actually produce are unlinked. This is deliberate: the existing v0.10.x
    radiation/*.parquet layout includes element files we don't yet touch in
    this PR (Z without any photon_evap data — e.g. very heavy nuclei). They
    stay on disk under their old content until the radioactive_decay (#71)
    + X-ray/Auger (#74) converters land and replace the rest.
    """
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)
    gammas, levels, elements = _load_inputs(
        data_dir,
        gammas_path=gammas_path,
        levels_path=levels_path,
        elements_path=elements_path,
    )

    df = transform(gammas, levels)

    # Sweep test: pin the absence of the v0.10.x IAEA-fetcher 2e+215
    # corruption (would surface here as a genuinely-large-but-still-bounded
    # G4 value at most).
    max_parent = df.select(pl.col("parent_level_keV").max()).item()
    if max_parent is not None and max_parent >= PARENT_LEVEL_KEV_CEILING:
        raise ValueError(
            f"parent_level_keV ceiling breach: max={max_parent} keV ≥ "
            f"{PARENT_LEVEL_KEV_CEILING:g} keV. G4 source values stay under "
            "50 MeV; investigate upstream parser."
        )

    out = out_dir or (data_dir / _OUT_DIR)
    paths = write_per_element(df, out, elements)
    logger.info(
        "Wrote %d per-element radiation/gamma files to %s (%d gamma rows)",
        len(paths),
        out,
        df.height,
    )
    return paths


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    paths = build(data_dir=args.data_dir)
    print(f"wrote {len(paths)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
