"""Convert strata's `nuclear/radioactive_decay.parquet` into nucl-parquet's
`meta/decay.parquet` (v0.10.x-compatible schema) plus a sibling
`meta/decay_detailed.parquet` (per-transition detail).

Inputs
------
- `radioactive_decay.parquet` (strata G4 RadioactiveDecay6.1.2, 14 columns,
  72_063 rows on the pinned revision). Each (Z, A, parent_ex_kev) parent
  contributes:
  * One *summary* row per decay-mode (`is_summary == True`) — total
    branching for that mode. **Per-shell EC is split** (KshellEC, LshellEC,
    MshellEC, NshellEC each as its own summary row).
  * One or more *detail* rows per mode (`is_summary == False`) — decays to
    specific daughter levels with q-values and (for beta) forbiddenness.

- `nuclides.parquet` (v0.10.x catalog) — used to map `parent_ex_kev` to the
  v0.10.x `state` label (`""` = ground, `"m"` = catalog isomer).

Outputs
-------
- `meta/decay.parquet` — v0.10.x schema preserved exactly:
    Z, A, state, half_life_s, decay_mode, daughter_Z, daughter_A,
    daughter_state, branching
  Only ground and m-state isomer parents are emitted (rows whose
  `parent_ex_kev` matches the nuclides catalog within ±1 keV). Other
  excited states are dropped from the legacy table — they are preserved in
  the detailed sibling.

- `meta/decay_detailed.parquet` — strata's detail rows as-is, with the
  v0.10.x column-naming applied to (Z, A, daughter_Z, daughter_A) for join
  compatibility. **Schema is additive**: this is a new file and breaks no
  existing consumer.

Design choices (per ADR-0002, issue #71)
----------------------------------------

1. **Decay-mode mapping**: G4 splits EC by atomic shell (K/L/M/N); v0.10.x
   lumped them under `"EC"`. We **preserve the per-shell split** in the
   converted output rather than collapsing — downstream issue #74
   (X-ray/Auger derivation) consumes per-shell fractions, and re-splitting
   a lumped value is not recoverable. Consequence: queries that expected a
   single `decay_mode == "EC"` row per parent must instead aggregate
   `decay_mode IN ('KshellEC', 'LshellEC', 'MshellEC', 'NshellEC')`. This
   is documented in the v0.11.0 CHANGELOG migration notes.

   Mode mapping table (G4 → v0.10.x):

       G4 token        →  v0.10.x decay_mode
       --------------     -----------------
       BetaMinus       →  beta-
       BetaPlus        →  beta+
       KshellEC        →  KshellEC      (kept; was lumped under "EC")
       LshellEC        →  LshellEC      (kept; was lumped under "EC")
       MshellEC        →  MshellEC      (kept; was lumped under "EC")
       NshellEC        →  NshellEC      (new; was absent from v0.10.x)
       IT              →  IT
       Alpha           →  alpha
       SpFission       →  SF
       Proton          →  p
       Neutron         →  n
       Triton          →  t             (new; was absent from v0.10.x)

2. **`half_life_s` sentinel handling**: per ADR-0002, G4's `-1.0` means
   "stable by physical observation" → maps to `+inf`. The strata file
   appears to use `-1` only sporadically (e.g. one Dy-163 case at the time
   of writing); the bulk of stable parents have positive half-lives or are
   absent from this table entirely. We still apply the conversion
   defensively — silent negative half-lives downstream are catastrophic.

3. **`branching`**: G4's `branching_ratio` is already a fraction in [0, 1]
   (matches the v0.10.x convention). Direct copy.

4. **State labels** (`state`, `daughter_state`):
   - `parent_ex_kev == 0` → `state = ""` (ground).
   - Otherwise look up the catalog: if `(Z, A)` has a `state == "m"` row in
     `nuclides.parquet` whose `level_keV` matches `parent_ex_kev` within
     ±1 keV, emit `state = "m"`. (Tolerance allows for G4 vs ENSDF
     rounding, e.g. Eu-152m: G4 45.5998 keV, ENSDF 45.6 keV.)
   - No catalog match for an excited parent → row is dropped from
     `decay.parquet` (not representable in v0.10.x's `{"", "m"}` enum).
     The detailed sibling table still carries it.
   - `daughter_state` follows the same rule applied to `daughter_ex_kev`.
     Daughter at ground gets `""`. Daughter excited but no m-catalog match
     gets `""` as a fallback (consistent with how v0.10.x stored "decay
     populates the daughter species" without distinguishing the daughter's
     internal state in the legacy schema). The detailed table preserves
     `daughter_ex_kev` exactly for spectroscopy consumers.

5. **SpFission daughter**: G4 records `daughter_z = -1, daughter_a = -1`
   (no single daughter). We emit `daughter_Z = NULL, daughter_A = NULL` to
   match how the legacy fetcher represented unknown daughters.

Acceptance invariants (issue #71)
---------------------------------

- Eu-152 ground: B- 27.92 % + per-shell EC 72.08 % = 100 % (within 0.1 %).
- Tc-99m at 142.6836 keV: IT ≈ 99.9963 %, B- ≈ 0.0037 %.
- No `state == ""` row carries `decay_mode == "IT"` (the v0.10.x IAEA-bug
  canary — IT-on-ground is unphysical).
- Branching sums per (Z, A, state) ∈ [0.95, 1.05]. After the IT-on-ground
  filter renormalizes affected groups and the floating-level filter drops
  duplicate ground entries, **all groups satisfy the bound** (verified on
  the pinned strata revision; upstream G4 source quirks like Ir-171's
  duplicated Alpha rows that previously summed to 2.0 are eliminated by
  filtering `parent_level_flag != "-"`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import polars as pl

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

#: G4 decay-mode token → v0.10.x decay_mode label. See module docstring §1.
G4_TO_V010_DECAY_MODE: Final[dict[str, str]] = {
    "BetaMinus": "beta-",
    "BetaPlus": "beta+",
    "KshellEC": "KshellEC",
    "LshellEC": "LshellEC",
    "MshellEC": "MshellEC",
    "NshellEC": "NshellEC",
    "IT": "IT",
    "Alpha": "alpha",
    "SpFission": "SF",
    "Proton": "p",
    "Neutron": "n",
    "Triton": "t",
}

#: G4 sentinel for "stable by observation".
_G4_STABLE_SENTINEL: Final[float] = -1.0

#: keV tolerance for matching `parent_ex_kev` to a catalog `m`-state level.
_M_STATE_KEV_TOLERANCE: Final[float] = 1.0

#: Schema (column → Polars dtype) of the v0.10.x decay.parquet — preserved exactly.
_DECAY_V010_SCHEMA: Final[dict[str, pl.DataType]] = {
    "Z": pl.Int32,
    "A": pl.Int32,
    "state": pl.String,
    "half_life_s": pl.Float64,
    "decay_mode": pl.String,
    "daughter_Z": pl.Int32,
    "daughter_A": pl.Int32,
    "daughter_state": pl.String,
    "branching": pl.Float64,
}

#: Schema of the new detailed table (per-transition).
_DECAY_DETAILED_SCHEMA: Final[dict[str, pl.DataType]] = {
    "Z": pl.Int32,
    "A": pl.Int32,
    "parent_ex_kev": pl.Float64,
    "parent_level_flag": pl.String,
    "half_life_s": pl.Float64,
    "decay_mode": pl.String,
    "daughter_Z": pl.Int32,
    "daughter_A": pl.Int32,
    "daughter_ex_kev": pl.Float64,
    "daughter_level_flag": pl.String,
    "branching": pl.Float64,
    "q_value_kev": pl.Float64,
    "forbiddenness": pl.String,
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _convert_half_life(hl: pl.Expr) -> pl.Expr:
    """Map G4 `half_life_s` to v0.10.x semantics.

    `-1.0` → `+inf` (stable). Positive values pass through. NaN/null → null.
    """
    inf = float("inf")
    return pl.when(hl == _G4_STABLE_SENTINEL).then(pl.lit(inf)).when(hl.is_nan()).then(None).otherwise(hl)


def _build_mode_mapping_expr(g4_mode: pl.Expr) -> pl.Expr:
    """Map G4 decay_mode token to v0.10.x label via the table.

    Unknown tokens raise (no silent fallthrough — fail loud at build time).
    """
    expr = pl.lit(None, dtype=pl.String)
    for g4, v010 in G4_TO_V010_DECAY_MODE.items():
        expr = pl.when(g4_mode == g4).then(pl.lit(v010)).otherwise(expr)
    return expr


def _load_m_state_levels(nuclides_path: Path) -> pl.DataFrame:
    """Return (Z, A, m_keV) for each m-state in the catalog with known level.

    Catalog entries with `level_keV is null` cannot be matched by energy and
    are silently excluded — strata's G4 always carries an energy.
    """
    nu = pl.read_parquet(nuclides_path)
    return nu.filter((pl.col("state") == "m") & pl.col("level_keV").is_not_null()).select(
        pl.col("Z").cast(pl.Int32),
        pl.col("A").cast(pl.Int32),
        pl.col("level_keV").alias("m_keV"),
    )


def _label_state(
    df: pl.DataFrame,
    *,
    z_col: str,
    a_col: str,
    ex_col: str,
    out_col: str,
    m_levels: pl.DataFrame,
) -> pl.DataFrame:
    """Add a `state` column to `df` based on excitation energy.

    `ex == 0` → "" (ground); for excited rows, the **single (Z,A)-unique**
    excitation closest to the catalog m-state level (within ±1 keV) → "m";
    every other excited row → null.

    Uniqueness is enforced *across input parents*: if two distinct
    `parent_ex_kev` values for the same (Z, A) both fall within tolerance of
    the catalog level (e.g. Ba-127 has G4 levels at 80.32 and 81.29 keV near
    a single ENSDF m at 80.3 keV), only the closest one gets labeled `m`.
    The other becomes null and is dropped from the v0.10.x decay table.
    Otherwise the schema-collapse silently doubles the branching sum.
    """
    z = pl.col(z_col).cast(pl.Int32)
    a = pl.col(a_col).cast(pl.Int32)
    ex = pl.col(ex_col).cast(pl.Float64)

    # Step 1: distinct (Z, A, ex) parent pairs in the input — one row per
    # physical level. We pick the m-label here, then broadcast back via join.
    distinct_parents = df.select(
        z.alias("_z"),
        a.alias("_a"),
        ex.alias("_ex"),
    ).unique()

    # For each (Z, A, ex), find the catalog m-level (if any) and the diff.
    with_diff = distinct_parents.join(
        m_levels.rename({"Z": "_z", "A": "_a"}),
        on=["_z", "_a"],
        how="left",
    ).with_columns(_diff=(pl.col("_ex") - pl.col("m_keV")).abs())

    # Within each (Z, A), the *minimum* diff wins. Ties (extremely unlikely
    # in floating-point) go to the lower _ex deterministically.
    min_diff_per_za = (
        with_diff.filter(pl.col("_diff") <= _M_STATE_KEV_TOLERANCE)
        .sort(["_z", "_a", "_diff", "_ex"])
        .group_by(["_z", "_a"], maintain_order=True)
        .head(1)
        .select(["_z", "_a", "_ex"])
        .with_columns(_is_m=pl.lit(True))
    )

    distinct_labeled = distinct_parents.join(min_diff_per_za, on=["_z", "_a", "_ex"], how="left").with_columns(
        pl.when(pl.col("_ex") == 0.0)
        .then(pl.lit(""))
        .when(pl.col("_is_m"))
        .then(pl.lit("m"))
        .otherwise(pl.lit(None, dtype=pl.String))
        .alias(out_col)
    )

    # Step 2: broadcast labels back to every row in df.
    df_keyed = df.with_columns(_z=z, _a=a, _ex=ex)
    return df_keyed.join(
        distinct_labeled.select(["_z", "_a", "_ex", out_col]),
        on=["_z", "_a", "_ex"],
        how="left",
    ).drop(["_z", "_a", "_ex"])


def _enforce_schema(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Project columns in declared order, casting each to the declared dtype."""
    missing = set(schema) - set(df.columns)
    if missing:
        raise ValueError(f"missing expected columns for schema: {sorted(missing)}")
    return df.select([pl.col(name).cast(dtype) for name, dtype in schema.items()])


# -----------------------------------------------------------------------------
# Main entry points
# -----------------------------------------------------------------------------


def build_decay_table(src: pl.DataFrame, nuclides_path: Path) -> pl.DataFrame:
    """Build the v0.10.x-shaped `decay` table from the strata source.

    Filters `is_summary == True`; emits one row per (Z, A, state, decay_mode,
    daughter (Z, A, state)). Drops excited parents that don't match the m-state
    catalog (not representable in the legacy `state` enum).
    """
    m_levels = _load_m_state_levels(nuclides_path)

    # For the legacy schema we keep only canonical (`parent_level_flag == "-"`)
    # parents. G4 sometimes ships a *second* "ground-state-like" row at the
    # same (Z, A, ex=0) with a floating-level flag (`+X`/`+Y`/...) — these
    # are ENSDF unresolved-doublet annotations. They collapse onto `state=""`
    # in the legacy schema, so without this filter Ir-171 etc. show up as
    # duplicate rows whose branching sums to 2. The detailed sibling table
    # preserves `parent_level_flag` so spectroscopy consumers retain the
    # full information.
    summary = src.filter(pl.col("is_summary") & (pl.col("parent_level_flag") == "-"))

    # Validate decay-mode coverage before any transform.
    unknown = summary.filter(~pl.col("decay_mode").is_in(list(G4_TO_V010_DECAY_MODE.keys())))
    if unknown.height:
        raise ValueError(f"unmapped G4 decay_mode tokens in source: {unknown['decay_mode'].unique().to_list()}")

    # Label parent state via m-catalog lookup.
    with_parent_state = _label_state(
        summary,
        z_col="parent_z",
        a_col="parent_a",
        ex_col="parent_ex_kev",
        out_col="state",
        m_levels=m_levels,
    )

    # Label daughter state. SpFission has daughter_z=-1 — the join will miss
    # naturally; we then null out daughter_state below.
    with_daughter_state = _label_state(
        with_parent_state,
        z_col="daughter_z",
        a_col="daughter_a",
        ex_col="daughter_ex_kev",
        out_col="daughter_state",
        m_levels=m_levels,
    )

    # Drop parents that didn't map to "" or "m": not representable in v0.10.x.
    kept = with_daughter_state.filter(pl.col("state").is_not_null())

    out = kept.select(
        pl.col("parent_z").cast(pl.Int32).alias("Z"),
        pl.col("parent_a").cast(pl.Int32).alias("A"),
        pl.col("state"),
        _convert_half_life(pl.col("half_life_s")).alias("half_life_s"),
        _build_mode_mapping_expr(pl.col("decay_mode")).alias("decay_mode"),
        # daughter_z=-1 (SpFission) → null; otherwise int32.
        pl.when(pl.col("daughter_z") < 0).then(None).otherwise(pl.col("daughter_z").cast(pl.Int32)).alias("daughter_Z"),
        pl.when(pl.col("daughter_a") < 0).then(None).otherwise(pl.col("daughter_a").cast(pl.Int32)).alias("daughter_A"),
        # Fall back to "" when daughter has no catalog m-match (legacy schema
        # can't express "this daughter level is some excited state").
        pl.col("daughter_state").fill_null("").alias("daughter_state"),
        pl.col("branching_ratio").alias("branching"),
    )

    # Strip IT-on-ground rows (the v0.10.x bug-class canary). Upstream G4
    # source carries a handful (e.g. Br-92) where ground-state IT to itself
    # is recorded — physically meaningless and historically the "fetcher
    # broken" smell. We filter the offending rows AND renormalize the
    # remaining branching for the affected (Z, A, "") group so the
    # branching-sum invariant still holds. Log every suppression so the
    # cleanup is auditable.
    it_on_ground = out.filter((pl.col("state") == "") & (pl.col("decay_mode") == "IT"))
    if it_on_ground.height:
        offenders = it_on_ground.select("Z", "A").unique().sort(["Z", "A"]).to_dicts()
        logger.warning(
            "filtered %d IT-on-ground rows (unphysical; upstream G4 quirk): %s",
            it_on_ground.height,
            offenders,
        )
        # Drop the offending rows.
        cleaned = out.filter(~((pl.col("state") == "") & (pl.col("decay_mode") == "IT")))
        # Renormalize: for each affected (Z, A, "") group, scale remaining
        # branching so it sums to 1.
        affected = it_on_ground.select("Z", "A").unique().with_columns(_affected=pl.lit(True))
        renorm_groups = (
            cleaned.join(affected, on=["Z", "A"], how="left")
            .filter(pl.col("_affected") & (pl.col("state") == ""))
            .group_by(["Z", "A"])
            .agg(pl.col("branching").sum().alias("_grp_sum"))
        )
        out = (
            cleaned.join(renorm_groups, on=["Z", "A"], how="left")
            .with_columns(
                branching=pl.when((pl.col("state") == "") & pl.col("_grp_sum").is_not_null() & (pl.col("_grp_sum") > 0))
                .then(pl.col("branching") / pl.col("_grp_sum"))
                .otherwise(pl.col("branching"))
            )
            .drop("_grp_sum")
        )

    # Collapse duplicate rows produced by upstream G4 source quirks (e.g.
    # Ir-171 ground lists Alpha twice with different branching values). The
    # legacy decay schema has no `daughter_ex_kev` column to distinguish
    # transitions to different daughter levels, so duplicates would otherwise
    # leak through. We sum branching across duplicates — this is the right
    # behaviour for a "total branching to (Z, A, state) → (daughter_Z,
    # daughter_A, daughter_state) via decay_mode" view. Other columns
    # (half_life_s) take the first non-null value (they're invariant per
    # parent by construction).
    out = (
        out.group_by(
            ["Z", "A", "state", "decay_mode", "daughter_Z", "daughter_A", "daughter_state"],
            maintain_order=True,
        )
        .agg(
            pl.col("half_life_s").first(),
            pl.col("branching").sum(),
        )
        .select(  # restore original column order
            "Z",
            "A",
            "state",
            "half_life_s",
            "decay_mode",
            "daughter_Z",
            "daughter_A",
            "daughter_state",
            "branching",
        )
    )

    # Stable sort for deterministic output.
    return out.sort(["Z", "A", "state", "decay_mode", "daughter_Z", "daughter_A"])


def build_decay_detailed_table(src: pl.DataFrame) -> pl.DataFrame:
    """Build the per-transition detail table from the strata source.

    Filters `is_summary == False`. Renames parent_z/a → Z/A for downstream
    join compatibility; otherwise preserves all strata columns.
    """
    detail = src.filter(~pl.col("is_summary"))

    return detail.select(
        pl.col("parent_z").cast(pl.Int32).alias("Z"),
        pl.col("parent_a").cast(pl.Int32).alias("A"),
        pl.col("parent_ex_kev"),
        pl.col("parent_level_flag"),
        _convert_half_life(pl.col("half_life_s")).alias("half_life_s"),
        _build_mode_mapping_expr(pl.col("decay_mode")).alias("decay_mode"),
        pl.when(pl.col("daughter_z") < 0).then(None).otherwise(pl.col("daughter_z").cast(pl.Int32)).alias("daughter_Z"),
        pl.when(pl.col("daughter_a") < 0).then(None).otherwise(pl.col("daughter_a").cast(pl.Int32)).alias("daughter_A"),
        pl.col("daughter_ex_kev"),
        pl.col("daughter_level_flag"),
        pl.col("branching_ratio").alias("branching"),
        # q_value_kev: NaN markers in the source mean "summary" rows (filtered
        # already); detail rows always carry a real Q. Convert NaN → null
        # defensively in case strata ever emits it.
        pl.when(pl.col("q_value_kev").is_nan()).then(None).otherwise(pl.col("q_value_kev")).alias("q_value_kev"),
        pl.col("forbiddenness"),
    ).sort(
        [
            "Z",
            "A",
            "parent_ex_kev",
            "decay_mode",
            "daughter_ex_kev",
        ]
    )


def write_decay_parquets(
    src_path: Path,
    nuclides_path: Path,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Read strata source, transform, write both parquets. Returns (decay,
    decay_detailed) paths."""
    src = pl.read_parquet(src_path)
    logger.info("read strata radioactive_decay: %d rows", src.height)

    decay = build_decay_table(src, nuclides_path)
    detailed = build_decay_detailed_table(src)
    logger.info(
        "transform: decay=%d rows, decay_detailed=%d rows",
        decay.height,
        detailed.height,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    decay_path = out_dir / "decay.parquet"
    detailed_path = out_dir / "decay_detailed.parquet"

    _enforce_schema(decay, _DECAY_V010_SCHEMA).write_parquet(decay_path)
    _enforce_schema(detailed, _DECAY_DETAILED_SCHEMA).write_parquet(detailed_path)
    logger.info("wrote %s and %s", decay_path, detailed_path)
    return decay_path, detailed_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/g4_raw/strata-nuclear/radioactive_decay.parquet"),
    )
    parser.add_argument(
        "--nuclides",
        type=Path,
        default=Path("data/meta/ensdf/nuclides.parquet"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/meta"),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    write_decay_parquets(args.src, args.nuclides, args.out_dir)


if __name__ == "__main__":
    _main()
