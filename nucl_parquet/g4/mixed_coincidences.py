"""Mixed-emission coincidence pairs: extend γ-γ cascades with parent-side emissions.

Per issue #170 / ADR-0002 (additive schema). This module supersedes the v0.11
γ-γ-only build in :mod:`nucl_parquet.g4.coincidences` and writes the augmented
table to the same location (``data/meta/ensdf/coincidences/{Symbol}.parquet``).

Each output row is one **(emission₁, emission₂) pair** observed simultaneously
in a single parent-nucleus decay event:

- γ-γ cascade pairs (preserved verbatim from the v0.11 build, with the
  parent-decay-mode column back-filled).
- β⁻ / β⁺ endpoint ⊗ γ from the fed daughter level.
- 511 keV annihilation ⊗ γ (for β⁺ branches; intensity = 2 × β⁺ branching).
- EC K/L/M/N-shell X-ray + Auger ⊗ γ from the corresponding fed level.

The same emission-type-agnostic schema accommodates internal-conversion
electrons in a follow-up; not synthesized in this initial cut.

v0.10.x columns are preserved (``gamma_energy_keV``, ``coinc_energy_keV``,
``gamma1_*``, ``gamma2_*``, ``parent_level_keV``, ``intermediate_level_keV``,
``final_level_keV``, ``pair_intensity``, ``gamma{1,2}_icc_total``,
``dataset``) — populated for γ-γ rows and NULL for mixed-emission rows.

New columns (nullable, additive per ADR-0002):
    parent_state                Utf8     '' | 'm' | 'm2'  (from parent_ex_kev)
    parent_decay_mode           Utf8     'beta-' | 'beta+' | 'KshellEC' | ...
    daughter_ex_keV             Float64  level fed by parent decay (NULL for γ-γ before back-fill)
    emission1_rad_type          Utf8     'gamma' | 'beta' | 'annihilation_511' | 'xray' | 'auger'
    emission1_energy_keV        Float64
    emission1_intensity         Float32  per parent decay (β/annihilation) or normalized line intensity (X-ray)
    emission1_shell             Utf8     'K' | 'L' | 'M' | 'N' (X-ray/Auger only)
    emission2_rad_type          Utf8     ...
    emission2_energy_keV        Float64
    emission2_intensity         Float32
    emission2_shell             Utf8

Level-energy fuzzy-match tolerance: ``daughter_ex_kev`` (from decay_detailed)
↔ ``nudex_levels.energy_MeV * 1000`` matched within ``_LEVEL_KEV_TOL`` = 1.0 keV.
Mismatches are logged at INFO; they are typically high-lying levels where
G4 and NUDEX disagree on the precise energy.

Pair-intensity normalization
----------------------------

``pair_intensity`` is the relative coincidence probability per parent decay,
following the v0.11 convention. For each pair type:

- γ-γ: ``i₁ × i₂ / 100`` (preserved from v0.11).
- β ⊗ γ: ``channel_branching × γ_intensity``. The β is emitted in 100% of
  decays through this (decay_mode, fed_level) channel, so this is the
  per-parent-decay probability.
- annihilation ⊗ γ: ``2 × β⁺_channel_branching × γ_intensity`` (two 511 keV
  photons per β⁺ decay).
- X-ray/Auger ⊗ γ: ``(xray_intensity_pct / 100) × γ_intensity ×
  (channel_branching / total_shell_branching)``. The first factor is the
  per-parent-decay X-ray yield (summed across all daughter levels); the
  ratio re-conditions on this specific (shell EC, fed level) channel.

For absolute probabilities, normalize per ``(Z, A, parent_state)`` downstream.

Usage
-----
::

    uv run python -m nucl_parquet.g4.mixed_coincidences

Input files (same as ``coincidences.build()`` plus three meta tables):

- ``data/g4_raw/strata-nuclear/photon_evap_gammas.parquet`` (γ-γ self-join input)
- ``data/g4_raw/strata-nuclear/photon_evap_levels.parquet`` (level energy lookup)
- ``data/meta/decay_detailed.parquet`` (parent decay channel + daughter level fed)
- ``data/meta/nudex_levels.parquet`` (daughter level scheme, high fidelity)
- ``data/meta/nudex_level_gammas.parquet`` (daughter cascade γ)
- ``data/meta/ensdf/radiation/{Symbol}.parquet`` (EC X-ray + Auger lines)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

import polars as pl

from nucl_parquet.g4.photon_evap_levels import Z_TO_SYMBOL

logger = logging.getLogger(__name__)

# Fuzzy tolerance for matching decay_detailed.daughter_ex_kev (keV)
# ↔ nudex_levels.energy_MeV (* 1000 → keV). Co-60 and Y-86 match sub-keV;
# 1 keV covers high-lying-level disagreements between G4 PhotonEvap and NUDEX.
_LEVEL_KEV_TOL: Final[float] = 1.0

# Parent isomer-state fuzzy tolerance (parent_ex_kev → 'm'/'m2' label).
# Matches the convention in radioactive_decay.py.
_PARENT_STATE_KEV_TOL: Final[float] = 1.0

_EC_SHELL_MODES: Final[dict[str, str]] = {
    "KshellEC": "K",
    "LshellEC": "L",
    "MshellEC": "M",
    "NshellEC": "N",
}

# Drop pair_intensity below this — less than 1 in 10⁹ decays is below the
# detection threshold for any realistic spectroscopy/imaging application,
# and the tail (mostly low-yield Auger × deep-cascade γ) inflates row count
# by ~4× on heavy elements. γ-γ rows are exempted from this filter to keep
# the v0.11 output stable.
_DEFAULT_MIN_PAIR_INTENSITY: Final[float] = 1e-9

# Pre-filter parent-side X-ray/Auger lines below this intensity_pct *before*
# the cascade cross-product. Sub-1e-4 % per decay = sub-ppm per decay = below
# any realistic detection threshold; keeping them in the cross-product blows
# memory by 3–10× on heavy elements (Yb, W, Tl) without producing rows that
# survive ``min_pair_intensity = 1e-9``.
_MIN_PARENT_XRAY_INTENSITY_PCT: Final[float] = 1e-4


def _label_parent_state(parent_ex_kev: pl.Expr) -> pl.Expr:
    """Coarse parent-state label from parent_ex_kev.

    Matches the convention used by ``decay.parquet``: ground state is ``''``
    (parent_ex_kev ≈ 0); excited states are labelled ``'m'`` (any non-zero
    excitation). A more granular ``'m'`` vs ``'m2'`` mapping would require
    cross-referencing ``nuclides.parquet`` ordering; that lookup is handled
    by the post-build state-merge stage (out of scope for this module).
    """
    return (
        pl.when(parent_ex_kev.abs() < _PARENT_STATE_KEV_TOL)
        .then(pl.lit(""))
        .otherwise(pl.lit("m"))
        .alias("parent_state")
    )


def load_daughter_cascade(nudex_levels: pl.DataFrame, nudex_gammas: pl.DataFrame) -> pl.DataFrame:
    """Cascade γ keyed by (daughter Z, A, source_level_idx).

    Returns a frame with the γ that depopulate each level (one row per γ).
    No transitive closure here — the (parent ⊗ γ_immediate) pairs are
    emitted from this set, and (parent ⊗ γ_deeper) chains are obtained
    later by joining through the γ-γ table.
    """
    levels = nudex_levels.select(
        pl.col("Z").alias("daughter_Z"),
        pl.col("A").alias("daughter_A"),
        pl.col("level_idx"),
        (pl.col("energy_MeV") * 1000.0).alias("level_energy_keV"),
    )
    gammas = nudex_gammas.select(
        pl.col("Z").alias("daughter_Z"),
        pl.col("A").alias("daughter_A"),
        pl.col("source_level_idx").alias("level_idx"),
        pl.col("dest_level_idx"),
        (pl.col("gamma_energy_MeV") * 1000.0).alias("gamma_energy_keV"),
        pl.col("intensity").alias("gamma_intensity"),
    )
    return gammas.join(levels, on=["daughter_Z", "daughter_A", "level_idx"], how="left")


def match_fed_levels(decay_detailed: pl.DataFrame, nudex_levels: pl.DataFrame) -> pl.DataFrame:
    """Fuzzy-match decay_detailed.daughter_ex_kev → nudex_levels.level_idx.

    Per-(daughter_Z, daughter_A) we pick the single closest level within
    ``_LEVEL_KEV_TOL``. Rows with no match are logged and dropped.
    """
    levels = nudex_levels.select(
        pl.col("Z").alias("daughter_Z"),
        pl.col("A").alias("daughter_A"),
        pl.col("level_idx"),
        (pl.col("energy_MeV") * 1000.0).alias("level_energy_keV"),
    )
    joined = (
        decay_detailed.lazy()
        .join(levels.lazy(), on=["daughter_Z", "daughter_A"], how="left")
        .with_columns((pl.col("daughter_ex_kev") - pl.col("level_energy_keV")).abs().alias("_dE"))
        .filter(pl.col("_dE") <= _LEVEL_KEV_TOL)
        .sort(
            [
                "Z",
                "A",
                "parent_ex_kev",
                "decay_mode",
                "daughter_ex_kev",
                "_dE",
            ]
        )
        # Pick the closest match for each parent-channel row.
        .group_by(
            [
                "Z",
                "A",
                "parent_ex_kev",
                "decay_mode",
                "daughter_Z",
                "daughter_A",
                "daughter_ex_kev",
                "branching",
            ],
            maintain_order=True,
        )
        .agg(
            pl.col("level_idx").first(),
            pl.col("level_energy_keV").first(),
            pl.col("q_value_kev").first(),
        )
        .collect()
    )
    matched = joined.height
    unmatched = decay_detailed.height - matched
    if unmatched:
        logger.info(
            "fuzzy-match: %d/%d decay_detailed rows matched to nudex level (±%.1f keV); %d high-lying rows skipped",
            matched,
            decay_detailed.height,
            _LEVEL_KEV_TOL,
            unmatched,
        )
    return joined.rename({"Z": "Z_parent", "A": "A_parent"})


def synthesize_parent_emissions(
    matched: pl.DataFrame,
    radiation: pl.DataFrame | None,
) -> pl.DataFrame:
    """One row per (parent decay channel, parent-side emission line).

    Output schema (per row):
        Z_parent, A_parent, parent_ex_kev, parent_state, parent_decay_mode,
        daughter_Z, daughter_A, daughter_ex_kev, branching, fed_level_idx,
        emission1_rad_type, emission1_energy_keV, emission1_intensity_per_decay,
        emission1_shell.

    ``emission1_intensity_per_decay`` is the **per-parent-decay** probability
    of this specific (channel, line) — directly usable as the e1 factor in
    ``pair_intensity = e1 × e2``.
    """
    out_rows: list[pl.DataFrame] = []

    # β⁻ and β⁺ endpoints: one row per (decay_mode, fed_level), intensity = branching.
    beta = matched.filter(pl.col("decay_mode").is_in(["beta-", "beta+"])).select(
        "Z_parent",
        "A_parent",
        "parent_ex_kev",
        pl.col("decay_mode").alias("parent_decay_mode"),
        "daughter_Z",
        "daughter_A",
        "daughter_ex_kev",
        "branching",
        pl.col("level_idx").alias("fed_level_idx"),
        pl.lit("beta").alias("emission1_rad_type"),
        pl.col("q_value_kev").alias("emission1_energy_keV"),
        pl.col("branching").cast(pl.Float32).alias("emission1_intensity_per_decay"),
        pl.lit(None, dtype=pl.Utf8).alias("emission1_shell"),
    )
    out_rows.append(beta)

    # β⁺ annihilation: 2× 511 keV per β⁺ decay through this channel.
    annih = matched.filter(pl.col("decay_mode") == "beta+").select(
        "Z_parent",
        "A_parent",
        "parent_ex_kev",
        pl.lit("beta+").alias("parent_decay_mode"),
        "daughter_Z",
        "daughter_A",
        "daughter_ex_kev",
        "branching",
        pl.col("level_idx").alias("fed_level_idx"),
        pl.lit("annihilation_511").alias("emission1_rad_type"),
        pl.lit(511.0).alias("emission1_energy_keV"),
        (pl.col("branching") * 2.0).cast(pl.Float32).alias("emission1_intensity_per_decay"),
        pl.lit(None, dtype=pl.Utf8).alias("emission1_shell"),
    )
    out_rows.append(annih)

    # K/L/M/N-shell EC X-rays + Auger:
    # For each (Z_parent, A_parent, parent_state, shell) channel, gather the
    # X-ray/Auger lines (decay_mode='EC' in radiation; subtype-prefix selects
    # the vacancy shell). Multiply through by channel branching / total shell
    # branching to re-condition the per-parent-decay X-ray yield onto this
    # specific (shell-EC, fed-level) channel.
    if radiation is not None and radiation.height:
        # Restrict radiation to EC-driven atomic emissions, and drop sub-ppm
        # lines before the cross-product to keep memory bounded on heavy
        # elements. See _MIN_PARENT_XRAY_INTENSITY_PCT for rationale.
        rad_ec = radiation.filter(
            (pl.col("rad_type").is_in(["xray", "auger"]))
            & (pl.col("decay_mode") == "EC")
            & (pl.col("intensity_pct") > _MIN_PARENT_XRAY_INTENSITY_PCT)
        ).with_columns(
            pl.when(pl.col("rad_subtype").str.starts_with("K"))
            .then(pl.lit("K"))
            .when(pl.col("rad_subtype").str.starts_with("L"))
            .then(pl.lit("L"))
            .when(pl.col("rad_subtype").str.starts_with("M"))
            .then(pl.lit("M"))
            .when(pl.col("rad_subtype").str.starts_with("N"))
            .then(pl.lit("N"))
            .otherwise(pl.lit(None))
            .alias("emission1_shell"),
        )

        # Total branching per (parent Z, A, parent_ex_kev, shell-EC mode) — used to
        # convert per-parent-decay X-ray yield to per-channel yield.
        shell_totals = matched.with_columns(
            pl.col("decay_mode").replace_strict(_EC_SHELL_MODES, default=None).alias("emission1_shell")
        ).filter(pl.col("emission1_shell").is_not_null())
        totals = shell_totals.group_by(["Z_parent", "A_parent", "parent_ex_kev", "emission1_shell"]).agg(
            pl.col("branching").sum().alias("total_shell_branching")
        )

        # Join: for every (channel × matching X-ray/Auger line × matched-state radiation row).
        channels = matched.with_columns(
            pl.col("decay_mode").replace_strict(_EC_SHELL_MODES, default=None).alias("emission1_shell")
        ).filter(pl.col("emission1_shell").is_not_null())

        # radiation rows are keyed by (Z, A, state); parent_ex_kev=0 ↔ state='' in our model.
        # Coarse-map parent state for the join.
        channels_with_state = channels.with_columns(_label_parent_state(pl.col("parent_ex_kev")))
        rad_keyed = rad_ec.select(
            pl.col("Z").alias("Z_parent"),
            pl.col("A").alias("A_parent"),
            pl.col("state").alias("parent_state"),
            pl.col("rad_type").alias("emission1_rad_type"),
            pl.col("energy_keV").alias("emission1_energy_keV"),
            (pl.col("intensity_pct") / 100.0).alias("xray_per_parent_decay"),
            "emission1_shell",
        )
        ec_joined = (
            channels_with_state.join(
                totals,
                on=["Z_parent", "A_parent", "parent_ex_kev", "emission1_shell"],
                how="left",
            )
            .join(
                rad_keyed,
                on=["Z_parent", "A_parent", "parent_state", "emission1_shell"],
                how="inner",
            )
            .with_columns(
                (pl.col("xray_per_parent_decay") * pl.col("branching") / pl.col("total_shell_branching"))
                .cast(pl.Float32)
                .alias("emission1_intensity_per_decay")
            )
            .select(
                "Z_parent",
                "A_parent",
                "parent_ex_kev",
                pl.col("decay_mode").alias("parent_decay_mode"),
                "daughter_Z",
                "daughter_A",
                "daughter_ex_kev",
                "branching",
                pl.col("level_idx").alias("fed_level_idx"),
                "emission1_rad_type",
                "emission1_energy_keV",
                "emission1_intensity_per_decay",
                "emission1_shell",
            )
        )
        out_rows.append(ec_joined)

    return pl.concat(out_rows, how="vertical_relaxed") if out_rows else _empty_parent_emissions()


def _empty_parent_emissions() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "Z_parent": pl.Int32,
            "A_parent": pl.Int32,
            "parent_ex_kev": pl.Float64,
            "parent_decay_mode": pl.Utf8,
            "daughter_Z": pl.Int32,
            "daughter_A": pl.Int32,
            "daughter_ex_kev": pl.Float64,
            "branching": pl.Float64,
            "fed_level_idx": pl.Int32,
            "emission1_rad_type": pl.Utf8,
            "emission1_energy_keV": pl.Float64,
            "emission1_intensity_per_decay": pl.Float32,
            "emission1_shell": pl.Utf8,
        }
    )


def build_mixed_pairs(parent_emissions: pl.DataFrame, cascade: pl.DataFrame) -> pl.DataFrame:
    """(parent emission × daughter γ from fed level) cross-product.

    Output schema is the v0.10.x ``coincidences`` columns (mostly NULL for
    mixed rows) plus the new emission-agnostic columns.
    """
    if parent_emissions.height == 0:
        return _empty_mixed_pairs()

    pairs = (
        parent_emissions.join(
            cascade.rename({"level_idx": "fed_level_idx"}),
            on=["daughter_Z", "daughter_A", "fed_level_idx"],
            how="inner",
        )
        .with_columns(
            (pl.col("emission1_intensity_per_decay").cast(pl.Float64) * pl.col("gamma_intensity").cast(pl.Float64))
            .cast(pl.Float32)
            .alias("pair_intensity"),
            _label_parent_state(pl.col("parent_ex_kev")),
        )
        .select(
            # Output Z, A reflect the daughter (where the γ is emitted) — matches
            # the v0.11 coincidences convention. parent_Z/A live in metadata cols.
            pl.col("daughter_Z").cast(pl.Int64).alias("Z"),
            pl.col("daughter_A").cast(pl.Int64).alias("A"),
            pl.lit(1, dtype=pl.Int64).alias("dataset"),
            # v0.10.x compat columns — NULL for non-γ-γ pairs.
            pl.lit(None, dtype=pl.Float64).alias("gamma_energy_keV"),
            pl.lit(None, dtype=pl.Float64).alias("coinc_energy_keV"),
            pl.lit(None, dtype=pl.Float64).alias("gamma1_energy_keV"),
            pl.lit(None, dtype=pl.Float32).alias("gamma1_intensity"),
            pl.lit(None, dtype=pl.Float64).alias("gamma2_energy_keV"),
            pl.lit(None, dtype=pl.Float32).alias("gamma2_intensity"),
            pl.col("daughter_ex_kev").alias("parent_level_keV"),
            pl.col("level_energy_keV").alias("intermediate_level_keV"),
            pl.lit(None, dtype=pl.Float64).alias("final_level_keV"),
            pl.col("pair_intensity"),
            pl.lit(None, dtype=pl.Float32).alias("gamma1_icc_total"),
            pl.lit(None, dtype=pl.Float32).alias("gamma2_icc_total"),
            # New emission-agnostic columns.
            "parent_state",
            "parent_decay_mode",
            pl.col("daughter_ex_kev").alias("daughter_ex_keV"),
            "emission1_rad_type",
            "emission1_energy_keV",
            pl.col("emission1_intensity_per_decay").alias("emission1_intensity"),
            "emission1_shell",
            pl.lit("gamma").alias("emission2_rad_type"),
            pl.col("gamma_energy_keV").alias("emission2_energy_keV"),
            pl.col("gamma_intensity").cast(pl.Float32).alias("emission2_intensity"),
            pl.lit(None, dtype=pl.Utf8).alias("emission2_shell"),
        )
    )
    return pairs


def _empty_mixed_pairs() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "Z": pl.Int64,
            "A": pl.Int64,
            "dataset": pl.Int64,
            "gamma_energy_keV": pl.Float64,
            "coinc_energy_keV": pl.Float64,
            "gamma1_energy_keV": pl.Float64,
            "gamma1_intensity": pl.Float32,
            "gamma2_energy_keV": pl.Float64,
            "gamma2_intensity": pl.Float32,
            "parent_level_keV": pl.Float64,
            "intermediate_level_keV": pl.Float64,
            "final_level_keV": pl.Float64,
            "pair_intensity": pl.Float32,
            "gamma1_icc_total": pl.Float32,
            "gamma2_icc_total": pl.Float32,
            "parent_state": pl.Utf8,
            "parent_decay_mode": pl.Utf8,
            "daughter_ex_keV": pl.Float64,
            "emission1_rad_type": pl.Utf8,
            "emission1_energy_keV": pl.Float64,
            "emission1_intensity": pl.Float32,
            "emission1_shell": pl.Utf8,
            "emission2_rad_type": pl.Utf8,
            "emission2_energy_keV": pl.Float64,
            "emission2_intensity": pl.Float32,
            "emission2_shell": pl.Utf8,
        }
    )


def annotate_gamma_gamma(gg_pairs: pl.DataFrame, parent_channels: pl.DataFrame) -> pl.DataFrame:
    """Augment γ-γ pairs with new emission-agnostic columns + parent decay mode.

    For each γ-γ row, find the (parent decay mode, fed level) channel that
    populates the γ₁ parent level. The cascade head matches by level energy
    ↔ daughter_ex_kev (fuzzy, ±1 keV).
    """
    # Build a (daughter Z, A, level_keV) → parent_decay_mode mapping.
    # If multiple parent channels feed the same level, pick the highest-branching.
    parent_lookup = (
        parent_channels.lazy()
        .with_columns(_label_parent_state(pl.col("parent_ex_kev")))
        .group_by(
            ["daughter_Z", "daughter_A", "level_energy_keV", "parent_state"],
            maintain_order=True,
        )
        .agg(
            pl.col("decay_mode").sort_by("branching", descending=True).first().alias("parent_decay_mode"),
            pl.col("daughter_ex_kev").first(),
        )
        .collect()
    )

    # Fuzzy-match parent_level_keV (gg, from photon_evap excitation_kev) to
    # level_energy_keV (parent_lookup, from nudex energy_MeV*1000). They agree
    # sub-keV on canonical cascades but can diverge ~0.1 keV for high-lying
    # levels. Tag each gg row with a row_idx, cross-join within (Z, A), keep
    # the closest match per gg row, then left-join back so unmatched rows
    # survive with parent_decay_mode = NULL.
    pl_keyed = parent_lookup.select(
        pl.col("daughter_Z").alias("Z"),
        pl.col("daughter_A").alias("A"),
        "level_energy_keV",
        "parent_state",
        "parent_decay_mode",
        pl.col("daughter_ex_kev").alias("daughter_ex_keV"),
    )

    gg_idx = gg_pairs.with_row_index(name="_row_idx")
    matches = (
        gg_idx.lazy()
        .select("_row_idx", "Z", "A", "parent_level_keV")
        .join(pl_keyed.lazy(), on=["Z", "A"], how="inner")
        .with_columns((pl.col("parent_level_keV") - pl.col("level_energy_keV")).abs().alias("_dE"))
        .filter(pl.col("_dE") <= _LEVEL_KEV_TOL)
        .sort("_row_idx", "_dE")
        .group_by("_row_idx", maintain_order=True)
        .agg(
            pl.col("parent_state").first(),
            pl.col("parent_decay_mode").first(),
            pl.col("daughter_ex_keV").first(),
        )
        .collect()
    )

    annotated = (
        gg_idx.join(matches, on="_row_idx", how="left")
        .drop("_row_idx")
        .with_columns(
            pl.col("parent_state").fill_null(""),
            pl.lit("gamma").alias("emission1_rad_type"),
            pl.col("gamma1_energy_keV").alias("emission1_energy_keV"),
            pl.col("gamma1_intensity").alias("emission1_intensity"),
            pl.lit(None, dtype=pl.Utf8).alias("emission1_shell"),
            pl.lit("gamma").alias("emission2_rad_type"),
            pl.col("gamma2_energy_keV").alias("emission2_energy_keV"),
            pl.col("gamma2_intensity").alias("emission2_intensity"),
            pl.lit(None, dtype=pl.Utf8).alias("emission2_shell"),
        )
    )
    return annotated


def build_table(
    gg: pl.DataFrame,
    decay_detailed: pl.DataFrame,
    nudex_levels: pl.DataFrame,
    nudex_gammas: pl.DataFrame,
    radiation_dir: Path | None,
    min_pair_intensity: float = _DEFAULT_MIN_PAIR_INTENSITY,
) -> pl.DataFrame:
    """Build the full augmented coincidences table (all elements, all pairs).

    Convenience wrapper around :func:`build_for_daughter_z` that processes
    every daughter Z found in ``gg`` and concats the results. The per-element
    streaming entry point (used by :func:`build`) is preferred for production
    builds — it caps memory at one element's slice rather than materializing
    the global cross-product.
    """
    daughter_zs = sorted(set(gg["Z"].unique().to_list()))
    frames: list[pl.DataFrame] = []
    for z in daughter_zs:
        gg_z = gg.filter(pl.col("Z") == z)
        frame = build_for_daughter_z(
            int(z),
            gg_z,
            decay_detailed,
            nudex_levels,
            nudex_gammas,
            radiation_dir,
            min_pair_intensity,
        )
        if frame.height:
            frames.append(frame)
    if not frames:
        return _empty_mixed_pairs()
    return pl.concat(frames, how="vertical_relaxed")


def _load_radiation_slice(radiation_dir: Path | None, parent_zs: list[int]) -> pl.DataFrame | None:
    """Load just the per-element radiation files for the requested parent Zs."""
    if radiation_dir is None or not radiation_dir.exists() or not parent_zs:
        return None
    frames: list[pl.DataFrame] = []
    for z in parent_zs:
        symbol = Z_TO_SYMBOL.get(int(z))
        if symbol is None:
            continue
        p = radiation_dir / f"{symbol}.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p))
    return pl.concat(frames, how="vertical_relaxed") if frames else None


def build_for_daughter_z(
    daughter_z: int,
    gg_z: pl.DataFrame,
    decay_detailed: pl.DataFrame,
    nudex_levels: pl.DataFrame,
    nudex_gammas: pl.DataFrame,
    radiation_dir: Path | None,
    min_pair_intensity: float = _DEFAULT_MIN_PAIR_INTENSITY,
) -> pl.DataFrame:
    """Build the augmented coincidence rows for a single daughter Z.

    Memory peak is bounded by one element's slice rather than the global
    cross-product. ``decay_detailed``, ``nudex_levels``, ``nudex_gammas`` are
    accepted whole (they are small, ~10–50 MB combined) — the function slices
    them internally.
    """
    dd_z = decay_detailed.filter(pl.col("daughter_Z") == daughter_z)
    nl_z = nudex_levels.filter(pl.col("Z") == daughter_z)
    ng_z = nudex_gammas.filter(pl.col("Z") == daughter_z)

    if dd_z.is_empty() and gg_z.is_empty():
        return _empty_mixed_pairs()

    matched = match_fed_levels(dd_z, nl_z)

    parent_zs = matched["Z_parent"].unique().to_list() if not matched.is_empty() else []
    radiation = _load_radiation_slice(radiation_dir, parent_zs)

    parent_emissions = synthesize_parent_emissions(matched, radiation)
    cascade = load_daughter_cascade(nl_z, ng_z)

    mixed = build_mixed_pairs(parent_emissions, cascade)
    if min_pair_intensity > 0 and mixed.height:
        mixed = mixed.filter(pl.col("pair_intensity") > min_pair_intensity)

    gg_annotated = annotate_gamma_gamma(gg_z, matched)

    # Force identical column order before concat.
    final_cols = mixed.columns
    gg_annotated = gg_annotated.select(final_cols)
    return pl.concat([gg_annotated, mixed], how="vertical_relaxed").sort(
        ["Z", "A", "parent_decay_mode", "parent_level_keV", "emission1_energy_keV", "emission2_energy_keV"],
        nulls_last=True,
    )


def write_per_element(df: pl.DataFrame, out_dir: Path) -> list[Path]:
    """Partition by Z → write ``meta/ensdf/coincidences/{Symbol}.parquet``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for z, group in sorted(df.group_by("Z"), key=lambda kg: kg[0][0]):
        z_int = int(z[0])
        symbol = Z_TO_SYMBOL.get(z_int)
        if symbol is None:
            logger.warning("Skipping Z=%d: no symbol mapping", z_int)
            continue
        path = out_dir / f"{symbol}.parquet"
        path.unlink(missing_ok=True)
        group.sort(
            ["A", "parent_decay_mode", "parent_level_keV", "emission1_energy_keV", "emission2_energy_keV"],
            nulls_last=True,
        ).write_parquet(path)
        written.append(path)
    return written


def build(
    decay_detailed_src: Path,
    nudex_levels_src: Path,
    nudex_gammas_src: Path,
    radiation_dir: Path,
    coincidences_dir: Path,
    min_pair_intensity: float = _DEFAULT_MIN_PAIR_INTENSITY,
) -> list[Path]:
    """End-to-end: stream per-element and rewrite each ``{Symbol}.parquet`` in place.

    Reads ``coincidences_dir`` for the existing γ-γ pairs (v0.11 output of
    :mod:`nucl_parquet.g4.coincidences`), augments + appends mixed-emission
    rows, and writes back to the same directory. Processes one element at a
    time so peak memory is bounded by a single daughter Z's cross-product
    rather than the global one.
    """
    for label, p in [
        ("decay_detailed", decay_detailed_src),
        ("nudex_levels", nudex_levels_src),
        ("nudex_level_gammas", nudex_gammas_src),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} input not found at {p}")

    decay_detailed = pl.read_parquet(decay_detailed_src)
    nudex_levels = pl.read_parquet(nudex_levels_src)
    nudex_gammas = pl.read_parquet(nudex_gammas_src)

    symbol_to_z = {sym: z for z, sym in Z_TO_SYMBOL.items()}
    gg_paths = sorted(coincidences_dir.glob("*.parquet"))
    if not gg_paths:
        raise FileNotFoundError(
            f"No γ-γ coincidence parquet files found in {coincidences_dir} — "
            "run `python -m nucl_parquet.g4.coincidences` first."
        )

    import gc

    written: list[Path] = []
    total_rows = 0
    for p in gg_paths:
        symbol = p.stem
        z = symbol_to_z.get(symbol)
        if z is None:
            logger.warning("skipping %s: no Z mapping for symbol %s", p.name, symbol)
            continue
        gg_z = pl.read_parquet(p)
        # Idempotency: if this file was already augmented by a previous mixed
        # build, strip the mixed-emission rows so we don't compound them on
        # re-run. v0.11 γ-γ-only files (which lack the emission1_rad_type
        # column) read through unchanged.
        if "emission1_rad_type" in gg_z.columns:
            gg_z = gg_z.filter(
                pl.col("emission1_rad_type").is_null()
                | ((pl.col("emission1_rad_type") == "gamma") & (pl.col("emission2_rad_type") == "gamma"))
            ).select(
                [
                    c
                    for c in gg_z.columns
                    if c
                    not in {
                        "parent_state",
                        "parent_decay_mode",
                        "daughter_ex_keV",
                        "emission1_rad_type",
                        "emission1_energy_keV",
                        "emission1_intensity",
                        "emission1_shell",
                        "emission2_rad_type",
                        "emission2_energy_keV",
                        "emission2_intensity",
                        "emission2_shell",
                    }
                ]
            )
        frame = build_for_daughter_z(
            z,
            gg_z,
            decay_detailed,
            nudex_levels,
            nudex_gammas,
            radiation_dir,
            min_pair_intensity,
        )
        # Atomic-ish write: overwrite each file with its augmented version.
        p.unlink(missing_ok=True)
        frame.write_parquet(p)
        written.append(p)
        total_rows += frame.height
        logger.info("[%s] Z=%d: %d rows", symbol, z, frame.height)
        # Eagerly drop per-element intermediates. polars' Arrow buffers don't
        # always return memory to the OS without explicit GC, and the
        # high-water mark across all 105 elements is what bites us.
        del gg_z, frame
        gc.collect()
    logger.info(
        "Wrote %d per-element mixed-coincidence files (%d pairs total) to %s",
        len(written),
        total_rows,
        coincidences_dir,
    )
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    repo_root = Path(__file__).resolve().parents[2]
    meta = repo_root / "data" / "meta"
    parser.add_argument("--decay-detailed", type=Path, default=meta / "decay_detailed.parquet")
    parser.add_argument("--nudex-levels", type=Path, default=meta / "nudex_levels.parquet")
    parser.add_argument("--nudex-gammas", type=Path, default=meta / "nudex_level_gammas.parquet")
    parser.add_argument("--radiation-dir", type=Path, default=meta / "ensdf" / "radiation")
    parser.add_argument(
        "--coincidences-dir",
        type=Path,
        default=meta / "ensdf" / "coincidences",
        help="Directory to read existing γ-γ pairs from and write augmented output to.",
    )
    parser.add_argument(
        "--min-pair-intensity",
        type=float,
        default=_DEFAULT_MIN_PAIR_INTENSITY,
        help=(
            "Drop mixed-emission rows below this pair_intensity threshold (default 1e-9; γ-γ rows are not filtered)."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    build(
        args.decay_detailed,
        args.nudex_levels,
        args.nudex_gammas,
        args.radiation_dir,
        args.coincidences_dir,
        args.min_pair_intensity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
