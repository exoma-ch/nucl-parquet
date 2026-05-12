"""Derive ``meta/ensdf/coincidences/{Symbol}.parquet`` (gamma cascade pairs)
from strata's PhotonEvaporation6.1.2 gammas + level scheme.

Per epic #66 / issue #73 / ADR-0002.

Two gammas (γ₁, γ₂) form a *coincidence pair* when γ₁ deexcites a parent level
to an intermediate level, and γ₂ then deexcites that same intermediate level
to a final level (often, but not necessarily, the ground state). They are the
two photons emitted in a single nuclear-deexcitation cascade and therefore
arrive in coincidence at a detector.

Method
------
For each (Z, A), self-join ``photon_evap_gammas`` on
``daughter_level_1 == parent_level_2``. Each surviving row is a direct
two-step cascade pair. We then LEFT JOIN ``photon_evap_levels`` on
``(z, a, level_idx)`` three times to materialize the parent / intermediate /
final excitation energies in keV.

n-step cascades (γ₁ → A → B → γ₃ with B intermediate) are *not* enumerated
here — direct pairs are the canonical tabulation; consumers reconstruct
longer chains by transitive closure on this table if needed (#73 pitfall).

Schema (output, per element ``meta/ensdf/coincidences/{Symbol}.parquet``)
------------------------------------------------------------------------

Per ADR-0002 the v0.10.x columns are preserved verbatim so existing consumers
(Rust crate / TS SDK / parquet-direct external readers) keep working
unchanged. The richer G4-derived columns are added alongside as nullable.

v0.10.x compat columns (preserved):
- ``Z``                       Int64
- ``A``                       Int64
- ``dataset``                 Int64 — ENSDF-source identifier in v0.10.x; G4
  PhotonEvaporation has a single source per nuclide so we ship constant 1.
- ``gamma_energy_keV``        Float64 — alias of ``gamma1_energy_keV``
- ``coinc_energy_keV``        Float64 — alias of ``gamma2_energy_keV``

G4-derived bonus columns (additive per ADR-0002):
- ``gamma1_energy_keV``       Float64 — energy of γ₁ (parent → intermediate)
- ``gamma1_intensity``        Float32 — relative per-cascade intensity (G4 raw, max ~9000)
- ``gamma2_energy_keV``       Float64 — energy of γ₂ (intermediate → final)
- ``gamma2_intensity``        Float32
- ``parent_level_keV``        Float64 — energy of the level γ₁ originates from
- ``intermediate_level_keV``  Float64 — energy of the level γ₁ ends at / γ₂ starts from
- ``final_level_keV``         Float64 — energy of the level γ₂ lands at (often 0 = ground)
- ``pair_intensity``          Float32 — see "Normalization" below
- ``gamma1_icc_total``        Float32 — for pure-γ pair correction
  ``(1 + icc1)⁻¹ × (1 + icc2)⁻¹`` in dose calc downstream.
- ``gamma2_icc_total``        Float32

Normalization
-------------

``pair_intensity`` ships as ``intensity1 × intensity2 / 100`` — a *relative*
quantity, not an absolute coincidence probability per parent decay. G4
PhotonEvaporation intensities are normalized per cascade level (max observed
~9000), not as percentages. Consumers needing absolute probabilities must
renormalize per ``(Z, A, parent_level_keV)``. Worked example in DuckDB:

::

    -- For each parent level, normalize gamma1 to unity and compute
    -- pair_intensity_norm as i1_norm × i2_norm.
    WITH normalized AS (
      SELECT *,
        gamma1_intensity / SUM(gamma1_intensity) OVER (
          PARTITION BY Z, A, parent_level_keV
        ) AS i1_norm,
        gamma2_intensity / SUM(gamma2_intensity) OVER (
          PARTITION BY Z, A, intermediate_level_keV
        ) AS i2_norm
      FROM coincidences
    )
    SELECT Z, A, gamma1_energy_keV, gamma2_energy_keV,
           i1_norm * i2_norm AS pair_intensity_per_parent_decay
    FROM normalized
    WHERE Z = 28 AND A = 60 -- Co-60 → Ni-60 cascade

For pure-γ pair correction (excluding internal-conversion losses), multiply
by ``1 / ((1 + gamma1_icc_total) × (1 + gamma2_icc_total))``.

Pitfalls
--------
- **Self-join scale**: the input table is ~297k rows. A naive
  ``join(on=['z', 'a'])`` Cartesian explodes within (Z,A) groups but is
  feasible at ~100k–1M output rows total. We post-filter on
  ``daughter_level_1 == parent_level_2`` to keep only physical cascade
  pairs.
- **Self-pairs**: a row joining to itself (same (parent_level, daughter_level)
  on both sides) is unphysical (γ self-coincidence). Filtered explicitly.
- **Intermediate naming**: γ₁'s daughter level *is* γ₂'s parent level by
  construction. We commit on the names ``parent / intermediate / final``
  for the output schema; comments in the join keep the mapping explicit.

Usage
-----
::

    uv run python -m nucl_parquet.g4.coincidences
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import polars as pl

from nucl_parquet.g4.photon_evap_levels import Z_TO_SYMBOL

logger = logging.getLogger(__name__)


def build_pairs(gammas: pl.DataFrame, levels: pl.DataFrame) -> pl.DataFrame:
    """Self-join ``gammas`` on shared intermediate level → cascade pairs.

    Parameters
    ----------
    gammas
        Strata's ``photon_evap_gammas.parquet`` schema:
        ``z, a, parent_level, daughter_level, gamma_energy_kev,
        intensity, multipolarity, mixing_ratio, icc_total``.
    levels
        Strata's ``photon_evap_levels.parquet`` schema (used to look up
        excitation energies): ``z, a, level_idx, excitation_kev, ...``.

    Returns
    -------
    Polars DataFrame in the output schema documented in the module docstring,
    sorted by ``(Z, A, parent_level_keV, intermediate_level_keV,
    final_level_keV)`` for stable, browseable output.

    Notes
    -----
    The self-join is grouped by ``(z, a)`` (the join keys), so the worst-case
    Cartesian is bounded per nuclide rather than across the whole table.
    """
    # Shape both sides explicitly so the suffix doesn't collide with anything
    # else and the column names map cleanly to the output schema.
    side_one = gammas.select(
        pl.col("z"),
        pl.col("a"),
        pl.col("parent_level").alias("parent_level_1"),
        pl.col("daughter_level").alias("daughter_level_1"),
        pl.col("gamma_energy_kev").alias("gamma1_energy_keV"),
        pl.col("intensity").alias("gamma1_intensity"),
        pl.col("icc_total").alias("gamma1_icc_total"),
    )
    side_two = gammas.select(
        pl.col("z"),
        pl.col("a"),
        pl.col("parent_level").alias("parent_level_2"),
        pl.col("daughter_level").alias("daughter_level_2"),
        pl.col("gamma_energy_kev").alias("gamma2_energy_keV"),
        pl.col("intensity").alias("gamma2_intensity"),
        pl.col("icc_total").alias("gamma2_icc_total"),
    )

    # Inner join on (z, a), then post-filter on the intermediate level
    # equality. Polars' ``join_where`` would let us push the filter into the
    # join; the equi-join + filter form is simpler and the size is bounded
    # by per-(Z,A) group cardinality which is small for most nuclides.
    pairs = side_one.join(side_two, on=["z", "a"]).filter(
        # Cascade contract: γ₁'s daughter level == γ₂'s parent level.
        pl.col("daughter_level_1") == pl.col("parent_level_2"),
        # Reject γ self-coincidence (a row joined to itself, parent_level_1 ==
        # parent_level_2 AND daughter_level_1 == daughter_level_2). This can't
        # happen for distinct rows because the cascade contract above forces
        # parent_level_2 == daughter_level_1, so equality of parent_level_1
        # and parent_level_2 implies a single-step γ that begins and ends on
        # the same level — unphysical.
        pl.col("parent_level_1") != pl.col("parent_level_2"),
    )

    # Materialize level energies via three LEFT JOINs into the levels table.
    # Each lookup is keyed on (z, a, level_idx); we re-alias each time to
    # produce parent / intermediate / final.
    level_lookup = levels.select(
        pl.col("z"),
        pl.col("a"),
        pl.col("level_idx"),
        pl.col("excitation_kev"),
    )

    pairs = (
        pairs.join(
            level_lookup.rename({"level_idx": "parent_level_1", "excitation_kev": "parent_level_keV"}),
            on=["z", "a", "parent_level_1"],
            how="left",
        )
        .join(
            level_lookup.rename({"level_idx": "daughter_level_1", "excitation_kev": "intermediate_level_keV"}),
            on=["z", "a", "daughter_level_1"],
            how="left",
        )
        .join(
            level_lookup.rename({"level_idx": "daughter_level_2", "excitation_kev": "final_level_keV"}),
            on=["z", "a", "daughter_level_2"],
            how="left",
        )
    )

    # Joint emission probability — relative scale (see module docstring).
    # Cast to Float32 to match the schema.
    pairs = pairs.with_columns(
        (pl.col("gamma1_intensity").cast(pl.Float64) * pl.col("gamma2_intensity").cast(pl.Float64) / 100.0)
        .cast(pl.Float32)
        .alias("pair_intensity"),
        pl.col("z").cast(pl.Int32).alias("Z"),
        pl.col("a").cast(pl.Int32).alias("A"),
    )

    return pairs.select(
        # v0.10.x-compat columns — preserved verbatim per ADR-0002 so the Rust
        # crate / TS SDK / parquet-direct consumers keep working unchanged.
        # (gamma_energy_keV ↔ gamma1_energy_keV, coinc_energy_keV ↔ gamma2_energy_keV.
        # `dataset` was an ENSDF-source identifier in v0.10.x; G4
        # PhotonEvaporation has a single source per nuclide so we ship 1.)
        pl.col("Z").cast(pl.Int64),
        pl.col("A").cast(pl.Int64),
        pl.lit(1, dtype=pl.Int64).alias("dataset"),
        pl.col("gamma1_energy_keV").alias("gamma_energy_keV"),
        pl.col("gamma2_energy_keV").alias("coinc_energy_keV"),
        # Bonus columns from G4 (additive per ADR-0002): cascade-level
        # provenance, individual intensities, internal-conversion coefficients,
        # and the joint-intensity scaling helper.
        "gamma1_energy_keV",
        "gamma1_intensity",
        "gamma2_energy_keV",
        "gamma2_intensity",
        "parent_level_keV",
        "intermediate_level_keV",
        "final_level_keV",
        "pair_intensity",
        "gamma1_icc_total",
        "gamma2_icc_total",
    ).sort(
        [
            "Z",
            "A",
            "parent_level_keV",
            "intermediate_level_keV",
            "final_level_keV",
            "gamma1_energy_keV",
        ]
    )


def write_per_element(df: pl.DataFrame, out_dir: Path) -> list[Path]:
    """Partition by Z → write ``meta/ensdf/coincidences/{Symbol}.parquet``.

    Returns the list of paths written. Filenames use the IUPAC capitalized
    symbol (``H``, ``He``, ``N``, ``Co``); on case-insensitive filesystems
    we ``unlink`` first so a stray lowercase ``n.parquet`` from an older
    build doesn't capture fresh writes for nitrogen (mirrors the case-FS
    fix in :mod:`nucl_parquet.g4.photon_evap_levels`).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for z, group in sorted(df.group_by("Z"), key=lambda kg: kg[0][0]):
        z_int = int(z[0])
        symbol = Z_TO_SYMBOL.get(z_int)
        if symbol is None:
            logger.warning("Skipping Z=%d: no symbol mapping (extend Z_TO_SYMBOL?)", z_int)
            continue
        path = out_dir / f"{symbol}.parquet"
        path.unlink(missing_ok=True)
        group.sort(["A", "parent_level_keV", "intermediate_level_keV", "final_level_keV"]).write_parquet(path)
        written.append(path)
    return written


def build(
    gammas_src: Path,
    levels_src: Path,
    out_dir: Path,
) -> list[Path]:
    """End-to-end: read strata inputs, build pairs, write per-element parquet."""
    if not gammas_src.exists():
        raise FileNotFoundError(f"strata gammas not found at {gammas_src}; run scripts/fetch_strata_nuclear.py first")
    if not levels_src.exists():
        raise FileNotFoundError(f"strata levels not found at {levels_src}; run scripts/fetch_strata_nuclear.py first")
    gammas = pl.read_parquet(gammas_src)
    levels = pl.read_parquet(levels_src)
    pairs = build_pairs(gammas, levels)
    paths = write_per_element(pairs, out_dir)
    logger.info(
        "Wrote %d per-element coincidence files (%d pairs total) to %s",
        len(paths),
        pairs.height,
        out_dir,
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--gammas",
        type=Path,
        default=repo_root / "data" / "g4_raw" / "strata-nuclear" / "photon_evap_gammas.parquet",
        help="Path to strata's photon_evap_gammas.parquet",
    )
    parser.add_argument(
        "--levels",
        type=Path,
        default=repo_root / "data" / "g4_raw" / "strata-nuclear" / "photon_evap_levels.parquet",
        help="Path to strata's photon_evap_levels.parquet (for level energy lookup)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "data" / "meta" / "ensdf" / "coincidences",
        help="Directory to write per-element parquet files into",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    build(args.gammas, args.levels, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
