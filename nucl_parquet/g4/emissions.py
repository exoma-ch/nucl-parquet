"""Derive ``meta/ensdf/emissions/{Symbol}.parquet`` — absolute per-decay photon
emission intensities, parent-keyed.

Per issue #196 — ``intensity_pct`` in the radiation table is relative per cascade
level (G4 PhotonEvaporation semantics), not absolute per parent decay.  This
module materializes the **NuDat-equivalent** table: for each parent nuclide, what
gammas are emitted and at what absolute probability per decay.

Algorithm
---------

For each parent (Z, A, state) that has known decay channels:

1. **Build feeding map** — daughter level → population fraction:
   - From ``decay_detailed``: β⁻/EC/α transitions with per-level branching.
   - From ``decay`` + ``nuclides``: IT entries (maps isomeric state → level energy).
2. **Walk the daughter gamma cascade** top-down (highest energy level first):
   - Per level: branch fraction = ``intensity × (1+ICC) / Σ(intensity × (1+ICC))``.
   - Propagate: ``population[daughter_level] += population[level] × branch_frac``.
3. **Absolute photon intensity** per gamma:
   ``abs_pct = population[level] × branch_frac × 1/(1+ICC) × 100``.

Key insight: G4's ``intensity_pct`` is the *photon* emission probability
(relative per level).  Total transition rate (photon + IC) =
``intensity × (1 + ICC)``.  Normalizing branch fractions per level requires the
total rate, not photon-only.

Schema (output, per element ``meta/ensdf/emissions/{Symbol}.parquet``)
----------------------------------------------------------------------

Filed by **parent** element symbol (Co-60 emissions live in ``Co.parquet``).

    parent_Z              Int32     — decaying nucleus
    parent_A              Int32
    parent_state          Utf8      — '' | 'm' | 'm2'
    decay_mode            Utf8      — 'beta-' | 'KshellEC' | 'IT' | …
    daughter_Z            Int32     — product nucleus
    daughter_A            Int32
    rad_type              Utf8      — 'gamma'
    energy_keV            Float64
    intensity_pct         Float64   — absolute per-decay photon emission (0–100%)
    icc_total             Float32   — internal conversion coefficient
    parent_level_keV      Float64   — cascade level in daughter
    daughter_level_keV    Float64   — destination level
    multipolarity         Int32     — from radiation table

Usage::

    uv run python -m nucl_parquet.g4.emissions
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import polars as pl

from nucl_parquet.g4.photon_evap_levels import Z_TO_SYMBOL

logger = logging.getLogger(__name__)

# Output column order — stable across all elements.
_OUTPUT_COLUMNS = [
    "parent_Z",
    "parent_A",
    "parent_state",
    "decay_mode",
    "daughter_Z",
    "daughter_A",
    "rad_type",
    "energy_keV",
    "intensity_pct",
    "icc_total",
    "parent_level_keV",
    "daughter_level_keV",
    "multipolarity",
]

_OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "parent_Z": pl.Int32,
    "parent_A": pl.Int32,
    "parent_state": pl.Utf8,
    "decay_mode": pl.Utf8,
    "daughter_Z": pl.Int32,
    "daughter_A": pl.Int32,
    "rad_type": pl.Utf8,
    "energy_keV": pl.Float64,
    "intensity_pct": pl.Float64,
    "icc_total": pl.Float32,
    "parent_level_keV": pl.Float64,
    "daughter_level_keV": pl.Float64,
    "multipolarity": pl.Int32,
}


def _empty_output() -> pl.DataFrame:
    """Empty DataFrame with the full output schema."""
    return pl.DataFrame(schema=_OUTPUT_SCHEMA)


# ---------------------------------------------------------------------------
# Core cascade algorithm
# ---------------------------------------------------------------------------


def _build_cascade_map(
    rad_df: pl.DataFrame, daughter_z: int, daughter_a: int,
) -> dict[float, list[dict]]:
    """Build per-level gamma transition list for a specific daughter isotope.

    Returns mapping: parent_level_keV → [{daughter_level, energy, intensity_pct,
    icc_total, multipolarity}, …].
    """
    gammas = rad_df.filter(
        (pl.col("Z") == daughter_z)
        & (pl.col("A") == daughter_a)
        & (pl.col("rad_type") == "gamma")
    )
    cascade: dict[float, list[dict]] = defaultdict(list)
    for row in gammas.iter_rows(named=True):
        cascade[row["parent_level_keV"]].append({
            "daughter_level": row["daughter_level_keV"],
            "energy": row["energy_keV"],
            "intensity_pct": row["intensity_pct"],
            "icc_total": row["icc_total"] if row["icc_total"] is not None else 0.0,
            "multipolarity": row.get("multipolarity"),
        })
    return dict(cascade)


def compute_absolute_intensities(
    feeding: dict[float, float],
    cascade: dict[float, list[dict]],
    min_intensity_pct: float = 1e-6,
) -> list[dict]:
    """Compute absolute photon emission intensities for a single decay channel.

    Parameters
    ----------
    feeding
        Mapping of daughter level energy (keV) → population fraction per parent
        decay (sum of all feeding entries should be ≤ 1.0).
    cascade
        Per-level gamma transitions as returned by :func:`_build_cascade_map`.
    min_intensity_pct
        Minimum absolute intensity (%) to include in output.

    Returns
    -------
    List of dicts with keys: energy, parent_level, daughter_level,
    intensity_pct, icc_total, multipolarity.
    """
    # Process levels top-down: propagate population through the cascade.
    # The level list is static — new daughter levels added during propagation
    # are terminal (no outgoing gammas in `cascade`) so need not be visited.
    all_levels = sorted(set(feeding.keys()) | set(cascade.keys()), reverse=True)
    population: dict[float, float] = {lev: feeding.get(lev, 0.0) for lev in all_levels}

    results: list[dict] = []

    for lev in all_levels:
        pop = population.get(lev, 0.0)
        if pop == 0 or lev not in cascade:
            continue

        gammas = cascade[lev]
        # Total transition rate = Σ(photon_intensity × (1 + ICC))
        total_transition = sum(
            g["intensity_pct"] * (1.0 + g["icc_total"]) for g in gammas
        )
        if total_transition == 0:
            continue

        for g in gammas:
            transition_rate = g["intensity_pct"] * (1.0 + g["icc_total"])
            branch_frac = transition_rate / total_transition
            photon_frac = 1.0 / (1.0 + g["icc_total"])
            abs_photon_pct = pop * branch_frac * photon_frac * 100.0

            # Propagate population to daughter level
            daughter = g["daughter_level"]
            if daughter not in population:
                population[daughter] = 0.0
            population[daughter] += pop * branch_frac

            if abs_photon_pct >= min_intensity_pct:
                results.append({
                    "energy": g["energy"],
                    "parent_level": lev,
                    "daughter_level": g["daughter_level"],
                    "intensity_pct": abs_photon_pct,
                    "icc_total": g["icc_total"],
                    "multipolarity": g["multipolarity"],
                })

    return results


def _build_feeding_from_detailed(
    decay_detailed: pl.DataFrame,
    parent_z: int, parent_a: int, parent_ex_kev: float,
) -> dict[tuple[int, int], dict[str, dict[float, float]]]:
    """Build per-daughter, per-decay-mode feeding maps from decay_detailed.

    Returns mapping: (daughter_Z, daughter_A) → {decay_mode → {level_keV → branching}}.
    """
    rows = decay_detailed.filter(
        (pl.col("Z") == parent_z)
        & (pl.col("A") == parent_a)
        & (pl.col("parent_ex_kev") == parent_ex_kev)
    )
    result: dict[tuple[int, int], dict[str, dict[float, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    for row in rows.iter_rows(named=True):
        dz, da = row["daughter_Z"], row["daughter_A"]
        if dz is None or da is None:
            continue  # fission / unknown daughter
        mode = row["decay_mode"]
        level = row["daughter_ex_kev"]
        result[(dz, da)][mode][level] += row["branching"]
    return dict(result)


def _build_feeding_from_it(
    decay_summary: pl.DataFrame,
    nuclides: pl.DataFrame,
    parent_z: int, parent_a: int, parent_state: str,
) -> dict[str, dict[float, float]] | None:
    """Build feeding map for IT (isomeric transition) decay.

    IT rows exist in decay.parquet (summary) but not decay_detailed.
    Returns {decay_mode → {level_keV → branching}} or None if no IT found.
    """
    it_rows = decay_summary.filter(
        (pl.col("Z") == parent_z)
        & (pl.col("A") == parent_a)
        & (pl.col("state") == parent_state)
        & (pl.col("decay_mode") == "IT")
    )
    if it_rows.is_empty():
        return None

    # Look up the isomeric level energy
    nuc = nuclides.filter(
        (pl.col("Z") == parent_z)
        & (pl.col("A") == parent_a)
        & (pl.col("state") == parent_state)
    )
    if nuc.is_empty():
        logger.warning(
            "IT decay for Z=%d A=%d state=%s but no nuclide catalog entry",
            parent_z, parent_a, parent_state,
        )
        return None

    level_kev = nuc["level_keV"][0]
    branching = it_rows["branching"][0]
    # IT feeds the isomeric level of the SAME nuclide (Z, A)
    return {"IT": {level_kev: branching}}


def build_for_parent(
    parent_z: int,
    parent_a: int,
    parent_state: str,
    decay_detailed: pl.DataFrame,
    decay_summary: pl.DataFrame,
    nuclides: pl.DataFrame,
    radiation_cache: dict[int, pl.DataFrame],
) -> pl.DataFrame:
    """Build absolute emission rows for a single parent nuclide.

    Parameters
    ----------
    radiation_cache
        Mapping of daughter_Z → radiation DataFrame (all isotopes for that element).
    """
    # Determine parent excitation energy
    if parent_state == "":
        parent_ex_kev = 0.0
    else:
        nuc = nuclides.filter(
            (pl.col("Z") == parent_z)
            & (pl.col("A") == parent_a)
            & (pl.col("state") == parent_state)
        )
        if nuc.is_empty():
            return _empty_output()
        parent_ex_kev = nuc["level_keV"][0]

    # Collect feeding per daughter per decay_mode
    # (daughter_Z, daughter_A) → {mode → {level → branching}}
    all_feeding: dict[tuple[int, int], dict[str, dict[float, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )

    # From decay_detailed (β⁻, EC, α, etc.)
    detailed = _build_feeding_from_detailed(
        decay_detailed, parent_z, parent_a, parent_ex_kev,
    )
    for key, modes in detailed.items():
        for mode, levels in modes.items():
            for lev, br in levels.items():
                all_feeding[key][mode][lev] += br

    # From IT (only in decay.parquet summary)
    it_feeding = _build_feeding_from_it(
        decay_summary, nuclides, parent_z, parent_a, parent_state,
    )
    if it_feeding is not None:
        # IT daughter is same nuclide
        key = (parent_z, parent_a)
        for mode, levels in it_feeding.items():
            for lev, br in levels.items():
                all_feeding[key][mode][lev] += br

    if not all_feeding:
        return _empty_output()

    # Process each daughter × decay_mode
    rows: list[dict] = []
    for (daughter_z, daughter_a), modes in all_feeding.items():
        rad_df = radiation_cache.get(daughter_z)
        if rad_df is None:
            continue
        cascade = _build_cascade_map(rad_df, daughter_z, daughter_a)
        if not cascade:
            continue

        for mode, feeding_map in modes.items():
            gammas = compute_absolute_intensities(dict(feeding_map), cascade)
            for g in gammas:
                rows.append({
                    "parent_Z": parent_z,
                    "parent_A": parent_a,
                    "parent_state": parent_state,
                    "decay_mode": mode,
                    "daughter_Z": daughter_z,
                    "daughter_A": daughter_a,
                    "rad_type": "gamma",
                    "energy_keV": g["energy"],
                    "intensity_pct": g["intensity_pct"],
                    "icc_total": g["icc_total"],
                    "parent_level_keV": g["parent_level"],
                    "daughter_level_keV": g["daughter_level"],
                    "multipolarity": g["multipolarity"],
                })

    if not rows:
        return _empty_output()

    return pl.DataFrame(rows, schema=_OUTPUT_SCHEMA).sort(
        ["parent_Z", "parent_A", "parent_state", "decay_mode", "intensity_pct"],
        descending=[False, False, False, False, True],
        nulls_last=True,
    )


# ---------------------------------------------------------------------------
# End-to-end build
# ---------------------------------------------------------------------------


def build(
    decay_detailed_path: Path,
    decay_summary_path: Path,
    nuclides_path: Path,
    radiation_dir: Path,
    out_dir: Path,
) -> list[Path]:
    """Read inputs, compute absolute emissions for all parents, write per-element.

    Files in ``out_dir`` are keyed by **parent** element symbol.
    """
    decay_detailed = pl.read_parquet(decay_detailed_path)
    decay_summary = pl.read_parquet(decay_summary_path)
    nuclides = pl.read_parquet(nuclides_path)

    # Build radiation cache: Z → DataFrame
    symbol_to_z = {sym: z for z, sym in Z_TO_SYMBOL.items()}
    radiation_cache: dict[int, pl.DataFrame] = {}
    for p in sorted(radiation_dir.glob("*.parquet")):
        z = symbol_to_z.get(p.stem)
        if z is not None:
            radiation_cache[z] = pl.read_parquet(p)

    # Enumerate all parents from decay_summary (ground + metastable states)
    parents = (
        decay_summary.select("Z", "A", "state")
        .unique()
        .sort(["Z", "A", "state"])
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate rows per parent element
    per_element: dict[int, list[pl.DataFrame]] = defaultdict(list)
    total_rows = 0

    for row in parents.iter_rows(named=True):
        pz, pa, ps = row["Z"], row["A"], row["state"]
        result = build_for_parent(
            pz, pa, ps,
            decay_detailed, decay_summary, nuclides, radiation_cache,
        )
        if not result.is_empty():
            per_element[pz].append(result)
            total_rows += result.height

    # Write per parent-element files
    written: list[Path] = []
    for z in sorted(per_element.keys()):
        symbol = Z_TO_SYMBOL.get(z)
        if symbol is None:
            continue
        frames = per_element[z]
        combined = pl.concat(frames, how="vertical_relaxed") if len(frames) > 1 else frames[0]
        out_path = out_dir / f"{symbol}.parquet"
        out_path.unlink(missing_ok=True)
        combined.write_parquet(out_path, compression="zstd")
        written.append(out_path)
        logger.info("[%s] Z=%d: %d emission rows", symbol, z, combined.height)

    logger.info(
        "Wrote %d per-element emission files (%d rows total) to %s",
        len(written), total_rows, out_dir,
    )
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    repo_root = Path(__file__).resolve().parents[2]
    meta = repo_root / "data" / "meta"
    parser.add_argument(
        "--decay-detailed",
        type=Path,
        default=meta / "decay_detailed.parquet",
    )
    parser.add_argument(
        "--decay-summary",
        type=Path,
        default=meta / "decay.parquet",
    )
    parser.add_argument(
        "--nuclides",
        type=Path,
        default=meta / "ensdf" / "nuclides.parquet",
    )
    parser.add_argument(
        "--radiation-dir",
        type=Path,
        default=meta / "ensdf" / "radiation",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=meta / "ensdf" / "emissions",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    build(
        args.decay_detailed,
        args.decay_summary,
        args.nuclides,
        args.radiation_dir,
        args.out_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
