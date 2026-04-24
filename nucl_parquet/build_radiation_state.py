"""Add state column to radiation parquet files.

Assigns each radiation row to a parent nuclear state ("", "m", "m2")
based on parent_level_keV matched against known isomeric level energies.

This makes the radiation table consistent with the (Z, A, state) keying
used in decay.parquet, dose_constants.parquet, and nuclides.parquet.

Usage:
    uv run python -m nucl_parquet.build_radiation_state
"""

from __future__ import annotations

import warnings
from pathlib import Path

from .build_nuclides import get_state_map
from .download import data_dir as _resolve_data_dir

# Max keV between radiation's parent_level_keV and the nearest known isomeric
# level before we consider the match suspicious. ENSDF parent levels are
# usually quoted to better than 1 keV; anything worse warrants a warning.
_LEVEL_MATCH_TOLERANCE_KEV = 2.0


def build(data_dir: Path | None = None) -> None:
    """Add state column to all radiation/*.parquet files."""
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    rad_dir = data_dir / "meta" / "ensdf" / "radiation"
    if not rad_dir.exists():
        raise FileNotFoundError(f"Radiation data not found: {rad_dir}")

    state_map = get_state_map(data_dir)
    _assert_unique_levels(state_map)

    import polars as pl

    files = sorted(rad_dir.glob("*.parquet"))
    total_rows = 0
    total_iso = 0
    orphans: set[tuple[int, int]] = set()
    fuzzy_matches: list[tuple[int, int, float, str, float]] = []

    for f in files:
        df = pl.read_parquet(f)

        # Drop existing state column if re-running
        if "state" in df.columns:
            df = df.drop("state")

        states = _assign_states(df, state_map, orphans, fuzzy_matches)

        # Insert state column after A
        state_series = pl.Series("state", states, dtype=pl.Utf8)
        cols = df.columns
        a_idx = cols.index("A") + 1
        df = df.with_columns(state_series)
        # Reorder: put state right after A
        new_order = cols[:a_idx] + ["state"] + cols[a_idx:]
        df = df.select(new_order)

        df.write_parquet(f, compression="zstd")

        n_iso = sum(1 for s in states if s != "")
        total_rows += len(df)
        total_iso += n_iso

    print(f"Processed {len(files)} radiation files, {total_rows} rows")
    print(f"  {total_iso} rows assigned to isomeric states")
    print(f"  {total_rows - total_iso} rows assigned to ground state")

    if orphans:
        warnings.warn(
            f"{len(orphans)} (Z,A) nuclides have radiation but no entry in "
            f"nuclides.parquet; non-zero parent_level_keV rows labelled 'm' by "
            f"fallback. Examples: {sorted(orphans)[:5]}",
            stacklevel=2,
        )
    if fuzzy_matches:
        warnings.warn(
            f"{len(fuzzy_matches)} radiation rows matched to an isomeric level "
            f"further than {_LEVEL_MATCH_TOLERANCE_KEV} keV — possible silent "
            f"mis-label. Examples: {fuzzy_matches[:3]}",
            stacklevel=2,
        )

    _validate(data_dir)


def _assign_states(
    df,
    state_map: dict[tuple[int, int], list[tuple[str, float]]],
    orphans: set[tuple[int, int]],
    fuzzy_matches: list[tuple[int, int, float, str, float]],
) -> list[str]:
    """Assign a state label ('', 'm', 'm2', ...) to each radiation row.

    NULL or zero parent_level_keV → ground state ('').  Non-zero levels are
    matched to the nearest known isomeric level from `state_map`.  Accumulates
    diagnostics into `orphans` and `fuzzy_matches` for the caller to surface.
    """
    states: list[str] = []
    for row in df.iter_rows(named=True):
        z, a = int(row["Z"]), int(row["A"])
        pl_raw = row["parent_level_keV"]
        pl_keV = float(pl_raw) if pl_raw is not None else 0.0

        if pl_keV == 0.0:
            states.append("")
            continue

        labels = state_map.get((z, a))
        if labels is None:
            orphans.add((z, a))
            states.append("m")
            continue

        best_label = "m"
        best_dist = float("inf")
        for label, energy in labels:
            dist = abs(energy - pl_keV)
            if dist < best_dist:
                best_dist = dist
                best_label = label

        if best_dist > _LEVEL_MATCH_TOLERANCE_KEV:
            fuzzy_matches.append((z, a, pl_keV, best_label, best_dist))

        states.append(best_label)
    return states


def _assert_unique_levels(
    state_map: dict[tuple[int, int], list[tuple[str, float]]],
) -> None:
    """Guard: no two isomeric labels for the same (Z,A) should share an energy."""
    for (z, a), labels in state_map.items():
        energies = [e for _, e in labels]
        for i, e in enumerate(energies):
            for j in range(i + 1, len(energies)):
                if abs(e - energies[j]) < 0.01:
                    raise ValueError(
                        f"Duplicate isomeric level at Z={z} A={a}: {labels[i]} and {labels[j]} within 0.01 keV"
                    )


def _validate(data_dir: Path) -> None:
    """Check specific nuclides have correct state assignments."""
    import duckdb

    db = duckdb.connect()
    rad_dir = data_dir / "meta" / "ensdf" / "radiation"

    checks = [
        # Eu-152: 121.78 keV should be ground state, 841.63 keV should be "m"
        (63, 152, 121.78, "", "Eu-152 121.78 keV → ground"),
        (63, 152, 841.63, "m", "Eu-152m 841.63 keV → m"),
        # Tc-99m: 140.5 keV is the isomer
        (43, 99, 140.5, "m", "Tc-99m 140.5 keV → m"),
    ]

    print("\nRadiation state validation:")
    for z, a, e_target, expected_state, desc in checks:
        rows = db.sql(f"""
            SELECT state, energy_keV, intensity_pct
            FROM read_parquet('{rad_dir}/*.parquet')
            WHERE Z={z} AND A={a}
              AND ABS(energy_keV - {e_target}) < 1.0
              AND rad_type = 'gamma'
            LIMIT 5
        """).fetchall()
        if not rows:
            print(f"  NOT FOUND: {desc}")
        else:
            for row in rows:
                actual = row[0]
                status = "OK" if actual == expected_state else f"WRONG (got '{actual}')"
                print(f"  {desc}: state='{actual}' E={row[1]:.2f} I={row[2]:.2f}% → {status}")


if __name__ == "__main__":
    build()
