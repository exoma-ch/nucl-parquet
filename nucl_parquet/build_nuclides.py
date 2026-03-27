"""Build nuclides.parquet from ground_states + decay.parquet isomers.

Extends the ground-state-only nuclide table with isomeric state entries
(Tc-99m, Sc-44m, Eu-152m, etc.) so that every long-lived nuclear state
is a first-class entry keyed on (Z, A, state).

Primary source for isomers: decay.parquet (comprehensive, all Z).
Fallback for level energies: levels/*.parquet (partial Z coverage).

Output: meta/ensdf/nuclides.parquet

Usage:
    uv run python -m nucl_parquet.build_nuclides
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from .download import data_dir as _resolve_data_dir


def build(data_dir: Path | None = None) -> None:
    """Build meta/ensdf/nuclides.parquet."""
    if data_dir is None:
        data_dir = _resolve_data_dir()
    data_dir = Path(data_dir)

    gs_path = data_dir / "meta" / "ensdf" / "ground_states.parquet"
    decay_path = data_dir / "meta" / "decay.parquet"
    levels_dir = data_dir / "meta" / "ensdf" / "levels"

    if not gs_path.exists():
        raise FileNotFoundError(f"Ground states not found: {gs_path}")
    if not decay_path.exists():
        raise FileNotFoundError(f"Decay data not found: {decay_path}")

    db = duckdb.connect()

    # --- Ground states ---
    gs_cols = db.sql(f"""
        SELECT Z, A, symbol, jp, half_life_s,
               decay_1, decay_1_pct, decay_2, decay_2_pct
        FROM read_parquet('{gs_path}')
        WHERE Z > 0
        ORDER BY Z, A
    """).fetchall()

    # Symbol lookup
    symbol_lookup: dict[int, str] = {}
    for row in gs_cols:
        symbol_lookup[int(row[0])] = row[2]

    # --- Isomeric states from decay.parquet ---
    # decay.parquet has (Z, A, state, half_life_s, decay_mode, branching)
    # Group by (Z, A, state) to get unique isomers with their properties
    iso_from_decay = db.sql(f"""
        SELECT Z, A, state, half_life_s,
               FIRST(decay_mode) AS decay_1,
               FIRST(branching) AS decay_1_pct
        FROM (
            SELECT Z, A, state, half_life_s, decay_mode, branching
            FROM read_parquet('{decay_path}')
            WHERE state != '' AND Z > 0
            ORDER BY Z, A, state, branching DESC
        )
        GROUP BY Z, A, state, half_life_s
        ORDER BY Z, A, state
    """).fetchall()

    # Get second decay mode for each isomer
    iso_decay2 = {}
    decay2_rows = db.sql(f"""
        WITH ranked AS (
            SELECT Z, A, state, decay_mode, branching,
                   ROW_NUMBER() OVER (PARTITION BY Z, A, state ORDER BY branching DESC) AS rn
            FROM read_parquet('{decay_path}')
            WHERE state != '' AND Z > 0
        )
        SELECT Z, A, state, decay_mode, branching
        FROM ranked WHERE rn = 2
    """).fetchall()
    for row in decay2_rows:
        iso_decay2[(int(row[0]), int(row[1]), row[2])] = (row[3], float(row[4]) if row[4] is not None else None)

    # --- Level energies from levels/*.parquet (partial coverage) ---
    level_energy_lookup: dict[tuple[int, int], float] = {}
    if levels_dir.exists() and list(levels_dir.glob("*.parquet")):
        level_rows = db.sql(f"""
            SELECT Z, A, energy_keV
            FROM read_parquet('{levels_dir}/*.parquet')
            WHERE energy_keV > 0 AND half_life_s > 0.001
            ORDER BY Z, A, energy_keV
        """).fetchall()
        for row in level_rows:
            key = (int(row[0]), int(row[1]))
            if key not in level_energy_lookup:  # take lowest isomeric level
                level_energy_lookup[key] = float(row[2])

    # --- Also get level energy from radiation parent_level_keV ---
    rad_dir = data_dir / "meta" / "ensdf" / "radiation"
    if rad_dir.exists():
        rad_levels = db.sql(f"""
            SELECT DISTINCT Z, A, ROUND(parent_level_keV, 1) AS level_keV
            FROM read_parquet('{rad_dir}/*.parquet')
            WHERE parent_level_keV > 0 AND Z > 0
            ORDER BY Z, A, level_keV
        """).fetchall()
        for row in rad_levels:
            key = (int(row[0]), int(row[1]))
            if key not in level_energy_lookup:
                level_energy_lookup[key] = float(row[2])

    # --- Build output ---
    import polars as pl

    out_Z, out_A, out_state, out_sym = [], [], [], []
    out_jp, out_hl, out_level = [], [], []
    out_d1, out_d1p, out_d2, out_d2p = [], [], [], []

    # Ground states
    for row in gs_cols:
        z, a, sym, jp, hl, d1, d1p, d2, d2p = row
        out_Z.append(int(z))
        out_A.append(int(a))
        out_state.append("")
        out_sym.append(sym)
        out_jp.append(jp)
        out_hl.append(float(hl) if hl is not None else None)
        out_level.append(0.0)
        out_d1.append(d1)
        out_d1p.append(float(d1p) if d1p is not None else None)
        out_d2.append(d2)
        out_d2p.append(float(d2p) if d2p is not None else None)

    # Isomeric states from decay.parquet
    seen_iso: set[tuple[int, int, str]] = set()
    for row in iso_from_decay:
        z, a, state, hl, d1, d1p = row
        z, a = int(z), int(a)
        if (z, a, state) in seen_iso:
            continue
        seen_iso.add((z, a, state))

        sym = symbol_lookup.get(z, f"Z{z}")
        level_keV = level_energy_lookup.get((z, a), None)

        d2_info = iso_decay2.get((z, a, state))
        d2, d2p = d2_info if d2_info else (None, None)

        out_Z.append(z)
        out_A.append(a)
        out_state.append(state)
        out_sym.append(sym)
        out_jp.append(None)  # jp not in decay.parquet
        out_hl.append(float(hl) if hl is not None else None)
        out_level.append(level_keV)
        out_d1.append(d1)
        out_d1p.append(float(d1p) if d1p is not None else None)
        out_d2.append(d2)
        out_d2p.append(float(d2p) if d2p is not None else None)

    # Enrich jp from levels data where available
    if levels_dir.exists() and list(levels_dir.glob("*.parquet")):
        jp_lookup: dict[tuple[int, int, float], str] = {}
        jp_rows = db.sql(f"""
            SELECT Z, A, energy_keV, jp
            FROM read_parquet('{levels_dir}/*.parquet')
            WHERE energy_keV > 0 AND half_life_s > 0.001 AND jp IS NOT NULL
        """).fetchall()
        for row in jp_rows:
            jp_lookup[(int(row[0]), int(row[1]), float(row[2]))] = row[3]

        # Back-fill jp for isomeric entries that have a level energy match
        for i in range(len(out_Z)):
            if out_state[i] != "" and out_jp[i] is None and out_level[i] is not None:
                for (z, a, e), jp in jp_lookup.items():
                    if z == out_Z[i] and a == out_A[i] and abs(e - out_level[i]) < 1.0:
                        out_jp[i] = jp
                        break

    df = pl.DataFrame(
        {
            "Z": pl.Series(out_Z, dtype=pl.Int32),
            "A": pl.Series(out_A, dtype=pl.Int32),
            "state": pl.Series(out_state, dtype=pl.Utf8),
            "symbol": pl.Series(out_sym, dtype=pl.Utf8),
            "jp": pl.Series(out_jp, dtype=pl.Utf8),
            "half_life_s": pl.Series(out_hl, dtype=pl.Float64),
            "level_keV": pl.Series(out_level, dtype=pl.Float64),
            "decay_1": pl.Series(out_d1, dtype=pl.Utf8),
            "decay_1_pct": pl.Series(out_d1p, dtype=pl.Float64),
            "decay_2": pl.Series(out_d2, dtype=pl.Utf8),
            "decay_2_pct": pl.Series(out_d2p, dtype=pl.Float64),
        }
    ).sort("Z", "A", "state")

    out_path = data_dir / "meta" / "ensdf" / "nuclides.parquet"
    df.write_parquet(out_path, compression="zstd")

    n_ground = (df["state"] == "").sum()
    n_iso = (df["state"] != "").sum()
    print(f"Wrote {len(df)} nuclides to {out_path}")
    print(f"  {n_ground} ground states, {n_iso} isomeric states")

    _validate(df)


def _validate(df) -> None:
    """Check known isomeric states are present with correct properties."""
    import polars as pl

    checks = [
        (43, 99, "m", (2.0e4, 2.5e4), "Tc-99m (6.01 h)"),
        (21, 44, "m", (1.5e5, 2.5e5), "Sc-44m (58.6 h)"),
        (49, 113, "m", (5.9e3, 6.1e3), "In-113m (1.66 h)"),
        (47, 108, "m", (1.2e10, 1.5e10), "Ag-108m (418 yr)"),
        (63, 152, "m", (3.0e4, 3.6e4), "Eu-152m (9.31 h)"),
        (56, 137, "m", (1.4e2, 1.6e2), "Ba-137m (2.55 min)"),
    ]

    print("\nValidation of known isomeric states:")
    for z, a, state, (t_lo, t_hi), desc in checks:
        row = df.filter(
            (pl.col("Z") == z) & (pl.col("A") == a) & (pl.col("state") == state)
        )
        if len(row) == 0:
            print(f"  MISSING: {desc}")
        else:
            hl = row["half_life_s"][0]
            ok = t_lo <= hl <= t_hi if hl is not None else False
            status = "OK" if ok else f"CHECK (T½={hl:.3e} s)"
            print(f"  {desc}: {status}")


def get_state_map(data_dir: Path) -> dict[tuple[int, int], list[tuple[str, float]]]:
    """Return (Z, A) → [(state_label, energy_keV), ...] for radiation state assignment.

    Combines levels data (has energy_keV) with radiation parent_level_keV
    to build a comprehensive mapping.
    """
    from collections import defaultdict

    db = duckdb.connect()
    za_levels: dict[tuple[int, int], list[float]] = defaultdict(list)

    # From levels (most precise)
    levels_dir = data_dir / "meta" / "ensdf" / "levels"
    if levels_dir.exists() and list(levels_dir.glob("*.parquet")):
        rows = db.sql(f"""
            SELECT Z, A, energy_keV
            FROM read_parquet('{levels_dir}/*.parquet')
            WHERE energy_keV > 0 AND half_life_s > 0.001
            ORDER BY Z, A, energy_keV
        """).fetchall()
        for z, a, e in rows:
            za_levels[(int(z), int(a))].append(float(e))

    # From radiation parent_level_keV (covers nuclides not in levels)
    rad_dir = data_dir / "meta" / "ensdf" / "radiation"
    if rad_dir.exists():
        rows = db.sql(f"""
            SELECT DISTINCT Z, A, ROUND(parent_level_keV, 1) AS level_keV
            FROM read_parquet('{rad_dir}/*.parquet')
            WHERE parent_level_keV > 0 AND Z > 0
            ORDER BY Z, A, level_keV
        """).fetchall()
        for z, a, e in rows:
            key = (int(z), int(a))
            e = float(e)
            # Only add if not already covered by levels data (within tolerance)
            existing = za_levels.get(key, [])
            if not any(abs(ex - e) < 1.0 for ex in existing):
                za_levels[key].append(e)

    result: dict[tuple[int, int], list[tuple[str, float]]] = {}
    for (z, a), energies in za_levels.items():
        energies.sort()
        labels = []
        for i, e in enumerate(energies):
            label = "m" if i == 0 else f"m{i + 1}"
            labels.append((label, e))
        result[(z, a)] = labels
    return result


if __name__ == "__main__":
    build()
