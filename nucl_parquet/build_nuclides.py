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

    # --- Rescue IT-from-ground-state orphans (#63) ---
    # The IAEA-LiveChart-derived decay.parquet has 49 rows with state='' AND
    # decay_mode='IT', which is semantically impossible: an Isomeric Transition
    # decay by definition has a metastable parent. The original fetcher script
    # (`scripts/fetch_ensdf.py`, deleted in commit e84ba66) lost the parent
    # state label on these rows. Each one represents a real isomer (m2, m3,
    # etc.) for a (Z,A) that already has an `m` entry in decay.parquet — we
    # synthesize the corrected row here so the catalog is complete without
    # mutating decay.parquet itself (which preserves the raw IAEA payload).
    # NB: level_energy_lookup is built below before the rescue is consumed.

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

    # Now level_energy_lookup is populated — synthesize the rescued isomers.
    rescued_isomers = _rescue_it_orphan_isomers(db, decay_path, data_dir, level_energy_lookup)

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

    # Append rescued IT-orphan isomers (#63) — see _rescue_it_orphan_isomers.
    n_rescued = 0
    for r in rescued_isomers:
        z, a, state, hl, level_keV, d1, d1p = r
        if (z, a, state) in seen_iso:
            continue  # already covered by decay.parquet (defensive)
        seen_iso.add((z, a, state))
        out_Z.append(z)
        out_A.append(a)
        out_state.append(state)
        out_sym.append(symbol_lookup.get(z, f"Z{z}"))
        out_jp.append(None)
        out_hl.append(float(hl) if hl is not None else None)
        out_level.append(float(level_keV) if level_keV is not None else None)
        out_d1.append(d1)
        out_d1p.append(float(d1p) if d1p is not None else None)
        out_d2.append(None)
        out_d2p.append(None)
        n_rescued += 1
    if n_rescued:
        print(f"  Rescued {n_rescued} IT-from-ground-state isomers (see #63).")

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
        row = df.filter((pl.col("Z") == z) & (pl.col("A") == a) & (pl.col("state") == state))
        if len(row) == 0:
            print(f"  MISSING: {desc}")
        else:
            hl = row["half_life_s"][0]
            ok = t_lo <= hl <= t_hi if hl is not None else False
            status = "OK" if ok else f"CHECK (T½={hl:.3e} s)"
            print(f"  {desc}: {status}")


def get_state_map(data_dir: Path) -> dict[tuple[int, int], list[tuple[str, float]]]:
    """Return (Z, A) → [(state_label, level_keV), ...] for radiation state assignment.

    Sourced exclusively from `nuclides.parquet` — the canonical isomer catalog
    introduced in v0.9.0. Only isomers with a known `level_keV` are included;
    isomers with NULL `level_keV` can't be matched against radiation rows on
    parent level energy and are skipped (empirically none of those have any
    radiation rows referring to them).

    Why not levels.parquet or radiation.parent_level_keV as fallback? Both
    over-include: levels.parquet has many short-lived "isomeric" levels that
    aren't real metastables, and radiation parent_level_keV references any
    cascade parent — including non-isomeric short-lived levels that ENSDF
    radiation tables tabulate. Mixing those in fabricates phantom isomer
    labels (m, m2, ...) absent from the catalog. See #58.

    Build pipeline order: build_nuclides must run before build_radiation_state
    (the only caller of this function). Raises FileNotFoundError otherwise.
    """
    nuc_path = data_dir / "meta" / "ensdf" / "nuclides.parquet"
    if not nuc_path.exists():
        raise FileNotFoundError(
            f"nuclides.parquet not found at {nuc_path}; run `python -m nucl_parquet.build_nuclides` first."
        )

    db = duckdb.connect()
    rows = db.sql(f"""
        SELECT Z, A, state, level_keV
        FROM read_parquet('{nuc_path}')
        WHERE state != '' AND Z > 0 AND level_keV IS NOT NULL
        ORDER BY Z, A, state
    """).fetchall()

    result: dict[tuple[int, int], list[tuple[str, float]]] = {}
    for z, a, state, level_keV in rows:
        result.setdefault((int(z), int(a)), []).append((state, float(level_keV)))
    return result


def _rescue_it_orphan_isomers(
    db: duckdb.DuckDBPyConnection,
    decay_path: Path,
    data_dir: Path,
    m_level_lookup: dict[tuple[int, int], float],
) -> list[tuple[int, int, str, float, float | None, str, float]]:
    """Reconstruct correct state labels for IAEA-fetcher's IT-from-ground rows.

    The IAEA LiveChart fetcher (deleted in commit e84ba66 alongside the v1.0
    package restructure) lost the parent-state field on Isomeric-Transition
    decay rows for higher-order isomers (m2, m3, ...). decay.parquet ships
    49 of these as `state='' AND decay_mode='IT'` — semantically impossible.
    For each such row, every (Z,A) already has a sibling `state='m'` entry
    in decay.parquet, so the broken row is the next isomer up (m2 in all
    observed cases). See issue #63 for the empirical breakdown.

    For each IT-orphan we synthesize a corrected isomer entry:
      * state: next available label after the existing m (m2 in practice).
      * level_keV: extracted from radiation.parent_level_keV by picking the
        parent-level value that does NOT match the existing m's level_keV.
        If no candidate is available, level_keV stays None.
      * half_life_s, decay_mode, branching: copied from the broken row.

    Returns list of (Z, A, state, half_life_s, level_keV, decay_mode, branching).
    """
    rad_dir = data_dir / "meta" / "ensdf" / "radiation"

    # All IT-from-ground rows (the broken set).
    broken = db.sql(f"""
        SELECT Z, A, half_life_s, decay_mode, branching
        FROM read_parquet('{decay_path}')
        WHERE state = '' AND decay_mode = 'IT' AND Z > 0
    """).fetchall()

    # All distinct radiation parent_level_keV per (Z,A).
    rad_pls: dict[tuple[int, int], list[float]] = {}
    if rad_dir.exists():
        for z, a, pl_keV in db.sql(f"""
            SELECT Z, A, parent_level_keV
            FROM read_parquet('{rad_dir}/*.parquet')
            WHERE parent_level_keV > 0 AND Z > 0
        """).fetchall():
            rad_pls.setdefault((int(z), int(a)), []).append(float(pl_keV))

    # All existing isomer labels per (Z,A) from decay.parquet — we must not
    # collide with these when assigning the rescued label.
    existing_labels: dict[tuple[int, int], set[str]] = {}
    for z, a, state in db.sql(f"""
        SELECT DISTINCT Z, A, state FROM read_parquet('{decay_path}')
        WHERE state != '' AND Z > 0
    """).fetchall():
        existing_labels.setdefault((int(z), int(a)), set()).add(state)

    rescued: list[tuple[int, int, str, float, float | None, str, float]] = []
    for z, a, hl, decay_mode, branching in broken:
        z, a = int(z), int(a)
        labels = existing_labels.get((z, a), set())

        # Next-available label in canonical m, m2, m3, ... order.
        next_label = "m"
        i = 1
        while next_label in labels:
            i += 1
            next_label = f"m{i}"

        # Pick a level_keV from radiation that doesn't match the existing m
        # level. Tolerance: 1 keV (matches the assigner's drift envelope for
        # near-degenerate isomer detection).
        m_level = m_level_lookup.get((z, a))
        candidates = sorted(set(rad_pls.get((z, a), [])))
        if m_level is not None:
            candidates = [pl_keV for pl_keV in candidates if abs(pl_keV - m_level) > 1.0]
        # When no m_level is known, we can't disambiguate — leave level NULL
        # rather than guess. The assigner will then skip these rows in
        # nearest-isomer matching, which is safe.
        rescued_level = candidates[0] if candidates and m_level is not None else None

        rescued.append(
            (
                z,
                a,
                next_label,
                float(hl) if hl is not None else None,
                rescued_level,
                decay_mode,
                float(branching) if branching is not None else None,
            )
        )

    return rescued


if __name__ == "__main__":
    build()
