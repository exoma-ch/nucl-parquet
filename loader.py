"""DuckDB loader for nucl-parquet data.

Registers all Parquet files as lazy DuckDB views for zero-copy querying
with predicate pushdown. No data is loaded into memory until queried.

Usage:
    import loader
    db = loader.connect()

    # Cross-section query:
    db.sql("SELECT * FROM tendl_2024 WHERE target_A=63 AND residual_Z=30")

    # Compare all libraries:
    db.sql("SELECT library, energy_MeV, xs_mb FROM xs WHERE target_A=63 AND residual_Z=30")

    # Decay radiation:
    db.sql("SELECT * FROM radiation WHERE Z=27 AND A=60 AND rad_type='gamma'")

    # Gamma coincidences:
    db.sql("SELECT * FROM coincidences WHERE Z=27 AND A=60")

    # Decay chain (recursive):
    db.sql(loader.DECAY_CHAIN_SQL, params={"parent_z": 92, "parent_a": 238})
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).parent


def connect(data_dir: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with all nucl-parquet data registered as views.

    Args:
        data_dir: Path to the nucl-parquet data directory.
                  Defaults to the directory containing this file.

    Returns:
        A DuckDB connection with lazy Parquet views.
    """
    data_dir = Path(data_dir) if data_dir else ROOT
    db = duckdb.connect()

    catalog_path = data_dir / "catalog.json"
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text())
    else:
        catalog = {"libraries": {}, "shared": {}}

    # --- Cross-section libraries ---
    lib_views: list[str] = []

    for lib_key, lib_info in catalog.get("libraries", {}).items():
        lib_dir = data_dir / lib_info["path"]
        if not lib_dir.exists() or not list(lib_dir.glob("*.parquet")):
            continue

        # View name: tendl-2024 -> tendl_2024, endfb-8.1 -> endfb_8_1
        view_name = lib_key.replace("-", "_").replace(".", "_")
        glob_path = str(lib_dir / "*.parquet")

        if lib_info.get("data_type") == "experimental_cross_sections":
            # EXFOR has a different schema — register as-is
            db.execute(f"""
                CREATE VIEW {view_name} AS
                SELECT *, '{lib_key}' AS library
                FROM read_parquet('{glob_path}', filename=true)
            """)
        else:
            db.execute(f"""
                CREATE VIEW {view_name} AS
                SELECT *, '{lib_key}' AS library
                FROM read_parquet('{glob_path}', filename=true)
            """)
            lib_views.append(view_name)

    # Unified xs view: UNION ALL of all evaluated libraries
    if lib_views:
        union_sql = " UNION ALL ".join(f"SELECT * FROM {v}" for v in lib_views)
        db.execute(f"CREATE VIEW xs AS {union_sql}")

    # --- Shared meta files ---
    _register_parquet(db, data_dir / "meta" / "abundances.parquet", "abundances")
    _register_parquet(db, data_dir / "meta" / "decay.parquet", "decay")
    _register_parquet(db, data_dir / "meta" / "elements.parquet", "elements")

    # --- Stopping powers ---
    _register_parquet(db, data_dir / "stopping" / "stopping.parquet", "stopping")

    # --- ENSDF data ---
    _register_parquet(db, data_dir / "meta" / "ensdf" / "ground_states.parquet", "ground_states")
    _register_glob(db, data_dir / "meta" / "ensdf" / "gammas", "ensdf_gammas")
    _register_glob(db, data_dir / "meta" / "ensdf" / "levels", "ensdf_levels")
    _register_glob(db, data_dir / "meta" / "ensdf" / "radiation", "radiation")
    _register_glob(db, data_dir / "meta" / "ensdf" / "coincidences", "coincidences")

    return db


def _register_parquet(
    db: duckdb.DuckDBPyConnection, path: Path, view_name: str,
) -> None:
    """Register a single Parquet file as a view if it exists."""
    if path.exists():
        db.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{path}')")


def _register_glob(
    db: duckdb.DuckDBPyConnection, directory: Path, view_name: str,
) -> None:
    """Register a directory of per-element Parquet files as a single view."""
    if directory.exists() and list(directory.glob("*.parquet")):
        glob_path = str(directory / "*.parquet")
        db.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{glob_path}')")


# ---------------------------------------------------------------------------
# Pre-built SQL helpers
# ---------------------------------------------------------------------------

DECAY_CHAIN_SQL = """
WITH RECURSIVE chain AS (
    -- Seed: the parent nuclide
    SELECT Z, A, symbol, half_life_s,
           decay_1 AS decay_mode, decay_1_pct AS branching_pct,
           -- Compute daughter Z,A from decay mode
           CASE WHEN decay_1 = 'A' THEN Z - 2
                WHEN decay_1 IN ('B-', 'B-N') THEN Z + 1
                WHEN decay_1 IN ('EC', 'B+', 'EC+B+') THEN Z - 1
                WHEN decay_1 = 'IT' THEN Z
                WHEN decay_1 = 'P' THEN Z - 1
                WHEN decay_1 = 'N' THEN Z
           END AS daughter_Z,
           CASE WHEN decay_1 = 'A' THEN A - 4
                WHEN decay_1 = 'B-' THEN A
                WHEN decay_1 = 'B-N' THEN A - 1
                WHEN decay_1 IN ('EC', 'B+', 'EC+B+') THEN A
                WHEN decay_1 = 'IT' THEN A
                WHEN decay_1 = 'P' THEN A - 1
                WHEN decay_1 = 'N' THEN A - 1
           END AS daughter_A,
           1 AS generation
    FROM ground_states
    WHERE Z = $parent_z AND A = $parent_a

    UNION ALL

    SELECT gs.Z, gs.A, gs.symbol, gs.half_life_s,
           gs.decay_1, gs.decay_1_pct,
           CASE WHEN gs.decay_1 = 'A' THEN gs.Z - 2
                WHEN gs.decay_1 IN ('B-', 'B-N') THEN gs.Z + 1
                WHEN gs.decay_1 IN ('EC', 'B+', 'EC+B+') THEN gs.Z - 1
                WHEN gs.decay_1 = 'IT' THEN gs.Z
                WHEN gs.decay_1 = 'P' THEN gs.Z - 1
                WHEN gs.decay_1 = 'N' THEN gs.Z
           END,
           CASE WHEN gs.decay_1 = 'A' THEN gs.A - 4
                WHEN gs.decay_1 = 'B-' THEN gs.A
                WHEN gs.decay_1 = 'B-N' THEN gs.A - 1
                WHEN gs.decay_1 IN ('EC', 'B+', 'EC+B+') THEN gs.A
                WHEN gs.decay_1 = 'IT' THEN gs.A
                WHEN gs.decay_1 = 'P' THEN gs.A - 1
                WHEN gs.decay_1 = 'N' THEN gs.A - 1
           END,
           c.generation + 1
    FROM ground_states gs
    JOIN chain c ON gs.Z = c.daughter_Z AND gs.A = c.daughter_A
    WHERE c.generation < 30
      AND c.decay_mode IS NOT NULL
      AND c.decay_mode != ''
      AND c.daughter_Z IS NOT NULL
)
SELECT Z, A, symbol, half_life_s, decay_mode, branching_pct,
       daughter_Z, daughter_A, generation
FROM chain
ORDER BY generation
"""

GAMMA_LINES_SQL = """
SELECT r.Z, r.A, gs.symbol, r.energy_keV, r.intensity_pct,
       r.decay_mode, r.rad_subtype, r.dose_MeV_per_Bq_s,
       gs.half_life_s
FROM radiation r
JOIN ground_states gs ON r.Z = gs.Z AND r.A = gs.A
WHERE r.rad_type = 'gamma'
  AND r.intensity_pct > $min_intensity
ORDER BY r.intensity_pct DESC
"""

IDENTIFY_GAMMA_SQL = """
SELECT r.Z, r.A, gs.symbol, r.energy_keV, r.intensity_pct,
       r.decay_mode, gs.half_life_s,
       ABS(r.energy_keV - $energy) AS delta_keV
FROM radiation r
JOIN ground_states gs ON r.Z = gs.Z AND r.A = gs.A
WHERE r.rad_type = 'gamma'
  AND r.energy_keV BETWEEN ($energy - $tolerance) AND ($energy + $tolerance)
  AND r.intensity_pct > 0.1
ORDER BY delta_keV ASC, r.intensity_pct DESC
LIMIT 20
"""

COINCIDENCE_SQL = """
SELECT DISTINCT
       c.gamma_energy_keV AS E_gamma_1,
       c.coinc_energy_keV AS E_gamma_2,
       r1.intensity_pct   AS intensity_1,
       r2.intensity_pct   AS intensity_2,
       ROUND(r1.intensity_pct / 100.0 * r2.intensity_pct / 100.0 * 100, 6) AS coinc_prob_pct
FROM coincidences c
LEFT JOIN (
    SELECT Z, A, energy_keV, MAX(intensity_pct) AS intensity_pct
    FROM radiation WHERE rad_type = 'gamma'
    GROUP BY Z, A, energy_keV
) r1 ON c.Z = r1.Z AND c.A = r1.A
    AND ABS(c.gamma_energy_keV - r1.energy_keV) < 0.5
LEFT JOIN (
    SELECT Z, A, energy_keV, MAX(intensity_pct) AS intensity_pct
    FROM radiation WHERE rad_type = 'gamma'
    GROUP BY Z, A, energy_keV
) r2 ON c.Z = r2.Z AND c.A = r2.A
    AND ABS(c.coinc_energy_keV - r2.energy_keV) < 0.5
WHERE c.Z = $z AND c.A = $a
  AND c.gamma_energy_keV < c.coinc_energy_keV  -- avoid symmetric duplicates
ORDER BY coinc_prob_pct DESC NULLS LAST
"""


# ---------------------------------------------------------------------------
# Stopping power computation
# ---------------------------------------------------------------------------

# Projectile properties: (A, Z, reference_source)
_PROJECTILES = {
    "p":   (1, 1, "PSTAR"),
    "d":   (2, 1, "PSTAR"),
    "t":   (3, 1, "PSTAR"),
    "h":   (3, 2, "ASTAR"),   # ³He
    "he3": (3, 2, "ASTAR"),
    "a":   (4, 2, "ASTAR"),
    "he4": (4, 2, "ASTAR"),
}

# Cache: (source, target_Z) -> (log_E, log_S) arrays
_stopping_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}


def _get_stopping_table(
    db: duckdb.DuckDBPyConnection, source: str, target_Z: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Get log-log stopping power arrays for an element, cached."""
    key = (source, target_Z)
    if key not in _stopping_cache:
        result = db.sql(
            "SELECT energy_MeV, dedx FROM stopping "
            "WHERE source = $src AND target_Z = $z ORDER BY energy_MeV",
            params={"src": source, "z": target_Z},
        ).fetchnumpy()
        E = result["energy_MeV"]
        S = result["dedx"]
        if len(E) == 0:
            _stopping_cache[key] = (np.array([]), np.array([]))
        else:
            _stopping_cache[key] = (np.log(E), np.log(S))
    return _stopping_cache[key]


def _interp_loglog(
    log_E: np.ndarray, log_S: np.ndarray, energy_MeV: np.ndarray,
) -> np.ndarray:
    """Log-log interpolation of stopping power."""
    return np.exp(np.interp(np.log(energy_MeV), log_E, log_S))


def elemental_dedx(
    db: duckdb.DuckDBPyConnection,
    projectile: str,
    target_Z: int,
    energy_MeV: float | np.ndarray,
) -> np.ndarray:
    """Mass stopping power [MeV cm²/g] for a projectile in a pure element.

    Supports p, d, t, ³He (h), α (a). Deuteron/triton are velocity-scaled
    from PSTAR; ³He is velocity-scaled from ASTAR.

    Args:
        db: DuckDB connection from connect().
        projectile: One of 'p', 'd', 't', 'h'/'he3', 'a'/'he4'.
        target_Z: Target element atomic number.
        energy_MeV: Projectile kinetic energy [MeV].

    Returns:
        Mass stopping power [MeV cm²/g].
    """
    energy_MeV = np.atleast_1d(np.asarray(energy_MeV, dtype=float))
    proj_A, proj_Z, ref_source = _PROJECTILES[projectile.lower()]
    log_E, log_S = _get_stopping_table(db, ref_source, target_Z)

    if len(log_E) == 0:
        return np.full_like(energy_MeV, np.nan)

    if ref_source == "PSTAR":
        # PSTAR is per-nucleon for protons (A=1), so for d/t:
        # same velocity means E_p = E_proj / A_proj
        lookup_E = energy_MeV / proj_A
        return _interp_loglog(log_E, log_S, lookup_E)
    else:
        # ASTAR is for alpha (A=4, Z=2), velocity-scale for ³He:
        # same velocity means E_alpha = E_proj * (4 / A_proj)
        lookup_E = energy_MeV * (4.0 / proj_A)
        return _interp_loglog(log_E, log_S, lookup_E)


def compound_dedx(
    db: duckdb.DuckDBPyConnection,
    projectile: str,
    composition: list[tuple[int, float]],
    energy_MeV: float | np.ndarray,
) -> np.ndarray:
    """Compound stopping power via Bragg additivity.

    S_compound(E) = Σ wᵢ × Sᵢ(E)

    Args:
        db: DuckDB connection from connect().
        projectile: Projectile type ('p', 'd', 't', 'h', 'a').
        composition: List of (Z, mass_fraction) pairs. Should sum to ~1.0.
        energy_MeV: Projectile energy [MeV].

    Returns:
        Compound mass stopping power [MeV cm²/g].
    """
    energy_MeV = np.atleast_1d(np.asarray(energy_MeV, dtype=float))
    total = np.zeros_like(energy_MeV)
    for Z, w in composition:
        total += w * elemental_dedx(db, projectile, Z, energy_MeV)
    return total


def linear_dedx(
    db: duckdb.DuckDBPyConnection,
    projectile: str,
    composition: list[tuple[int, float]],
    density_g_cm3: float,
    energy_MeV: float | np.ndarray,
) -> np.ndarray:
    """Linear stopping power [MeV/cm] = S [MeV cm²/g] × ρ [g/cm³]."""
    return compound_dedx(db, projectile, composition, energy_MeV) * density_g_cm3


if __name__ == "__main__":
    db = connect()

    # Print summary of registered views
    views = db.sql("SELECT table_name FROM information_schema.tables WHERE table_type='VIEW' ORDER BY table_name").fetchall()
    print(f"Registered {len(views)} views:")
    for (name,) in views:
        try:
            count = db.sql(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            print(f"  {name:25s} {count:>12,} rows")
        except Exception:
            print(f"  {name:25s} (empty or unavailable)")
