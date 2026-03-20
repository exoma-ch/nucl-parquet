"""DuckDB loader for nucl-parquet data.

Registers all Parquet files as lazy DuckDB views for zero-copy querying
with predicate pushdown. No data is loaded into memory until queried.

Usage:
    import nucl_parquet
    db = nucl_parquet.connect()

    # Cross-section query:
    db.sql("SELECT * FROM tendl_2024 WHERE target_A=63 AND residual_Z=30")

    # Compare all libraries:
    db.sql("SELECT library, energy_MeV, xs_mb FROM xs WHERE target_A=63 AND residual_Z=30")

    # Decay radiation:
    db.sql("SELECT * FROM radiation WHERE Z=27 AND A=60 AND rad_type='gamma'")

    # Gamma coincidences:
    db.sql("SELECT * FROM coincidences WHERE Z=27 AND A=60")

    # Decay chain (recursive):
    db.sql(nucl_parquet.DECAY_CHAIN_SQL, params={"parent_z": 92, "parent_a": 238})
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import numpy as np

from .download import data_dir as _resolve_data_dir

# Element symbol → Z lookup for dynamic heavy-ion projectile resolution
_SYMBOL_TO_Z: dict[str, int] = {
    sym.lower(): z for z, sym in enumerate([
        "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
        "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
        "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U",
    ], start=1)
}


def connect(data_dir: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with all nucl-parquet data registered as views.

    Args:
        data_dir: Path to the nucl-parquet data directory.
                  Defaults to automatic resolution via download.data_dir().

    Returns:
        A DuckDB connection with lazy Parquet views.
    """
    if data_dir is not None:
        data_dir = Path(data_dir)
    else:
        data_dir = _resolve_data_dir()

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

        data_type = lib_info.get("data_type", "cross_sections")
        db.execute(f"""
            CREATE VIEW {view_name} AS
            SELECT *, '{lib_key}' AS library
            FROM read_parquet('{glob_path}', filename=true)
        """)
        if data_type == "cross_sections":
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
    _register_parquet(db, data_dir / "stopping" / "catima.parquet", "catima_stopping")

    # --- ENSDF data ---
    _register_parquet(db, data_dir / "meta" / "ensdf" / "ground_states.parquet", "ground_states")
    _register_glob(db, data_dir / "meta" / "ensdf" / "gammas", "ensdf_gammas")
    _register_glob(db, data_dir / "meta" / "ensdf" / "levels", "ensdf_levels")
    _register_glob(db, data_dir / "meta" / "ensdf" / "radiation", "radiation")
    _register_glob(db, data_dir / "meta" / "ensdf" / "coincidences", "coincidences")

    # --- Dose constants ---
    _register_parquet(db, data_dir / "meta" / "dose_constants.parquet", "dose_constants")

    # --- XCOM photon attenuation ---
    _register_parquet(db, data_dir / "meta" / "xcom_elements.parquet", "xcom_elements")
    _register_parquet(db, data_dir / "meta" / "xcom_compounds.parquet", "xcom_compounds")

    # --- EPDL97 photon interaction data ---
    _register_glob(db, data_dir / "meta" / "epdl97" / "photon_xs", "epdl_photon_xs")
    _register_glob(db, data_dir / "meta" / "epdl97" / "form_factors", "epdl_form_factors")
    _register_glob(db, data_dir / "meta" / "epdl97" / "scattering_fn", "epdl_scattering_fn")
    _register_glob(db, data_dir / "meta" / "epdl97" / "anomalous", "epdl_anomalous")
    _register_glob(db, data_dir / "meta" / "epdl97" / "subshell_pe", "epdl_subshell_pe")

    # --- EADL atomic relaxation / fluorescence ---
    _register_glob(db, data_dir / "meta" / "eadl", "eadl_transitions")

    # --- EEDL electron interaction data ---
    _register_glob(db, data_dir / "meta" / "eedl", "eedl_electron_xs")

    return db


def _register_parquet(
    db: duckdb.DuckDBPyConnection,
    path: Path,
    view_name: str,
) -> None:
    """Register a single Parquet file as a view if it exists."""
    if path.exists():
        db.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{path}')")


def _register_glob(
    db: duckdb.DuckDBPyConnection,
    directory: Path,
    view_name: str,
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
# (A, Z, source) for light projectiles covered by NIST tables
_PROJECTILES: dict[str, tuple[int, int, str]] = {
    "p":   (1, 1,  "PSTAR"),
    "d":   (2, 1,  "PSTAR"),
    "t":   (3, 1,  "PSTAR"),
    "h":   (3, 2,  "ASTAR"),  # 3He
    "he3": (3, 2,  "ASTAR"),
    "a":   (4, 2,  "ASTAR"),
    "he4": (4, 2,  "ASTAR"),
    "e":   (0, -1, "ESTAR"),  # electron
    "e-":  (0, -1, "ESTAR"),
}

_CATIMA_PATTERN = re.compile(r"^([a-z]+)(\d+)$")


def _resolve_projectile(name: str) -> tuple[int, int, str]:
    """Return (A, proj_Z, source) for a projectile name.

    Light projectiles (p, d, t, h/he3, a/he4, e) use NIST tables.
    Heavy ions (e.g. 'c12', 'pb208', 'xe132') use the catima table;
    any isotope of element Z works since catima stores data in MeV/u.
    """
    key = name.lower()
    if key in _PROJECTILES:
        return _PROJECTILES[key]

    m = _CATIMA_PATTERN.match(key)
    if m:
        sym, a = m.group(1), int(m.group(2))
        z = _SYMBOL_TO_Z.get(sym)
        if z is not None:
            return (a, z, "catima")

    raise KeyError(f"Unknown projectile {name!r}. Use 'p','d','t','h','a','e' or e.g. 'c12','pb208'.")

# Cache: (source, target_Z) -> (log_E, log_S) arrays
_stopping_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

# Cache: (proj_Z, target_Z) -> (log_E_MeV_u, log_S) arrays
_catima_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}


def _get_stopping_table(
    db: duckdb.DuckDBPyConnection,
    source: str,
    target_Z: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Get log-log stopping power arrays for an element, cached."""
    key = (source, target_Z)
    if key not in _stopping_cache:
        result = db.sql(
            "SELECT energy_MeV, dedx FROM stopping WHERE source = $src AND target_Z = $z ORDER BY energy_MeV",
            params={"src": source, "z": target_Z},
        ).fetchnumpy()
        E = result["energy_MeV"]
        S = result["dedx"]
        if len(E) == 0:
            _stopping_cache[key] = (np.array([]), np.array([]))
        else:
            _stopping_cache[key] = (np.log(E), np.log(S))
    return _stopping_cache[key]


def _get_catima_table(
    db: duckdb.DuckDBPyConnection,
    proj_Z: int,
    target_Z: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Get log-log catima stopping arrays (energy in MeV/u), cached."""
    key = (proj_Z, target_Z)
    if key not in _catima_cache:
        result = db.sql(
            "SELECT energy_MeV_u, dedx FROM catima_stopping WHERE proj_Z = $pz AND target_Z = $tz ORDER BY energy_MeV_u",
            params={"pz": proj_Z, "tz": target_Z},
        ).fetchnumpy()
        E = result["energy_MeV_u"]
        S = result["dedx"]
        if len(E) == 0:
            _catima_cache[key] = (np.array([]), np.array([]))
        else:
            _catima_cache[key] = (np.log(E), np.log(S))
    return _catima_cache[key]


def _interp_loglog(
    log_E: np.ndarray,
    log_S: np.ndarray,
    energy_MeV: np.ndarray,
) -> np.ndarray:
    """Log-log interpolation of stopping power."""
    return np.exp(np.interp(np.log(energy_MeV), log_E, log_S))


def elemental_dedx(
    db: duckdb.DuckDBPyConnection,
    projectile: str,
    target_Z: int,
    energy_MeV: float | np.ndarray,
) -> np.ndarray:
    """Mass stopping power [MeV cm2/g] for a projectile in a pure element.

    Supports all projectiles:
    - Light ions via NIST tables: p, d, t, h/he3, a/he4, e/e-
    - Any heavy ion via catima: 'c12', 'pb208', 'xe132', 'fe56', etc.
      Any isotope of a given element works — catima stores data in MeV/u
      and the lookup divides total energy by A automatically.

    Args:
        db: DuckDB connection from connect().
        projectile: Projectile name. Light ions: 'p','d','t','h','a','e'.
                    Heavy ions: element symbol + mass number, e.g. 'c12',
                    'pb208', 'xe132' (any isotope of Z=1-92).
        target_Z: Target element atomic number (1-92).
        energy_MeV: Total projectile kinetic energy [MeV].

    Returns:
        Mass stopping power [MeV cm2/g].
    """
    energy_MeV = np.atleast_1d(np.asarray(energy_MeV, dtype=float))
    proj_A, proj_Z, ref_source = _resolve_projectile(projectile)

    if ref_source == "catima":
        log_E, log_S = _get_catima_table(db, proj_Z, target_Z)
        if len(log_E) == 0:
            return np.full_like(energy_MeV, np.nan)
        # catima table is in MeV/u — convert total MeV by dividing by A
        return _interp_loglog(log_E, log_S, energy_MeV / proj_A)

    log_E, log_S = _get_stopping_table(db, ref_source, target_Z)
    if len(log_E) == 0:
        return np.full_like(energy_MeV, np.nan)

    if ref_source == "ESTAR":
        return _interp_loglog(log_E, log_S, energy_MeV)
    elif ref_source == "PSTAR":
        # velocity-scale for d/t: same velocity → E_p = E_proj / A
        return _interp_loglog(log_E, log_S, energy_MeV / proj_A)
    else:
        # ASTAR for alpha (A=4); velocity-scale for 3He: E_alpha = E_proj * (4/A)
        return _interp_loglog(log_E, log_S, energy_MeV * (4.0 / proj_A))


def compound_dedx(
    db: duckdb.DuckDBPyConnection,
    projectile: str,
    composition: list[tuple[int, float]],
    energy_MeV: float | np.ndarray,
) -> np.ndarray:
    """Compound stopping power via Bragg additivity.

    S_compound(E) = sum(wi * Si(E))

    Args:
        db: DuckDB connection from connect().
        projectile: Projectile type ('p', 'd', 't', 'h', 'a').
        composition: List of (Z, mass_fraction) pairs. Should sum to ~1.0.
        energy_MeV: Projectile energy [MeV].

    Returns:
        Compound mass stopping power [MeV cm2/g].
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
    """Linear stopping power [MeV/cm] = S [MeV cm2/g] * rho [g/cm3]."""
    return compound_dedx(db, projectile, composition, energy_MeV) * density_g_cm3
