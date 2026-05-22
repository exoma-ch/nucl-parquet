"""DuckDB loader for nucl-parquet data.

Registers all Parquet files as lazy DuckDB views for zero-copy querying
with predicate pushdown. No data is loaded into memory until queried.

Usage:
    import nucl_parquet
    db = nucl_parquet.connect()

    # Cross-section query:
    db.sql("SELECT * FROM tendl_2023_iso WHERE target_A=63 AND residual_Z=30")

    # Compare all libraries:
    db.sql("SELECT library, energy_MeV, xs_mb FROM xs WHERE target_A=63 AND residual_Z=30")

    # Decay radiation:
    db.sql("SELECT * FROM radiation WHERE Z=27 AND A=60 AND rad_type='gamma'")

    # Gamma coincidences:
    db.sql("SELECT * FROM coincidences WHERE Z=27 AND A=60")

    # Decay chain (recursive):
    db.sql(nucl_parquet.DECAY_CHAIN_SQL, params={"parent_z": 92, "parent_a": 238})

    # Photon-matter interaction (v0.12+, G4EMLOW8.8):
    db.sql("SELECT * FROM photon_pe WHERE Z=82 AND shell=0")        # Pb K σ_PE
    db.sql("SELECT * FROM photon_compton WHERE Z=29")               # Cu σ_C
    db.sql("SELECT * FROM photon_pair WHERE channel='total'")       # σ_pair
    db.sql("SELECT * FROM atomic_relaxation WHERE Z=53")            # I cascade
    db.sql("SELECT * FROM fluorescence WHERE Z=82 AND vacancy_shell='K'")

    # Detailed nuclear data — NUDEX (v0.14+, G4NUDEXLIB1.0):
    db.sql("SELECT * FROM nudex_levels WHERE Z=27 AND A=60")        # Co-60 full level scheme
    db.sql("SELECT * FROM nudex_level_gammas WHERE Z=82 AND A=208") # Pb-208 transitions
    db.sql("SELECT * FROM capture_gammas WHERE Z=27 AND A=60")      # 59Co(n,γ)60Co
    db.sql("SELECT alpha FROM icc_factors WHERE Z=82 AND shell='K' AND multipolarity='E1'")
    db.sql("SELECT * FROM psf_e1 WHERE Z=82 AND A=208")             # IAEA SMLO E1 GDR
    db.sql("SELECT * FROM level_density_bfm WHERE Z=82 AND A=208")  # BFM params
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
    sym.lower(): z
    for z, sym in enumerate(
        [
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
            "Fr",
            "Ra",
            "Ac",
            "Th",
            "Pa",
            "U",
        ],
        start=1,
    )
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
        # Some catalog entries (e.g. strata-data-nuclear) are build-time
        # provenance records, not queryable cross-section directories — they
        # have no `path`. Skip them here; the loader's job is to mount
        # queryable libraries, not enumerate every catalog entry.
        if "path" not in lib_info:
            continue
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

    # --- Catalog-driven view registration ---
    # All views declared in catalog.json::views — single source of truth.
    # New data tables become queryable by adding an entry to catalog.json,
    # no code changes needed in any client (Python, TypeScript, Rust).
    for view_name, view_def in catalog.get("views", {}).items():
        view_path = view_def["path"]
        view_type = view_def.get("type", "file")
        if view_type == "glob":
            _register_glob(db, data_dir / view_path, view_name)
        else:
            _register_parquet(db, data_dir / view_path, view_name)

    # --- Special views that need logic beyond simple registration ---

    # ground_states: when nuclides.parquet exists, override the file-based
    # ground_states view with a filtered view of nuclides (state='').
    nuclides_path = data_dir / "meta" / "ensdf" / "nuclides.parquet"
    if nuclides_path.exists():
        db.execute("CREATE OR REPLACE VIEW ground_states AS SELECT * FROM nuclides WHERE state = ''")

    # EADL aliases: eadl_transitions (v0.11 compat) + fluorescence (radiative subset)
    eadl_dir = data_dir / "meta" / "eadl"
    if eadl_dir.exists() and list(eadl_dir.glob("*.parquet")):
        db.execute("CREATE VIEW eadl_transitions AS SELECT * FROM atomic_relaxation")
        db.execute("CREATE VIEW fluorescence AS SELECT * FROM atomic_relaxation WHERE transition_type = 'radiative'")

    return db


def _register_parquet(
    db: duckdb.DuckDBPyConnection,
    path: Path,
    view_name: str,
) -> None:
    """Register a single Parquet file as a view, fetching lazily if needed."""
    if not path.exists():
        _try_lazy_fetch(path)
    if path.exists():
        db.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{path}')")


def _register_glob(
    db: duckdb.DuckDBPyConnection,
    directory: Path,
    view_name: str,
) -> None:
    """Register a directory of per-element Parquet files as a single view."""
    if not directory.exists():
        _try_lazy_fetch_glob(directory)
    if directory.exists() and list(directory.glob("*.parquet")):
        glob_path = str(directory / "*.parquet")
        db.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_parquet('{glob_path}')")


def _try_lazy_fetch(path: Path) -> None:
    """If lazy fetch is configured, download a single missing file."""
    from .download import fetch_file

    # Walk up to find data root (contains catalog.json)
    for parent in path.parents:
        if (parent / "catalog.json").exists():
            rel = path.relative_to(parent).as_posix()
            try:
                fetch_file(parent, rel)
            except FileNotFoundError:
                pass  # No lazy base URL — normal tarball mode
            return


def _try_lazy_fetch_glob(directory: Path) -> None:
    """If lazy fetch is configured, download all parquet files for a glob directory.

    Note: glob views (per-element files) cannot be lazily fetched without a
    manifest listing individual files. The catalog only declares the directory
    path. For now, glob views are skipped in lazy mode — they'll be registered
    once the user fetches the full tarball or specific files are cached.
    """
    import warnings

    # Check if we're actually in lazy mode (marker file exists)
    for parent in directory.parents:
        if (parent / ".lazy_base_url").exists():
            rel = directory.relative_to(parent).as_posix()
            warnings.warn(
                f"View '{rel}' is a glob directory and cannot be lazily fetched. "
                "Run nucl_parquet.download() for the full dataset, or manually "
                "fetch the needed element files.",
                stacklevel=4,
            )
            return


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

# ---------------------------------------------------------------------------
# Gamma-line SQL constants
#
# Two flavours per query:
#   * `*_SQL`     — filter `state = $state` (default ground state). Safe default.
#   * `*_ALL_SQL` — no state filter; caller must filter downstream if needed.
#
# For most callers, prefer the `gamma_lines()` / `identify_gamma()` helper
# functions below; they mirror the Rust `DecayDb::modes(z, a, state)` API.
# ---------------------------------------------------------------------------

GAMMA_LINES_SQL = """
SELECT r.Z, r.A, r.state, n.symbol, r.energy_keV, r.intensity_pct,
       r.decay_mode, r.rad_subtype, r.dose_MeV_per_Bq_s,
       n.half_life_s
FROM radiation r
JOIN nuclides n ON r.Z = n.Z AND r.A = n.A AND r.state = n.state
WHERE r.rad_type = 'gamma'
  AND r.state = $state
  AND r.intensity_pct > $min_intensity
ORDER BY r.intensity_pct DESC
"""

GAMMA_LINES_ALL_SQL = """
SELECT r.Z, r.A, r.state, n.symbol, r.energy_keV, r.intensity_pct,
       r.decay_mode, r.rad_subtype, r.dose_MeV_per_Bq_s,
       n.half_life_s
FROM radiation r
JOIN nuclides n ON r.Z = n.Z AND r.A = n.A AND r.state = n.state
WHERE r.rad_type = 'gamma'
  AND r.intensity_pct > $min_intensity
ORDER BY r.intensity_pct DESC
"""

IDENTIFY_GAMMA_SQL = """
SELECT r.Z, r.A, r.state, n.symbol, r.energy_keV, r.intensity_pct,
       r.decay_mode, n.half_life_s,
       ABS(r.energy_keV - $energy) AS delta_keV
FROM radiation r
JOIN nuclides n ON r.Z = n.Z AND r.A = n.A AND r.state = n.state
WHERE r.rad_type = 'gamma'
  AND r.state = $state
  AND r.energy_keV BETWEEN ($energy - $tolerance) AND ($energy + $tolerance)
  AND r.intensity_pct > 0.1
ORDER BY delta_keV ASC, r.intensity_pct DESC
LIMIT 20
"""

IDENTIFY_GAMMA_ALL_SQL = """
SELECT r.Z, r.A, r.state, n.symbol, r.energy_keV, r.intensity_pct,
       r.decay_mode, n.half_life_s,
       ABS(r.energy_keV - $energy) AS delta_keV
FROM radiation r
JOIN nuclides n ON r.Z = n.Z AND r.A = n.A AND r.state = n.state
WHERE r.rad_type = 'gamma'
  AND r.energy_keV BETWEEN ($energy - $tolerance) AND ($energy + $tolerance)
  AND r.intensity_pct > 0.1
ORDER BY delta_keV ASC, r.intensity_pct DESC
LIMIT 20
"""

# NOTE: `coincidences/*.parquet` carries both γ-γ cascade pairs and mixed-
# emission pairs (β/EC X-ray/Auger/511 keV annihilation ⊗ γ) since #170.
# Mixed rows have NULL gamma_energy_keV/coinc_energy_keV but populated
# emission{1,2}_* columns; the `WHERE emission1_rad_type='gamma' AND
# emission2_rad_type='gamma'` filter below preserves γ-γ-only behavior.
#
# `$state` still scopes the radiation intensity lookup. For metastable
# parents whose cascades differ from the ground state, prefer the
# `coincidences()` Python helper (filters on `parent_state` + `parent_decay_mode`
# directly).
COINCIDENCE_SQL = """
SELECT DISTINCT
       c.gamma_energy_keV AS E_gamma_1,
       c.coinc_energy_keV AS E_gamma_2,
       r1.intensity_pct   AS intensity_1,
       r2.intensity_pct   AS intensity_2,
       ROUND(r1.intensity_pct / 100.0 * r2.intensity_pct / 100.0 * 100, 6) AS coinc_prob_pct
FROM coincidences c
LEFT JOIN (
    SELECT Z, A, state, energy_keV, MAX(intensity_pct) AS intensity_pct
    FROM radiation WHERE rad_type = 'gamma'
    GROUP BY Z, A, state, energy_keV
) r1 ON c.Z = r1.Z AND c.A = r1.A AND r1.state = $state
    AND ABS(c.gamma_energy_keV - r1.energy_keV) < 0.5
LEFT JOIN (
    SELECT Z, A, state, energy_keV, MAX(intensity_pct) AS intensity_pct
    FROM radiation WHERE rad_type = 'gamma'
    GROUP BY Z, A, state, energy_keV
) r2 ON c.Z = r2.Z AND c.A = r2.A AND r2.state = $state
    AND ABS(c.coinc_energy_keV - r2.energy_keV) < 0.5
WHERE c.Z = $z AND c.A = $a
  AND COALESCE(c.parent_state, '') = $state
  AND c.emission1_rad_type = 'gamma' AND c.emission2_rad_type = 'gamma'
  AND c.gamma_energy_keV < c.coinc_energy_keV  -- avoid symmetric duplicates
ORDER BY coinc_prob_pct DESC NULLS LAST
"""


# ---------------------------------------------------------------------------
# Gamma-line helper functions — blessed API (mirrors Rust `DecayDb::modes`).
# ---------------------------------------------------------------------------


def gamma_lines(
    db: duckdb.DuckDBPyConnection,
    z: int | None = None,
    a: int | None = None,
    state: str = "",
    min_intensity: float = 0.0,
) -> duckdb.DuckDBPyRelation:
    """Gamma lines for a parent nuclide, scoped to a single nuclear state.

    Mirrors the Rust `DecayDb::modes(z, a, state)` shape: `state=""` is
    ground state, `"m"`/`"m2"` are isomeric states. An aged calibration
    source corresponds to `state=""`; a freshly activated isomer source to
    `state="m"`.

    Returns a DuckDB relation — call `.pl()` for Polars, `.df()` for Pandas,
    `.fetchall()` for tuples, `.arrow()` for an Arrow table.
    """
    where = ["r.rad_type = 'gamma'", "r.state = ?", "r.intensity_pct > ?"]
    params: list[object] = [state, float(min_intensity)]
    if z is not None:
        where.append("r.Z = ?")
        params.append(int(z))
    if a is not None:
        where.append("r.A = ?")
        params.append(int(a))
    sql = f"""
        SELECT r.Z, r.A, r.state, n.symbol, r.energy_keV, r.intensity_pct,
               r.decay_mode, r.rad_subtype, r.dose_MeV_per_Bq_s,
               n.half_life_s
        FROM radiation r
        JOIN nuclides n ON r.Z = n.Z AND r.A = n.A AND r.state = n.state
        WHERE {" AND ".join(where)}
        ORDER BY r.intensity_pct DESC
    """
    return db.sql(sql, params=params)


def coincidences(
    db: duckdb.DuckDBPyConnection,
    z: int,
    a: int,
    parent_state: str = "",
    parent_decay_mode: str | None = None,
    emission1_rad_type: str | None = None,
    emission2_rad_type: str | None = None,
    min_intensity: float = 0.0,
) -> duckdb.DuckDBPyRelation:
    """Coincidence pairs for a parent nuclide, filterable by emission type (#170).

    Each row is one (emission₁, emission₂) pair from a single parent decay.

    - ``parent_state``: ``""`` (ground) | ``"m"`` (isomer) — selects which parent
      state's decay channels populate the cascade.
    - ``parent_decay_mode``: ``"beta-"`` | ``"beta+"`` | ``"KshellEC"`` |
      ``"LshellEC"`` | ... — restrict to a single parent decay channel.
    - ``emission{1,2}_rad_type``: ``"gamma"`` | ``"beta"`` | ``"xray"`` |
      ``"auger"`` | ``"annihilation_511"`` — filter by emission kind. Pass
      ``"gamma"`` for both to recover legacy γ-γ-only behavior.
    - ``min_intensity``: filter on ``pair_intensity`` (relative — see schema docs).

    Returns a DuckDB relation; call ``.pl()`` / ``.df()`` / ``.arrow()`` as usual.
    """
    where = ["c.Z = ?", "c.A = ?", "c.pair_intensity > ?"]
    params: list[object] = [int(z), int(a), float(min_intensity)]
    # Parent-state filter — coalesce NULL to '' so γ-γ rows without a fed-level
    # match (parent_decay_mode unknown) still surface when caller asks for "".
    where.append("COALESCE(c.parent_state, '') = ?")
    params.append(parent_state)
    if parent_decay_mode is not None:
        where.append("c.parent_decay_mode = ?")
        params.append(parent_decay_mode)
    if emission1_rad_type is not None:
        where.append("c.emission1_rad_type = ?")
        params.append(emission1_rad_type)
    if emission2_rad_type is not None:
        where.append("c.emission2_rad_type = ?")
        params.append(emission2_rad_type)
    sql = f"""
        SELECT c.Z, c.A, c.parent_state, c.parent_decay_mode, c.daughter_ex_keV,
               c.emission1_rad_type, c.emission1_energy_keV, c.emission1_intensity, c.emission1_shell,
               c.emission2_rad_type, c.emission2_energy_keV, c.emission2_intensity, c.emission2_shell,
               c.pair_intensity
        FROM coincidences c
        WHERE {" AND ".join(where)}
        ORDER BY c.pair_intensity DESC NULLS LAST
    """
    return db.sql(sql, params=params)


def identify_gamma(
    db: duckdb.DuckDBPyConnection,
    energy: float,
    tolerance: float = 2.0,
    state: str = "",
    min_intensity: float = 0.1,
) -> duckdb.DuckDBPyRelation:
    """Candidate nuclides emitting a gamma near `energy` (keV), scoped by state.

    Default `state=""` (ground) is correct for aged calibration sources and
    most spectroscopy workflows. Pass `state="m"` etc. for isomer lookups.
    """
    sql = """
        SELECT r.Z, r.A, r.state, n.symbol, r.energy_keV, r.intensity_pct,
               r.decay_mode, n.half_life_s,
               ABS(r.energy_keV - ?) AS delta_keV
        FROM radiation r
        JOIN nuclides n ON r.Z = n.Z AND r.A = n.A AND r.state = n.state
        WHERE r.rad_type = 'gamma'
          AND r.state = ?
          AND r.energy_keV BETWEEN (? - ?) AND (? + ?)
          AND r.intensity_pct > ?
        ORDER BY delta_keV ASC, r.intensity_pct DESC
        LIMIT 20
    """
    return db.sql(
        sql,
        params=[
            float(energy),
            state,
            float(energy),
            float(tolerance),
            float(energy),
            float(tolerance),
            float(min_intensity),
        ],
    )


def summing_partners(
    db: duckdb.DuckDBPyConnection,
    z: int,
    a: int,
    primary_energy_keV: float | None = None,
    tolerance_keV: float = 0.5,
    parent_state: str = "",
    emission1_rad_type: str | None = None,
) -> duckdb.DuckDBPyRelation:
    """ICC-corrected summing partners for HPGe TCS corrections (#177).

    Returns all emission pairs that can sum in a close-geometry HPGe detector
    for the specified nuclide. Each row carries ``icc_correction_factor`` and
    ``pure_emission_joint_intensity`` pre-computed.

    Parameters
    ----------
    db
        DuckDB connection from :func:`connect`.
    z, a
        Daughter nucleus (filing convention — Co-60 cascades are under Ni-60).
    primary_energy_keV
        If given, filter to pairs where *either* emission matches this energy
        within ``tolerance_keV``. Typical use: find all summing partners of a
        specific gamma line.
    tolerance_keV
        Energy-match tolerance (default 0.5 keV, typical HPGe FWHM at ~1 MeV).
    parent_state
        ``""`` (ground) | ``"m"`` | ``"m2"``.
    emission1_rad_type
        Filter emission side 1: ``"gamma"`` | ``"xray"`` | ``"auger"``.
        Pass ``"gamma"`` for γ-γ only; omit for all pair types.

    Returns
    -------
    DuckDB relation; call ``.pl()`` / ``.df()`` / ``.arrow()`` as usual.
    """
    where = ["s.Z = ?", "s.A = ?", "COALESCE(s.parent_state, '') = ?"]
    params: list[object] = [int(z), int(a), parent_state]

    if primary_energy_keV is not None:
        where.append("(ABS(s.emission1_energy_keV - ?) < ? OR ABS(s.emission2_energy_keV - ?) < ?)")
        params.extend(
            [float(primary_energy_keV), float(tolerance_keV), float(primary_energy_keV), float(tolerance_keV)]
        )

    if emission1_rad_type is not None:
        where.append("s.emission1_rad_type = ?")
        params.append(emission1_rad_type)

    sql = f"""
        SELECT s.*
        FROM summing_partners s
        WHERE {" AND ".join(where)}
        ORDER BY s.pure_emission_joint_intensity DESC NULLS LAST
    """
    return db.sql(sql, params=params)


def emissions(
    db: duckdb.DuckDBPyConnection,
    parent_z: int,
    parent_a: int,
    parent_state: str = "",
    decay_mode: str | None = None,
    energy_keV: float | None = None,
    tolerance_keV: float = 0.5,
    min_intensity_pct: float = 0.0,
) -> duckdb.DuckDBPyRelation:
    """Absolute per-decay photon emission intensities (#196).

    Returns all gamma emissions for the specified parent nuclide with absolute
    intensities (NuDat-equivalent). Filed by parent, not daughter.

    Parameters
    ----------
    db
        DuckDB connection from :func:`connect`.
    parent_z, parent_a
        Parent nucleus (the decaying nuclide — e.g. Z=27, A=60 for Co-60).
    parent_state
        ``""`` (ground) | ``"m"`` | ``"m2"``.
    decay_mode
        Filter by decay mode: ``"beta-"`` | ``"KshellEC"`` | ``"IT"`` | etc.
    energy_keV
        If given, filter to gammas near this energy within ``tolerance_keV``.
    tolerance_keV
        Energy-match tolerance (default 0.5 keV).
    min_intensity_pct
        Minimum absolute intensity (%) to include (default 0 = all).

    Returns
    -------
    DuckDB relation; call ``.pl()`` / ``.df()`` / ``.arrow()`` as usual.
    """
    where = ["e.parent_Z = ?", "e.parent_A = ?", "e.parent_state = ?"]
    params: list[object] = [int(parent_z), int(parent_a), parent_state]

    if decay_mode is not None:
        where.append("e.decay_mode = ?")
        params.append(decay_mode)

    if energy_keV is not None:
        where.append("ABS(e.energy_keV - ?) < ?")
        params.extend([float(energy_keV), float(tolerance_keV)])

    if min_intensity_pct > 0:
        where.append("e.intensity_pct >= ?")
        params.append(float(min_intensity_pct))

    sql = f"""
        SELECT e.*
        FROM emissions e
        WHERE {" AND ".join(where)}
        ORDER BY e.intensity_pct DESC
    """
    return db.sql(sql, params=params)


# ---------------------------------------------------------------------------
# Stopping power computation
# ---------------------------------------------------------------------------

# Projectile properties: (A, Z, reference_source)
# (A, Z, source) for light projectiles covered by NIST tables
_PROJECTILES: dict[str, tuple[int, int, str]] = {
    "p": (1, 1, "PSTAR"),
    "d": (2, 1, "PSTAR"),
    "t": (3, 1, "PSTAR"),
    # α uses NIST ASTAR (ICRU-49 reference; reproducible via build_stopping.py).
    # ³He has no NIST table — routes through catima. Both pre-#137 paths used
    # the broken ASTAR.parquet that was Z²-scaled from PSTAR at the wrong axis.
    "h": (3, 2, "catima"),  # 3He
    "he3": (3, 2, "catima"),
    "a": (4, 2, "ASTAR"),
    "he4": (4, 2, "ASTAR"),
    "e": (0, -1, "ESTAR"),  # electron
    "e-": (0, -1, "ESTAR"),
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

# Cache: (proj_Z, proj_A, target_Z) -> (log_E_MeV_u, log_S) arrays
_catima_cache: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}

# Cache: proj_Z -> sorted list of available proj_A values (for error messages)
_catima_isotopes_cache: dict[int, list[int]] = {}


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


def _get_catima_isotopes(
    db: duckdb.DuckDBPyConnection,
    proj_Z: int,
) -> list[int]:
    """Return sorted list of proj_A values available in catima_stopping for proj_Z."""
    if proj_Z not in _catima_isotopes_cache:
        result = db.sql(
            "SELECT DISTINCT proj_A FROM catima_stopping WHERE proj_Z = $pz ORDER BY proj_A",
            params={"pz": proj_Z},
        ).fetchall()
        _catima_isotopes_cache[proj_Z] = [int(row[0]) for row in result]
    return _catima_isotopes_cache[proj_Z]


def _get_catima_table(
    db: duckdb.DuckDBPyConnection,
    proj_Z: int,
    proj_A: int,
    target_Z: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Get log-log catima stopping arrays (energy in MeV/u) for a specific isotope, cached."""
    key = (proj_Z, proj_A, target_Z)
    if key not in _catima_cache:
        result = db.sql(
            "SELECT energy_MeV_u, dedx FROM catima_stopping "
            "WHERE proj_Z = $pz AND proj_A = $pa AND target_Z = $tz "
            "ORDER BY energy_MeV_u",
            params={"pz": proj_Z, "pa": proj_A, "tz": target_Z},
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

    Source routing (post-#137):
    - p, d, t          → NIST PSTAR (ICRU-49); d/t velocity-scaled (E_p = E/A)
    - α (a, he4)       → NIST ASTAR (ICRU-49)
    - ³He (h, he3)     → catima (no NIST ³He table)
    - e, e-            → NIST ESTAR (ICRU-37)
    - heavy ions       → catima (any isotope of Z=1-92)

    NIST PSTAR/ASTAR only publishes 25 elemental materials; for target_Z not
    in NIST's table (e.g. Tc, Pm, Po, Rn), the loader falls back to catima.

    Catima tables are tabulated per-isotope (proj_Z, proj_A): the energy axis
    is MeV/u, but the rows differ between isotopes of the same Z at low energy
    where nuclear stopping (reduced-mass-dependent) dominates. Above ~0.1 MeV/u
    isotope differences are <0.1%; below ~0.01 MeV/u they reach up to ~15% for
    light projectiles. Querying an A not present in the table raises KeyError.

    Args:
        db: DuckDB connection from connect().
        projectile: Projectile name. Light ions: 'p','d','t','h','a','e'.
                    Heavy ions: element symbol + mass number, e.g. 'c12',
                    'pb208', 'xe132'. Must be a tabulated isotope; see
                    nucl_parquet.build_heavy_ions for the inventory.
        target_Z: Target element atomic number (1-92).
        energy_MeV: Total projectile kinetic energy [MeV].

    Returns:
        Mass stopping power [MeV cm2/g].

    Raises:
        KeyError: if `projectile` is unknown or its (Z, A) is not in the
            catima table.
    """
    energy_MeV = np.atleast_1d(np.asarray(energy_MeV, dtype=float))
    proj_A, proj_Z, ref_source = _resolve_projectile(projectile)

    if ref_source == "catima":
        available_As = _get_catima_isotopes(db, proj_Z)
        if proj_A not in available_As:
            raise KeyError(
                f"catima table has no row for Z={proj_Z} A={proj_A}. "
                f"Available isotopes for Z={proj_Z}: {available_As}. "
                "Use a tabulated isotope or rebuild via build_heavy_ions.py "
                "with an extended allowlist."
            )
        log_E, log_S = _get_catima_table(db, proj_Z, proj_A, target_Z)
        if len(log_E) == 0:
            return np.full_like(energy_MeV, np.nan)
        # catima table is in MeV/u — convert total MeV by dividing by A
        return _interp_loglog(log_E, log_S, energy_MeV / proj_A)

    log_E, log_S = _get_stopping_table(db, ref_source, target_Z)
    if len(log_E) == 0:
        # NIST tables only cover 25 elemental materials; fall back to catima
        # (Bethe-Bloch) for elements NIST doesn't publish (Tc, Pm, Po, Rn, …).
        if ref_source in ("PSTAR", "ASTAR"):
            available_As = _get_catima_isotopes(db, proj_Z)
            if proj_A in available_As:
                log_E_c, log_S_c = _get_catima_table(db, proj_Z, proj_A, target_Z)
                if len(log_E_c) > 0:
                    return _interp_loglog(log_E_c, log_S_c, energy_MeV / proj_A)
        return np.full_like(energy_MeV, np.nan)

    if ref_source == "ESTAR":
        return _interp_loglog(log_E, log_S, energy_MeV)
    if ref_source == "ASTAR":
        # ASTAR is keyed on total α kinetic energy (A=4); no rescaling needed.
        return _interp_loglog(log_E, log_S, energy_MeV)
    # PSTAR: velocity-scale for d/t (same velocity → E_p = E / A_proj).
    return _interp_loglog(log_E, log_S, energy_MeV / proj_A)


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
        projectile: Projectile name. Light ions: 'p','d','t','h','a','e'.
                    Heavy ions: e.g. 'c12','pb208','xe132' (any isotope of Z=1-92).
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
