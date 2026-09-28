"""nucl-parquet MCP server — wraps the nucl_parquet client library.

Delegates all data access to ``nucl_parquet.connect()`` which registers
70+ Parquet-backed DuckDB views with lazy loading and predicate pushdown.
This is the SSoT refactor (epic #173, Sub-D #178): the MCP is a thin shell
over the client library, not a re-implementation of parquet I/O.
"""

from __future__ import annotations

import json
import re
import threading
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

import duckdb
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import nucl_parquet

# ---------------------------------------------------------------------------
# Data bootstrap
# ---------------------------------------------------------------------------

_db: duckdb.DuckDBPyConnection | None = None
_db_lock = threading.Lock()


def _get_db() -> duckdb.DuckDBPyConnection:
    """Lazy-connect to the nucl-parquet DuckDB instance (thread-safe)."""
    global _db  # noqa: PLW0603
    if _db is not None:
        return _db

    with _db_lock:
        if _db is not None:
            return _db

        # Try to find local data; download if not present.
        try:
            data_path = nucl_parquet.data_dir()
        except FileNotFoundError:
            nucl_parquet.download()
            data_path = nucl_parquet.data_dir()

        conn = nucl_parquet.connect(data_path)
        _harden(conn, data_path)
        _db = conn
        return _db


class RequestError(ToolError, ValueError):
    """A caller's mistake, reported to the MCP client with its message.

    mcp 2 surfaces a tool's exception text only for `ToolError`. Anything else
    becomes `UnexpectedToolError("Error executing tool <name>")` and the message
    ("Only read queries are allowed…", "Unknown material… Available: …") never
    reaches the caller who needs it to fix the call. Also a `ValueError`, so
    code importing these functions directly still catches what it did under mcp 1.
    """


def _harden(conn: duckdb.DuckDBPyConnection, data_path: Path) -> None:
    """Confine the connection to reading the data tree, then freeze its settings.

    `sql_query` hands callers a SQL prompt, so the connection itself must not
    be able to reach anything else. Order matters:

    * `allowed_directories` first. The views read Parquet files, which is itself
      external access, so switching access off without an exception breaks every
      tool with "Permission Error: Cannot access file".
    * `enable_external_access = false`: no files outside `data_path`, no network,
      no INSTALL.
    * Extension auto-install/auto-load off. This used to be spelled
      `allow_extensions_autoloading`, which is not a DuckDB setting. The `SET`
      raised, so the first database access failed and every data tool errored.
    * `lock_configuration` last, so a caller cannot `SET` any of this back.

    Writes *inside* `data_path` (COPY, ATTACH) are still permitted at this
    level, so `sql_query` also admits only a single read statement; see
    `_require_read_only`.
    """
    conn.execute("SET allowed_directories = [?]", [str(data_path)])
    conn.sql("SET enable_external_access = false")
    conn.sql("SET autoinstall_known_extensions = false")
    conn.sql("SET autoload_known_extensions = false")
    conn.sql("SET lock_configuration = true")


#: Leading keywords `sql_query` accepts. Necessary but not sufficient: see
#: `_require_read_only`.
_READ_KEYWORDS = frozenset({"SELECT", "WITH", "FROM", "EXPLAIN", "DESCRIBE", "SHOW", "SUMMARIZE"})
_EXPLAIN_PREFIX = re.compile(r"^\s*EXPLAIN(\s+ANALYZE)?\s+", re.IGNORECASE)


def _require_read_only(sql: str) -> None:
    """Raise unless *sql* is exactly one read statement, as DuckDB's parser sees it.

    A leading-keyword check alone is not enough. `EXPLAIN ANALYZE` *executes* the
    statement it wraps, so `EXPLAIN ANALYZE COPY (...) TO '<data_path>/x.parquet'`
    starts with an allowed word and writes into the data tree. The parser's
    statement type closes that: one statement, of type SELECT (which is also how
    DuckDB classifies DESCRIBE, SHOW and SUMMARIZE), and an EXPLAIN only when
    what it wraps is itself a single SELECT.
    """
    stripped = sql.strip()
    if not stripped:
        raise RequestError("Empty SQL query")
    first_word = stripped.split()[0].upper()
    if first_word not in _READ_KEYWORDS:
        raise RequestError(f"Only read queries are allowed (SELECT, WITH, EXPLAIN, DESCRIBE). Got: {first_word}")
    try:
        statements = duckdb.extract_statements(stripped)
    except duckdb.Error as e:
        raise RequestError(f"SQL error: {e}") from e
    if len(statements) != 1:
        raise RequestError(f"Only read queries are allowed, one statement at a time. Got {len(statements)} statements.")
    kind = statements[0].type
    if kind == duckdb.StatementType.EXPLAIN:
        _require_read_only(_EXPLAIN_PREFIX.sub("", stripped, count=1))
        return
    if kind != duckdb.StatementType.SELECT:
        raise RequestError(f"Only read queries are allowed (SELECT, WITH, EXPLAIN, DESCRIBE). Got a {kind.name} statement.")


def _records(rel: duckdb.DuckDBPyRelation) -> list[dict[str, Any]]:
    """Rows as dicts, straight from DuckDB's Python values.

    This used to round-trip through pandas, which had two problems. pandas was
    never a declared dependency, so an installed server failed every data tool
    with "'pandas' is required". And a null's survival depended on pandas
    picking a nullable dtype: a numpy int would have turned a null `residual_Z`
    into 0 (#362). `fetchall()` returns SQL NULL as `None` with no dtype choice
    in between.
    """
    columns = rel.columns
    return [dict(zip(columns, row, strict=True)) for row in rel.fetchall()]


def _materials(db: duckdb.DuckDBPyConnection) -> list[str]:
    return [m for (m,) in db.sql("SELECT DISTINCT material FROM compound_compositions ORDER BY material").fetchall()]


def _query(sql: str, params: dict[str, Any] | None = None, max_rows: int = 500) -> dict[str, Any]:
    """Run a read-only SQL query and return {total, truncated, rows}."""
    db = _get_db()
    rel = db.sql(sql, params=params or {})
    total = rel.count("*").fetchone()[0]
    rows = _records(rel.limit(max_rows))
    truncated = total > max_rows
    return {"total": total, "truncated": truncated, "rows": rows}


# ---------------------------------------------------------------------------
# Catalog (read from the actual data directory, not hardcoded)
# ---------------------------------------------------------------------------


def _load_catalog() -> dict[str, Any]:
    """Load catalog.json from the data directory."""
    data_path = nucl_parquet.data_dir()
    catalog_path = Path(data_path) / "catalog.json"
    with open(catalog_path) as f:
        return json.load(f)


_catalog: dict[str, Any] | None = None

# Kept as a module-level reference for tests that import it.
CATALOG: dict[str, Any] = {}


def _ensure_catalog() -> dict[str, Any]:
    global _catalog, CATALOG  # noqa: PLW0603
    if _catalog is None:
        _catalog = _load_catalog()
        CATALOG.update(_catalog)
    return _catalog


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

try:
    _VERSION = _pkg_version("nucl-parquet-mcp")
except PackageNotFoundError:
    _VERSION = "0.0.0-dev"

mcp = MCPServer("nucl-parquet", version=_VERSION)


# ---------------------------------------------------------------------------
# Library / cross-section tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_libraries() -> str:
    """List all available nuclear data libraries with projectiles and descriptions."""
    cat = _ensure_catalog()
    libs = [
        {
            "id": lib_id,
            "name": lib["name"],
            "description": lib["description"],
            "projectiles": lib["projectiles"],
            "version": lib["version"],
            "data_type": lib["data_type"],
        }
        for lib_id, lib in cat["libraries"].items()
        if "projectiles" in lib  # skip non-library entries like strata-data-nuclear
    ]
    return json.dumps(libs, indent=2)


@mcp.tool()
async def list_isotopes(library: str, projectile: str) -> str:
    """List available target elements for a library and projectile.

    Args:
        library: Library ID, e.g. 'tendl-2025', 'endfb-8.1'.
        projectile: Projectile type: n, p, d, t, h, a, g.
    """
    cat = _ensure_catalog()
    lib = cat["libraries"].get(library)
    if lib is None:
        raise RequestError(f"Unknown library: {library}. Use list_libraries to see options.")
    if projectile not in lib.get("projectiles", []):
        raise RequestError(
            f"Projectile '{projectile}' not in {library}. Available: {', '.join(lib.get('projectiles', []))}"
        )
    # Manifests are JSON files alongside the parquet data — read from disk.
    data_path = nucl_parquet.data_dir()
    lib_path = lib["path"]
    if not lib_path.endswith("xs/"):
        raise RequestError(f"Library {library} path '{lib_path}' does not end with 'xs/' — cannot derive manifest path")
    manifest_path = Path(data_path) / lib_path.replace("xs/", "manifest.json")
    if not manifest_path.exists():
        raise RequestError(f"Manifest not found at {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    elements = manifest.get("elements", [])
    return json.dumps(
        {"library": library, "projectile": projectile, "elements": elements, "count": len(elements)},
        indent=2,
    )


@mcp.tool()
async def get_cross_sections(
    library: str,
    projectile: str,
    element: str,
    max_rows: int = 500,
) -> str:
    """Get nuclear reaction cross-section data for a target element.

    Returns energy (MeV) and cross-section (mb) with reaction product info.

    Args:
        library: Library ID, e.g. 'tendl-2025'.
        projectile: Projectile: n, p, d, t, h, a, g.
        element: Target element symbol, e.g. 'Cu', 'Fe'.
        max_rows: Maximum rows to return (default 500).
    """
    # Validate inputs against strict patterns to prevent path traversal / injection.
    if not re.match(r"^[npdthag]$", projectile):
        raise RequestError(f"Invalid projectile: {projectile!r}. Must be one of: n, p, d, t, h, a, g")
    if not re.match(r"^[A-Z][a-z]?$", element):
        raise RequestError(f"Invalid element symbol: {element!r}. Must be 1-2 letters (e.g. 'Cu', 'Fe')")

    cat = _ensure_catalog()
    lib = cat["libraries"].get(library)
    if lib is None:
        raise RequestError(f"Unknown library: {library}")

    # Read the parquet file for this projectile + element combination.
    data_path = nucl_parquet.data_dir()
    parquet_path = Path(data_path) / f"{lib['path']}{projectile}_{element}.parquet"
    if not parquet_path.exists():
        raise RequestError(f"No data for {projectile}_{element} in {library}")
    db = _get_db()
    rel = db.sql("SELECT * FROM read_parquet($path)", params={"path": str(parquet_path)})
    total = rel.count("*").fetchone()[0]
    rows = _records(rel.limit(max_rows))
    truncated = total > max_rows

    return json.dumps(
        {
            "library": library,
            "projectile": projectile,
            "element": element,
            "total": total,
            "truncated": truncated,
            "rows": rows,
        },
        indent=2,
        default=str,
    )


# ---------------------------------------------------------------------------
# Nuclear structure tools (DuckDB views)
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_decay_data(z: int | None = None, a: int | None = None) -> str:
    """Get radioactive decay data (half-lives, decay modes, daughters).

    Args:
        z: Atomic number (e.g. 92 for U).
        a: Mass number (e.g. 238).
    """
    if z is None and a is None:
        raise RequestError("Provide at least z or a to filter decay data.")

    conditions = []
    params: dict[str, Any] = {}
    if z is not None:
        conditions.append("Z = $z")
        params["z"] = z
    if a is not None:
        conditions.append("A = $a")
        params["a"] = a

    where = " AND ".join(conditions)
    result = _query(f"SELECT * FROM decay WHERE {where}", params)
    return json.dumps(
        {"z": z, "a": a, "count": result["total"], "rows": result["rows"]},
        indent=2,
        default=str,
    )


@mcp.tool()
async def get_abundances(z: int) -> str:
    """Get natural isotope abundances and atomic masses for an element.

    Args:
        z: Atomic number (e.g. 29 for Cu).
    """
    result = _query("SELECT * FROM abundances WHERE Z = $z", {"z": z})
    return json.dumps(
        {"z": z, "count": result["total"], "isotopes": result["rows"]},
        indent=2,
    )


@mcp.tool()
async def get_stopping_power(source: str, target_z: int) -> str:
    """Get mass stopping power (dE/dx) for a projectile in a target element.

    Args:
        source: Data source: PSTAR (protons), ASTAR (α via NIST ICRU-49),
                ESTAR (electrons), dSTAR/tSTAR (velocity-scaled deuteron/triton).
                ³He routes through the catima master table (no NIST table exists).
        target_z: Target element atomic number.
    """
    # Map source names to DuckDB view names
    view_map = {
        "PSTAR": "stopping",
        "ASTAR": "stopping",
        "ESTAR": "stopping",
        "dSTAR": "stopping",
        "tSTAR": "stopping",
        "catima": "catima_stopping",
    }
    if source not in view_map:
        raise RequestError(f"Unknown source {source!r}. Valid: PSTAR, ASTAR, ESTAR, dSTAR, tSTAR, catima")

    view = view_map[source]
    if source == "catima":
        result = _query(
            f"SELECT * FROM {view} WHERE target_Z = $tz",
            {"tz": target_z},
        )
    else:
        result = _query(
            f"SELECT * FROM {view} WHERE source = $src AND target_Z = $tz",
            {"src": source, "tz": target_z},
        )
    return json.dumps(
        {"source": source, "target_z": target_z, "count": result["total"], "rows": result["rows"]},
        indent=2,
    )


# ---------------------------------------------------------------------------
# Radiation / coincidence / spectra tools (DuckDB views)
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_radiation(z: int, a: int | None = None, max_rows: int = 500) -> str:
    """Get radiation emissions (gammas, X-rays, Auger electrons, conversion electrons) for a nuclide.

    Args:
        z: Atomic number of the parent nuclide.
        a: Mass number (optional — omit to get all isotopes of element Z).
        max_rows: Maximum rows to return (default 500).
    """
    conditions = ["Z = $z"]
    params: dict[str, Any] = {"z": z}
    if a is not None:
        conditions.append("A = $a")
        params["a"] = a
    where = " AND ".join(conditions)
    result = _query(f"SELECT * FROM radiation WHERE {where}", params, max_rows)
    return json.dumps(
        {"z": z, "a": a, "total": result["total"], "truncated": result["truncated"], "rows": result["rows"]},
        indent=2,
    )


@mcp.tool()
async def get_coincidences(z: int, a: int | None = None, max_rows: int = 500) -> str:
    """Get gamma-gamma and mixed-emission coincidence pairs for a nuclide.

    Returns pairs of emissions that occur in the same cascade (useful for
    coincidence gating in spectroscopy). Includes gamma-gamma pairs and
    mixed pairs (beta/EC/X-ray/Auger/511 keV annihilation paired with gammas).

    Args:
        z: Atomic number of the parent nuclide.
        a: Mass number (optional — omit for all isotopes of element Z).
        max_rows: Maximum rows to return (default 500).
    """
    conditions = ["Z = $z"]
    params: dict[str, Any] = {"z": z}
    if a is not None:
        conditions.append("A = $a")
        params["a"] = a
    where = " AND ".join(conditions)
    result = _query(f"SELECT * FROM coincidences WHERE {where}", params, max_rows)
    return json.dumps(
        {"z": z, "a": a, "total": result["total"], "truncated": result["truncated"], "rows": result["rows"]},
        indent=2,
    )


@mcp.tool()
async def get_summing_partners(
    z: int,
    a: int,
    primary_energy_keV: float | None = None,
    tolerance_keV: float = 0.5,
    emission1_rad_type: str | None = None,
    max_rows: int = 500,
) -> str:
    """Get ICC-corrected summing partners for HPGe true-coincidence-summing (TCS).

    Returns all emission pairs that can sum in a close-geometry HPGe detector.
    Each row carries ``icc_correction_factor`` and ``pure_emission_joint_intensity``
    pre-computed. Includes gamma-gamma pairs and X-ray/Auger-gamma pairs.

    Args:
        z: Atomic number of the daughter nuclide (filing convention).
        a: Mass number.
        primary_energy_keV: Filter to pairs matching this energy (either side).
        tolerance_keV: Energy match tolerance in keV (default 0.5).
        emission1_rad_type: Filter emission side 1 ('gamma', 'xray', 'auger').
        max_rows: Maximum rows to return (default 500).
    """
    conditions = ["Z = $z", "A = $a"]
    params: dict[str, Any] = {"z": z, "a": a}
    if primary_energy_keV is not None:
        conditions.append("(ABS(emission1_energy_keV - $energy) < $tol OR ABS(emission2_energy_keV - $energy) < $tol)")
        params["energy"] = primary_energy_keV
        params["tol"] = tolerance_keV
    if emission1_rad_type is not None:
        conditions.append("emission1_rad_type = $e1type")
        params["e1type"] = emission1_rad_type
    where = " AND ".join(conditions)
    result = _query(
        f"SELECT * FROM summing_partners WHERE {where} ORDER BY pure_emission_joint_intensity DESC",
        params,
        max_rows,
    )
    return json.dumps(
        {
            "z": z,
            "a": a,
            "primary_energy_keV": primary_energy_keV,
            "total": result["total"],
            "truncated": result["truncated"],
            "rows": result["rows"],
        },
        indent=2,
    )


@mcp.tool()
async def get_emissions(
    parent_z: int,
    parent_a: int,
    parent_state: str = "",
    decay_mode: str | None = None,
    energy_keV: float | None = None,
    tolerance_keV: float = 0.5,
    min_intensity_pct: float = 0.0,
    max_rows: int = 500,
) -> str:
    """Get absolute per-decay photon emission intensities (NuDat-equivalent).

    Returns all gamma emissions for a parent nuclide with absolute intensities
    (photon emission probability per decay, 0-100%). Filed by parent, not daughter.

    Args:
        parent_z: Atomic number of the decaying parent nuclide (e.g. 27 for Co-60).
        parent_a: Mass number of the parent (e.g. 60 for Co-60).
        parent_state: Nuclear state ('' = ground, 'm' = metastable, 'm2' = 2nd isomer).
        decay_mode: Filter by decay mode ('beta-', 'KshellEC', 'IT', etc.).
        energy_keV: Filter to gammas near this energy.
        tolerance_keV: Energy match tolerance in keV (default 0.5).
        min_intensity_pct: Minimum absolute intensity (%) to include (default 0).
        max_rows: Maximum rows to return (default 500).
    """
    conditions = ["parent_Z = $z", "parent_A = $a", "parent_state = $state"]
    params: dict[str, Any] = {"z": parent_z, "a": parent_a, "state": parent_state}
    if decay_mode is not None:
        conditions.append("decay_mode = $mode")
        params["mode"] = decay_mode
    if energy_keV is not None:
        conditions.append("ABS(energy_keV - $energy) < $tol")
        params["energy"] = energy_keV
        params["tol"] = tolerance_keV
    if min_intensity_pct > 0:
        conditions.append("intensity_pct >= $min_int")
        params["min_int"] = min_intensity_pct
    where = " AND ".join(conditions)
    result = _query(
        f"SELECT * FROM emissions WHERE {where} ORDER BY intensity_pct DESC",
        params,
        max_rows,
    )
    return json.dumps(
        {
            "parent_z": parent_z,
            "parent_a": parent_a,
            "parent_state": parent_state,
            "total": result["total"],
            "truncated": result["truncated"],
            "rows": result["rows"],
        },
        indent=2,
    )


@mcp.tool()
async def get_beta_spectrum(z: int, a: int, max_rows: int = 500) -> str:
    """Get the continuous beta-decay kinetic-energy spectrum for a nuclide.

    Returns pre-tabulated Fermi-function spectra (dN/dE, normalized to 1)
    for all beta-minus and beta-plus transitions of the nuclide.

    Args:
        z: Atomic number of the parent nuclide.
        a: Mass number of the parent nuclide.
        max_rows: Maximum rows to return (default 500).
    """
    result = _query(
        "SELECT * FROM beta_spectra WHERE Z = $z AND A = $a",
        {"z": z, "a": a},
        max_rows,
    )
    return json.dumps(
        {"z": z, "a": a, "total": result["total"], "truncated": result["truncated"], "rows": result["rows"]},
        indent=2,
    )


@mcp.tool()
async def get_compound_compositions(material: str | None = None) -> str:
    """Get elemental compositions (weight fractions) for NIST XCOM standard materials.

    Returns Z and weight_fraction for each element in the compound.
    Useful for Bragg-additive cross-section calculations.

    Args:
        material: Material key (e.g. 'water', 'air', 'concrete'). Omit to list all materials.
    """
    db = _get_db()
    if material is None:
        materials = _materials(db)
        return json.dumps({"count": len(materials), "materials": materials}, indent=2)

    result = _query(
        "SELECT * FROM compound_compositions WHERE material = $mat",
        {"mat": material},
    )
    if result["total"] == 0:
        raise RequestError(f"Unknown material: {material!r}. Available: {_materials(db)}")
    return json.dumps(
        {"material": material, "count": result["total"], "composition": result["rows"]},
        indent=2,
    )


@mcp.tool()
async def get_electron_stopping(
    target: str | None = None,
    target_z: int | None = None,
    max_rows: int = 500,
) -> str:
    """Get electron stopping power with collision/radiative split.

    Richer than ESTAR — includes ~183 compounds plus all elements Z=1..98.
    Filter by element (target_z) or compound name (target).

    Args:
        target: Compound name (e.g. 'G4_WATER', 'G4_AIR'). For elements use target_z instead.
        target_z: Atomic number for elemental targets.
        max_rows: Maximum rows to return (default 500).
    """
    if target is None and target_z is None:
        raise RequestError("Provide target (compound name) or target_z (atomic number).")

    if target_z is not None:
        result = _query(
            "SELECT * FROM electron_stopping WHERE target_Z = $tz",
            {"tz": target_z},
            max_rows,
        )
    else:
        result = _query(
            "SELECT * FROM electron_stopping WHERE name = $t OR g4_name = $t",
            {"t": target},
            max_rows,
        )
    return json.dumps(
        {
            "target": target,
            "target_z": target_z,
            "total": result["total"],
            "truncated": result["truncated"],
            "rows": result["rows"],
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# SQL escape hatch (Sub-E #179)
# ---------------------------------------------------------------------------


@mcp.tool()
async def sql_query(sql: str, max_rows: int = 10000) -> str:
    """Execute read-only SQL against all 70+ nuclear data tables.

    Supports JOINs, aggregations, window functions — anything DuckDB supports.
    Use describe_schema() to discover available tables and columns.

    Args:
        sql: Read-only SQL query. DDL/DML (DROP, CREATE, INSERT, UPDATE) will be rejected.
        max_rows: Maximum rows to return (default 10000).
    """
    # Two layers: `_require_read_only` admits one parsed read statement, and the
    # connection (`_harden`) can only read inside the data tree with its
    # configuration locked.
    _require_read_only(sql)

    db = _get_db()
    try:
        rel = db.sql(sql)
    except duckdb.Error as e:
        raise RequestError(f"SQL error: {e}") from e

    total = rel.count("*").fetchone()[0]
    rows = _records(rel.limit(max_rows))
    truncated = total > max_rows
    return json.dumps(
        {"total": total, "truncated": truncated, "rows": rows},
        indent=2,
        default=str,
    )


@mcp.tool()
async def describe_schema() -> str:
    """List all available tables/views with their column names and types.

    Use this to discover what data is available before writing SQL queries.
    """
    db = _get_db()
    tables_rel = db.sql("SHOW TABLES")
    table_names = sorted(r[0] for r in tables_rel.fetchall())

    schema: dict[str, list[dict[str, str]]] = {}
    for tbl in table_names:
        try:
            cols_rel = db.sql(f"DESCRIBE {tbl}")
            cols = [{"name": r[0], "type": r[1]} for r in cols_rel.fetchall()]
            schema[tbl] = cols
        except duckdb.Error:
            schema[tbl] = []

    return json.dumps({"tables": len(schema), "schema": schema}, indent=2)


@mcp.tool()
async def list_tables() -> str:
    """List all available table/view names (short form of describe_schema)."""
    db = _get_db()
    tables = sorted(r[0] for r in db.sql("SHOW TABLES").fetchall())
    return json.dumps({"count": len(tables), "tables": tables}, indent=2)
