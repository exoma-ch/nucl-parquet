"""nucl-parquet MCP server — wraps the nucl_parquet client library.

Delegates all data access to ``nucl_parquet.connect()`` which registers
70+ Parquet-backed DuckDB views with lazy loading and predicate pushdown.
This is the SSoT refactor (epic #173, Sub-D #178): the MCP is a thin shell
over the client library, not a re-implementation of parquet I/O.
"""

from __future__ import annotations

import json
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

import duckdb
from mcp.server.fastmcp import FastMCP

import nucl_parquet

# ---------------------------------------------------------------------------
# Data bootstrap
# ---------------------------------------------------------------------------

_db: duckdb.DuckDBPyConnection | None = None


def _get_db() -> duckdb.DuckDBPyConnection:
    """Lazy-connect to the nucl-parquet DuckDB instance."""
    global _db  # noqa: PLW0603
    if _db is not None:
        return _db

    # Try to find local data; download if not present.
    try:
        data_path = nucl_parquet.data_dir()
    except FileNotFoundError:
        # No local data — download the latest release tarball.
        nucl_parquet.download()
        data_path = nucl_parquet.data_dir()

    _db = nucl_parquet.connect(data_path)
    return _db


def _query(sql: str, params: dict[str, Any] | None = None, max_rows: int = 500) -> dict[str, Any]:
    """Run a read-only SQL query and return {total, truncated, rows}."""
    db = _get_db()
    rel = db.sql(sql, params=params or {})
    total = rel.count("*").fetchone()[0]
    rows = rel.limit(max_rows).fetchdf().to_dict(orient="records")
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


def _get_catalog() -> dict[str, Any]:
    global _catalog  # noqa: PLW0603
    if "_catalog" not in globals() or _catalog is None:
        globals()["_catalog"] = _load_catalog()
    return _catalog


_catalog: dict[str, Any] | None = None

# Kept as a module-level reference for tests that import it.
CATALOG: dict[str, Any] = {}


def _ensure_catalog() -> dict[str, Any]:
    global CATALOG  # noqa: PLW0603
    if not CATALOG:
        CATALOG.update(_get_catalog())
    return CATALOG


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

try:
    _VERSION = _pkg_version("nucl-parquet-mcp")
except PackageNotFoundError:
    _VERSION = "0.0.0-dev"

mcp = FastMCP("nucl-parquet", version=_VERSION)


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
        library: Library ID, e.g. 'tendl-2024', 'endfb-8.1'.
        projectile: Projectile type: n, p, d, t, h, a, g.
    """
    cat = _ensure_catalog()
    lib = cat["libraries"].get(library)
    if lib is None:
        raise ValueError(f"Unknown library: {library}. Use list_libraries to see options.")
    if projectile not in lib.get("projectiles", []):
        raise ValueError(
            f"Projectile '{projectile}' not in {library}. "
            f"Available: {', '.join(lib.get('projectiles', []))}"
        )
    # Manifests are JSON files alongside the parquet data — read from disk.
    data_path = nucl_parquet.data_dir()
    manifest_path = Path(data_path) / lib["path"].replace("xs/", "manifest.json")
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
        library: Library ID, e.g. 'tendl-2024'.
        projectile: Projectile: n, p, d, t, h, a, g.
        element: Target element symbol, e.g. 'Cu', 'Fe'.
        max_rows: Maximum rows to return (default 500).
    """
    cat = _ensure_catalog()
    lib = cat["libraries"].get(library)
    if lib is None:
        raise ValueError(f"Unknown library: {library}")

    # Cross-section tables are registered as DuckDB views named after the library.
    # The view name is the library ID with hyphens replaced by underscores.
    view_name = library.replace("-", "_").replace(".", "_")
    db = _get_db()

    # Check if view exists
    tables = [r[0] for r in db.sql("SHOW TABLES").fetchall()]
    if view_name not in tables:
        # Fall back to reading the parquet file directly
        data_path = nucl_parquet.data_dir()
        parquet_path = Path(data_path) / f"{lib['path']}{projectile}_{element}.parquet"
        if not parquet_path.exists():
            raise ValueError(f"No data for {projectile}_{element} in {library}")
        rel = db.sql(f"SELECT * FROM read_parquet('{parquet_path}')")
        total = rel.count("*").fetchone()[0]
        rows = rel.limit(max_rows).fetchdf().to_dict(orient="records")
        truncated = total > max_rows
    else:
        result = _query(
            f"SELECT * FROM {view_name} WHERE target_symbol = $element",
            {"element": element},
            max_rows,
        )
        total, truncated, rows = result["total"], result["truncated"], result["rows"]

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
        raise ValueError("Provide at least z or a to filter decay data.")

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
        return json.dumps(
            {"error": f"unknown source {source!r}; valid: PSTAR, ASTAR, ESTAR, dSTAR, tSTAR, catima"},
            indent=2,
        )

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
        material: Material name (e.g. 'Water, Liquid', 'Air, Dry'). Omit to list all materials.
    """
    db = _get_db()
    if material is None:
        rows = db.sql("SELECT DISTINCT material FROM compound_compositions ORDER BY material").fetchdf()
        materials = rows["material"].tolist()
        return json.dumps({"count": len(materials), "materials": materials}, indent=2)

    result = _query(
        "SELECT * FROM compound_compositions WHERE material = $mat",
        {"mat": material},
    )
    if result["total"] == 0:
        all_mats = db.sql("SELECT DISTINCT material FROM compound_compositions ORDER BY material").fetchdf()
        raise ValueError(f"Unknown material: {material!r}. Available: {all_mats['material'].tolist()}")
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
        raise ValueError("Provide target (compound name) or target_z (atomic number).")

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
    forbidden = {"DROP", "CREATE", "INSERT", "UPDATE", "DELETE", "ALTER", "TRUNCATE", "COPY"}
    first_word = sql.strip().split()[0].upper() if sql.strip() else ""
    if first_word in forbidden:
        raise ValueError(f"Write operations are not allowed: {first_word}")

    db = _get_db()
    try:
        rel = db.sql(sql)
    except duckdb.Error as e:
        raise ValueError(f"SQL error: {e}") from e

    total = rel.count("*").fetchone()[0]
    rows = rel.limit(max_rows).fetchdf().to_dict(orient="records")
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
