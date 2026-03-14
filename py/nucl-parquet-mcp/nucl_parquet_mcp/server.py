"""nucl-parquet MCP server — lazy-loads Parquet files from GitHub."""

from __future__ import annotations

import io
import os
from typing import Any

import httpx
import pyarrow.parquet as pq
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get(
    "NUCL_PARQUET_BASE_URL",
    "https://raw.githubusercontent.com/exoma-ch/nucl-parquet/main/",
)

# ---------------------------------------------------------------------------
# Embedded catalog
# ---------------------------------------------------------------------------

CATALOG: dict[str, Any] = {
    "libraries": {
        "tendl-2024": {
            "name": "TENDL-2024",
            "description": "TALYS Evaluated Nuclear Data Library 2024 (IAEA/PSI)",
            "projectiles": ["p", "d", "t", "h", "a"],
            "data_type": "cross_sections",
            "version": "2024",
            "path": "tendl-2024/xs/",
        },
        "endfb-8.1": {
            "name": "ENDF/B-VIII.1",
            "description": "US Evaluated Nuclear Data File (NNDC/BNL)",
            "projectiles": ["n", "p", "d", "t", "h", "a"],
            "data_type": "cross_sections",
            "version": "VIII.1",
            "path": "endfb-8.1/xs/",
        },
        "jeff-4.0": {
            "name": "JEFF-4.0",
            "description": "Joint Evaluated Fission and Fusion File (NEA)",
            "projectiles": ["n", "p"],
            "data_type": "cross_sections",
            "version": "4.0",
            "path": "jeff-4.0/xs/",
        },
        "jendl-5": {
            "name": "JENDL-5",
            "description": "Japanese Evaluated Nuclear Data Library (JAEA)",
            "projectiles": ["n", "p", "d", "a"],
            "data_type": "cross_sections",
            "version": "5",
            "path": "jendl-5/xs/",
        },
        "tendl-2025": {
            "name": "TENDL-2025",
            "description": "TALYS Evaluated Nuclear Data Library 2025 (PSI)",
            "projectiles": ["n", "p", "d", "t", "h", "a"],
            "data_type": "cross_sections",
            "version": "2025",
            "path": "tendl-2025/xs/",
        },
        "cendl-3.2": {
            "name": "CENDL-3.2",
            "description": "Chinese Evaluated Nuclear Data Library (CIAE)",
            "projectiles": ["n"],
            "data_type": "cross_sections",
            "version": "3.2",
            "path": "cendl-3.2/xs/",
        },
        "brond-3.1": {
            "name": "BROND-3.1",
            "description": "Russian Evaluated Nuclear Data Library (IPPE)",
            "projectiles": ["n"],
            "data_type": "cross_sections",
            "version": "3.1",
            "path": "brond-3.1/xs/",
        },
        "fendl-3.2": {
            "name": "FENDL-3.2",
            "description": "Fusion Evaluated Nuclear Data Library (IAEA)",
            "projectiles": ["n"],
            "data_type": "cross_sections",
            "version": "3.2",
            "path": "fendl-3.2/xs/",
        },
        "eaf-2010": {
            "name": "EAF-2010",
            "description": "European Activation File (CCFE)",
            "projectiles": ["n"],
            "data_type": "cross_sections",
            "version": "2010",
            "path": "eaf-2010/xs/",
        },
        "irdff-2": {
            "name": "IRDFF-II",
            "description": "International Reactor Dosimetry and Fusion File (IAEA)",
            "projectiles": ["n"],
            "data_type": "cross_sections",
            "version": "II",
            "path": "irdff-2/xs/",
        },
        "iaea-medical": {
            "name": "IAEA-Medical",
            "description": "Medical isotope production cross-sections (IAEA)",
            "projectiles": ["p", "d", "h", "a"],
            "data_type": "cross_sections",
            "version": "latest",
            "path": "iaea-medical/xs/",
        },
        "jendl-ad-2017": {
            "name": "JENDL/AD-2017",
            "description": "Activation/Dosimetry Library (JAEA)",
            "projectiles": ["n", "p"],
            "data_type": "cross_sections",
            "version": "2017",
            "path": "jendl-ad-2017/xs/",
        },
        "jendl-deu-2020": {
            "name": "JENDL-DEU-2020",
            "description": "Dedicated deuteron-induced reaction library (JAEA)",
            "projectiles": ["d"],
            "data_type": "cross_sections",
            "version": "2020",
            "path": "jendl-deu-2020/xs/",
        },
        "iaea-pd-2019": {
            "name": "IAEA-PD-2019",
            "description": "Photonuclear Data Library (IAEA)",
            "projectiles": ["g"],
            "data_type": "cross_sections",
            "version": "2019",
            "path": "iaea-pd-2019/xs/",
        },
        "exfor": {
            "name": "EXFOR",
            "description": "Experimental nuclear reaction data (IAEA NDS)",
            "projectiles": ["n", "p", "d", "t", "h", "a"],
            "data_type": "experimental_cross_sections",
            "version": "latest",
            "path": "exfor/",
        },
    },
    "shared": {
        "meta": {
            "path": "meta/",
            "files": {"abundances": "abundances.parquet", "decay": "decay.parquet", "elements": "elements.parquet"},
        },
        "stopping": {
            "path": "stopping/",
            "files": {"stopping": "stopping.parquet"},
            "sources": ["PSTAR", "ASTAR", "ICRU73", "MSTAR"],
        },
    },
}

# ---------------------------------------------------------------------------
# Parquet fetch + cache
# ---------------------------------------------------------------------------

_cache: dict[str, list[dict[str, Any]]] = {}


def _get_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(follow_redirects=True, timeout=60.0)


async def fetch_parquet_rows(
    relative_path: str,
    base_url: str = BASE_URL,
) -> list[dict[str, Any]]:
    """Fetch a parquet file from GitHub and return rows as dicts."""
    if relative_path in _cache:
        return _cache[relative_path]

    url = base_url.rstrip("/") + "/" + relative_path.lstrip("/")
    async with _get_client() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.content

    table = pq.read_table(io.BytesIO(data))
    rows = table.to_pylist()
    _cache[relative_path] = rows
    return rows


async def _fetch_manifest(library_id: str) -> dict[str, Any]:
    """Fetch the manifest.json for a library."""
    lib = CATALOG["libraries"].get(library_id)
    if lib is None:
        raise ValueError(f"Unknown library: {library_id}")
    manifest_path = lib["path"].replace("xs/", "manifest.json")
    url = BASE_URL.rstrip("/") + "/" + manifest_path
    async with _get_client() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP("nucl-parquet")


@mcp.tool()
async def list_libraries() -> str:
    """List all available nuclear data libraries with projectiles and descriptions."""
    libs = [
        {
            "id": lib_id,
            "name": lib["name"],
            "description": lib["description"],
            "projectiles": lib["projectiles"],
            "version": lib["version"],
            "data_type": lib["data_type"],
        }
        for lib_id, lib in CATALOG["libraries"].items()
    ]
    import json

    return json.dumps(libs, indent=2)


@mcp.tool()
async def list_isotopes(library: str, projectile: str) -> str:
    """List available target elements for a library and projectile.

    Args:
        library: Library ID, e.g. 'tendl-2024', 'endfb-8.1'.
        projectile: Projectile type: n, p, d, t, h, a, g.
    """
    lib = CATALOG["libraries"].get(library)
    if lib is None:
        raise ValueError(f"Unknown library: {library}. Use list_libraries to see options.")
    if projectile not in lib["projectiles"]:
        raise ValueError(f"Projectile '{projectile}' not in {library}. Available: {', '.join(lib['projectiles'])}")
    manifest = await _fetch_manifest(library)
    elements = manifest.get("elements", [])
    import json

    return json.dumps(
        {"library": library, "projectile": projectile, "elements": elements, "count": len(elements)}, indent=2
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
    lib = CATALOG["libraries"].get(library)
    if lib is None:
        raise ValueError(f"Unknown library: {library}")

    parquet_path = f"{lib['path']}{projectile}_{element}.parquet"
    rows = await fetch_parquet_rows(parquet_path)

    truncated = len(rows) > max_rows
    import json

    return json.dumps(
        {
            "library": library,
            "projectile": projectile,
            "element": element,
            "total": len(rows),
            "truncated": truncated,
            "rows": rows[:max_rows],
        },
        indent=2,
    )


@mcp.tool()
async def get_decay_data(z: int | None = None, a: int | None = None) -> str:
    """Get radioactive decay data (half-lives, decay modes, daughters).

    Args:
        z: Atomic number (e.g. 92 for U).
        a: Mass number (e.g. 238).
    """
    if z is None and a is None:
        raise ValueError("Provide at least z or a to filter decay data.")

    meta = CATALOG["shared"]["meta"]
    path = meta["path"] + meta["files"]["decay"]
    rows = await fetch_parquet_rows(path)

    filtered = [row for row in rows if (z is None or row.get("Z") == z) and (a is None or row.get("A") == a)]

    import json

    return json.dumps({"z": z, "a": a, "count": len(filtered), "rows": filtered}, indent=2, default=str)


@mcp.tool()
async def get_abundances(z: int) -> str:
    """Get natural isotope abundances and atomic masses for an element.

    Args:
        z: Atomic number (e.g. 29 for Cu).
    """
    meta = CATALOG["shared"]["meta"]
    path = meta["path"] + meta["files"]["abundances"]
    rows = await fetch_parquet_rows(path)
    filtered = [row for row in rows if row.get("Z") == z]

    import json

    return json.dumps({"z": z, "count": len(filtered), "isotopes": filtered}, indent=2)


@mcp.tool()
async def get_stopping_power(source: str, target_z: int) -> str:
    """Get mass stopping power (dE/dx) for a projectile in a target element.

    Args:
        source: Data source: PSTAR, ASTAR, ICRU73, or MSTAR.
        target_z: Target element atomic number.
    """
    sp = CATALOG["shared"]["stopping"]
    path = sp["path"] + sp["files"]["stopping"]
    rows = await fetch_parquet_rows(path)

    filtered = [row for row in rows if row.get("source") == source and row.get("target_Z") == target_z]

    import json

    return json.dumps({"source": source, "target_z": target_z, "count": len(filtered), "rows": filtered}, indent=2)
