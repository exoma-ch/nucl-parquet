"""Tests for nucl-parquet MCP server."""

from __future__ import annotations

import json

import pytest
from nucl_parquet_mcp.server import (
    CATALOG,
    fetch_parquet_rows,
    get_abundances,
    get_cross_sections,
    get_decay_data,
    list_isotopes,
    list_libraries,
)

# ---------------------------------------------------------------------------
# Catalog tests (no network)
# ---------------------------------------------------------------------------


class TestCatalog:
    def test_has_all_libraries(self):
        libs = CATALOG["libraries"]
        assert len(libs) >= 15
        assert "tendl-2024" in libs
        assert "endfb-8.1" in libs
        assert "exfor" in libs

    def test_all_libraries_have_projectiles(self):
        for lib_id, lib in CATALOG["libraries"].items():
            assert len(lib["projectiles"]) > 0, f"{lib_id} has no projectiles"

    def test_all_paths_end_with_slash(self):
        for lib_id, lib in CATALOG["libraries"].items():
            assert lib["path"].endswith("/"), f"{lib_id} path should end with /"

    def test_shared_meta_files(self):
        meta = CATALOG["shared"]["meta"]
        assert "abundances" in meta["files"]
        assert "decay" in meta["files"]
        assert "elements" in meta["files"]

    def test_stopping_sources(self):
        sources = CATALOG["shared"]["stopping"]["sources"]
        assert "PSTAR" in sources
        assert "ASTAR" in sources


class TestUrlConstruction:
    def test_cross_section_path(self):
        lib = CATALOG["libraries"]["tendl-2024"]
        path = f"{lib['path']}p_Cu.parquet"
        assert path == "tendl-2024/xs/p_Cu.parquet"

    def test_meta_path(self):
        meta = CATALOG["shared"]["meta"]
        path = meta["path"] + meta["files"]["decay"]
        assert path == "meta/decay.parquet"

    def test_manifest_path(self):
        lib = CATALOG["libraries"]["tendl-2024"]
        path = lib["path"].replace("xs/", "manifest.json")
        assert path == "tendl-2024/manifest.json"


# ---------------------------------------------------------------------------
# Integration tests (require network)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestIntegration:
    async def test_list_libraries(self):
        result = await list_libraries()
        data = json.loads(result)
        assert len(data) >= 15
        ids = [lib["id"] for lib in data]
        assert "tendl-2024" in ids

    async def test_list_isotopes_invalid_library(self):
        with pytest.raises(ValueError, match="Unknown library"):
            await list_isotopes("nonexistent", "p")

    async def test_list_isotopes_invalid_projectile(self):
        with pytest.raises(ValueError, match="not in"):
            await list_isotopes("tendl-2024", "n")

    async def test_get_decay_data_requires_filter(self):
        with pytest.raises(ValueError, match="at least z or a"):
            await get_decay_data()

    @pytest.mark.timeout(30)
    async def test_fetch_abundances(self):
        rows = await fetch_parquet_rows("meta/abundances.parquet")
        assert len(rows) > 0
        cu63 = [r for r in rows if r.get("Z") == 29 and r.get("A") == 63]
        assert len(cu63) == 1
        assert 0.5 < cu63[0]["abundance"] < 0.8

    @pytest.mark.timeout(30)
    async def test_get_abundances(self):
        result = await get_abundances(29)
        data = json.loads(result)
        assert data["z"] == 29
        assert data["count"] >= 2  # Cu has at least Cu-63 and Cu-65

    @pytest.mark.timeout(30)
    async def test_get_cross_sections(self):
        result = await get_cross_sections("fendl-3.2", "n", "Cu", max_rows=10)
        data = json.loads(result)
        assert data["library"] == "fendl-3.2"
        assert data["total"] > 0
        assert len(data["rows"]) <= 10

    @pytest.mark.timeout(30)
    async def test_get_decay_data(self):
        result = await get_decay_data(z=27, a=60)
        data = json.loads(result)
        assert data["count"] >= 1  # Co-60 should exist

    @pytest.mark.timeout(30)
    async def test_cache_works(self):
        """Second fetch should use cache."""
        import time

        t0 = time.monotonic()
        await fetch_parquet_rows("meta/elements.parquet")
        t1 = time.monotonic()
        await fetch_parquet_rows("meta/elements.parquet")
        t2 = time.monotonic()

        # Cached call should be much faster
        assert (t2 - t1) < (t1 - t0) + 0.01
