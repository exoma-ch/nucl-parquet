"""Tests for nucl-parquet MCP server (SSoT refactor — DuckDB-backed)."""

from __future__ import annotations

import json

import pytest
from nucl_parquet_mcp.server import (
    _ensure_catalog,
    _get_db,
    describe_schema,
    get_abundances,
    get_beta_spectrum,
    get_coincidences,
    get_compound_compositions,
    get_cross_sections,
    get_decay_data,
    get_electron_stopping,
    get_radiation,
    list_libraries,
    list_isotopes,
    list_tables,
    mcp,
    sql_query,
)

# ---------------------------------------------------------------------------
# Version tests
# ---------------------------------------------------------------------------


class TestVersion:
    def test_server_version_from_metadata(self):
        import importlib.metadata

        expected = importlib.metadata.version("nucl-parquet-mcp")
        assert mcp.settings.version == expected
        assert expected  # not empty


# ---------------------------------------------------------------------------
# Catalog tests (loaded from disk, not hardcoded)
# ---------------------------------------------------------------------------


class TestCatalog:
    def test_has_all_libraries(self):
        cat = _ensure_catalog()
        libs = cat["libraries"]
        assert len(libs) >= 15
        assert "tendl-2024" in libs
        assert "endfb-8.1" in libs
        assert "exfor" in libs

    def test_all_xs_libraries_have_projectiles(self):
        cat = _ensure_catalog()
        for lib_id, lib in cat["libraries"].items():
            if "projectiles" in lib:
                assert len(lib["projectiles"]) > 0, f"{lib_id} has no projectiles"


# ---------------------------------------------------------------------------
# DuckDB integration tests (local data, no network)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDuckDB:
    def test_db_connects(self):
        db = _get_db()
        tables = [r[0] for r in db.sql("SHOW TABLES").fetchall()]
        assert "decay" in tables
        assert "abundances" in tables
        assert "radiation" in tables

    async def test_list_libraries(self):
        result = await list_libraries()
        data = json.loads(result)
        assert len(data) >= 15
        ids = [lib["id"] for lib in data]
        assert "tendl-2024" in ids

    async def test_list_isotopes_invalid_library(self):
        with pytest.raises(ValueError, match="Unknown library"):
            await list_isotopes("nonexistent", "p")

    async def test_get_decay_data_requires_filter(self):
        with pytest.raises(ValueError, match="at least z or a"):
            await get_decay_data()

    async def test_get_abundances(self):
        result = await get_abundances(29)
        data = json.loads(result)
        assert data["z"] == 29
        assert data["count"] >= 2  # Cu-63 and Cu-65

    async def test_get_decay_data(self):
        result = await get_decay_data(z=27, a=60)
        data = json.loads(result)
        assert data["count"] >= 1  # Co-60 should exist

    async def test_get_radiation(self):
        result = await get_radiation(z=27, a=60)
        data = json.loads(result)
        assert data["total"] > 0
        # Co-60 has the famous 1173 keV gamma
        energies = [r.get("energy_keV") for r in data["rows"] if r.get("rad_type") == "gamma"]
        assert any(abs(e - 1173.2) < 1.0 for e in energies if e is not None)

    async def test_get_coincidences(self):
        result = await get_coincidences(z=27, a=60)
        data = json.loads(result)
        assert data["total"] > 0

    async def test_get_beta_spectrum(self):
        result = await get_beta_spectrum(z=15, a=32)
        data = json.loads(result)
        assert data["total"] > 0  # P-32 has a beta transition

    async def test_get_compound_compositions_list(self):
        result = await get_compound_compositions()
        data = json.loads(result)
        assert data["count"] > 0
        assert any("Water" in m for m in data["materials"])

    async def test_get_stopping_power(self):
        result = await get_stopping_power("PSTAR", 29)
        data = json.loads(result)
        assert data["count"] > 0

    async def test_sql_query(self):
        result = await sql_query("SELECT COUNT(*) AS n FROM decay WHERE Z = 27")
        data = json.loads(result)
        assert data["total"] == 1
        assert data["rows"][0]["n"] > 0

    async def test_sql_query_rejects_ddl(self):
        with pytest.raises(ValueError, match="Only read queries"):
            await sql_query("DROP TABLE decay")

    async def test_sql_query_rejects_copy(self):
        with pytest.raises(ValueError, match="Only read queries"):
            await sql_query("COPY radiation TO '/tmp/exfil.csv'")

    async def test_sql_query_rejects_attach(self):
        with pytest.raises(ValueError, match="Only read queries"):
            await sql_query("ATTACH '/tmp/evil.db' AS x")

    async def test_sql_query_rejects_export(self):
        with pytest.raises(ValueError, match="Only read queries"):
            await sql_query("EXPORT DATABASE '/tmp/dump'")

    async def test_sql_query_rejects_install(self):
        with pytest.raises(ValueError, match="Only read queries"):
            await sql_query("INSTALL httpfs")

    async def test_sql_query_rejects_pragma(self):
        with pytest.raises(ValueError, match="Only read queries"):
            await sql_query("PRAGMA version")

    async def test_get_electron_stopping(self):
        result = await get_electron_stopping(target_z=29)
        data = json.loads(result)
        assert data["total"] > 0

    async def test_get_stopping_power_catima(self):
        result = await get_stopping_power("catima", 29)
        data = json.loads(result)
        assert data["count"] > 0

    async def test_get_stopping_power_invalid_source(self):
        with pytest.raises(ValueError, match="Unknown source"):
            await get_stopping_power("INVALID", 29)

    async def test_get_cross_sections_invalid_element(self):
        with pytest.raises(ValueError, match="Invalid element"):
            await get_cross_sections("tendl-2024", "p", "'; DROP TABLE decay; --")

    async def test_get_cross_sections_invalid_projectile(self):
        with pytest.raises(ValueError, match="Invalid projectile"):
            await get_cross_sections("tendl-2024", "x", "Cu")

    async def test_describe_schema(self):
        result = await describe_schema()
        data = json.loads(result)
        assert data["tables"] > 0
        assert "decay" in data["schema"]

    async def test_list_tables(self):
        result = await list_tables()
        data = json.loads(result)
        assert "decay" in data["tables"]
        assert "radiation" in data["tables"]
        assert data["count"] >= 20
