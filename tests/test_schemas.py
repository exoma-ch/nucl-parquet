"""Schema validation — verify parquet files match expected column schemas.

All tests require data files and are marked with @pytest.mark.data.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from nucl_parquet._schemas import (
    ABUNDANCES_SCHEMA,
    CANONICAL_XS_SCHEMA,
    DECAY_SCHEMA,
    ELEMENTS_SCHEMA,
    PENDING_COLUMN_ADDITION,
    STOPPING_SCHEMA,
)

# Every cross-section table now shares one shape, so the map is no longer a
# per-data_type dispatch to a different set of columns — it just records which
# data_types are cross-section tables at all.
_XS_SCHEMA_BY_TYPE: dict[str, dict | None] = {
    "cross_sections": CANONICAL_XS_SCHEMA,
    "transport_cross_sections": CANONICAL_XS_SCHEMA,
    "production_cross_sections": CANONICAL_XS_SCHEMA,
    "total_reaction_cross_sections": CANONICAL_XS_SCHEMA,
    "experimental_cross_sections": CANONICAL_XS_SCHEMA,
}

# DuckDB type name mapping for comparison
_DTYPE_MAP = {
    "Int32": "INTEGER",
    "Utf8": "VARCHAR",
    "Float64": "DOUBLE",
}


def _check_schema(path: Path, expected: dict[str, str]) -> None:
    """Assert parquet file columns match expected schema.

    Columns in `PENDING_COLUMN_ADDITION` are skipped: the builders write them but
    the shipped parquets predate the rebuild that fills them. Read from the one
    ledger rather than listed here, so there is a single place an exemption
    exists and a single place it is cleaned up (`_schemas.py`).
    """
    db = duckdb.connect()
    cols = db.sql(f"SELECT name, duckdb_type FROM parquet_schema('{path}') WHERE name != 'root'").fetchall()
    col_map = {name: dtype for name, dtype in cols}
    for col_name, expected_type in expected.items():
        if col_name in PENDING_COLUMN_ADDITION:
            continue
        assert col_name in col_map, f"Missing column '{col_name}' in {path.name}"
        duckdb_type = _DTYPE_MAP.get(expected_type, expected_type)
        assert col_map[col_name] == duckdb_type, (
            f"{path.name}: column '{col_name}' is {col_map[col_name]}, expected {duckdb_type}"
        )


@pytest.mark.data
def test_abundances_schema(data_dir_path: Path) -> None:
    _check_schema(data_dir_path / "meta" / "abundances.parquet", ABUNDANCES_SCHEMA)


@pytest.mark.data
def test_decay_schema(data_dir_path: Path) -> None:
    _check_schema(data_dir_path / "meta" / "decay.parquet", DECAY_SCHEMA)


@pytest.mark.data
def test_elements_schema(data_dir_path: Path) -> None:
    _check_schema(data_dir_path / "meta" / "elements.parquet", ELEMENTS_SCHEMA)


@pytest.mark.data
def test_stopping_schema(data_dir_path: Path) -> None:
    _check_schema(data_dir_path / "stopping" / "PSTAR.parquet", STOPPING_SCHEMA)


@pytest.mark.data
def test_xs_schema_sample(data_dir_path: Path) -> None:
    """Check schema of first available xs parquet file per evaluated library."""
    catalog = json.loads((data_dir_path / "catalog.json").read_text())
    checked = 0
    for lib_key, lib_info in catalog["libraries"].items():
        data_type = lib_info.get("data_type", "cross_sections")
        schema = _XS_SCHEMA_BY_TYPE.get(data_type)
        if schema is None:
            continue
        lib_dir = data_dir_path / lib_info["path"]
        files = sorted(lib_dir.glob("*.parquet"))
        if files:
            _check_schema(files[0], schema)
            checked += 1
    assert checked > 0, "No xs parquet files found to validate"


@pytest.mark.data
def test_exfor_schema(data_dir_path: Path) -> None:
    """Check EXFOR schema if data is present."""
    exfor_dir = data_dir_path / "exfor"
    if not exfor_dir.exists():
        pytest.skip("EXFOR data not present")
    files = sorted(exfor_dir.glob("*.parquet"))
    if not files:
        pytest.skip("No EXFOR parquet files")
    _check_schema(files[0], CANONICAL_XS_SCHEMA)
