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
    DECAY_SCHEMA,
    ELEMENTS_SCHEMA,
    EXFOR_SCHEMA,
    STOPPING_SCHEMA,
    XS_SCHEMA,
)

# DuckDB type name mapping for comparison
_DTYPE_MAP = {
    "Int32": "INTEGER",
    "Utf8": "VARCHAR",
    "Float64": "DOUBLE",
}


def _check_schema(path: Path, expected: dict[str, str]) -> None:
    """Assert parquet file columns match expected schema."""
    db = duckdb.connect()
    cols = db.sql(f"SELECT name, duckdb_type FROM parquet_schema('{path}') WHERE name != 'root'").fetchall()
    col_map = {name: dtype for name, dtype in cols}
    for col_name, expected_type in expected.items():
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
    _check_schema(data_dir_path / "stopping" / "stopping.parquet", STOPPING_SCHEMA)


@pytest.mark.data
def test_xs_schema_sample(data_dir_path: Path) -> None:
    """Check schema of first available xs parquet file per evaluated library."""
    catalog = json.loads((data_dir_path / "catalog.json").read_text())
    checked = 0
    for lib_key, lib_info in catalog["libraries"].items():
        if lib_info.get("data_type") == "experimental_cross_sections":
            continue
        lib_dir = data_dir_path / lib_info["path"]
        files = sorted(lib_dir.glob("*.parquet"))
        if files:
            _check_schema(files[0], XS_SCHEMA)
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
    _check_schema(files[0], EXFOR_SCHEMA)
