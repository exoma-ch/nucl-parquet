"""Shared fixtures for nucl-parquet tests."""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent


def _find_data_dir() -> Path | None:
    """Find data directory, or None if not available."""
    if (_REPO_ROOT / "catalog.json").exists():
        return _REPO_ROOT
    return None


DATA_DIR = _find_data_dir()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "data: requires parquet data files")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item],
) -> None:
    if DATA_DIR is not None:
        return
    skip = pytest.mark.skip(reason="nucl-parquet data not available")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture()
def data_dir_path() -> Path:
    """Return the data directory, skipping if unavailable."""
    if DATA_DIR is None:
        pytest.skip("nucl-parquet data not available")
    return DATA_DIR
