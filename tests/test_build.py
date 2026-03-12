"""Smoke tests for nucl-parquet validation and import."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).parent.parent


def _load_build():
    """Import build.py as a module."""
    spec = importlib.util.spec_from_file_location("build", ROOT / "build.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _create_test_parquets(base: Path) -> None:
    """Create minimal Parquet files matching expected schemas."""
    meta = base / "meta"
    meta.mkdir(parents=True)

    pl.DataFrame(
        {"Z": [29, 26], "A": [63, 56], "symbol": ["Cu", "Fe"],
         "abundance": [0.6915, 0.9175], "atomic_mass": [62.93, 55.93]},
        schema={"Z": pl.Int32, "A": pl.Int32, "symbol": pl.Utf8,
                "abundance": pl.Float64, "atomic_mass": pl.Float64},
    ).write_parquet(meta / "abundances.parquet", compression="zstd")

    pl.DataFrame(
        {"Z": [30], "A": [65], "state": [""], "half_life_s": [211.0],
         "decay_mode": ["beta+"], "daughter_Z": [29], "daughter_A": [65],
         "daughter_state": [""], "branching": [1.0]},
        schema={"Z": pl.Int32, "A": pl.Int32, "state": pl.Utf8,
                "half_life_s": pl.Float64, "decay_mode": pl.Utf8,
                "daughter_Z": pl.Int32, "daughter_A": pl.Int32,
                "daughter_state": pl.Utf8, "branching": pl.Float64},
    ).write_parquet(meta / "decay.parquet", compression="zstd")

    pl.DataFrame(
        {"Z": [29, 26], "symbol": ["Cu", "Fe"]},
        schema={"Z": pl.Int32, "symbol": pl.Utf8},
    ).write_parquet(meta / "elements.parquet", compression="zstd")

    stopping = base / "stopping"
    stopping.mkdir()
    pl.DataFrame(
        {"source": ["PSTAR", "PSTAR"], "target_Z": [29, 29],
         "energy_MeV": [1.0, 10.0], "dedx": [50.0, 20.0]},
        schema={"source": pl.Utf8, "target_Z": pl.Int32,
                "energy_MeV": pl.Float64, "dedx": pl.Float64},
    ).write_parquet(stopping / "stopping.parquet", compression="zstd")

    xs = base / "xs"
    xs.mkdir()
    pl.DataFrame(
        {"target_A": [63, 63, 65], "residual_Z": [30, 30, 30],
         "residual_A": [63, 63, 65], "state": ["", "", ""],
         "energy_MeV": [10.0, 20.0, 10.0], "xs_mb": [100.0, 200.0, 50.0]},
        schema={"target_A": pl.Int32, "residual_Z": pl.Int32,
                "residual_A": pl.Int32, "state": pl.Utf8,
                "energy_MeV": pl.Float64, "xs_mb": pl.Float64},
    ).write_parquet(xs / "p_Cu.parquet", compression="zstd")


@pytest.fixture()
def imported_data(tmp_path: Path) -> Path:
    """Create source data and import it via build.py."""
    source = tmp_path / "source"
    source.mkdir()
    _create_test_parquets(source)

    build = _load_build()
    dest = tmp_path / "dest"
    dest.mkdir()
    build.import_all(source, dest)
    return dest


def test_meta_files_exist(imported_data: Path) -> None:
    assert (imported_data / "meta" / "abundances.parquet").exists()
    assert (imported_data / "meta" / "decay.parquet").exists()
    assert (imported_data / "meta" / "elements.parquet").exists()


def test_stopping_exists(imported_data: Path) -> None:
    assert (imported_data / "stopping" / "stopping.parquet").exists()


def test_xs_files_exist(imported_data: Path) -> None:
    xs_dir = imported_data / "tendl-2024" / "xs"
    assert xs_dir.exists()
    files = sorted(f.name for f in xs_dir.glob("*.parquet"))
    assert "p_Cu.parquet" in files


def test_abundances_content(imported_data: Path) -> None:
    df = pl.read_parquet(imported_data / "meta" / "abundances.parquet")
    assert len(df) == 2
    cu = df.filter(pl.col("Z") == 29)
    assert len(cu) == 1
    assert abs(cu["abundance"][0] - 0.6915) < 1e-6


def test_xs_content(imported_data: Path) -> None:
    df = pl.read_parquet(imported_data / "tendl-2024" / "xs" / "p_Cu.parquet")
    assert len(df) == 3
    assert set(df["target_A"].unique().to_list()) == {63, 65}


def test_manifest_written(imported_data: Path) -> None:
    manifest_path = imported_data / "tendl-2024" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["library"] == "tendl-2024"
    assert manifest["files"] == 1
    assert "p" in manifest["projectiles"]


def test_catalog_schema_valid() -> None:
    """Verify catalog.json has expected structure."""
    catalog = json.loads((ROOT / "catalog.json").read_text())
    assert catalog["version"] == 1
    assert "tendl-2024" in catalog["libraries"]
    lib = catalog["libraries"]["tendl-2024"]
    assert "p" in lib["projectiles"]
    assert lib["path"].endswith("/")


def test_schema_validation() -> None:
    """Verify that schema validation catches wrong column types."""
    build = _load_build()
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
        # Write with wrong type (Utf8 instead of Int32 for target_A)
        pl.DataFrame(
            {"target_A": ["wrong"], "residual_Z": [1], "residual_A": [1],
             "state": [""], "energy_MeV": [1.0], "xs_mb": [1.0]},
        ).write_parquet(f.name)

        errors = build._validate_parquet(Path(f.name), build.XS_SCHEMA)
        assert any("target_A" in e for e in errors)
