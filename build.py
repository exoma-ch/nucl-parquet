"""Validate and assemble nucl-parquet data.

This script verifies that all Parquet files conform to the expected schema
and that the catalog is consistent. It can also import Parquet files from
an external source directory into the correct layout.

Usage:
    # Verify existing data against catalog:
    python build.py --verify

    # Import cross-section parquet files from an external source:
    python build.py --import-xs /path/to/xs/ --library tendl-2024

    # Import all data (meta + stopping + xs) from an external parquet dir:
    python build.py --import-all /path/to/parquet/
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent

# Expected schemas for validation
XS_SCHEMA = {
    "target_A": pl.Int32,
    "residual_Z": pl.Int32,
    "residual_A": pl.Int32,
    "state": pl.Utf8,
    "energy_MeV": pl.Float64,
    "xs_mb": pl.Float64,
}

STOPPING_SCHEMA = {
    "source": pl.Utf8,
    "target_Z": pl.Int32,
    "energy_MeV": pl.Float64,
    "dedx": pl.Float64,
}

ABUNDANCES_SCHEMA = {
    "Z": pl.Int32,
    "A": pl.Int32,
    "symbol": pl.Utf8,
    "abundance": pl.Float64,
    "atomic_mass": pl.Float64,
}

DECAY_SCHEMA = {
    "Z": pl.Int32,
    "A": pl.Int32,
    "state": pl.Utf8,
    "half_life_s": pl.Float64,
    "decay_mode": pl.Utf8,
    "daughter_Z": pl.Int32,
    "daughter_A": pl.Int32,
    "daughter_state": pl.Utf8,
    "branching": pl.Float64,
}

ELEMENTS_SCHEMA = {
    "Z": pl.Int32,
    "symbol": pl.Utf8,
}


def _validate_parquet(path: Path, expected_schema: dict[str, pl.DataType]) -> list[str]:
    """Validate a Parquet file against expected column names and types."""
    errors: list[str] = []
    try:
        df = pl.read_parquet(path)
    except Exception as e:
        errors.append(f"Cannot read {path}: {e}")
        return errors

    for col, dtype in expected_schema.items():
        if col not in df.columns:
            errors.append(f"{path}: missing column '{col}'")
        elif df[col].dtype != dtype:
            errors.append(f"{path}: column '{col}' is {df[col].dtype}, expected {dtype}")

    return errors


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def verify(data_dir: Path) -> None:
    """Verify data against catalog.json and expected schemas."""
    catalog_path = ROOT / "catalog.json"
    catalog = json.loads(catalog_path.read_text())

    errors: list[str] = []

    # Check shared meta files
    meta_schemas = {
        "abundances": ABUNDANCES_SCHEMA,
        "decay": DECAY_SCHEMA,
        "elements": ELEMENTS_SCHEMA,
    }
    for name, filename in catalog["shared"]["meta"]["files"].items():
        path = data_dir / catalog["shared"]["meta"]["path"] / filename
        if not path.exists():
            errors.append(f"Missing meta file: {path}")
        else:
            errors.extend(_validate_parquet(path, meta_schemas.get(name, {})))
            df = pl.read_parquet(path)
            logger.info("  %-20s %6d rows  %6d KB", name, len(df), path.stat().st_size // 1024)

    # Check stopping
    for name, filename in catalog["shared"]["stopping"]["files"].items():
        path = data_dir / catalog["shared"]["stopping"]["path"] / filename
        if not path.exists():
            errors.append(f"Missing stopping file: {path}")
        else:
            errors.extend(_validate_parquet(path, STOPPING_SCHEMA))
            df = pl.read_parquet(path)
            logger.info("  %-20s %6d rows  %6d KB", name, len(df), path.stat().st_size // 1024)

    # Check library xs files
    for lib_key, lib_info in catalog["libraries"].items():
        xs_dir = data_dir / lib_info["path"]
        if not xs_dir.exists():
            errors.append(f"Missing library xs dir: {xs_dir}")
            continue
        files = list(xs_dir.glob("*.parquet"))
        if not files:
            errors.append(f"No parquet files in {xs_dir}")
        else:
            # Validate a sample
            for f in files[:5]:
                errors.extend(_validate_parquet(f, XS_SCHEMA))
            total_size = sum(f.stat().st_size for f in files)
            logger.info(
                "  %-20s %6d files  %6.1f MB",
                lib_key, len(files), total_size / 1024 / 1024,
            )

    if errors:
        for e in errors:
            logger.error(e)
        raise SystemExit(1)

    logger.info("Verification passed.")


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def import_all(source_dir: Path, data_dir: Path) -> None:
    """Import a complete parquet directory (meta/ + stopping/ + xs/) into data_dir."""
    for subdir in ("meta", "stopping"):
        src = source_dir / subdir
        dst = data_dir / subdir
        if src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob("*.parquet"):
                shutil.copy2(f, dst / f.name)
                logger.info("Copied %s → %s", f, dst / f.name)

    # xs/ goes into default library dir
    xs_src = source_dir / "xs"
    if xs_src.exists():
        import_xs(xs_src, data_dir, "tendl-2024")


def import_xs(source_dir: Path, data_dir: Path, library: str) -> None:
    """Import cross-section Parquet files into a library directory."""
    xs_dir = data_dir / library / "xs"
    xs_dir.mkdir(parents=True, exist_ok=True)

    files = list(source_dir.glob("*.parquet"))
    if not files:
        logger.warning("No .parquet files found in %s", source_dir)
        return

    total_rows = 0
    projectiles: set[str] = set()
    elements: set[str] = set()

    for f in files:
        shutil.copy2(f, xs_dir / f.name)
        # Parse filename: {projectile}_{element}.parquet
        stem = f.stem
        parts = stem.split("_", 1)
        if len(parts) == 2:
            projectiles.add(parts[0])
            elements.add(parts[1])
        df = pl.read_parquet(f)
        total_rows += len(df)

    # Write manifest
    manifest = {
        "library": library,
        "files": len(files),
        "total_rows": total_rows,
        "projectiles": sorted(projectiles),
        "elements": sorted(elements),
    }
    manifest_path = data_dir / library / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    logger.info(
        "Imported %d cross-section files (%d total rows) → %s/",
        len(files), total_rows, library,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and assemble nucl-parquet data.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify existing output against catalog.json and expected schemas",
    )
    parser.add_argument(
        "--import-all", type=Path, metavar="DIR",
        help="Import all data (meta + stopping + xs) from a parquet directory",
    )
    parser.add_argument(
        "--import-xs", type=Path, metavar="DIR",
        help="Import cross-section parquet files from a directory",
    )
    parser.add_argument(
        "--library", default="tendl-2024",
        help="Library identifier for cross-section import (default: tendl-2024)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT,
        help="Data directory (default: repo root)",
    )
    args = parser.parse_args()

    if args.verify:
        verify(args.data_dir)
    elif args.import_all:
        import_all(args.import_all, args.data_dir)
        verify(args.data_dir)
    elif args.import_xs:
        import_xs(args.import_xs, args.data_dir, args.library)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
