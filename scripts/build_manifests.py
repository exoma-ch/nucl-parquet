"""Generate `manifest.json` for every cross-section library that ships one.

A manifest is the per-library counterpart of `catalog.json`: it records what was
actually written (files, rows, projectiles, elements) so a consumer can see the
shape of a library without opening 600 parquet files, and so a rebuild that
silently drops half a library is visible in a diff.

Three libraries shipped without one — `endfb-8.0`, `hi-xs`, `hi-xs-prod` — because
each was added by a builder that did not write manifests, and nothing checked.
`tests/test_manifests.py` now does.

Derived entirely from the data on disk, so it is safe to re-run after any build.

Usage:
    nix develop -c .venv/bin/python scripts/build_manifests.py
    nix develop -c .venv/bin/python scripts/build_manifests.py --check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

XS_TYPES = {
    "cross_sections",
    "transport_cross_sections",
    "production_cross_sections",
    "total_reaction_cross_sections",
    "experimental_cross_sections",
}


def library_dirs(data_dir: Path) -> list[tuple[str, Path, Path]]:
    """Yield (library key, parquet dir, manifest path) for every xs library."""
    catalog = json.loads((data_dir / "catalog.json").read_text())
    out = []
    for key, info in catalog.get("libraries", {}).items():
        if info.get("data_type") not in XS_TYPES or "path" not in info:
            continue
        pq_dir = data_dir / info["path"]
        if not pq_dir.exists() or not any(pq_dir.glob("*.parquet")):
            continue
        # One manifest per library, at a path that cannot collide with another
        # library's. The established convention is `data/<lib>/manifest.json`
        # beside `xs/`, but two libraries can share a root (`endfb-8.0/xs/` and
        # `endfb-8.0/channels/`) and some hold parquets directly (`exfor/`).
        # Walking up unconditionally makes the second library silently overwrite
        # the first — so only walk up when the parent directory *is* this
        # library, and otherwise keep the manifest beside its own data.
        parent = pq_dir.parent
        lib_root = parent if parent.name == key else pq_dir
        out.append((key, pq_dir, lib_root / "manifest.json"))
    return out


def build_manifest(key: str, pq_dir: Path) -> dict:
    import polars as pl

    files = sorted(pq_dir.glob("*.parquet"))
    total_rows = 0
    projectiles: set[str] = set()
    elements: set[str] = set()

    for f in files:
        # Read only what the manifest summarises — the row count comes from
        # parquet metadata, so this stays cheap on a 600-file library.
        lf = pl.scan_parquet(f)
        total_rows += lf.select(pl.len()).collect().item()
        cols = lf.collect_schema().names()
        if "projectile" in cols:
            projectiles |= set(lf.select("projectile").unique().collect().to_series().to_list())
        # <projectile>_<Element>.parquet, or <Element>.parquet for meta tables.
        stem = f.stem
        elements.add(stem.split("_", 1)[1] if "_" in stem else stem)

    return {
        "library": key,
        "files": len(files),
        "total_rows": total_rows,
        "projectiles": sorted(projectiles),
        "elements": sorted(elements),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument(
        "--check",
        action="store_true",
        help="report drift without writing (for CI)",
    )
    args = ap.parse_args()

    drift: list[str] = []
    for key, pq_dir, manifest_path in library_dirs(args.data_dir):
        fresh = build_manifest(key, pq_dir)
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            # Preserve fields this builder does not derive (e.g. `source_files`,
            # `sublibrary`) so re-running never loses provenance a builder wrote.
            merged = {**existing, **fresh}
        else:
            merged = fresh

        if manifest_path.exists() and json.loads(manifest_path.read_text()) == merged:
            continue
        drift.append(f"{key}: {'stale' if manifest_path.exists() else 'missing'}")
        if not args.check:
            manifest_path.write_text(json.dumps(merged, indent=2) + "\n")
            logger.info("%-18s %d files, %s rows", key, merged["files"], f"{merged['total_rows']:,}")

    if args.check and drift:
        logger.error("manifests out of date:\n  %s", "\n  ".join(drift))
        raise SystemExit(1)
    logger.info("%d manifest(s) %s", len(drift), "would change" if args.check else "written")


if __name__ == "__main__":
    main()
