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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import DATA_DIR, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT))  # so `nucl_parquet` imports from the checkout

# Which libraries ship a manifest, and where it lives, is defined once — the
# builders resolve the same path when they write their provenance stamp, and
# `tests/test_builder_staleness.py` walks the same set when it audits them. Two
# copies of that rule is how `endfb-8.0/xs/` and `endfb-8.0/channels/` came to
# overwrite each other's manifest in the first place.
from nucl_parquet.builder_stamp import library_dirs  # noqa: E402


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


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser, separately from running it (#363)."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument(
        "--check",
        action="store_true",
        help="report drift without writing (for CI)",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()

    drift: list[str] = []
    for key, pq_dir, manifest_path in library_dirs(args.data_dir):
        fresh = build_manifest(key, pq_dir)
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            # Preserve fields this builder does not derive (`source_files`,
            # `sublibrary`, and above all `builder`) so re-running never loses
            # provenance a builder wrote.
            #
            # `builder` is deliberately never *written* here: this script
            # regenerates manifests from data that may be years old, and
            # stamping it with today's builder digest would attest that today's
            # code produced yesterday's parquets — the exact lie the stamp
            # exists to detect (#342). Only a real ingest may stamp.
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
