"""Lift every cross-section table into `CANONICAL_XS_SCHEMA`.

The shipped 6-column xs schema stores reaction identity in the *file path*:
`data/<library>/xs/<projectile>_<Element>.parquet`. Nothing in a row says which
library, which projectile, or which target element it came from. Two bugs follow
directly:

  #273  the unified `xs` view interleaved isobaric targets (Nd/Pm/Sm-145 all
        reaching Nd-143). Fixed at query time by regexing the filename and
        joining a symbol->Z table — a workaround for missing data.
  ---   that same view still merges five projectiles. `SELECT ... FROM xs WHERE
        target_Z=29 AND target_A=63 AND residual_Z=30` — the example in
        loader.py's own docstring — returns (p,n), (d,2n), (a,x), (h,x) and
        (t,x) rows superposed as if they were one reaction.

This migration puts identity in the data, so the loader can stop reconstructing
it and the projectile bug becomes unrepresentable.

Idempotent: a file already carrying `library` is left alone.

Usage:
    nix develop -c .venv/bin/python scripts/migrate_xs_schema.py --dry-run
    nix develop -c .venv/bin/python scripts/migrate_xs_schema.py --library tendl-2025
    nix develop -c .venv/bin/python scripts/migrate_xs_schema.py
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


COMPRESSION = "zstd"

# The canonical row vocabulary and the frame transform now live in
# `_canonical.py`, so the ingest can produce canonical output directly instead of
# depending on this migration being remembered afterwards (#359). Re-exported
# here because these names are this module's public surface (tests import
# `parse_stem`, `_RENAMES`).
from _canonical import LIGHT_ION, SYMBOL_TO_Z, canonical_frame, parse_stem  # noqa: E402
from _canonical import RENAMES as _RENAMES  # noqa: E402

__all__ = [
    "LIGHT_ION",
    "SYMBOL_TO_Z",
    "_RENAMES",
    "canonical_frame",
    "migrate_file",
    "parse_stem",
]


def migrate_file(path: Path, library: str, kind: str, dry_run: bool) -> tuple[int, str]:
    """Rewrite one parquet file in canonical form. Returns (rows, status)."""
    import polars as pl

    parsed = parse_stem(path.stem)
    if parsed is None:
        return 0, "unparseable-stem"
    projectile, proj_z, proj_a, target_z = parsed

    try:
        df = pl.read_parquet(path)
    except Exception as e:  # a corrupt shard must not abort the whole migration
        logger.warning("  unreadable %s: %s", path, str(e)[:80])
        return 0, "unreadable"
    if "library" in df.columns:
        return df.height, "already-canonical"

    df = canonical_frame(
        df,
        library=library,
        kind=kind,
        projectile=projectile,
        proj_z=proj_z,
        proj_a=proj_a,
        target_z=target_z,
    ).sort("target_A", "residual_Z", "residual_A", "energy_MeV")

    if not dry_run:
        df.write_parquet(path, compression=COMPRESSION)
    return df.height, "migrated"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--library", help="migrate only this library key")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    catalog = json.loads((args.data_dir / "catalog.json").read_text())
    # `transport_cross_sections` is deliberately absent: those tables are written
    # canonical by scripts/build_channels.py and have no legacy form to lift.
    xs_types = {
        "cross_sections",
        "production_cross_sections",
        "total_reaction_cross_sections",
        "experimental_cross_sections",
    }

    totals: dict[str, int] = {}
    rows_total = 0
    for lib_key, info in catalog.get("libraries", {}).items():
        if info.get("data_type") not in xs_types or "path" not in info:
            continue
        if args.library and lib_key != args.library:
            continue
        lib_dir = args.data_dir / info["path"]
        if not lib_dir.exists():
            continue
        # Every legacy table is a production sum — `xs_types` above excludes the
        # channel libraries, which are written canonical by their own builders.
        kind = "production"
        files = sorted(lib_dir.glob("*.parquet"))
        for f in files:
            rows, status = migrate_file(f, lib_key, kind, args.dry_run)
            totals[status] = totals.get(status, 0) + 1
            rows_total += rows
        logger.info("%-18s %4d files", lib_key, len(files))

    logger.info("---")
    for status, n in sorted(totals.items()):
        logger.info("%-20s %5d files", status, n)
    logger.info("%d rows total%s", rows_total, " (dry run)" if args.dry_run else "")


if __name__ == "__main__":
    main()
