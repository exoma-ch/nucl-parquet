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

Failure is loud (#361). A file this migration cannot handle — an unparseable
stem, an unreadable shard — raises `UnmigratableFile` and aborts the run. It used
to be tallied into a summary counter that nothing branched on, so a library could
come out of a multi-hour rebuild still in the legacy 6-column form with a log
line as the only evidence, and the process still exited 0.

That silence produced two confidently wrong answers during the review of #349, a
623-file deletion justified by a migration round-trip: a checker that copied each
file to a temp path named `f.parquet` got `unparseable-stem`, so `migrate_file`
left the legacy file untouched and the comparison read as a real mismatch. The
disagreement happened to point the safe way. It did not have to.

Usage:
    nix develop -c .venv/bin/python scripts/migrate_xs_schema.py --dry-run
    nix develop -c .venv/bin/python scripts/migrate_xs_schema.py --library tendl-2025
    nix develop -c .venv/bin/python scripts/migrate_xs_schema.py

    # Survey every problem file instead of stopping at the first. Still exits
    # non-zero, and names each path it skipped.
    nix develop -c .venv/bin/python scripts/migrate_xs_schema.py --skip-unmigratable
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

#: Statuses that mean the file is now in canonical form. Anything else means it
#: is not, and must not be reported as a successful run.
SUCCESS_STATUSES = frozenset({"migrated", "already-canonical"})


class UnmigratableFile(RuntimeError):
    """A file the migration could not process — it is still in its legacy shape.

    Raised by `migrate_file` unless the caller explicitly opts into tolerant mode
    with `strict=False`. The alternative, returning a status string, is what #361
    was about: `migrate_file` had four return statuses, two of them failures, and
    its only consumer incremented a counter. Nothing could distinguish "600 files
    migrated" from "600 files skipped" without reading a log line.
    """

    def __init__(self, path: Path, status: str, detail: str = "") -> None:
        self.path = path
        self.status = status
        self.detail = detail
        super().__init__(f"{path}: {status}" + (f" — {detail}" if detail else ""))


def _not_migrated(path: Path, status: str, detail: str, *, strict: bool) -> tuple[int, str]:
    """Raise, or return the failure status if the caller opted into tolerance.

    Tolerant mode still logs the individual path rather than only counting it —
    "3 unparseable-stem" tells you nothing you can act on.
    """
    if strict:
        raise UnmigratableFile(path, status, detail)
    logger.warning("NOT migrated: %s (%s)%s", path, status, f" — {detail}" if detail else "")
    return 0, status


__all__ = [
    "LIGHT_ION",
    "SYMBOL_TO_Z",
    "_RENAMES",
    "canonical_frame",
    "migrate_file",
    "parse_stem",
]


def migrate_file(path: Path, library: str, kind: str, dry_run: bool, *, strict: bool = True) -> tuple[int, str]:
    """Rewrite one parquet file in canonical form. Returns (rows, status).

    Raises `UnmigratableFile` when the file cannot be processed, unless
    `strict=False`. Strict is the default deliberately: the failure modes here
    are *silent no-ops* — the file is left exactly as it was — so a caller that
    ignores the outcome cannot tell a migration from a skip by looking at the
    result. Anything reasoning about migrated data (a round-trip proof, a rebuild
    chain) must not be able to opt out by accident.
    """
    import polars as pl

    parsed = parse_stem(path.stem)
    if parsed is None:
        return _not_migrated(
            path,
            "unparseable-stem",
            f"stem {path.stem!r} is not <projectile>_<Element>, so the row identity "
            "this migration adds cannot be derived",
            strict=strict,
        )
    projectile, proj_z, proj_a, target_z = parsed

    try:
        df = pl.read_parquet(path)
    except Exception as e:
        return _not_migrated(path, "unreadable", str(e)[:120], strict=strict)
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


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser, separately from running it.

    So the defaults can be asserted without executing a migration — the same
    reason the fetch scripts expose one (#341, #363).
    """
    ap = argparse.ArgumentParser(description="Lift legacy cross-section tables into CANONICAL_XS_SCHEMA.")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--library", help="migrate only this library key")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--skip-unmigratable",
        action="store_true",
        help=(
            "Do not stop at the first file that cannot be migrated: process the rest, "
            "then list every skipped path. Still exits non-zero — this surveys the "
            "damage, it does not accept it."
        ),
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()

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
    # Every path that came out still in its legacy shape. A count is not enough
    # to act on, and a count is all this used to produce (#361).
    not_migrated: list[tuple[Path, str]] = []

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
            try:
                rows, status = migrate_file(f, lib_key, kind, args.dry_run, strict=not args.skip_unmigratable)
            except UnmigratableFile as exc:
                logger.error("%s", exc)
                logger.error(
                    "aborting: this file is still in the legacy schema. Re-run with "
                    "--skip-unmigratable to process the rest and list every such file."
                )
                raise SystemExit(1) from exc
            totals[status] = totals.get(status, 0) + 1
            rows_total += rows
            if status not in SUCCESS_STATUSES:
                not_migrated.append((f, status))
        logger.info("%-18s %4d files", lib_key, len(files))

    logger.info("---")
    for status, n in sorted(totals.items()):
        logger.info("%-20s %5d files", status, n)
    logger.info("%d rows total%s", rows_total, " (dry run)" if args.dry_run else "")

    if not_migrated:
        logger.error("---")
        logger.error("%d file(s) are STILL IN THE LEGACY SCHEMA:", len(not_migrated))
        for path, status in not_migrated:
            logger.error("  %s  (%s)", path, status)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
