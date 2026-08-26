"""Rewrite shipped parquets into the `state` vocabulary of #357/#367.

The builders now emit the vocabulary directly, so most tables are fixed by a
re-ingest. Four are not, and this script is the only thing that can fix them:

    iaea-pd-2019, jendl-ad-2017, jendl-deu-2020, tendl-2023-iso

Those are the #346 external-builder libraries — `data/builder_stamp_exemptions.json`
records that no ingest script for them exists in this repository. There is
nothing to re-run, so an in-place migration is not a shortcut here, it is the
only path. That is also why this script raises rather than counting: a
migration that silently skips a table is how those tables came to disagree with
everything else in the first place.

What it does, per table kind (see `nucl_parquet.state_vocabulary`):

    evaluated xs   ''   -> 'sum'   (the MF=3 "summed over states" claim)
    measured xs    ''   -> NULL    ("the measurement did not say")
                   'm1' -> 'm'     (X4 synonym; one spelling reaches disk)
    stopping/em    column `state` -> `phase`  (phase of matter, never a state)

The nuclide-identity tables under `meta/` are deliberately **not** handled. Their
`''` means "the ground state" for 3,148 of 3,161 rows and something unresolved
for the other 13 — levels between 124.5 and 2166.1 keV that carry no isomer
flag. Mapping those to `'g'` would assert "ground state" about a 2 MeV level,
which is the same defect class this migration exists to remove. See the
follow-up issue named in `PENDING_MIGRATION`.

Idempotent: a table already in the new vocabulary is reported `already-migrated`
and its bytes are left untouched.

Usage:
    nix develop -c uv run python scripts/migrate_state_vocabulary.py --dry-run
    nix develop -c uv run python scripts/migrate_state_vocabulary.py --table tendl-2023-iso/xs
    nix develop -c uv run python scripts/migrate_state_vocabulary.py
    nix develop -c uv run python scripts/migrate_state_vocabulary.py --verify
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import DATA_DIR, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT))

from nucl_parquet.state_vocabulary import (  # noqa: E402
    LEGACY_UNSPECIFIED,
    MEASURED_XS_STATES,
    PENDING_COLUMN_RENAME,
    PENDING_MIGRATION,
    SUM,
    TABLE_STATES,
    allowed_states,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

COMPRESSION = "zstd"

#: Statuses that mean the table is now in the new vocabulary. Anything else is
#: not, and must not be reported as a successful run (#361).
SUCCESS_STATUSES = frozenset({"migrated", "already-migrated"})


class UnmigratableTable(RuntimeError):
    """A table the migration could not process — it is still in its old shape.

    Raised rather than counted, deliberately. Every failure mode here is a
    *silent no-op* — the file is left exactly as it was — so a caller that only
    tallied statuses could not tell "17 tables migrated" from "17 tables
    skipped", which is precisely what #361 removed from `migrate_xs_schema`.
    """

    def __init__(self, table: str, status: str, detail: str = "") -> None:
        self.table = table
        self.status = status
        self.detail = detail
        super().__init__(f"{table}: {status}" + (f" — {detail}" if detail else ""))


def shards(data_dir: Path, table: str) -> list[Path]:
    """Every parquet shard of `table`, which is a directory under `data/`."""
    directory = data_dir / table
    if not directory.is_dir():
        raise UnmigratableTable(table, "missing", f"{directory} is not a directory")
    return sorted(directory.glob("*.parquet"))


def migrated_state_column(table: str, states: pl.Series) -> pl.Series:
    """Map one table's `state` column onto the new vocabulary.

    Raises for a value the vocabulary cannot place, rather than passing it
    through or defaulting it. An unknown state is the thing this whole change is
    about; quietly carrying it forward would migrate the spelling and keep the
    defect.
    """
    target = TABLE_STATES[table]
    measured = target == MEASURED_XS_STATES

    mapping: dict[str, str | None] = {LEGACY_UNSPECIFIED: None if measured else SUM}
    if measured:
        mapping["m1"] = "m"

    unknown = sorted(v for v in states.unique().to_list() if v is not None and v not in target and v not in mapping)
    if unknown:
        raise UnmigratableTable(table, "unknown-state", f"values the vocabulary cannot place: {unknown}")

    return states.replace_strict(mapping, default=pl.first(), return_dtype=pl.Utf8)


def migrate_table(data_dir: Path, table: str, *, dry_run: bool) -> tuple[int, str]:
    """Rewrite every shard of `table`. Returns (rows_changed, status)."""
    paths = shards(data_dir, table)
    if not paths:
        raise UnmigratableTable(table, "no-shards", "declared in TABLE_STATES but ships no parquet")

    allowed_now = allowed_states(table)
    legacy = PENDING_MIGRATION[table].legacy if table in PENDING_MIGRATION else frozenset()

    changed = 0
    for path in paths:
        df = pl.read_parquet(path)
        if "state" not in df.columns:
            raise UnmigratableTable(table, "no-state-column", str(path))

        present = {v for v in df["state"].unique().to_list() if v is not None}
        stray = sorted(present - allowed_now)
        if stray:
            raise UnmigratableTable(table, "unknown-state", f"{path.name}: {stray}")
        if not (present & legacy):
            continue  # this shard is already in the new vocabulary

        new_state = migrated_state_column(table, df["state"])
        n = int((df["state"] != new_state).sum() + (new_state.is_null() & df["state"].is_not_null()).sum())
        changed += n
        if not dry_run:
            df.with_columns(new_state.alias("state")).write_parquet(path, compression=COMPRESSION)

    if changed == 0:
        return 0, "already-migrated"
    return changed, "migrated"


def rename_state_to_phase(data_dir: Path, target: str, *, dry_run: bool) -> tuple[int, str]:
    """`density_effect_params`' `state` column holds solid/liquid/gas.

    Rename, never revalue: the values were always correct, the *name* was the
    defect. `target` is a file path relative to `data/`, because the directory
    also holds a parquet that never had the column.
    """
    path = data_dir / target
    if not path.is_file():
        raise UnmigratableTable(target, "missing", f"{path} is not a file")

    df = pl.read_parquet(path)
    if "phase" in df.columns and "state" not in df.columns:
        return 0, "already-migrated"
    if "state" not in df.columns:
        raise UnmigratableTable(target, "no-state-column", str(path))
    if "phase" in df.columns:
        raise UnmigratableTable(target, "both-columns", "has `state` and `phase`")

    if not dry_run:
        df.rename({"state": "phase"}).write_parquet(path, compression=COMPRESSION)
    return df.height, "migrated"


def verify(data_dir: Path) -> list[str]:
    """Every complaint the new vocabulary has about the tree. Empty is success."""
    problems: list[str] = []
    for table in sorted(TABLE_STATES):
        directory = data_dir / table
        if not directory.is_dir():
            problems.append(f"{table}: declared but absent")
            continue
        seen: set[str | None] = set()
        for path in sorted(directory.glob("*.parquet")):
            seen.update(pl.read_parquet(path, columns=["state"])["state"].unique().to_list())
        outside = sorted(v for v in seen if v is not None and v not in TABLE_STATES[table])
        if outside:
            problems.append(f"{table}: {outside} outside its vocabulary")
    for target in sorted(PENDING_COLUMN_RENAME):
        path = data_dir / target
        if path.is_file() and "state" in pl.read_parquet_schema(path):
            problems.append(f"{target}: still has a `state` column")
    return problems


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Migrate shipped parquets into the #357 `state` vocabulary.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--table", help="migrate only this table (e.g. tendl-2023-iso/xs)")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="check the tree against the vocabulary and exit; writes nothing",
    )
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.verify:
        problems = verify(args.data_dir)
        for problem in problems:
            logger.error("  %s", problem)
        if problems:
            raise SystemExit(f"{len(problems)} table(s) do not match the vocabulary")
        logger.info("every table matches the #357 vocabulary")
        return

    wanted = [args.table] if args.table else [*sorted(PENDING_MIGRATION), *sorted(PENDING_COLUMN_RENAME)]
    unknown = [t for t in wanted if t not in TABLE_STATES and t not in PENDING_COLUMN_RENAME]
    if unknown:
        raise SystemExit(f"not a declared table: {unknown}")

    total = 0
    statuses: dict[str, str] = {}
    for table in wanted:
        # meta/* is deliberately out of scope — see the module docstring.
        if table.startswith("meta"):
            logger.info("skipping %s: the nuclide-identity tables need their own pass", table)
            continue
        migrate = rename_state_to_phase if table in PENDING_COLUMN_RENAME else migrate_table
        rows, status = migrate(args.data_dir, table, dry_run=args.dry_run)
        statuses[table] = status
        total += rows
        logger.info("  %-28s %-16s %8d row(s)", table, status, rows)

    bad = {t: s for t, s in statuses.items() if s not in SUCCESS_STATUSES}
    if bad:
        raise SystemExit(f"tables not in the new vocabulary: {bad}")

    verb = "would change" if args.dry_run else "changed"
    logger.info("%s %d row(s) across %d table(s)", verb, total, len(statuses))
    if not args.dry_run:
        logger.info("re-run with --verify to confirm, then recompute catalog.json::data_sha256")


if __name__ == "__main__":
    main()
