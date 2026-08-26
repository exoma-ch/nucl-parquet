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
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import DATA_DIR, ROOT  # noqa: E402

sys.path.insert(0, str(ROOT))  # so `nucl_parquet` imports from the checkout

from nucl_parquet._schemas import CANONICAL_XS_SCHEMA  # noqa: E402

COMPRESSION = "zstd"

# Legacy -> canonical column names. `exfor_entry` is source-specific naming for
# what is really "which record did this datum come from"; the canonical schema
# generalises it so measurements from any source share one provenance column.
_RENAMES = {"exfor_entry": "source_entry"}

# Light-ion projectile code -> (Z, A). Photons carry ZA = 0, per ENDF.
LIGHT_ION: dict[str, tuple[int, int]] = {
    "n": (0, 1),
    "p": (1, 1),
    "d": (1, 2),
    "t": (1, 3),
    "h": (2, 3),
    "a": (2, 4),
    "g": (0, 0),
}

_ELEMENTS = (
    "n H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl "
    "Mc Lv Ts Og"
).split()
SYMBOL_TO_Z = {s.lower(): i for i, s in enumerate(_ELEMENTS)}

# Heavy-ion projectile stem, e.g. 'ar40' -> ('Ar', 40).
_HEAVY_ION = re.compile(r"^([a-z]{1,2})(\d{1,3})$")
# File stem: <projectile>_<Element>. The target is either an element symbol or,
# for elements the builders have no symbol for (transuranics, Tc, Pm), the
# explicit 'Z<number>' form — e.g. 'p_Z61', 'd_Z105'.
_STEM = re.compile(r"^([a-z]{1,2}\d{0,3})_(Z\d{1,3}|[A-Za-z]{1,2})$")


def parse_stem(stem: str) -> tuple[str, int, int, int] | None:
    """'p_Cu' -> ('p', 1, 1, 29);  'ar40_Ac' -> ('ar40', 18, 40, 89);
    'p_Z61' -> ('p', 1, 1, 61)."""
    m = _STEM.match(stem)
    if not m:
        return None
    proj, elem = m.group(1), m.group(2)
    if elem.startswith("Z") and elem[1:].isdigit():
        target_z = int(elem[1:])
    else:
        target_z = SYMBOL_TO_Z.get(elem.lower())
    if target_z is None:
        return None
    if proj in LIGHT_ION:
        pz, pa = LIGHT_ION[proj]
    else:
        hm = _HEAVY_ION.match(proj)
        if hm is None:
            return None
        pz = SYMBOL_TO_Z.get(hm.group(1).lower())
        if pz is None:
            return None
        pa = int(hm.group(2))
    return proj, pz, pa, target_z


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

    # Legacy column names that the canonical schema spells differently. Without
    # this the select-by-canonical-name below silently *drops* them — EXFOR entry
    # ids are the whole provenance chain back to the measurement.
    renames = {old: new for old, new in _RENAMES.items() if old in df.columns and new not in df.columns}
    if renames:
        df = df.rename(renames)

    have = set(df.columns)
    # hi-xs-prod already carries proj_Z/proj_A; prefer the file's own values.
    if "proj_Z" in have and "proj_A" in have:
        proj_expr = [pl.col("proj_Z").cast(pl.Int32), pl.col("proj_A").cast(pl.Int32)]
    else:
        proj_expr = [
            pl.lit(proj_z, dtype=pl.Int32).alias("proj_Z"),
            pl.lit(proj_a, dtype=pl.Int32).alias("proj_A"),
        ]

    df = df.with_columns(
        pl.lit(library, dtype=pl.Utf8).alias("library"),
        pl.lit(kind, dtype=pl.Utf8).alias("kind"),
        pl.lit(projectile, dtype=pl.Utf8).alias("projectile"),
        *proj_expr,
        # target_Z is absent from the legacy schema — it lived in the filename.
        (
            pl.col("target_Z").cast(pl.Int32)
            if "target_Z" in have
            else pl.lit(target_z, dtype=pl.Int32).alias("target_Z")
        ),
    )

    # Fill the rest of the canonical shape with typed nulls.
    for col, dtype in CANONICAL_XS_SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=getattr(pl, dtype)).alias(col))

    # A 0/0 residual is the legacy sentinel for "this channel names none".
    # Nulls say that truthfully and do not collide with a real Z=0 product.
    if {"residual_Z", "residual_A"} <= set(df.columns):
        no_residual = (pl.col("residual_Z") == 0) & (pl.col("residual_A") == 0)
        df = df.with_columns(
            pl.when(no_residual).then(None).otherwise(pl.col("residual_Z")).cast(pl.Int32).alias("residual_Z"),
            pl.when(no_residual).then(None).otherwise(pl.col("residual_A")).cast(pl.Int32).alias("residual_A"),
        )

    df = df.select([pl.col(c).cast(getattr(pl, t)) for c, t in CANONICAL_XS_SCHEMA.items()]).sort(
        "target_A", "residual_Z", "residual_A", "energy_MeV"
    )

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
