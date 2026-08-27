"""Invariants of `CANONICAL_XS_SCHEMA`.

Every cross-section table shares one shape so that evaluated production data,
transport channels and EXFOR measurements union without special cases. These
tests pin the properties that make that true — each one corresponds to a defect
the canonical schema was introduced to make unrepresentable.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest

from nucl_parquet._schemas import (
    CANONICAL_XS_REQUIRED,
    CANONICAL_XS_SCHEMA,
    PENDING_COLUMN_ADDITION,
)

DATA_DIR = Path(__file__).parent.parent / "data"

_XS_TYPES = {
    "cross_sections",
    "transport_cross_sections",
    "production_cross_sections",
    "total_reaction_cross_sections",
    "experimental_cross_sections",
}


def _xs_dirs_with_kind() -> list[tuple[str, Path, bool]]:
    """`_xs_dirs`, plus whether the library is *measured* rather than evaluated.

    EXFOR's MT is assigned by a compiler reading SF-fields, not authored against
    ENDF-102, so a handful of MT semantics differ there. Checks that assert ENDF
    meaning have to know which they are looking at.
    """
    catalog_path = DATA_DIR / "catalog.json"
    catalog = json.loads(catalog_path.read_text()) if catalog_path.exists() else {}
    experimental = {
        key
        for key, info in catalog.get("libraries", {}).items()
        if info.get("data_type") == "experimental_cross_sections"
    }
    return [(key, d, key in experimental) for key, d in _xs_dirs()]


def _xs_dirs() -> list[tuple[str, Path]]:
    catalog_path = DATA_DIR / "catalog.json"
    if not catalog_path.exists():
        return []
    catalog = json.loads(catalog_path.read_text())
    out = []
    for key, info in catalog.get("libraries", {}).items():
        if info.get("data_type") not in _XS_TYPES or "path" not in info:
            continue
        d = DATA_DIR / info["path"]
        if d.exists() and any(d.glob("*.parquet")):
            out.append((key, d))
    return out


pytestmark = pytest.mark.skipif(not _xs_dirs(), reason="no cross-section data")


@pytest.mark.data
def test_all_xs_tables_share_the_canonical_schema() -> None:
    """One shape, so `UNION ALL BY NAME` across sources is meaningful.

    Columns in `PENDING_COLUMN_ADDITION` are excused: the builder writes them but
    the shipped parquets predate the rebuild that fills them. The excuse is
    per-column and named, not "tolerate anything missing".
    """
    expected = set(CANONICAL_XS_SCHEMA) - set(PENDING_COLUMN_ADDITION)
    problems: list[str] = []
    con = duckdb.connect()
    for key, d in _xs_dirs():
        for sample in sorted(d.glob("*.parquet")):
            cols = {
                r[0] for r in con.sql(f"SELECT name FROM parquet_schema('{sample}') WHERE name != 'root'").fetchall()
            }
            missing = expected - cols
            if missing:
                problems.append(f"{key}/{sample.name}: missing {sorted(missing)}")
    assert not problems, "tables not in canonical form:\n  " + "\n  ".join(problems)


def test_pending_columns_are_actually_in_the_schema() -> None:
    """The ledger may only excuse columns the schema declares.

    An entry naming a column that no longer exists excuses nothing and hides a
    typo in the one place a typo would silently widen the exemption.
    """
    unknown = sorted(set(PENDING_COLUMN_ADDITION) - set(CANONICAL_XS_SCHEMA))
    assert not unknown, f"PENDING_COLUMN_ADDITION names columns not in CANONICAL_XS_SCHEMA: {unknown}"


def test_every_pending_column_names_an_issue() -> None:
    """A debt with no issue number is a decision nobody agreed to."""
    for column, reason in PENDING_COLUMN_ADDITION.items():
        assert "#" in reason, f"{column}: pending reason must cite the issue that clears it — got {reason!r}"


@pytest.mark.data
def test_the_pending_column_ledger_is_self_cleaning() -> None:
    """Once every table carries a pending column, its entry must be deleted.

    Otherwise the ledger silently becomes a permanent exemption, and the schema
    test stops checking the column it was added for. This is the same
    self-cleaning contract as `state_vocabulary.PENDING_MIGRATION`.
    """
    con = duckdb.connect()
    for column in sorted(PENDING_COLUMN_ADDITION):
        everywhere = True
        for _key, d in _xs_dirs():
            for sample in sorted(d.glob("*.parquet")):
                cols = {
                    r[0]
                    for r in con.sql(f"SELECT name FROM parquet_schema('{sample}') WHERE name != 'root'").fetchall()
                }
                if column not in cols:
                    everywhere = False
                    break
            if not everywhere:
                break
        assert not everywhere, (
            f"every shipped xs table now carries {column!r} — delete its "
            "PENDING_COLUMN_ADDITION entry so the schema test enforces it again."
        )


@pytest.mark.data
def test_identity_spine_is_never_null() -> None:
    """A row must always say which library, projectile and target it describes.

    `library` and `projectile` living only in the file path is what let the
    unified `xs` view merge five beams into one result set.
    """
    problems: list[str] = []
    con = duckdb.connect()
    checks = ", ".join(f"count(*) FILTER (WHERE {c} IS NULL) AS {c}_nulls" for c in CANONICAL_XS_REQUIRED)
    for key, d in _xs_dirs():
        # One glob per library rather than per file — same coverage, far fewer
        # round trips than 3,970 individual reads.
        row = con.sql(f"SELECT {checks} FROM read_parquet('{d}/*.parquet')").fetchone()
        for col, n in zip(CANONICAL_XS_REQUIRED, row):
            if n:
                problems.append(f"{key}: {col} has {n} nulls")
    assert not problems, "identity columns must never be null:\n  " + "\n  ".join(problems)


@pytest.mark.data
def test_no_zero_sentinel_residuals() -> None:
    """ "No residual" is NULL, never (0, 0).

    The 0-sentinel collides with a real Z=0 product and made (n,tot), (n,el) and
    (n,f) mutually indistinguishable across the 82% of EXFOR rows that name no
    residual (#279).
    """
    problems: list[str] = []
    con = duckdb.connect()
    for key, d in _xs_dirs():
        n = con.sql(
            f"SELECT count(*) FROM read_parquet('{d}/*.parquet') WHERE residual_Z = 0 AND residual_A = 0"
        ).fetchone()[0]
        if n:
            problems.append(f"{key}: {n} rows with residual (0,0)")
    assert not problems, "use NULL, not a 0 sentinel:\n  " + "\n  ".join(problems)


@pytest.mark.data
def test_kind_discriminates_production_from_channel() -> None:
    """`kind` must be one of the two known values, so unions cannot double-count.

    A production row is a sum over every channel reaching that residual; a
    channel row is one ENDF MT. Adding them together is wrong, so the
    distinction has to be queryable.
    """
    allowed = {"production", "channel"}
    problems: list[str] = []
    con = duckdb.connect()
    for key, d in _xs_dirs():
        kinds = {r[0] for r in con.sql(f"SELECT DISTINCT kind FROM read_parquet('{d}/*.parquet')").fetchall()}
        bad = kinds - allowed
        if bad:
            problems.append(f"{key}: unexpected kind {sorted(bad)}")
    assert not problems, "\n  ".join(problems)


@pytest.mark.data
def test_fission_rows_name_no_residual() -> None:
    """MT=18 is sigma_f. Fission has no single residual — it makes two fragments.

    A nuclide named after (n,f) is a fission *product yield*; ENDF keeps those in
    MF=8/MT=454 rather than MF=3/MT=18 for exactly this reason. 15,630 yield rows
    once carried MT=18, which made the obvious query -- `WHERE MT = 18` -- sum
    sigma_f with a dozen fragment curves. The fragments peak near 100 mb where
    U-235 sigma_f is thousands of barns, so it never looked wrong.
    """
    problems: list[str] = []
    con = duckdb.connect()
    for key, d in _xs_dirs():
        n = con.sql(
            f"SELECT count(*) FROM read_parquet('{d}/*.parquet') WHERE MT IN (18,19,20,21,38) AND residual_Z IS NOT NULL"
        ).fetchone()[0]
        if n:
            problems.append(f"{key}: {n} fission rows naming a residual")
    assert not problems, "fission yields are production rows, not MT=18:\n  " + "\n  ".join(problems)


@pytest.mark.data
def test_named_residuals_are_reachable_from_the_channel() -> None:
    """Where MT fixes the residual, the stored residual must agree with it.

    This is principle 2 held from the other side: MT is the primitive and the
    residual is derived, so a residual that the MT cannot produce means the SF4
    token was attached to the wrong reaction. Two cases are checkable without a
    full reaction table:

      * MT 1/2/3 (total, elastic, nonelastic) leave Z unchanged.
      * MT 4 is `(z,n)` — see below, it depends on who wrote the MT.
      * MT 102 (capture) absorbs the projectile whole.

    A=0 targets are natural elements, so only Z is constrained for them.
    """
    problems: list[str] = []
    con = duckdb.connect()
    for key, d, experimental in _xs_dirs_with_kind():
        # MT=4 means `(z,n)` — one neutron out — so it leaves Z alone only when
        # the projectile is itself a neutron. Charged-particle evaluations
        # transmute: TENDL's a+Al-30 MT=4 makes P-33, Z 13 -> 15, and that is the
        # definition working. This check asserted `residual_Z = target_Z` for all
        # of MT=1/2/3/4 and so flagged 329,928 correct rows the moment the rebuild
        # gave the charged-particle sublibraries their MT column.
        #
        # EXFOR does not author MTs, it derives them from SF-fields, and for
        # charged particles it files *inelastic scattering* under MT=4 — the
        # residual is the target, which is right for what was measured and not
        # what ENDF's MT=4 means. So the correct rule differs by provenance, and
        # each side is pinned to the single convention it actually uses rather
        # than both being widened to accept either:
        #
        #   evaluated       329,928 charged-particle MT=4 rows, all transmuting
        #   exfor-channels    6,735 charged-particle MT=4 rows, all Z-preserving
        #
        # Neither set has a single row on the other side of that line, so an
        # `OR` here would buy nothing and cost the ability to catch a real swap.
        mt4_rule = "residual_Z <> target_Z" if experimental else "residual_Z <> target_Z + proj_Z"
        row = con.sql(f"""
            SELECT
              count(*) FILTER (WHERE MT IN (1,2,3) AND residual_Z <> target_Z),
              count(*) FILTER (WHERE MT = 4 AND {mt4_rule}),
              count(*) FILTER (WHERE MT = 102 AND residual_Z <> target_Z + proj_Z),
              count(*) FILTER (WHERE MT = 102 AND target_A > 0 AND residual_A <> target_A + proj_A)
            FROM read_parquet('{d}/*.parquet')
            WHERE residual_Z IS NOT NULL
        """).fetchone()
        labels = (
            "total/elastic/nonelastic changing Z",
            "(z,n) not reaching " + ("the target it scattered off" if experimental else "target+projectile"),
            "capture changing Z",
            "capture changing A",
        )
        for n, what in zip(row, labels):
            if n:
                problems.append(f"{key}: {n} rows with {what}")
    assert not problems, "residual contradicts its MT:\n  " + "\n  ".join(problems)


@pytest.mark.data
def test_the_mt_residual_check_actually_reaches_the_rebuilt_channels() -> None:
    """The check above passes; this proves it passes on rows, not on an empty set.

    Its MT=4 clause was rewritten for the rebuild, and a filter that silently
    matches nothing is the failure mode a green rewrite hides. Both conventions
    must be present in the corpus for the split above to be doing any work.
    """
    con = duckdb.connect()
    counts = {}
    for key, d, experimental in _xs_dirs_with_kind():
        n = con.sql(f"""
            SELECT count(*) FROM read_parquet('{d}/*.parquet')
            WHERE residual_Z IS NOT NULL AND MT = 4 AND proj_Z > 0
        """).fetchone()[0]
        counts["experimental" if experimental else "evaluated"] = (
            counts.get("experimental" if experimental else "evaluated", 0) + n
        )
    assert counts.get("evaluated", 0) > 300_000, f"evaluated (z,n) rows vanished: {counts}"
    assert counts.get("experimental", 0) > 6_000, f"EXFOR (z,n) rows vanished: {counts}"


@pytest.mark.data
def test_channel_rows_carry_mt_and_production_rows_do_not() -> None:
    """The two kinds are distinguished by what identifies the datum."""
    con = duckdb.connect()
    for key, d in _xs_dirs():
        row = con.sql(f"""
            SELECT
              count(*) FILTER (WHERE kind='channel'    AND MT IS NULL),
              count(*) FILTER (WHERE kind='production' AND MT IS NOT NULL)
            FROM read_parquet('{d}/*.parquet')
        """).fetchone()
        assert row[0] == 0, f"{key}: {row[0]} channel rows without an MT"
        assert row[1] == 0, f"{key}: {row[1]} production rows carrying an MT"
