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

from nucl_parquet._schemas import CANONICAL_XS_REQUIRED, CANONICAL_XS_SCHEMA

DATA_DIR = Path(__file__).parent.parent / "data"

_XS_TYPES = {
    "cross_sections",
    "transport_cross_sections",
    "production_cross_sections",
    "total_reaction_cross_sections",
    "experimental_cross_sections",
}


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
    """One shape, so `UNION ALL BY NAME` across sources is meaningful."""
    expected = set(CANONICAL_XS_SCHEMA)
    problems: list[str] = []
    for key, d in _xs_dirs():
        sample = sorted(d.glob("*.parquet"))[0]
        cols = {
            r[0]
            for r in duckdb.connect()
            .sql(f"SELECT name FROM parquet_schema('{sample}') WHERE name != 'root'")
            .fetchall()
        }
        missing = expected - cols
        if missing:
            problems.append(f"{key}: missing {sorted(missing)}")
    assert not problems, "tables not in canonical form:\n  " + "\n  ".join(problems)


@pytest.mark.data
def test_identity_spine_is_never_null() -> None:
    """A row must always say which library, projectile and target it describes.

    `library` and `projectile` living only in the file path is what let the
    unified `xs` view merge five beams into one result set.
    """
    problems: list[str] = []
    for key, d in _xs_dirs():
        sample = sorted(d.glob("*.parquet"))[0]
        con = duckdb.connect()
        checks = ", ".join(f"count(*) FILTER (WHERE {c} IS NULL) AS {c}_nulls" for c in CANONICAL_XS_REQUIRED)
        row = con.sql(f"SELECT {checks} FROM read_parquet('{sample}')").fetchone()
        for col, n in zip(CANONICAL_XS_REQUIRED, row):
            if n:
                problems.append(f"{key}/{sample.name}: {col} has {n} nulls")
    assert not problems, "identity columns must never be null:\n  " + "\n  ".join(problems)


@pytest.mark.data
def test_no_zero_sentinel_residuals() -> None:
    """ "No residual" is NULL, never (0, 0).

    The 0-sentinel collides with a real Z=0 product and made (n,tot), (n,el) and
    (n,f) mutually indistinguishable across the 82% of EXFOR rows that name no
    residual (#279).
    """
    problems: list[str] = []
    for key, d in _xs_dirs():
        for sample in sorted(d.glob("*.parquet"))[:5]:
            n = (
                duckdb.connect()
                .sql(f"SELECT count(*) FROM read_parquet('{sample}') WHERE residual_Z = 0 AND residual_A = 0")
                .fetchone()[0]
            )
            if n:
                problems.append(f"{key}/{sample.name}: {n} rows with residual (0,0)")
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
    for key, d in _xs_dirs():
        sample = sorted(d.glob("*.parquet"))[0]
        kinds = {r[0] for r in duckdb.connect().sql(f"SELECT DISTINCT kind FROM read_parquet('{sample}')").fetchall()}
        bad = kinds - allowed
        if bad:
            problems.append(f"{key}: unexpected kind {sorted(bad)}")
    assert not problems, "\n  ".join(problems)


@pytest.mark.data
def test_channel_rows_carry_mt_and_production_rows_do_not() -> None:
    """The two kinds are distinguished by what identifies the datum."""
    con = duckdb.connect()
    for key, d in _xs_dirs():
        sample = sorted(d.glob("*.parquet"))[0]
        row = con.sql(f"""
            SELECT
              count(*) FILTER (WHERE kind='channel'    AND MT IS NULL),
              count(*) FILTER (WHERE kind='production' AND MT IS NOT NULL)
            FROM read_parquet('{sample}')
        """).fetchone()
        assert row[0] == 0, f"{key}: {row[0]} channel rows without an MT"
        assert row[1] == 0, f"{key}: {row[1]} production rows carrying an MT"
