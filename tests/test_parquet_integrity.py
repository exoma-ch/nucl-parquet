"""Every shipped .parquet must actually open.

Two files (`exfor/n_C`, `exfor/n_Mn`) shipped for at least five months with junk
bytes appended after a complete PAR1 footer — valid Parquet followed by a
duplicated tail write. They were committed, so every data release reproduced them
byte-identically, and nothing caught it until a consumer hit it (#279).

The failure mode is nastier than one bad file: in DuckDB the first read error
aborts the transaction, and every *subsequent* query on that connection then
fails with "Current transaction is aborted" — which makes the ~265 healthy files
in the same directory look broken too.

These tests are cheap (a 4-byte tail read for the sweep) and run without the
`data` marker so they gate every CI run.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

DATA_DIR = Path(__file__).parent.parent / "data"
MAGIC = b"PAR1"

# Deliberate placeholders that are not Parquet. Keep this list tiny and explicit.
_ALLOWED_NON_PARQUET = {"g4_raw/strata-nuclear/ensdfstate.parquet"}


def _shipped_parquets() -> list[Path]:
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.rglob("*.parquet"))


def _rel(path: Path) -> str:
    return path.relative_to(DATA_DIR).as_posix()


@pytest.mark.skipif(not DATA_DIR.exists(), reason="no data tree")
def test_every_parquet_has_valid_magic() -> None:
    """Header and footer must both be PAR1 — catches truncation and tail junk."""
    bad: list[str] = []
    for path in _shipped_parquets():
        rel = _rel(path)
        if rel in _ALLOWED_NON_PARQUET:
            continue
        with path.open("rb") as fh:
            head = fh.read(4)
            if path.stat().st_size < 8:
                bad.append(f"{rel}: {path.stat().st_size} bytes")
                continue
            fh.seek(-4, 2)
            tail = fh.read(4)
        if head != MAGIC or tail != MAGIC:
            bad.append(f"{rel}: head={head!r} tail={tail!r}")

    assert not bad, "Parquet files with invalid magic bytes:\n  " + "\n  ".join(bad)


@pytest.mark.data
@pytest.mark.skipif(not DATA_DIR.exists(), reason="no data tree")
def test_every_parquet_opens_and_counts() -> None:
    """Each file must return a row count.

    A fresh connection per file: a shared one would abort after the first
    failure and report every later file as broken too.
    """
    bad: list[str] = []
    for path in _shipped_parquets():
        rel = _rel(path)
        if rel in _ALLOWED_NON_PARQUET:
            continue
        try:
            con = duckdb.connect()
            con.sql(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()
            con.close()
        except Exception as e:  # noqa: BLE001 — report every failure, not the first
            bad.append(f"{rel}: {str(e).splitlines()[0][:100]}")

    assert not bad, "Parquet files that failed to open:\n  " + "\n  ".join(bad)
