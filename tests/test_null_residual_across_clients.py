"""Every client must keep a null distinguishable from a real value (#362, #380).

Two columns, one bug class.

`residual_Z = residual_A = NULL` means "this channel names no single product" —
(n,tot), (n,el), (n,f). Collapsing that onto 0 is the sentinel collision
representation principle 3 exists to prevent: Z=0 is a real value, so all three
of those reactions become indistinguishable from each other *and* from a real
Z=0 product, and their curves interleave under one bogus key.

`state` is the same shape and it is live rather than pending: #380 retired `''`
because that one token meant "summed over states", "not stated" and "the ground
state" depending on which table you read, and made absence `NULL`. `NULL` is now
a positive answer distinct from `'g'`, and the heavy-ion tables ship it on every
row — so a client substituting `''` merges them into the ~38M legacy rows today.

The TypeScript client did exactly that to both columns until #362
(`new Int32Array([1, null])` is `[1, 0]`; `?? ""` for the string). This module is
the audit that says the other clients do not, and says it in a way that keeps
being true — "I checked and they are fine" and "I did not check" are otherwise
indistinguishable a month later.

Per-client verification lives with each client where it can be specific:

* **Rust** — `clients/rs/nucl-parquet/src/xs.rs` skips a row when
  `rz.is_null(i) || ra.is_null(i) || ta.is_null(i)`, pinned by
  `null_residuals_are_skipped_not_keyed_as_zero`, `null_target_a_is_skipped`
  and `all_null_residuals_yields_an_empty_db_not_an_error`.
* **TypeScript** — `clients/ts/nucl-parquet/test/xs_nulls.test.ts` carries the
  same three names against the same fixtures.
* **Go** — has no cross-section reader at all; see the tripwire below.
* **Python** — verified here, because both Python paths go through DuckDB and
  the property worth pinning is that DuckDB's nulls survive the hand-off.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
_CLIENTS = _REPO_ROOT / "clients"

# The marker goes on the tests that actually read parquet, not on the module.
# The tripwires at the bottom read source files and want to run everywhere,
# including a checkout with no data tree. (#375 stopped ci.sh from using `data`
# as a CI filter, so this is now about graceful degradation rather than about
# being deselected — but a module-level marker would still wrongly skip the
# tripwires whenever the tree is absent.)


# --- Python: DuckDB nulls survive the hand-off ------------------------------


@pytest.mark.data
def test_python_loader_returns_none_not_zero_for_a_null_residual() -> None:
    """`nucl_parquet.loader` is DuckDB views, so nulls arrive as None.

    Nothing in the loader converts an integer column, which is exactly why it
    was never exposed to this bug — but "nothing converts it" is a property of
    the current implementation, not a guarantee, so pin it.
    """
    from nucl_parquet import loader

    db = loader.connect()
    rows = db.sql("SELECT residual_Z, residual_A, MT FROM channels WHERE residual_Z IS NULL LIMIT 5").fetchall()
    assert rows, "no null-residual rows found — this test would otherwise assert nothing"
    for residual_z, residual_a, mt in rows:
        assert residual_z is None, f"null residual_Z came back as {residual_z!r}"
        assert residual_a is None, f"null residual_A came back as {residual_a!r}"
        # The channel is still identifiable — by MT, its canonical identity.
        assert mt is not None


@pytest.mark.data
def test_python_loader_returns_none_not_empty_string_for_a_null_state() -> None:
    """`state` is the second nullable column of this class, and it is live.

    #380 retired `''` because one token meant "summed over states", "not stated"
    and "the ground state" depending on the table. `NULL` is now a positive
    answer — "not stated" — distinct from `'g'`. The heavy-ion tables ship it on
    every row today, so a client that substitutes `''` merges them into the
    legacy rows immediately rather than at rebuild time.
    """
    from nucl_parquet import loader

    db = loader.connect()
    rows = db.sql("SELECT state FROM read_parquet('data/hi-xs-prod/**/*.parquet') LIMIT 5").fetchall()
    assert rows, "no heavy-ion rows found — this test would otherwise assert nothing"
    for (state,) in rows:
        assert state is None, f"null state came back as {state!r}"


@pytest.mark.data
def test_python_loader_keeps_transport_channels_distinguishable() -> None:
    """The symptom, stated directly: (n,tot), (n,el) and (n,f) stay separate.

    This is the assertion the TS client failed. Grouping null-residual rows by
    MT must yield several distinct channels, not one merged blob.
    """
    from nucl_parquet import loader

    db = loader.connect()
    mts = {
        r[0] for r in db.sql("SELECT DISTINCT MT FROM channels WHERE residual_Z IS NULL AND MT IS NOT NULL").fetchall()
    }
    # 1 = total, 2 = elastic, 18 = fission — all name no single product.
    assert {1, 2, 18} <= mts, f"expected the transport channels to be present, got {sorted(mts)[:20]}"


@pytest.mark.data
def test_duckdb_to_pandas_preserves_the_null() -> None:
    """The MCP server's `fetchdf()` hop is the one place a null could become 0.

    DuckDB maps an INTEGER column with nulls to pandas' nullable `Int32`, so the
    null survives as `None`. Had it mapped to numpy int32 it would have become
    0 and reproduced #362 one client over; had it mapped to float64 the residual
    would silently become `30.0`. Neither is hypothetical enough to leave
    unpinned.
    """
    # Imported, not `importorskip`-ed. pandas is a declared dev dependency
    # precisely so this runs; skipping when it is absent would turn the audit
    # into the silence it exists to detect.
    #
    # Note this pins a DuckDB/pandas property, not ours — which is the point.
    # The Python MCP server relies on that mapping and would reproduce #362 if
    # it ever changed, without a line of our code being touched.
    import duckdb
    import pandas  # noqa: F401

    df = (
        duckdb.connect()
        .sql(
            f"SELECT residual_Z FROM read_parquet('{_REPO_ROOT}/data/endfb-8.0/channels/n_U.parquet') "
            "WHERE residual_Z IS NULL LIMIT 1"
        )
        .fetchdf()
    )
    assert str(df["residual_Z"].dtype) == "Int32", (
        f"expected pandas nullable Int32, got {df['residual_Z'].dtype} — "
        "a numpy int dtype would have turned the null into 0"
    )
    assert df.to_dict(orient="records")[0]["residual_Z"] is None


# --- The tripwire: who reads residual_Z at all ------------------------------

# Every client source file that reads `residual_Z`, and how each handles a null.
# This is an allowlist rather than a scan result: adding a residual reader to a
# client that does not have one is precisely the moment to think about nulls,
# and a silent pass here would be the wrong answer.
_RESIDUAL_READERS = {
    # Skips the row: a residual-keyed map has no key for a channel.
    "rs/nucl-parquet/src/xs.rs",
    # Values + validity mask; `residualKeyedIndices` reproduces the Rust skip.
    "ts/nucl-parquet/src/columns.ts",
    # The tests for both of the above.
    "ts/nucl-parquet/test/xs_nulls.test.ts",
}

_SOURCE_SUFFIXES = {".rs", ".ts", ".go", ".py"}


def test_no_client_reads_residual_z_without_a_documented_null_policy() -> None:
    """A new residual reader must come past this list.

    The Go client has no cross-section reader at all today, which is the only
    reason it is not on this list — not because it handles nulls. If someone
    adds one, this fails and points them at the two implementations that do.
    """
    # Git-tracked files only, rather than a walk with a denylist of directory
    # names. The denylist version listed node_modules/target/dist and still
    # scanned `clients/py/nucl-parquet-mcp/.venv/`, so simply running that
    # package's tests locally installed a copy of `nucl_parquet` and turned this
    # tripwire red on three vendored files. Asking git is not a longer denylist;
    # it is the property actually wanted — only files this repository authors.
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "clients/"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")

    found = set()
    for rel in tracked:
        if not rel:
            continue
        path = _REPO_ROOT / rel
        if path.suffix not in _SOURCE_SUFFIXES or not path.is_file():
            continue
        if re.search(r"\bresidual_Z\b", path.read_text(encoding="utf-8", errors="ignore")):
            found.add(str(path.relative_to(_CLIENTS)))
    assert found, "no client sources scanned — this test would otherwise assert nothing"

    assert found == _RESIDUAL_READERS, (
        f"the set of client files reading residual_Z changed.\n"
        f"  added:   {sorted(found - _RESIDUAL_READERS)}\n"
        f"  removed: {sorted(_RESIDUAL_READERS - found)}\n"
        "A null residual_Z means the row names no product; it must never be read as 0 "
        "(#362). See clients/rs/nucl-parquet/src/xs.rs for the skip-on-null approach and "
        "clients/ts/nucl-parquet/src/columns.ts for the validity-mask one, then add the "
        "new file here."
    )


def test_the_go_client_has_no_cross_section_reader() -> None:
    """States the reason Go is absent above, so it cannot be mistaken for an oversight."""
    go_sources = [
        _REPO_ROOT / rel
        for rel in subprocess.run(
            ["git", "ls-files", "-z", "clients/go/"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\0")
        if rel.endswith(".go")
    ]
    assert go_sources, "no Go sources found — this test would assert nothing"
    offenders = [
        str(p.relative_to(_CLIENTS))
        for p in go_sources
        if re.search(r"\b(residual_[ZA]|xs_mb|target_A)\b", p.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        f"{offenders} now read cross-section columns. The Go client had none when #362 was "
        "audited; give it the same null-residual handling as the Rust and TS clients and "
        "add it to _RESIDUAL_READERS."
    )
