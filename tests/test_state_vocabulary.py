"""Every `state` value in every shipped parquet, checked against one vocabulary.

`state` is supposed to name the isomeric state of a nuclide. Measured across the
tree it did not: four spellings of one concept, one spelling carrying *five*
different meanings, and a column of the same name in `stopping/em` holding phase
of matter (#357, #367).

Nothing asserted any of it, which is the whole reason it drifted. This file is
that assertion, and it is deliberately **per table** — a value that is
meaningful in one table can be meaningless in another, so "legal somewhere" is
not a check. `'sum'` on a measured EXFOR row would be a claim nobody made.

The vocabulary itself lives in `nucl_parquet/state_vocabulary.py`, imported by
both EXFOR builders and all three ENDF builders. One definition, so the two
allowed sets that #367 found in two builders cannot come back.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT))

from nucl_parquet.state_vocabulary import (  # noqa: E402
    GROUND,
    ISOMERS,
    JOINABLE_STATES,
    LEGACY_UNSPECIFIED,
    MAX_ISOMER,
    PENDING_MIGRATION,
    PENDING_RENAME_TABLES,
    STATES,
    SUM,
    TABLE_STATES,
    UNRESOLVED,
    allowed_states,
    is_valid_state,
    isomer_state,
    parse_x4_state,
)

pytestmark = pytest.mark.filterwarnings("ignore")


def _tables_with_state() -> dict[str, list[Path]]:
    """Every shipped table carrying a `state` column -> its shards."""
    tables: dict[str, list[Path]] = {}
    for path in sorted(DATA.rglob("*.parquet")):
        try:
            columns = pl.read_parquet_schema(path)
        except Exception:  # a corrupt shard is test_parquet_integrity's problem
            continue
        if "state" in columns:
            tables.setdefault(str(path.parent.relative_to(DATA)).replace("\\", "/"), []).append(path)
    return tables


@pytest.fixture(scope="module")
def tables() -> dict[str, list[Path]]:
    if not (DATA / "catalog.json").exists():
        pytest.skip("data tree not available")
    found = _tables_with_state()
    assert found, "no shipped table has a `state` column — the scan is broken, not the data"
    return found


# ---------------------------------------------------------------------------
# The vocabulary itself
# ---------------------------------------------------------------------------


def test_the_empty_string_is_not_a_value():
    """The point of #357. `''` meant 'summed over states' on an ENDF row, 'not
    stated' on an EXFOR row and 'the ground state' in meta/ensdf — three claims
    in four bytes, and a fourth reading in `stopping/em`, where the column was
    phase of matter. Absence is NULL; everything else says what it means."""
    assert "" not in STATES
    assert not is_valid_state("")
    assert is_valid_state(None), "NULL is how a row says nothing, and is always allowed"


def test_sum_is_a_word_not_an_empty_string():
    """`'sum'` is a claim about the quantity — 'this is the total over states'.
    Spelling a claim as `''` is how it got confused with the absence of one, and
    an empty string does not survive a CSV or pandas round-trip that coerces it
    to null."""
    assert SUM == "sum"
    assert SUM not in JOINABLE_STATES, "a sum names no nuclide state, so it joins nothing"


def test_ground_has_one_spelling():
    """ENDF MF=10 said `'g'`, meta/ensdf said `''`, for the same physical state.
    So `xs JOIN nuclides USING (Z, A, state)` missed every real ground-state row
    and instead matched the summed rows, silently."""
    assert GROUND == "g"
    assert GROUND in JOINABLE_STATES


def test_first_isomer_is_m_not_m1():
    """EXFOR shipped both. `'m'` is the spelling meta/ensdf uses, so it is the
    one that joins."""
    assert ISOMERS[0] == "m"
    assert "m1" not in STATES
    assert parse_x4_state("M1") == "m"
    assert parse_x4_state("M") == "m"


def test_unresolved_is_kept_and_is_not_null():
    """EXFOR's `L`: 'a metastable state is involved, but this measurement does
    not resolve which'. That is a real datum and is not the same as saying
    nothing, so it is not NULL and not dropped."""
    assert UNRESOLVED == "l"
    assert parse_x4_state("L") == "l"
    assert UNRESOLVED in STATES
    assert UNRESOLVED not in JOINABLE_STATES, "it names no specific state, so it joins nothing"


@pytest.mark.parametrize(("rank", "expected"), [(0, "g"), (1, "m"), (2, "m2"), (3, "m3")])
def test_isomer_ranks_have_one_spelling_each(rank, expected):
    assert isomer_state(rank) == expected


def test_an_unspellable_rank_raises_rather_than_inventing_one():
    """`{0:'g',1:'m',2:'m2'}.get(rank, '')` is the #367 defect in its other
    spelling — rank 3 fell through to `''` and landed on the ground-state key."""
    with pytest.raises(ValueError, match="refusing to invent"):
        isomer_state(MAX_ISOMER + 1)


# ---------------------------------------------------------------------------
# The shipped data
# ---------------------------------------------------------------------------


def test_every_table_with_a_state_column_declares_its_vocabulary(tables):
    """A new `state` column must say what its values mean. Inferring it from the
    path would silently classify the next table somebody adds, which is the
    benign-default failure this repo keeps paying for."""
    undeclared = sorted(set(tables) - set(TABLE_STATES) - PENDING_RENAME_TABLES)
    assert not undeclared, (
        f"tables with a `state` column and no TABLE_STATES entry: {undeclared}\n"
        "Add them to nucl_parquet/state_vocabulary.py, saying what their states mean."
    )


def test_declared_tables_still_exist(tables):
    """The other direction: a declaration for a table that is gone is stale, and
    stale declarations are how a ledger stops describing reality."""
    missing = sorted(set(TABLE_STATES) - set(tables))
    assert not missing, f"TABLE_STATES declares tables that ship no `state` column: {missing}"


def test_no_shipped_state_is_outside_its_table_vocabulary(tables):
    """The load-bearing gate.

    Per table, because `'sum'` is right for an evaluation and wrong for a
    measurement. Tables mid-migration additionally tolerate `''` — and only
    `''` — while their `PENDING_MIGRATION` entry stands.
    """
    violations: list[str] = []
    for table, shards in sorted(tables.items()):
        if table in PENDING_RENAME_TABLES:
            continue  # not an isomeric state at all — checked by the phase tests
        allowed = allowed_states(table)
        seen: set[str | None] = set()
        for shard in shards:
            seen.update(pl.read_parquet(shard, columns=["state"])["state"].unique().to_list())
        for value in sorted(v for v in seen if v is not None):
            if value not in allowed:
                violations.append(f"{table}: {value!r} not in {sorted(allowed)}")
    assert not violations, "state values outside their table's vocabulary:\n" + "\n".join(violations)


def test_the_pending_ledger_is_self_cleaning(tables):
    """An entry is a debt, not a decision. Once a table stops shipping `''` the
    entry must go, or the ledger quietly widens what the gate above permits —
    the same contract as data/builder_stamp_exemptions.json."""
    stale: list[str] = []
    for table in sorted(PENDING_MIGRATION):
        if table not in tables:
            stale.append(f"{table}: named in PENDING_MIGRATION but ships no `state` column")
            continue
        legacy = PENDING_MIGRATION[table].legacy
        seen: set[str | None] = set()
        for shard in tables[table]:
            seen.update(pl.read_parquet(shard, columns=["state"])["state"].unique().to_list())
        gone = sorted(legacy - {v for v in seen if v is not None})
        if gone == sorted(legacy):
            stale.append(f"{table}: no longer ships any of {sorted(legacy)} — delete its PENDING_MIGRATION entry")
    assert not stale, "\n".join(stale)


def test_every_pending_entry_names_an_issue():
    """So a debt cannot be recorded without saying who pays it."""
    for table, pending in sorted(PENDING_MIGRATION.items()):
        assert "#" in pending.reason, f"{table}: reason names no issue: {pending.reason!r}"
        assert pending.legacy, f"{table}: a pending entry must name the values it still ships"
        assert pending.legacy <= {LEGACY_UNSPECIFIED, "m1"}, (
            f"{table}: {sorted(pending.legacy)} — a migration debt covers retired spellings only"
        )


def test_stopping_em_builder_writes_phase_not_state():
    """`density_effect_params` held solid/liquid/gas in a column named `state`,
    while every other table meant isomeric state. Identical name, unrelated
    concept, and a consumer filtering `state` across a glob crossed the two.

    Asserted against the *builder*, because the shipped parquet keeps the old
    column until a stopping rebuild — that debt is `PENDING_COLUMN_RENAME`."""
    source = (ROOT / "nucl_parquet" / "build_em_stopping.py").read_text()
    assert 'pl.col("state").alias("phase")' in source, (
        "build_em_stopping.py must project strata's `state` as `phase` (#357)"
    )


def test_the_column_rename_ledger_is_self_cleaning(tables):
    """Once the rebuild lands and the column is gone, the entry must go too."""
    stale = [t for t in PENDING_RENAME_TABLES if t not in tables]
    assert not stale, (
        f"{stale} no longer ship a `state` column — delete their PENDING_COLUMN_RENAME "
        "entries and add them to TABLE_STATES only if they hold isomeric states"
    )


def test_phase_values_are_phases_not_states():
    """The counterpart: the renamed column must still hold what it always held,
    so this cannot be 'passed' by deleting the data."""
    path = DATA / "stopping" / "em" / "density_effect_params.parquet"
    if not path.exists():
        pytest.skip("stopping/em not available")
    columns = pl.read_parquet_schema(path)
    column = "phase" if "phase" in columns else "state"
    values = set(pl.read_parquet(path, columns=[column])[column].unique().to_list())
    assert values <= {"solid", "liquid", "gas"}, f"unexpected phase values: {sorted(values)}"
    assert {"solid", "liquid", "gas"} <= values, "a phase went missing"
    assert not values & STATES, "phase of matter and isomeric state must not share a value"


# ---------------------------------------------------------------------------
# Negative controls — does the gate above actually discriminate?
# ---------------------------------------------------------------------------


def test_the_gate_rejects_a_value_from_another_table_kind(monkeypatch):
    """`'sum'` is legal in an evaluated library and meaningless in EXFOR. If the
    gate were global rather than per table it would accept it here."""
    assert SUM not in allowed_states("exfor")
    assert SUM in allowed_states("tendl-2025/xs")


def test_the_gate_rejects_the_retired_spelling_once_migration_is_done(monkeypatch):
    """While a PENDING_MIGRATION entry stands, `''` is tolerated. Remove the
    entry and it must be rejected — otherwise the ledger is decorative."""
    assert LEGACY_UNSPECIFIED in allowed_states("exfor")
    monkeypatch.delitem(PENDING_MIGRATION, "exfor")
    assert LEGACY_UNSPECIFIED not in allowed_states("exfor")


def test_a_pending_entry_excuses_only_the_empty_string(monkeypatch):
    """A debt for `''` must not become a licence for an unrelated typo."""
    from nucl_parquet.state_vocabulary import PendingMigration

    monkeypatch.setitem(PENDING_MIGRATION, "exfor", PendingMigration(frozenset({LEGACY_UNSPECIFIED}), "#357: test"))
    allowed = allowed_states("exfor")
    assert LEGACY_UNSPECIFIED in allowed
    assert "m1" not in allowed, "an entry naming only '' must not also excuse 'm1'"
    assert "solid" not in allowed


def test_an_undeclared_table_raises():
    with pytest.raises(KeyError, match="no entry in TABLE_STATES"):
        allowed_states("some-new-library/xs")
