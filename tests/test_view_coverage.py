"""Two views of one library must agree about which elements exist (#329).

`endfb-8.0` ships twice: `xs/` is residual-production (what comes out) and
`channels/` is transport (total, elastic, inelastic, fission, capture). Same
source HDF5, same inventory, two builders — and for eighteen months they
disagreed. `xs/` had plutonium; `channels/` did not.

Nothing could see it. The catalog declares the same projectile for both views
and says nothing about element coverage, so "this library has no Pu" and "Pu's
fission channel was lost in a throttled download" were the same observation:
`SELECT … WHERE MT = 18` for plutonium returned nothing, silently, for the
second-most-important fissile element there is.

## Why this test reports sets and not counts

The build that caused it *did* count. It logged `"37 skipped"` and exited 0 —
twenty-two legitimate metastable skips and fifteen throttled downloads, summed
into one integer that says nothing about which. A count is what hid this. So
every assertion here names the elements on both sides of the difference, and
the direction, because "channels is missing Pu, Ra" and "xs is missing Xe" are
different faults with different fixes.

## Why both directions

The gap in #329 happened to be one-directional, but that is a property of that
incident and not of the class. A channels build that gained an element xs lacks
would be equally wrong and equally invisible, so the check is symmetric — and
the failure message says which way round it is rather than leaving the reader
to diff two sorted lists by eye.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DATA = Path(__file__).parent.parent / "data"

#: Library key -> the two view directories that must cover the same elements.
#:
#: Explicit rather than inferred from the catalog. A rule like "any two dirs
#: under the same library" would silently start policing the next pair somebody
#: adds, and a benign default where a declaration belonged is the failure this
#: repository keeps paying for (#334, #340, #351, #356, #367).
PAIRED_VIEWS: dict[str, tuple[str, str]] = {
    "endfb-8.0": ("endfb-8.0/xs", "endfb-8.0/channels"),
}

#: Elements a pair is allowed to disagree about, with the issue that closes it.
#:
#: Same self-cleaning contract as `PENDING_COLUMN_ADDITION` and
#: `builder_stamp_exemptions.json`: an entry is a debt, not a decision, and
#: `test_the_coverage_ledger_is_self_cleaning` fails once it stops being true.
#:
#: Pu and Ra are here because #329 is a *code* fix — the builders now refuse to
#: report success after losing a nuclide — and regenerating `endfb-8.0/channels`
#: is a data release that has to happen separately. The entry is what keeps this
#: gate switched on for every other element in the meantime.
KNOWN_VIEW_GAPS: dict[str, dict[str, str]] = {
    "endfb-8.0": {
        "Pu": "#329: lost to a throttled download in the #280 build; restored by regenerating channels/",
        "Ra": "#329: lost to a throttled download in the #280 build; restored by regenerating channels/",
    },
}


def _elements(view: str) -> set[str] | None:
    """Element symbols present in a view directory, from its filenames."""
    d = DATA / view
    if not d.is_dir():
        return None
    found = {p.stem.split("_", 1)[1] for p in d.glob("*.parquet") if "_" in p.stem}
    return found or None


def _pairs() -> list[tuple[str, str, str]]:
    out = []
    for lib, (a, b) in PAIRED_VIEWS.items():
        if _elements(a) and _elements(b):
            out.append((lib, a, b))
    return out


pytestmark = pytest.mark.skipif(not _pairs(), reason="paired views not available")


@pytest.mark.data
@pytest.mark.parametrize(("lib", "a", "b"), _pairs())
def test_paired_views_cover_the_same_elements(lib: str, a: str, b: str) -> None:
    """The load-bearing gate, and the one that would have caught #329 on day one.

    It needs nobody to suspect plutonium in particular: it compares the two sets
    and reports whatever is not in both.
    """
    left, right = _elements(a), _elements(b)
    allowed = set(KNOWN_VIEW_GAPS.get(lib, {}))

    only_a = sorted(left - right - allowed)
    only_b = sorted(right - left - allowed)

    problems = []
    if only_a:
        problems.append(f"in {a} but NOT {b}: {only_a}")
    if only_b:
        problems.append(f"in {b} but NOT {a}: {only_b}")
    assert not problems, (
        f"{lib}: the two views disagree about which elements exist (#329).\n  "
        + "\n  ".join(problems)
        + f"\n\n{a} has {len(left)} elements, {b} has {len(right)}. "
        "Either regenerate the short view, or add the element to KNOWN_VIEW_GAPS "
        "with the issue that restores it. Do not 'fix' this by deleting from the "
        "longer side — the union is what the source actually contains."
    )


@pytest.mark.data
@pytest.mark.parametrize(("lib", "a", "b"), _pairs())
def test_the_coverage_ledger_is_self_cleaning(lib: str, a: str, b: str) -> None:
    """An entry that is no longer a gap must go, or it hides the next one.

    Once `channels/` is regenerated with Pu and Ra, these entries stop
    describing reality and start excusing whatever goes missing next.
    """
    left, right = _elements(a), _elements(b)
    stale = sorted(sym for sym in KNOWN_VIEW_GAPS.get(lib, {}) if sym in left and sym in right)
    assert not stale, (
        f"{lib}: {stale} are present in BOTH views now — delete their KNOWN_VIEW_GAPS "
        "entries so the gate covers them again."
    )


@pytest.mark.data
@pytest.mark.parametrize(("lib", "a", "b"), _pairs())
def test_every_declared_gap_names_an_issue(lib: str, a: str, b: str) -> None:
    """So a debt cannot be recorded without saying who pays it."""
    for sym, reason in KNOWN_VIEW_GAPS.get(lib, {}).items():
        assert "#" in reason, f"{lib}/{sym}: reason names no issue: {reason!r}"


@pytest.mark.data
def test_the_gate_can_actually_see_the_329_gap() -> None:
    """The guard on the guard.

    A set-difference check that silently found nothing would look exactly like a
    clean tree — the failure mode this whole file is about. So assert the real,
    still-present #329 difference is detected when the ledger is not consulted.
    """
    left, right = _elements("endfb-8.0/xs"), _elements("endfb-8.0/channels")
    assert left is not None and right is not None
    assert sorted(left - right) == ["Pu", "Ra"], (
        "expected the #329 gap to still be present and detectable; if channels/ has "
        "been regenerated, delete the KNOWN_VIEW_GAPS entries and this control"
    )
    assert not (right - left), "channels/ has gained an element xs/ lacks — the other direction"


@pytest.mark.data
def test_the_missing_elements_really_are_missing_downstream() -> None:
    """What the gap costs, asserted rather than described.

    `channels/` is where MT is populated, so it is what a criticality or
    shielding consumer selects for fission. With Pu absent, MT=18 for plutonium
    returns nothing — and `xs/` carrying 85,480 Pu rows is what makes that look
    like a physics answer rather than a missing file.
    """
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect()
    channels = DATA / "endfb-8.0" / "channels"
    n_fission_pu = con.sql(
        f"SELECT count(*) FROM read_parquet('{channels}/*.parquet') WHERE MT = 18 AND target_Z = 94"
    ).fetchone()[0]
    assert n_fission_pu == 0, "Pu fission is back in channels/ — update this test and the ledger"

    xs = DATA / "endfb-8.0" / "xs"
    n_pu_xs = con.sql(f"SELECT count(*) FROM read_parquet('{xs}/n_Pu.parquet')").fetchone()[0]
    assert n_pu_xs > 80_000, f"xs/ should still carry plutonium, found {n_pu_xs} rows"
