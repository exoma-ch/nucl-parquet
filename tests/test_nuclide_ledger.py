"""A build that loses a requested nuclide must not report success (#329).

`endfb-8.0/channels/` shipped without plutonium for eighteen months because a
throttle window during the #280 build hit fifteen consecutive stems, the
exception was caught, downgraded to a warning, folded into an anonymous
`"37 skipped"`, and the run exited 0. Twenty-two of those thirty-seven were
correct — a metastable target has no shard slot — and fifteen were the defect.

The comment defending that behaviour said *"one bad nuclide must not sink the
build"*. It reads as resilience and is the #334 defect stated as a policy:
tolerating a transient failure is only correct if something later notices the
gap, and nothing did. The failure was transient; the tolerance made it permanent.

So these tests pin the two halves of the fix:

* the **distinction** — an expected skip and a loss must never be the same
  number again;
* the **refusal** — a loss raises, and `--allow-missing` surveys without
  accepting, still exiting non-zero.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from build_neutron_njoy import LostNuclides, NuclideLedger  # noqa: E402


def test_an_expected_skip_is_not_a_loss():
    """The distinction the old counter destroyed.

    A metastable target has no `n_<Sym>` slot, so producing nothing is the right
    answer and must not fail the build. Same split #383 made for provenance and
    #386 for band attribution, and load-bearing every time.
    """
    led = NuclideLedger()
    led.skip("Ag110_m1", "metastable/unknown — no n_<Sym> shard slot")
    led.ok("Fe56")
    assert led.lost == []
    led.check("endfb-8.0", allow_missing=False)  # must not raise


def test_a_failed_fetch_is_a_loss_and_raises():
    """The #329 mechanism exactly: the fetch raised, and the build went on."""
    led = NuclideLedger()
    led.ok("Pt198")
    led.fail("Pu239", RuntimeError("429 Too Many Requests"))
    led.ok("Rb85")
    assert led.lost == ["Pu239"]
    with pytest.raises(LostNuclides, match="Pu239"):
        led.check("endfb-8.0-channels", allow_missing=False)


def test_a_nuclide_that_produced_no_rows_is_also_a_loss():
    """`if not rows: skipped += 1` in build_channels.py had no log line at all,
    so this outcome left no trace anywhere — not even a warning to scroll past."""
    led = NuclideLedger()
    led.nothing("Ra226")
    assert led.lost == ["Ra226"]
    with pytest.raises(LostNuclides, match="Ra226"):
        led.check("endfb-8.0-channels", allow_missing=False)


def test_the_refusal_names_every_loss_not_just_a_count():
    """A count is what hid this for months. The message must carry the names, so
    a human reading CI output learns *which* nuclides left."""
    led = NuclideLedger()
    for stem in ("Pu236", "Pu237", "Pu238"):
        led.fail(stem, RuntimeError("429"))
    led.nothing("Ra226")
    with pytest.raises(LostNuclides) as exc:
        led.check("endfb-8.0-channels", allow_missing=False)
    message = str(exc.value)
    for stem in ("Pu236", "Pu237", "Pu238", "Ra226"):
        assert stem in message, f"{stem} is lost but unnamed in the refusal"
    assert "4 nuclide(s)" in message


def test_allow_missing_surveys_without_accepting():
    """Mirrors `migrate_xs_schema.py --skip-unmigratable`: process everything,
    report the whole picture, and still fail. Surveying damage is not accepting
    it, and the exit code is what keeps those apart."""
    led = NuclideLedger()
    led.fail("Pu239", RuntimeError("429"))
    led.check("endfb-8.0-channels", allow_missing=True)  # does not raise
    assert led.lost == ["Pu239"], "the loss must still be recorded for the exit code"


def test_the_summary_keeps_the_four_outcomes_apart():
    """`37 skipped` conflated three of these. The summary must not."""
    led = NuclideLedger()
    led.ok("Fe56")
    led.skip("Ag110_m1", "metastable")
    led.fail("Pu239", RuntimeError("429"))
    led.nothing("Ra226")
    s = led.summary()
    assert "1 built" in s
    assert "1 skipped as expected" in s
    assert "1 failed" in s
    assert "1 empty" in s


@pytest.mark.parametrize("module_name", ["build_channels", "build_neutron_njoy"])
def test_both_builders_refuse_by_default(module_name):
    """The flag is opt-in on both, so the safe behaviour is the one you get
    without thinking about it — which is the whole lesson of #329."""
    module = __import__(module_name)
    parser = module.build_parser()
    action = next((a for a in parser._actions if a.dest == "allow_missing"), None)
    assert action is not None, f"{module_name} has no --allow-missing"
    assert parser.parse_args([]).allow_missing is False, "refusing must be the default"


@pytest.mark.parametrize("module_name", ["build_channels", "build_neutron_njoy"])
def test_both_builders_return_the_ledger_so_main_can_set_an_exit_code(module_name):
    """`build()` returning the ledger is what lets `--allow-missing` write the
    library and still exit non-zero, with no module-level state to get stale."""
    import inspect

    module = __import__(module_name)
    sig = inspect.signature(module.build)
    assert "allow_missing" in sig.parameters
    assert sig.return_annotation is not None and "None" != str(sig.return_annotation)


# ---------------------------------------------------------------------------
# The same conflation, one level down
# ---------------------------------------------------------------------------


def test_only_a_metastable_stem_is_an_expected_skip():
    """`parse_nuclide_stem` returns None for three unrelated reasons and only one
    of them is a correct outcome (#329, again).

    A metastable target genuinely has no `n_<Sym>` slot. But an unknown element
    symbol means `_SYMBOL_TO_Z` has fallen behind the upstream inventory, and an
    unparseable stem means the naming convention moved — both are an element
    about to go missing quietly, which is the whole defect this file is about.

    All 22 stems the VIII.0 inventory skips today are metastable, so the other
    two branches are latent. That is precisely when to separate them, rather
    than after an inventory refresh has already dropped something.
    """
    from build_neutron_njoy import stem_skip_reason

    assert stem_skip_reason("Ag110_m1") == "metastable"
    assert stem_skip_reason("Am242_m1") == "metastable"
    assert stem_skip_reason("Zz42") == "unknown-symbol"
    assert stem_skip_reason("not-a-stem") == "unparseable"


def test_an_unnameable_stem_is_a_loss_not_a_skip():
    """The consequence: it must reach `failed`, so the build refuses."""
    led = NuclideLedger()
    led.skip("Ag110_m1", "metastable target — no n_<Sym> shard slot")
    led.fail("Zz42", "stem not understood (unknown-symbol)")
    assert led.lost == ["Zz42"], "an unnameable stem must count as lost, not skipped"
    with pytest.raises(LostNuclides, match="Zz42"):
        led.check("endfb-8.0-channels", allow_missing=False)
