"""The MT -> residual-product mapping in `scripts/fetch_endf_libs.py`, checked
against two oracles that are independent of it and of each other.

`MT_EMITTED_PARTICLES` says which particles leave a reaction channel;
`mt_to_residual` subtracts them from target + projectile to name the product.
Thirteen of ~30 entries carried the wrong (Z, A) for years, and every affected
row was filed under the wrong nuclide — not dropped, *misattributed*, so the row
looked fine and the cross-section was a plausible number (#351). MT=44 was
commented `n + 2p` and coded (2, 6); MT=111 `(x,2p)` was coded (1, 2), which is
a deuteron; MT=109 `(x,3α)` was coded (4, 11) where three alphas are (6, 12).

Nothing could catch that, because the table was the only statement of the fact.
A test written from the table would have agreed with it, right or wrong. So both
oracles here are external:

* **`endf.reaction.REACTION_NAME`** — a third-party transcription of ENDF-102's
  reaction names, shipped by the `endf` package. Checks *which particles* each
  MT emits. Covers every entry, needs no network, and cannot go stale.
* **MF=10's `IZAP`** — the product identifier real evaluators write down
  directly, read from committed excerpts of four real evaluations
  (`tests/fixtures/mf10/`). Checks the whole path: table, arithmetic, and all.
  This is the oracle that found #351.

Neither is derived from the table, and they disagree about nothing here, which
is the point of having both.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures" / "mf10"
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_endf_libs import (  # noqa: E402
    LEVEL_RANGE_PARTICLES,
    LEVEL_RANGES,
    MT_EMITTED_PARTICLES,
    MT_TO_EMISSION,
    NO_RESIDUAL_MTS,
    PARTICLE_ZA,
    emitted_za,
    mt_to_residual,
)

# ---------------------------------------------------------------------------
# Oracle 1: ENDF-102's reaction names, via the `endf` package
# ---------------------------------------------------------------------------
#
# `endf.reaction.REACTION_NAME` spells MT=11 '(n,2nd)' and MT=44 '(n,n2p)'.
# Parsing that into a particle multiset and comparing it against our tuple asks
# the one question the old table could not answer: does entry N describe the
# reaction that MT number actually means?

#: How `REACTION_NAME` spells each particle, mapped onto our symbols.
_NAME_TO_SYMBOL = {"n": "n", "p": "p", "d": "d", "t": "t", "3He": "h", "a": "a"}

#: The two names `REACTION_NAME` writes as words rather than as particle lists.
_WORD_NAMES = {"level": ("n",), "gamma": ("g",)}

#: Names covering MTs that emit no single well-defined set of particles, and so
#: are expected to be absent from our table. Stated here rather than read off
#: `NO_RESIDUAL_MTS`, so the test asserts an expectation instead of echoing the
#: code it is checking.
_NOT_A_SINGLE_PRODUCT = {
    "total",
    "elastic",
    "misc",  # MT=5, "anything": a different product in every evaluation
    "fission",
    "f",
    "nf",
    "2nf",
    "3nf",
    "absorption",
    "disappear",
    # The continuum MTs. Real channels, but `LEVEL_RANGE_PARTICLES` covers them
    # by MT range rather than one entry each, so they are checked separately.
    "nc",
    "pc",
    "dc",
    "tc",
    "3Hec",
    "ac",
    "2nc",
}


def _parse_reaction_name(name: str) -> tuple[str, ...] | None:
    """'(n,2nd)' -> ('n', 'n', 'd'). None if it names no single particle set."""
    import re

    body = name[name.index(",") + 1 : -1]
    if body in _NOT_A_SINGLE_PRODUCT:
        return None
    if body in _WORD_NAMES:
        return _WORD_NAMES[body]
    particles: list[str] = []
    pos = 0
    while pos < len(body):
        m = re.match(r"(\d*)(3He|[npdta])", body[pos:])
        if m is None:  # a discrete-level suffix such as '(n,p3)'
            return None
        count, symbol = m.groups()
        particles.extend([_NAME_TO_SYMBOL[symbol]] * int(count or 1))
        pos += m.end()
    return tuple(particles)


@pytest.fixture(scope="module")
def reaction_names() -> dict[int, str]:
    from endf.reaction import REACTION_NAME

    return REACTION_NAME


def test_the_name_parser_reads_endf_102_spellings():
    """The oracle adapter is itself worth pinning — a parser that silently
    returned () for everything would make every comparison below pass."""
    assert _parse_reaction_name("(n,2nd)") == ("n", "n", "d")
    assert _parse_reaction_name("(n,n2p)") == ("n", "p", "p")
    assert _parse_reaction_name("(n,3a)") == ("a", "a", "a")
    assert _parse_reaction_name("(n,2n3He)") == ("n", "n", "h")
    assert _parse_reaction_name("(n,n3He)") == ("n", "h")
    assert _parse_reaction_name("(n,nd2a)") == ("n", "d", "a", "a")
    assert _parse_reaction_name("(n,3n2pa)") == ("n", "n", "n", "p", "p", "a")
    assert _parse_reaction_name("(n,level)") == ("n",)  # MT=4, spelled as a word
    assert _parse_reaction_name("(n,gamma)") == ("g",)  # MT=102, likewise
    assert _parse_reaction_name("(n,fission)") is None
    assert _parse_reaction_name("(n,misc)") is None


def test_every_entry_emits_what_endf_102_says_it_emits(reaction_names):
    """The check the old table could not survive.

    MT=44 was coded (2, 6) beside a comment reading `n + 2p`. Ask ENDF-102 what
    MT=44 is and the answer is `(n,n2p)` — one neutron and two protons, (2, 3).
    """
    checked = 0
    wrong: list[str] = []
    for mt, particles in sorted(MT_EMITTED_PARTICLES.items()):
        name = reaction_names.get(mt)
        assert name is not None, f"MT={mt} is in the table but unknown to ENDF-102"
        expected = _parse_reaction_name(name)
        assert expected is not None, f"MT={mt} {name} names no single product — remove it"
        if sorted(particles) != sorted(expected):
            wrong.append(f"MT={mt} {name}: table says {particles}, ENDF-102 says {expected}")
        checked += 1

    # 85 entries as of #351: MT=4-117 plus ENDF-102's high-multiplicity block,
    # MT=152-200. A floor, so deleting entries cannot quietly make this pass.
    assert checked >= 85, f"only {checked} entries checked — the table shrank unexpectedly"
    assert not wrong, "table entries disagree with ENDF-102:\n" + "\n".join(wrong)


def test_no_channel_endf_102_names_is_silently_missing(reaction_names):
    """A missing entry makes `mt_to_residual` return None, and the ingest skips
    the channel without a word — how MT=11, 30 and 114 were lost until #351.

    Bounded to MT < 250 so the discrete-level partials (MT=51-91, 600-849),
    which `LEVEL_RANGE_PARTICLES` covers by range, are not double-counted here.
    """
    missing = [
        f"MT={mt} {name}"
        for mt, name in sorted(reaction_names.items())
        if mt < 250 and mt not in MT_EMITTED_PARTICLES and mt not in NO_RESIDUAL_MTS and _parse_reaction_name(name)
    ]
    assert not missing, (
        "ENDF-102 channels with a single residual that the table does not name:\n"
        + "\n".join(missing)
        + "\n\nAdd them to MT_EMITTED_PARTICLES, or to NO_RESIDUAL_MTS with a reason."
    )


def test_level_ranges_emit_what_their_mt_range_says(reaction_names):
    """MT=600-649 are (x,p) to discrete levels: one proton each, whichever level."""
    for (mt_lo, mt_hi), particles in LEVEL_RANGE_PARTICLES.items():
        for mt in (mt_lo, (mt_lo + mt_hi) // 2, mt_hi):
            name = reaction_names.get(mt)
            if name is None:
                continue
            # '(n,p12)' is the 12th discrete proton level; strip the index.
            body = name[name.index(",") + 1 : -1].rstrip("0123456789")
            expected = _parse_reaction_name(f"(n,{body})") if body else None
            if expected is None:  # the continuum MTs, spelled '(n,pc)' etc.
                continue
            assert sorted(expected) == sorted(particles), f"MT={mt} {name} vs {particles}"


# ---------------------------------------------------------------------------
# Oracle 2: MF=10's IZAP, from real evaluations
# ---------------------------------------------------------------------------


def _fixture_materials():
    """Parse every committed MF=10 excerpt. Yields (name, target_Z, target_A,
    {MT: product (Z, A)}) for the sections that name exactly one product."""
    import endf

    paths = sorted(FIXTURES.rglob("*.endf"))
    assert paths, f"no MF=10 fixtures under {FIXTURES} — see its README"
    for path in paths:
        material = endf.Material(io.StringIO(path.read_text()))
        sections = {mt: sec for (mf, mt), sec in material.section_data.items() if mf == 10}
        assert sections, f"{path.name} carries no MF=10 sections"

        head = next(iter(sections.values()))
        # The evaluation names its own target: ZA = Z*1000 + A, LIS = the
        # target's isomeric state. A metastable target would shift the balance,
        # so the fixture set is ground-state only and this asserts it.
        assert head["LIS"] == 0, f"{path.name} is a metastable target"
        target_z, target_a = head["ZA"] // 1000, head["ZA"] % 1000

        products: dict[int, tuple[int, int]] = {}
        for mt, section in sections.items():
            izaps = {int(level["IZAP"]) for level in section["levels"] if int(level["IZAP"]) > 0}
            if len(izaps) != 1:
                continue  # MT=5 lumps many products; no single residual to check
            izap = izaps.pop()
            products[mt] = (izap // 1000, izap % 1000)
        yield f"{path.parent.name}/{path.stem}", target_z, target_a, products


@pytest.fixture(scope="module")
def evaluations() -> list[tuple[str, int, int, dict[int, tuple[int, int]]]]:
    return list(_fixture_materials())


def _izap_disagreements(evaluations) -> tuple[list[str], int]:
    """Every (evaluation, MT) where mt_to_residual and IZAP name different
    nuclides, plus how many pairs were compared."""
    disagreements: list[str] = []
    checked = 0
    for name, target_z, target_a, products in evaluations:
        for mt, want in sorted(products.items()):
            if mt in NO_RESIDUAL_MTS:
                continue
            got = mt_to_residual(mt, target_z, target_a, 0, 1)
            if mt == 4:
                # Inelastic: the product IS the target, and mt_to_residual
                # deliberately returns None so it cannot swamp a real channel.
                assert want == (target_z, target_a), f"{name} MT=4 -> {want}, expected the target"
                assert got is None
                continue
            checked += 1
            if got != want:
                disagreements.append(f"{name} MT={mt}: mt_to_residual says {got}, IZAP says {want}")
    return disagreements, checked


def test_residuals_match_the_products_real_evaluators_name(evaluations):
    """`mt_to_residual` must land on the nuclide MF=10's IZAP names.

    This is the check that found #351 and the check that stops it returning.
    Under the committed table it failed for MT=23, 25, 29, 35, 36, 37, 44, 45,
    109, 111, 115, 116 and 117, and skipped MT=11, 30 and 114 entirely.
    """
    disagreements, checked = _izap_disagreements(evaluations)
    assert checked >= 50, f"only {checked} (MT, evaluation) pairs checked — did the fixtures shrink?"
    assert not disagreements, "products disagree with the evaluators:\n" + "\n".join(disagreements)


# --- negative controls -----------------------------------------------------
#
# Both oracles pass on the corrected table. So would an oracle that checks
# nothing. These put each defect back and require the check to notice, because
# "the suite is green" and "the suite can tell" are different claims and #351 is
# what the difference costs.


@pytest.mark.parametrize(
    ("mt", "committed_value", "reaction"),
    [
        (44, (2, 6), "(z,n2p)"),
        (111, (1, 2), "(z,2p)"),
        (45, (2, 9), "(z,npα)"),
        (37, (0, 5), "(z,4n) — coded as 5n"),
        (25, (0, 4), "(z,3nα) — coded as 4n"),
        (109, (4, 11), "(z,3α)"),
        (117, (3, 7), "(z,dα)"),
    ],
)
def test_the_izap_oracle_rejects_the_committed_values(evaluations, monkeypatch, mt, committed_value, reaction):
    """Put a wrong (Z, A) back for one MT; the evaluator oracle must object."""
    monkeypatch.setitem(MT_TO_EMISSION, mt, committed_value)
    disagreements, _checked = _izap_disagreements(evaluations)
    assert any(f"MT={mt}:" in d for d in disagreements), (
        f"the oracle accepted the committed value {committed_value} for MT={mt} {reaction} — "
        "it is not checking what it claims to"
    )


@pytest.mark.parametrize(("mt", "wrong"), [(44, ("n", "p")), (111, ("d",)), (11, ("n", "n", "n", "d"))])
def test_the_endf_102_name_oracle_rejects_wrong_particles(reaction_names, monkeypatch, mt, wrong):
    """Name the wrong particles for one MT; the name oracle must object."""
    monkeypatch.setitem(MT_EMITTED_PARTICLES, mt, wrong)
    with pytest.raises(AssertionError, match=f"MT={mt} "):
        test_every_entry_emits_what_endf_102_says_it_emits(reaction_names)


def test_the_missing_entry_check_rejects_a_deletion(reaction_names, monkeypatch):
    """Delete a channel and the completeness check must notice, rather than
    letting it fall through `mt_to_residual` as a silent skip the way MT=11,
    30 and 114 did."""
    monkeypatch.delitem(MT_EMITTED_PARTICLES, 114)
    with pytest.raises(AssertionError, match="MT=114"):
        test_no_channel_endf_102_names_is_silently_missing(reaction_names)


#: Corrected in #351 *and* witnessed by an MF=10 section in the fixtures.
_ORACLE_VERIFIED_CORRECTIONS = frozenset({23, 25, 29, 35, 36, 37, 44, 45, 109, 111, 115, 116, 117})

#: Added in #351 and likewise witnessed.
_ORACLE_VERIFIED_ADDITIONS = frozenset({11, 30, 114})

#: Corrected or added in #351 with **no** MF=10 witness in any of the 74
#: evaluations sampled, so the evaluator oracle cannot speak to them. They rest
#: on `test_every_entry_emits_what_endf_102_says_it_emits` alone — one notch
#: weaker, and named here so that stays visible rather than being assumed
#: covered. MT=113 is why the issue counted 13 wrong entries and this branch
#: corrects 14: `(z,t2α)` was coded (3, 8), and t + 2α is (5, 11).
_HAND_CHECKED_ONLY = frozenset({113}) | frozenset(range(152, 201))


def test_the_fixtures_cover_the_entries_the_oracle_can_reach(evaluations):
    """Each correction and addition the evaluator oracle *can* witness must
    actually be witnessed, or this file is testing something other than the bug."""
    covered = {mt for _n, _z, _a, products in evaluations for mt in products}
    for label, expected in (
        ("corrected", _ORACLE_VERIFIED_CORRECTIONS),
        ("added", _ORACLE_VERIFIED_ADDITIONS),
    ):
        assert expected <= covered, f"no evaluator witness for {label} {sorted(expected - covered)}"


def test_the_hand_checked_entries_are_declared_not_assumed():
    """An unverified entry presented as verified is the failure mode of #351
    itself, so the split is asserted rather than left to a comment: every
    hand-checked MT is in the table, and none of them is quietly claimed as
    oracle-verified."""
    assert _HAND_CHECKED_ONLY <= set(MT_EMITTED_PARTICLES), (
        f"declared hand-checked but not in the table: {sorted(_HAND_CHECKED_ONLY - set(MT_EMITTED_PARTICLES))}"
    )
    overlap = _HAND_CHECKED_ONLY & (_ORACLE_VERIFIED_CORRECTIONS | _ORACLE_VERIFIED_ADDITIONS)
    assert not overlap, f"claimed both hand-checked and oracle-verified: {sorted(overlap)}"


def test_the_hand_checked_entries_really_have_no_evaluator_witness(evaluations):
    """The other half: if a fixture ever *does* witness one of these, the
    bookkeeping above has gone stale and is overstating the gap."""
    covered = {mt for _n, _z, _a, products in evaluations for mt in products}
    now_witnessed = sorted(_HAND_CHECKED_ONLY & covered)
    assert not now_witnessed, (
        f"MT {now_witnessed} now has an MF=10 witness in the fixtures — move it out of "
        "_HAND_CHECKED_ONLY and into _ORACLE_VERIFIED_*, it is better verified than claimed."
    )


def test_the_fixtures_span_more_than_one_library(evaluations):
    """One evaluator's convention is a convention; three agreeing is a fact."""
    libraries = {name.split("/")[0] for name, _z, _a, _p in evaluations}
    assert len(libraries) >= 3, f"only {libraries} represented"


# ---------------------------------------------------------------------------
# The derivation itself
# ---------------------------------------------------------------------------


def test_emission_sums_are_derived_not_written():
    """`MT_TO_EMISSION` must be a view of `MT_EMITTED_PARTICLES`, so there is
    no second spelling of the fact that can drift from the first."""
    assert MT_TO_EMISSION == {mt: emitted_za(p) for mt, p in MT_EMITTED_PARTICLES.items()}
    assert LEVEL_RANGES == {r: emitted_za(p) for r, p in LEVEL_RANGE_PARTICLES.items()}


def test_every_symbol_used_is_a_known_particle():
    symbols = {s for particles in MT_EMITTED_PARTICLES.values() for s in particles}
    symbols |= {s for particles in LEVEL_RANGE_PARTICLES.values() for s in particles}
    assert symbols <= set(PARTICLE_ZA), f"unknown particle symbols: {symbols - set(PARTICLE_ZA)}"


@pytest.mark.parametrize(
    ("particles", "expected"),
    [
        (("n",), (0, 1)),
        (("p", "p"), (2, 2)),  # 2p, not the (1, 2) deuteron MT=111 used to claim
        (("a", "a", "a"), (6, 12)),  # 3α, not (4, 11)
        (("n", "p", "p"), (2, 3)),  # MT=44, not (2, 6)
        (("n", "p", "a"), (3, 6)),  # MT=45, not (2, 9)
        (("g",), (0, 0)),  # γ carries neither
        ((), (0, 0)),
    ],
)
def test_emitted_za_sums_particles(particles, expected):
    assert emitted_za(particles) == expected


@pytest.mark.parametrize(
    ("mt", "expected"),
    [
        # Fe-56 + n. Worked by hand from the reaction name, not from the table.
        (16, (26, 55)),  # (n,2n)  -> Fe-55
        (37, (26, 53)),  # (n,4n)  -> Fe-53, NOT Fe-52: the committed table
        #                             coded MT=37 as 5n and filed (n,4n) at A-5
        (25, (24, 50)),  # (n,3nα) -> Cr-50, which is where MT=37 used to land
        (102, (26, 57)),  # (n,γ)   -> Fe-57
        (103, (25, 56)),  # (n,p)   -> Mn-56
        (107, (24, 53)),  # (n,α)   -> Cr-53
        (111, (24, 55)),  # (n,2p)  -> Cr-55, not Mn-55 via a phantom deuteron
        (109, (20, 45)),  # (n,3α)  -> Ca-45
        (11, (25, 53)),  # (n,2nd) -> Mn-53, skipped entirely before #351
        (30, (22, 47)),  # (n,2n2α)-> Ti-47, likewise
        (114, (21, 47)),  # (n,d2α) -> Sc-47, likewise
        (113, (21, 46)),  # (n,t2α) -> Sc-46. No MF=10 witness in any sampled
        #                             evaluation; the committed table said
        #                             (3, 8), which is neither t+2α nor anything
        #                             else. Hand-checked, and covered by the
        #                             ENDF-102 name oracle above.
        (155, (23, 50)),  # (n,tα)  -> V-50
        (600, (25, 56)),  # (n,p) to the ground level -> Mn-56, as MT=103
        (849, (24, 53)),  # (n,α) continuum -> Cr-53, as MT=107
    ],
)
def test_residuals_for_a_worked_example(mt, expected):
    assert mt_to_residual(mt, 26, 56, 0, 1) == expected


def test_the_two_channels_that_swapped_products_no_longer_do():
    """MT=25 (z,3nα) and MT=37 (z,4n) were not arithmetic slips but the wrong
    reactions: they were commented `(x,4n)` and `(x,5n)`, coded (0, 4) and (0, 5).

    So MT=37's genuine (n,4n) product was filed one mass unit low, and MT=25's
    tiny (n,3nα) cross-section landed on the nuclide (n,4n) should have produced
    — the two channels swapped places rather than each being merely wrong.
    TENDL-2025's Br-80m → Br-77 was off by 6.5e7 because of it.
    """
    # Br-80 + n. The right answers:
    assert mt_to_residual(37, 35, 80, 0, 1) == (35, 77)  # (n,4n)  -> Br-77
    assert mt_to_residual(25, 35, 80, 0, 1) == (33, 74)  # (n,3nα) -> As-74
    # The committed table put both of these at Br-77, one of them wrongly.
    assert mt_to_residual(25, 35, 80, 0, 1) != (35, 77)
    assert mt_to_residual(37, 35, 80, 0, 1) != mt_to_residual(25, 35, 80, 0, 1)


def test_projectiles_other_than_neutrons_balance():
    """The table is projectile-independent — ENDF-102 writes these as (z,…) —
    so the same MT must balance for a proton or alpha beam too."""
    assert mt_to_residual(107, 26, 56, 1, 1) == (25, 53)  # Fe-56(p,α)Mn-53
    assert mt_to_residual(16, 26, 56, 2, 4) == (28, 58)  # Fe-56(α,2n)Ni-58
    assert mt_to_residual(103, 13, 27, 1, 2) == (13, 28)  # Al-27(d,p)Al-28


# ---------------------------------------------------------------------------
# What must stay out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mt", "why"),
    [
        (1, "total"),
        (2, "elastic — the projectile re-emerges, and (Z, A+1) collides with (n,γ)"),
        (3, "nonelastic, a sum"),
        (5, "'anything' — a different product in every evaluation"),
        (18, "fission"),
        (19, "first-chance fission"),
        (27, "absorption, a sum"),
        (101, "disappearance, a sum"),
        (201, "(x,Xn) — a particle yield, not a channel"),
        (202, "(x,Xγ) — likewise, and absent from REACTION_NAME, so only this asserts it"),
        (203, "(x,Xp) — likewise"),
        (207, "(x,Xα) — likewise"),
        (301, "heating"),
        (444, "damage energy"),
    ],
)
def test_mts_with_no_single_residual_stay_none(mt, why):
    assert mt_to_residual(mt, 26, 56, 0, 1) is None, why
    assert mt not in MT_EMITTED_PARTICLES, f"MT={mt} ({why}) must not be tabulated"


def test_mt_50_transmutes_for_a_charged_projectile_and_not_for_a_neutron():
    """The neutron level band starts at 50, and that is safe *because* of the
    residual==target guard rather than in spite of it (#335).

    MT=50 is (z,n₀), the transition leaving the residual in its ground state.
    For an incident neutron that is elastic scattering under another name — the
    residual is the target — so it must yield no production row, and widening
    the band from 51 to 50 is a no-op for every neutron library.

    For an incident charged particle it is a genuine transmutation and the
    dominant (p,n) channel at threshold. ⁹Be(p,n₀)⁹B and ⁷Li(p,n₀)⁷Be are the
    two that matter here: with the band starting at 51 both were dropped, which
    is what made natural beryllium a dead proton target (hyrr#668).
    """
    # Neutron: n₀ off Fe-56 leaves Fe-56.
    assert mt_to_residual(50, 26, 56, 0, 1) is None
    # Proton: (p,n₀) is a real product, and it is the isobar one Z up.
    assert mt_to_residual(50, 4, 9, 1, 1) == (5, 9), "⁹Be(p,n₀) must produce B-9"
    assert mt_to_residual(50, 3, 7, 1, 1) == (4, 7), "⁷Li(p,n₀) must produce Be-7"
    # And MT=50 is inside the band rather than an entry of its own, so the two
    # spellings cannot drift apart.
    assert 50 not in MT_EMITTED_PARTICLES
    assert any(lo <= 50 <= hi for lo, hi in LEVEL_RANGES), "MT=50 is not covered by any level range"


@pytest.mark.parametrize("mt", [4, 51, 91])
def test_inelastic_names_a_particle_but_not_a_new_nuclide(mt):
    """Inelastic scattering is tabulated — one neutron leaves — but the residual
    is the target, which decays back to it. `mt_to_residual` must return None so
    it never populates a product channel: at (Z, A) it would double the target's
    own row, and the elastic case (MT=2) would collide with the (n,γ) residual
    (Z, A+1), swamping a ~mb capture with ~barns of potential scattering.

    Metastable products of inelastic scattering are carried by MF=10 instead,
    which names them via IZAP.
    """
    assert mt_to_residual(mt, 26, 56, 0, 1) is None
    # But the emission itself is known, and is one neutron.
    emitted = MT_TO_EMISSION.get(mt) or next(e for (lo, hi), e in LEVEL_RANGES.items() if lo <= mt <= hi)
    assert emitted == (0, 1)


def test_mt_5_is_absent_because_its_product_is_not_derivable():
    """Not an oversight. Cr-50's MF=10 MT=5 in FENDL-3.2 names 26 different
    products in one section; MT alone cannot pick one. MF=10's IZAP can, and
    `parse_mf10_rows` uses it — which is why MT=5 rows are not lost."""
    fendl = FIXTURES / "fendl-3.2" / "n_024-Cr-50_2425.endf"
    import endf

    material = endf.Material(io.StringIO(fendl.read_text()))
    izaps = {int(level["IZAP"]) for level in material.section_data[10, 5]["levels"]}
    assert len(izaps) > 20, f"expected many products under MT=5, got {sorted(izaps)}"
    assert mt_to_residual(5, 24, 50, 0, 1) is None


def test_impossible_residuals_are_rejected():
    """Emitting more than the nucleus has cannot produce anything."""
    assert mt_to_residual(109, 2, 4, 0, 1) is None  # 3α out of He-5
    assert mt_to_residual(161, 1, 1, 0, 1) is None  # 8n out of H-1
