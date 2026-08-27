"""The one definition of what the `state` column may contain.

`state` names *which isomeric state of a nuclide a row is about*. It had four
spellings of that concept and, worse, one spelling carrying three unrelated
meanings (#357, #367).

## The collision this module removes

`''` meant three different things depending on which table you read:

| table | `''` meant | rows |
|---|---|---|
| ENDF cross-sections (`tendl-2025/xs`, …) | summed over every state — a *claim* | ~26.5 M |
| EXFOR (`exfor`, `exfor-channels`) | the measurement did not say | ~4.4 M |
| nuclide identity (`meta/ensdf/*`) | the ground state | ~4.4 M |

Two consequences, both silent:

* `SELECT … WHERE state = ''` over a glob returns a mixture of "this is a total",
  "we don't know" and "this is the ground state", with nothing to separate them.
* `JOIN … USING (Z, A, state)` between a cross-section table and
  `meta/ensdf/nuclides.parquet` attaches ground-state half-lives to
  summed-over-states cross-sections, *and* misses the real ground-state rows,
  because ENDF spells the ground state `'g'` and ENSDF spelled it `''`. The
  join the schema documentation promised did not work.

## The rule

Every value is a positive statement. Absence of information is `NULL`, never a
string. There is no value that means "empty" — that is what `NULL` is for, and
CLAUDE.md principle 3 (nulls, not sentinels) is the whole reason.

    'g'    the ground state
    'm'    the first isomer, 'm2' the second, 'm3' the third, …
    'l'    an isomer is involved but the measurement does not resolve which
    'sum'  summed over all states — an aggregate, not a peer of the others
    NULL   not stated

`'sum'` is the only value that is not a state at all, and it is deliberately a
word rather than `''`: it is a *claim about the quantity*, and spelling a claim
as an empty string is how it got confused with the absence of one. It also
survives a CSV or pandas round-trip that would coerce `''` to null and destroy
the distinction.

`'l'` is EXFOR's `L` flag, "level (isomer unresolved)". It is a real datum —
"a metastable state is involved, but this measurement does not say which" — and
is genuinely different from `NULL`, which says nothing at all. Keep it.

## Never sum across `'sum'` and the rest

An ENDF evaluation ships both the MF=3 channel total and the MF=10 split, so
Al-27(n,2n) is one 177 mb `'sum'` row *and* a 114 mb `'g'` + 65 mb `'m'` pair.
`GROUP BY residual_Z, residual_A` with `SUM(xs_mb)` double-counts. Filter
`state = 'sum'` for totals, or `state <> 'sum'` for the split — never both.
"""

from __future__ import annotations

from typing import NamedTuple

#: The ground state.
GROUND = "g"

#: Summed over every isomeric state. An aggregate, not a state — see the module
#: docstring. Never sum this together with the `'g'`/`'m'` rows beside it.
SUM = "sum"

#: An isomer is involved, but the measurement does not resolve which one.
#: EXFOR's `L` ("level") flag. Different from NULL, which asserts nothing.
UNRESOLVED = "l"

#: How many isomers above the ground state we are willing to spell. ENSDF ships
#: up to `m3` today; the cap exists so that a parser bug cannot mint `'m47'` and
#: have it silently accepted as a new state (which is how `'m1'` became a fourth
#: spelling of `'m'`).
MAX_ISOMER = 9

#: Isomers in ascending excitation: 'm', 'm2', 'm3', …
#: `'m'` rather than `'m1'` for the first, because that is the spelling
#: `meta/ensdf/nuclides.parquet` uses and therefore the one that joins.
ISOMERS: tuple[str, ...] = tuple("m" if i == 1 else f"m{i}" for i in range(1, MAX_ISOMER + 1))

#: Every value the `state` column may hold, besides NULL.
#:
#: NULL is not in this set and cannot be: it is the absence of a value, not a
#: value. `is_valid_state(None)` is True; `None in STATES` is False, on purpose.
STATES: frozenset[str] = frozenset({GROUND, SUM, UNRESOLVED, *ISOMERS})

#: The subset that names an actual isomeric state of a nuclide, so a row using
#: one of these can be joined against `meta/ensdf/nuclides.parquet`. `'sum'` is
#: excluded because it names no nuclide state, and `'l'` because it names an
#: unidentified one.
JOINABLE_STATES: frozenset[str] = frozenset({GROUND, *ISOMERS})

#: EXFOR's X4 nuclide-suffix spellings, mapped onto the vocabulary above.
#:
#: `'m1'` is X4's spelling of the first isomer and is *the same state* as `'m'`;
#: shipping both was one of the four spellings #357 counted. `'g'` and `'l'`
#: pass through. Anything else — including an absent suffix — is not a state
#: this repository can name, and becomes NULL rather than a guess.
#:
#: Derived from `ISOMERS` rather than written out. It was hand-written to `m4`
#: while `ISOMERS` ran to `m9`, so `parse_x4_state('M9')` returned None — "not
#: stated" — for a suffix that states the isomer perfectly clearly, while
#: `isomer_state(9)` happily produced `'m9'`. Two spellings of one cap that
#: disagreed: the same defect as the table this module replaced, one level up.
_X4_SUFFIXES: dict[str, str] = {
    GROUND: GROUND,
    UNRESOLVED: UNRESOLVED,
    "m1": "m",  # X4 synonym for 'm', normalised here so only one reaches disk
    **{isomer: isomer for isomer in ISOMERS},
}


# ---------------------------------------------------------------------------
# Which values each kind of table may hold
# ---------------------------------------------------------------------------
#
# Per *table*, not globally: a value that is meaningful in one table can be
# meaningless in another, and "legal somewhere" is not a check. `'sum'` on a
# measured EXFOR row would be a claim nobody made; `'l'` in an evaluated
# library would be one ENDF cannot express.

#: Evaluated cross-sections (ENDF and friends). MF=3 gives the channel total
#: over every state — `'sum'` — and MF=10 splits it into `'g'`/`'m'`/…
EVALUATED_XS_STATES: frozenset[str] = frozenset({SUM, GROUND, *ISOMERS})

#: Measured cross-sections (EXFOR). A measurement names the state it resolved,
#: says `'l'` when it knows an isomer is involved but not which, or says
#: nothing — NULL. It never asserts `'sum'`: EXFOR reports what was measured,
#: and "summed over states" is a claim about an evaluation, not a measurement.
MEASURED_XS_STATES: frozenset[str] = frozenset({GROUND, UNRESOLVED, *ISOMERS})

#: Nuclide-identity tables (`meta/ensdf/*`, `meta/decay.parquet`, …). A row is
#: *about* a nuclide in a given state.
#:
#: NULL is meaningful here, and is not the same as "this nuclide has no isomer".
#: It means **the state could not be established**, and two measured cases need
#: it (#378):
#:
#:   * `nuclides.parquet` carries 13 rows that are excited levels, not ground
#:     states — the builder labelled the lowest *listed* level per (Z, A) as
#:     ground, and G4ENSDFSTATE does not list those nuclides' ground states at
#:     all. Four of them carry ENSDF's `+X` floating flag, so their excitation
#:     is relative to an unknown offset and is not even a definite energy.
#:   * `radiation` carries 26 rows across 13 nuclides whose `state` was `''`
#:     ("the ground-band decay chain") but whose emitting level coincides with a
#:     catalogued isomer of the same nuclide at a *measured* energy. Whether
#:     those are ground-band cascade gammas or isomer decays cannot be settled
#:     from an energy coincidence; no table in this repository carries ENSDF's
#:     own band attribution, so resolving them needs upstream data (#386).
#:
#: `'g'` is a positive claim about which nuclear state a row belongs to.
#: Asserting it over a row measured as ambiguous is inventing a claim — the
#: defect class of #326, #351 and #377. NULL is the only value in this
#: vocabulary that is true of those rows.
#:
#: They stop joining, which is the correct consequence: a row that does not come
#: back is visible, and a row that comes back wrong is not.
NUCLIDE_STATES: frozenset[str] = frozenset({GROUND, *ISOMERS})


# ---------------------------------------------------------------------------
# The *target's* state (#353)
# ---------------------------------------------------------------------------
#
# `state` names the state of the *residual* — what comes out. `target_state`
# names the state of the nuclide that went in, and it is the same vocabulary,
# deliberately: Br-80m is one nuclide, and which side of a reaction it appears on
# does not change how this repository spells it. A second set of values for the
# target side would undo exactly what #357/#380 collapsed.
#
# The allowed *subsets* differ, because the two sides can say different things:

#: What an evaluated library's `target_state` may hold.
#:
#: An evaluation is *of one nuclide*, so it names a real state. Notably absent:
#:
#:   'sum' — an evaluation is never summed over target states. Two targets are
#:           two evaluations in two files (`n_035-Br-80` and `n_035-Br-80M`),
#:           which is the whole of #353. `'sum'` on the target side would assert
#:           an aggregate nobody computed, and would re-introduce the merge as a
#:           *value* right after the column was added to end it.
#:   'l'   — ENDF cannot express "some isomer, unresolved" about its own target.
TARGET_STATES: frozenset[str] = frozenset({GROUND, *ISOMERS})

#: What a measured table's `target_state` may hold.
#:
#: EXFOR *can* say `'l'`: a sample irradiated as a mixed isomeric target, with the
#: measurement unable to resolve which. That is a real datum and distinct from
#: NULL. Same argument as `MEASURED_XS_STATES` on the residual side.
MEASURED_TARGET_STATES: frozenset[str] = frozenset({GROUND, UNRESOLVED, *ISOMERS})


def target_state_for_natural_element() -> None:
    """A natural-element target (`target_A = 0`) has `target_state = NULL`.

    Written as a function because it is a *decision*, and decisions in this
    module get argued for where they are made rather than in a commit message.

    A natural element is not a nuclide: it is an abundance-weighted mixture of
    isotopes. "Which isomeric state" is not a question with an answer, so the
    answer is the absence of one — NULL, per CLAUDE.md principle 3.

    It is emphatically **not** `'sum'`. `'sum'` already means *summed over the
    isomeric states of one nuclide*, and a natural element is an aggregate over
    *isotopes*. Reusing the word for a second kind of aggregate is precisely the
    one-name-two-meanings collision #357 spent 26.5 M rows undoing, and it would
    make `WHERE target_state = 'sum'` return a mixture again.

    Nor is it `'g'`. Natural chlorine is not "chlorine in its ground state" — the
    claim would be false, and false in a way that joins: `JOIN nuclides USING
    (Z, A, state)` would attach Cl-0's non-existent ground state to something.
    """
    return None


#: ENDF filename isomer markers -> isomer rank. Used only to cross-check the
#: authoritative `LISO`, never as the primary source (see
#: `scripts/fetch_endf_libs.py::target_state_from_material`).
#:
#: Deliberately does not include `'n'`, which appears in some mirror listings.
#: Guessing that it means the second isomer is exactly the kind of invention
#: #334/#340/#351 were each caused by; `LISO` states the rank outright, so there
#: is nothing to guess.
ENDF_TARGET_MARKERS: dict[str, int] = {"": 0, "g": 0, "m": 1, "m1": 1, "m2": 2, "m3": 3}


#: The retired spelling itself: the empty string that meant three different
#: things depending on which table you read.
LEGACY_UNSPECIFIED = ""

#: Every shipped table carrying a `state` column, and what it may hold.
#:
#: Keyed by the table's directory under `data/`, because these tables are
#: sharded per element and every shard of a table shares one vocabulary.
#:
#: Explicit rather than inferred from the path. A rule like "anything under
#: `*/xs` is evaluated" would silently classify the next table somebody adds,
#: and a benign default where a declaration belonged is the failure mode this
#: repository keeps paying for (#334, #340, #351, #356, #367). A table with a
#: `state` column that is not listed here fails `tests/test_state_vocabulary.py`
#: until somebody says what its states mean.
TABLE_STATES: dict[str, frozenset[str]] = {
    # --- evaluated libraries
    "brond-3.1/xs": EVALUATED_XS_STATES,
    "cendl-3.2/xs": EVALUATED_XS_STATES,
    "endfb-8.0/channels": EVALUATED_XS_STATES,
    "endfb-8.0/xs": EVALUATED_XS_STATES,
    "endfb-8.1/xs": EVALUATED_XS_STATES,
    "fendl-3.2/xs": EVALUATED_XS_STATES,
    "iaea-medical/xs": EVALUATED_XS_STATES,
    "iaea-pd-2019/xs": EVALUATED_XS_STATES,
    "irdff-2/xs": EVALUATED_XS_STATES,
    "jeff-4.0/xs": EVALUATED_XS_STATES,
    "jendl-5/xs": EVALUATED_XS_STATES,
    "jendl-ad-2017/xs": EVALUATED_XS_STATES,
    "jendl-deu-2020/xs": EVALUATED_XS_STATES,
    "tendl-2023-iso/xs": EVALUATED_XS_STATES,
    "tendl-2025/xs": EVALUATED_XS_STATES,
    # Heavy-ion fragment production. Ships NULL throughout — the Geant4 run
    # names no isomeric state — which is exactly what NULL is for.
    "hi-xs/xs": EVALUATED_XS_STATES,
    "hi-xs-prod/xs": EVALUATED_XS_STATES,
    # --- measured
    "exfor": MEASURED_XS_STATES,
    "exfor-channels": MEASURED_XS_STATES,
    # --- nuclide identity
    "meta": NUCLIDE_STATES,
    "meta/ensdf": NUCLIDE_STATES,
    "meta/ensdf/beta_spectra": NUCLIDE_STATES,
    "meta/ensdf/radiation": NUCLIDE_STATES,
}


#: Tables whose *shipped* data still uses the pre-#357 spelling, with the issue
#: that clears the entry. The builders are fixed; the parquets are not, because
#: rewriting them is a data release and this is a code change.
#:
#: Same contract as `data/builder_stamp_exemptions.json`: an entry is a debt,
#: not a decision, and the ledger is self-cleaning — once a table passes on its
#: own, `tests/test_state_vocabulary.py` fails on the leftover entry until it is
#: deleted. Without this the vocabulary test could not be written at all until
#: the rebuild landed, and a test that cannot run yet is a test nobody writes.
class PendingMigration(NamedTuple):
    """A table still shipping retired spellings, and who clears the debt."""

    #: Exactly which retired values it still ships. Naming them, rather than
    #: tolerating "anything old", is what stops a pending entry from becoming a
    #: licence for an unrelated typo.
    legacy: frozenset[str]
    reason: str


def _pending(reason: str, *legacy: str) -> PendingMigration:
    return PendingMigration(frozenset(legacy), reason)


PENDING_MIGRATION: dict[str, PendingMigration] = {
    # `data/meta/spectrum_xs.parquet` alone still carries `''`. It is built by
    # `nucl_parquet/build_spectrum_xs.py`, which copies `state` straight from the
    # xs rows it averages — so the value is now `'sum'` at the source and the
    # file simply has not been regenerated. One rebuild of that table clears it.
    #
    # Every other entry that stood here was retired by the 2026.8.5 rebuild and
    # the `meta/ensdf` migration, both of which have now run. The ledger is
    # self-cleaning and this is it cleaning.
    "meta": _pending("#357: '' -> 'sum', pending a rebuild of meta/spectrum_xs.parquet", LEGACY_UNSPECIFIED),
}

#: Tables whose shipped `state` column is not an isomeric state at all and is
#: being renamed, rather than revalued. Same self-cleaning contract.
#:
#: `stopping/em`'s column holds solid/liquid/gas. Sharing the name `state` with
#: twenty tables that mean isomeric state is CLAUDE.md principle 1 at its
#: purest — not a variant spelling but an identical name for an unrelated
#: concept, so a consumer filtering `state` across a glob crossed phase of
#: matter with nuclear isomers and got no error.
#: Keyed by the parquet path relative to `data/`, not by directory: `stopping/em`
#: also holds `electron_stopping.parquet`, which never had the column. "This
#: file never had a `state`" and "this file lost its `state`" are different
#: facts and the ledger must not blur them.
#: Files whose `state` column never held an isomeric state, so the column must
#: be spelled `phase`. **Permanent, not a ledger.** The rename is done, but the
#: rule that these files must not grow a `state` column back outlives the debt
#: of doing it — `verify` and the migration CLI both key on this, so they keep
#: working once `PENDING_COLUMN_RENAME` is empty.
PHASE_NOT_STATE: dict[str, str] = {
    "stopping/em/density_effect_params.parquet": (
        "#357: holds phase of matter (solid/liquid/gas), not an isomeric state"
    ),
}

PENDING_COLUMN_RENAME: dict[str, str] = {
    # Empty: `stopping/em/density_effect_params.parquet` shipped `state` holding
    # phase of matter, and the rebuild renamed it to `phase`. The file no longer
    # has a `state` column at all, so the entry retired itself — which is the
    # contract this ledger is for. The invariant it was enforcing lives on in
    # `PHASE_NOT_STATE`; only the "still to do" part cleared.
}

#: The directories those files live in, for checks that work per table.
#:
#: Derived from the *debt*, not from `PHASE_NOT_STATE`: this set exempts a table
#: from the isomeric-state gate while it is mid-migration, and an exemption that
#: outlived its migration is exactly what the self-cleaning contract forbids.
PENDING_RENAME_TABLES: frozenset[str] = frozenset(f.rsplit("/", 1)[0] for f in PENDING_COLUMN_RENAME)


#: Every shipped table carrying a `target_state` column, and what it may hold.
#:
#: Derived from `TABLE_STATES` rather than retyped: a table's *kind* — evaluated,
#: measured, nuclide-identity — decides both vocabularies, so writing the list
#: twice would let the two drift into disagreeing about what a table is. The
#: nuclide-identity tables have no target at all and are excluded.
TABLE_TARGET_STATES: dict[str, frozenset[str]] = {
    table: (MEASURED_TARGET_STATES if states is MEASURED_XS_STATES else TARGET_STATES)
    for table, states in TABLE_STATES.items()
    if states is not NUCLIDE_STATES
}


def allowed_target_states(table: str) -> frozenset[str]:
    """The values `table.target_state` may hold.

    Raises for a table that has not declared itself, for the same reason
    `allowed_states` does: a new column arriving with an undeclared vocabulary is
    how a second spelling gets in.
    """
    if table not in TABLE_TARGET_STATES:
        raise KeyError(
            f"{table!r} has a `target_state` column but no entry in TABLE_TARGET_STATES. "
            "Declare what its target states mean before shipping it."
        )
    return TABLE_TARGET_STATES[table]


def allowed_states(table: str) -> frozenset[str]:
    """The values `table` may hold today, pending migrations included.

    Raises for a table that has not declared itself, so a new `state` column
    cannot arrive with an undeclared vocabulary.
    """
    if table not in TABLE_STATES:
        raise KeyError(
            f"{table!r} has a `state` column but no entry in TABLE_STATES. "
            "Declare what its states mean before shipping it."
        )
    allowed = TABLE_STATES[table]
    pending = PENDING_MIGRATION.get(table)
    if pending is not None:
        allowed = allowed | pending.legacy
    return allowed


def is_valid_state(value: str | None) -> bool:
    """True if `value` may appear in a `state` column. NULL always may."""
    return value is None or value in STATES


def parse_x4_state(suffix: str | None) -> str | None:
    """Map an EXFOR nuclide suffix onto the vocabulary. Unknown -> None.

    `'27-CO-58-M3'` has suffix `'M3'` and is the third isomer of Co-58, a state
    `meta/ensdf/nuclides.parquet` already ships. It must come back as `'m3'`.

    Before #367 the two EXFOR builders each had their own inline version of this
    and disagreed about the allowed set (`fetch_exfor.py` permitted five values,
    `fetch_exfor_master.py` six). `fetch_exfor_master.py` also lowercased the
    suffix and *then* tested `suffix.startswith("M")`, so that branch was dead
    and every unrecognised suffix — including `M3` — fell through to `''`,
    i.e. to the same key as the ground-state and summed rows for that nuclide.
    A third isomer was not mislabelled, it was made unrecoverable.

    Returning None for an unrecognised suffix is deliberate and is *not* the old
    behaviour: None says "this measurement does not tell us the state", which is
    true, and is a different key from `'g'` and from `'sum'`. The old `''`
    collided with both.
    """
    if not suffix:
        return None
    return _X4_SUFFIXES.get(suffix.strip().lower())


def isomer_state(rank: int) -> str:
    """The state naming the `rank`-th isomer above ground. `isomer_state(1)` is `'m'`.

    `rank` 0 is the ground state and returns `'g'`. Raises above `MAX_ISOMER`
    rather than minting an unheard-of spelling.
    """
    if rank == 0:
        return GROUND
    if not 1 <= rank <= MAX_ISOMER:
        raise ValueError(f"isomer rank {rank} is outside 1..{MAX_ISOMER}; refusing to invent a state")
    return ISOMERS[rank - 1]
