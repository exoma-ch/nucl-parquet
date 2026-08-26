"""Expected Parquet column schemas for validation."""

from __future__ import annotations

XS_SCHEMA = {
    "target_A": "Int32",
    "residual_Z": "Int32",
    "residual_A": "Int32",
    "state": "Utf8",
    "energy_MeV": "Float64",
    "xs_mb": "Float64",
}

# ---------------------------------------------------------------------------
# Canonical cross-section schema
# ---------------------------------------------------------------------------
# One shape for every sigma(E) table, so that evaluated production data,
# transport channels and EXFOR measurements union without special cases.
#
# It exists because the legacy 6-column XS_SCHEMA above encodes reaction
# identity *in the file path* rather than in the data. Two shipped bugs came
# straight out of that:
#
#   #273  the unified `xs` view interleaved isobaric targets, because target_Z
#         is not a column — the loader now regexes the filename and joins a
#         symbol->Z table at query time to recover it.
#   ---   the same view still silently merges FIVE projectiles: a query for
#         Cu-63 -> Zn-63 returns (p,n), (d,2n), (a,x), (h,x) and (t,x) rows
#         superposed, because `projectile` is likewise only in the filename.
#
# Identity therefore lives in columns. Nulls mean "not applicable" — never 0,
# which collides with real Z=0 products and made (n,tot)/(n,el)/(n,f)
# indistinguishable in the 82% of EXFOR rows that name no residual.
CANONICAL_XS_SCHEMA = {
    # --- provenance of the evaluation / measurement
    "library": "Utf8",
    # 'production' — summed over every channel reaching this residual (MT null)
    # 'channel'    — one ENDF reaction channel (MT set; residual may be null)
    #
    # These are two different quantities over the same data, and an evaluation
    # ships both, so `SUM(xs_mb)` without a `kind` filter double-counts. Pick
    # one: `kind='production'` to ask "what makes Fe-55", `kind='channel'` to
    # ask "what does MT=102 do". Within channels, ENDF's own redundancy still
    # applies — MT=1 contains MT=2 — and `endf_mt.redundant` says which:
    #
    #   SELECT * FROM xs JOIN endf_mt USING (MT)
    #   WHERE kind='channel' AND NOT endf_mt.redundant
    "kind": "Utf8",
    # --- reaction identity
    "projectile": "Utf8",  # n p d t h a g, or a heavy ion such as 'ar40'
    "proj_Z": "Int32",
    "proj_A": "Int32",
    "target_Z": "Int32",
    "target_A": "Int32",  # 0 = natural element (ENDF convention)
    # Which isomeric state of the target the row is about — the same vocabulary
    # as `state` below, defined once in `nucl_parquet/state_vocabulary.py`.
    #
    # Br-80 and Br-80m are two nuclides with different half-lives and different
    # cross-sections, shipped by ENDF as two evaluations in two files. Without
    # this column they land as `target_A = 80` in one shard, interleaved and
    # indistinguishable: 1,490 rows under 757 distinct keys in
    # `tendl-2025/xs/n_Br.parquet`, almost exactly 2x (#353). Identity lived in
    # the filename, the filename was consumed, and the rows did not carry it —
    # CLAUDE.md principle 5, and the same failure as `projectile` before #359.
    #
    #   'g'    the ground state. For ENDF this is a *claim*, not a default: an
    #          unmarked filename means the ground state, and MF=1/451 LISO=0 says
    #          so independently. Absence of a marker is information here.
    #   'm'…   the isomer, by ascending excitation, taken from LISO.
    #   'l'    measured tables only — a mixed isomeric target the measurement did
    #          not resolve. ENDF cannot express this about its own target.
    #   NULL   not stated, *or* not applicable: `target_A = 0` is a natural
    #          element, an isotopic mixture that has no isomeric state to name.
    #
    # There is deliberately no 'sum'. Two target states are two evaluations, not
    # one summed quantity, so nothing ever computes that aggregate — and reusing
    # 'sum' (which means "over isomeric states of one nuclide") for a natural
    # element (a mixture of *isotopes*) would rebuild the #357 collision on the
    # target side. See `state_vocabulary.target_state_for_natural_element`.
    "target_state": "Utf8",
    # ENDF's channel identity, and the primitive: MT -> residual is derivable,
    # residual -> MT is not. Null on production rows, which are a sum over
    # several MTs and so name none.
    "MT": "Int32",
    # Null — never 0/0 — when the channel names no single product. Total,
    # elastic, inelastic and fission all do; `WHERE residual_Z IS NULL` is how
    # you ask for them, and a 0/0 sentinel made them indistinguishable from each
    # other and from a real Z=0 product.
    "residual_Z": "Int32",
    "residual_A": "Int32",
    # Residual isomeric state. One vocabulary, defined once in
    # `nucl_parquet/state_vocabulary.py` and imported by every builder:
    #
    #   'g'    the ground state
    #   'm'    first isomer, 'm2' second, 'm3' third, … ascending excitation
    #   'l'    an isomer is involved, but the measurement did not resolve which
    #          (EXFOR's L flag) — a real datum, and not the same as NULL
    #   'sum'  summed over all states. An aggregate, NOT a peer of the others
    #   NULL   not stated
    #
    # There is no '' — see #357. It used to mean three different things
    # depending on which table you read: "summed over states" on an ENDF row,
    # "not stated" on an EXFOR row, and "the ground state" in meta/ensdf. So
    # `WHERE state = ''` over a glob returned a mixture of all three, and
    # nothing could separate them again.
    #
    # 'sum' is a word, not '', on purpose: it is a claim about the quantity, and
    # spelling a claim as an empty string is how it got confused with the
    # absence of one. It also survives a CSV or pandas round-trip that coerces
    # '' to null.
    #
    # NEVER SUM ACROSS 'sum' AND THE REST. ENDF gives the channel total in MF=3
    # and the ground/metastable split in MF=10, and both become rows: Al-27(n,2n)
    # is one 177 mb 'sum' row *and* a 114 mb 'g' + 65 mb 'm' pair. So
    # `GROUP BY residual_Z, residual_A` with `SUM(xs_mb)` double-counts. Filter
    # `state = 'sum'` for totals or `state <> 'sum'` for the split (#340).
    #
    # THE JOIN INVARIANT (#357). `state` is the third component of nuclide
    # identity, so it is a join key against `meta/ensdf/nuclides.parquet`:
    #
    #     JOIN nuclides USING (Z, A, state)
    #
    #   * 'g', 'm', 'm2', 'm3' name a nuclide state and JOIN.
    #   * 'sum' names no state and MUST JOIN NOTHING — it is an aggregate over
    #     the very rows it would otherwise match. A 'sum' row that acquires a
    #     half-life has silently been given the ground state's.
    #   * 'l' names an unidentified state and joins nothing.
    #   * NULL joins nothing, as NULL does.
    #
    # That invariant did not hold before #357: ENDF spelled the ground state 'g'
    # while meta/ensdf spelled it '', so the join missed every real ground-state
    # row *and* matched the summed rows instead — wrong in both directions at
    # once. `nucl_parquet.state_vocabulary.JOINABLE_STATES` is the set that may
    # appear on the left of that join, and `tests/test_state_vocabulary.py`
    # enforces it per table.
    "state": "Utf8",
    # --- the datum
    "energy_MeV": "Float64",
    "xs_mb": "Float64",
    "energy_err_MeV": "Float64",
    "xs_err_mb": "Float64",
    # --- experimental provenance (null for evaluations)
    "source_entry": "Utf8",
    "author": "Utf8",
    "year": "Int32",
}

#: Columns declared in `CANONICAL_XS_SCHEMA` that the *shipped* parquets do not
#: carry yet, with the issue that lands them.
#:
#: A schema addition and the rebuild that fills it are two changes: the builders
#: must write the column before there is any run that could produce it, so the
#: declaration necessarily leads the data. Without this ledger the choice would
#: be between a red suite for however long the rebuild takes and adding the
#: column silently — and a schema test that has been red for a week is one nobody
#: reads.
#:
#: Same contract as `state_vocabulary.PENDING_MIGRATION` and
#: `data/builder_stamp_exemptions.json`: an entry is a debt, not a decision, and
#: it is self-cleaning. Once every table carries the column,
#: `tests/test_canonical_schema.py` fails on the leftover entry until it is
#: deleted, so the ledger cannot quietly become permanent.
PENDING_COLUMN_ADDITION: dict[str, str] = {
    "target_state": (
        "#353: metastable targets are merged into their ground state. The builder "
        "writes the column; the shipped parquets gain it in the ENDF re-ingest."
    ),
}

# Columns that must never be null in a canonical table — the identity spine.
#
# `target_state` is deliberately absent: it is NULL for every natural-element
# target (`target_A = 0`), which is a real answer rather than a missing one.
CANONICAL_XS_REQUIRED = (
    "library",
    "kind",
    "projectile",
    "proj_Z",
    "proj_A",
    "target_Z",
    "target_A",
    "energy_MeV",
    "xs_mb",
)

# Heavy-ion production XS (hi-xs-prod): carries projectile identity
HI_XS_PROD_SCHEMA = {
    "proj_Z": "Int32",
    "proj_A": "Int32",
    "target_Z": "Int32",
    "target_A": "Int32",
    "residual_Z": "Int32",
    "residual_A": "Int32",
    "energy_MeV": "Float64",
    "xs_mb": "Float64",
}

# Heavy-ion total reaction XS (hi-xs): projectile-target pair only
HI_XS_SCHEMA = {
    "target_Z": "Int32",
    "target_A": "Int32",
    "energy_MeV": "Float64",
    "xs_mb": "Float64",
}


STOPPING_SCHEMA = {
    "source": "Utf8",
    "target_Z": "Int32",
    "energy_MeV": "Float64",
    "dedx": "Float64",
}

ABUNDANCES_SCHEMA = {
    "Z": "Int32",
    "A": "Int32",
    "symbol": "Utf8",
    "abundance": "Float64",
    "atomic_mass": "Float64",
}

DECAY_SCHEMA = {
    "Z": "Int32",
    "A": "Int32",
    "state": "Utf8",
    "half_life_s": "Float64",
    "decay_mode": "Utf8",
    "daughter_Z": "Int32",
    "daughter_A": "Int32",
    "daughter_state": "Utf8",
    "branching": "Float64",
}

ELEMENTS_SCHEMA = {
    "Z": "Int32",
    "symbol": "Utf8",
}

DOSE_CONSTANTS_SCHEMA = {
    "Z": "Int32",
    "A": "Int32",
    "state": "Utf8",
    "k_uSv_m2_MBq_h": "Float64",
    "dominant_gamma_keV": "Float64",
    "n_photon_lines": "Int32",
    "source": "Utf8",
}

XCOM_ELEMENTS_SCHEMA = {
    "Z": "Int32",
    "energy_MeV": "Float64",
    "mu_rho_cm2_g": "Float64",
    "mu_en_rho_cm2_g": "Float64",
}

XCOM_COMPOUNDS_SCHEMA = {
    "material": "Utf8",
    "energy_MeV": "Float64",
    "mu_rho_cm2_g": "Float64",
    "mu_en_rho_cm2_g": "Float64",
}

EPDL_PHOTON_XS_SCHEMA = {
    "Z": "Int32",
    "energy_MeV": "Float64",
    "process": "Utf8",
    "xs_barns": "Float64",
}

EPDL_FORM_FACTORS_SCHEMA = {
    "Z": "Int32",
    "momentum_transfer": "Float64",
    "form_factor": "Float64",
}

EPDL_SCATTERING_FN_SCHEMA = {
    "Z": "Int32",
    "momentum_transfer": "Float64",
    "scattering_fn": "Float64",
}

EPDL_ANOMALOUS_SCHEMA = {
    "Z": "Int32",
    "energy_MeV": "Float64",
    "component": "Utf8",
    "factor": "Float64",
}

EPDL_SUBSHELL_PE_SCHEMA = {
    "Z": "Int32",
    "energy_MeV": "Float64",
    "subshell": "Utf8",
    "xs_barns": "Float64",
    "edge_MeV": "Float64",
    "fluorescence_yield_eV": "Float64",
}

EADL_TRANSITIONS_SCHEMA = {
    "Z": "Int32",
    "vacancy_shell": "Utf8",
    "filling_shell": "Utf8",
    "transition_type": "Utf8",
    "energy_keV": "Float64",
    "probability": "Float64",
    "edge_keV": "Float64",
}

EEDL_ELECTRON_XS_SCHEMA = {
    "Z": "Int32",
    "energy_MeV": "Float64",
    "process": "Utf8",
    "xs_barns": "Float64",
}

NUCLIDES_SCHEMA = {
    "Z": "Int32",
    "A": "Int32",
    # Which state of this nuclide the row is about — same vocabulary as the xs
    # tables (`nucl_parquet.state_vocabulary`), which is what makes
    # `JOIN nuclides USING (Z, A, state)` work. `'sum'` never appears here: a
    # nuclide catalogue describes states, not aggregates over them.
    #
    # NULL means **the state could not be established**, which is NOT the same
    # as "this nuclide has no isomer". Two cases carry it (#378):
    #
    #   nuclides   13 rows that are excited levels rather than ground states —
    #              the builder labels the lowest *listed* level per (Z, A) as
    #              ground, and G4ENSDFSTATE does not list those nuclides'
    #              ground states. Four carry ENSDF's `+X` floating flag, so
    #              their excitation is relative to an unknown offset.
    #   radiation  13,106 rows across 45 nuclides whose `state` was `''`
    #              ("the ground-band decay chain") but whose emitting level
    #              coincides with a catalogued isomer of the same nuclide.
    #              Ground-band cascade gamma or isomer decay cannot be told
    #              apart from an energy coincidence, so neither is claimed.
    #              Resolving them needs ENSDF's own band assignment (#386).
    #
    # Those rows do not join, deliberately. A row that does not come back is
    # visible; a row that comes back wrong is not.
    "state": "Utf8",
    "symbol": "Utf8",
    "jp": "Utf8",
    "half_life_s": "Float64",
    "level_keV": "Float64",
    "decay_1": "Utf8",
    "decay_1_pct": "Float64",
    "decay_2": "Utf8",
    "decay_2_pct": "Float64",
}

# EXFOR now shares CANONICAL_XS_SCHEMA like every other cross-section table;
# `exfor_entry` became the source-agnostic `source_entry`. Kept as an alias so
# external callers importing it keep working.
EXFOR_SCHEMA = CANONICAL_XS_SCHEMA
