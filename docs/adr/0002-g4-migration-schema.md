# ADR 0002 — Geant4 migration: preserve current parquet schema, transform on import

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-04-30 |
| **Affected versions** | v0.10.x → v0.11.0 |
| **Related** | #66 (epic), #67 (this decision), #68–#79 (sub-tasks), ADR-0001 (state API) |

## Context

Issue #66 is migrating nucl-parquet's nuclear-structure data sources from the v0.10.x IAEA-LiveChart fetcher path to Geant4 ASCII data files (G4ENSDFSTATE3.0, PhotonEvaporation6.1.2, RadioactiveDecay6.1.2). The G4 data is published as Parquet by the strata project at `gerchowl/strata-data` on Hugging Face.

Strata's published schema differs in encoding and naming from nucl-parquet's current Parquet schema:

| Concept | nucl-parquet (v0.10.x) | strata G4 |
|---|---|---|
| Excitation energy | `level_keV` (Float64) | `excitation_kev` (Float64) — same units, different name |
| Spin & parity | `jp` string ("3-", "1/2+", "(3/2)") | `spin_x2` (Int16, J × 2) + `parity` (Int8, ±1) or `floating_level_flag` (String) |
| Half-life | `half_life_s` (Float64) | `mean_life_ns` (Float64) for ENSDFSTATE; `half_life_s` for photon_evap_levels |
| Magnetic moment | absent | `magnetic_moment_jt` (Float64, joule/tesla) — bonus |
| Multipolarity | absent in radiation | encoded integer per gamma transition — bonus |
| Internal conversion coeff. | absent in radiation | `icc_total` per gamma — bonus |

The migration must pick a target schema for the rebuilt `nuclides.parquet` / `decay.parquet` / `levels/*.parquet` / `radiation/*.parquet` files in nucl-parquet.

## Decision

**Adopt option (a): preserve nucl-parquet's current schema, transform on import.**

The G4-derived input files (strata's parquet) are read by build-time converter scripts (#69–#72), which compute:

- `level_keV` ← `excitation_kev` (rename only, identical Float64 keV semantics)
- `half_life_s` ← `mean_life_ns × ln(2) × 1e-9` (G4 ENSDFSTATE row stores mean lifetime in ns; convert via half-life formula)
- `jp` ← `encode(spin_x2, parity)` producing strings like "3-", "1/2+" with parenthesisation for uncertain values dropped (G4 strips ENSDF parens at parse time)
- `(Z, A, state)` ← derived per-(Z,A) from ascending `level_keV` of long-lived isomers, mirroring the v0.9.0+ state convention
- AME2020 / IUPAC auxiliary tables (already shipped on `feat/g4-data-migration` per commit `0abef28`) are LEFT-JOINed into `ground_states.parquet` for mass-excess and isotopic-abundance columns

**Bonus columns are added additively, not substituted**: `magnetic_moment_jt`, `multipolarity`, `mixing_ratio`, `icc_total` get new optional columns in the existing tables. Nullable. Documented but not required.

## Alternatives considered

### B. Adopt strata's schema verbatim (rejected)

Write `excitation_kev` / `mean_life_ns` / `spin_x2` / `parity` etc. directly into nucl-parquet's output, dropping the v0.10.x column names.

**Rejected** — every downstream consumer breaks in lockstep with the migration:

- The Rust crate at `clients/rs/nucl-parquet/src/meta.rs` (`DecayDb::modes`, `NuclidesDb::lookup`) reads `level_keV`/`half_life_s`/`jp`. Re-deriving per-shell decoders would mean a coordinated nucl-parquet + crate release.
- The TypeScript SDK `clients/ts/core/{decayColumns,nuclidesColumns}` types match the current parquet schema.
- The MCP server queries against the Python loader's view definitions (`GAMMA_LINES_SQL`, `IDENTIFY_GAMMA_SQL`).
- External consumers of v0.10.x (fd5, smelt, theranostics docs) implicitly depend on the existing column set.

A coordinated breaking-change release is appropriate at v1.0; not at v0.11. The v0.10 → v0.11 transition's *primary* value is data-quality (eliminating the IAEA-fetcher bug class — see ADR-0001 / PRs #57, #61, #62, #64). Bundling an API break dilutes that value and complicates user upgrades.

### C. Dual-publish (both schemas as separate tables) (rejected)

Ship both `nuclides.parquet` (current schema) and `nuclides_g4.parquet` (strata schema). Users opt into the new schema on their own timeline.

**Rejected** — doubles storage, doubles tests, doubles audit surface, and provides no migration runway because v0.11 still ships v0.10.x columns. If we want a v1.0 schema break, a single deprecation cycle (v0.11 deprecates → v1.0 removes) is cleaner.

## Consequences

### Positive

- Zero breaking changes for the Rust crate, TS SDK, MCP server, and external Python consumers — they upgrade to v0.11 and immediately get correct data with no code changes.
- The migration's *primary* claim ("the data is now correct") is presented to users without an API-churn distractor.
- Bonus columns (`magnetic_moment_jt`, `multipolarity`, `mixing_ratio`, `icc_total`) are pure additions — pre-1.0 consumers ignore them; opportunistic consumers get richer data.
- AME2020 mass-excess + IUPAC abundance integration into `ground_states.parquet` is also additive (previously NULL columns get populated).

### Negative

- The transform layer in #69–#72 carries unit-conversion code (`mean_life_ns → half_life_s`) and encoding code (`spin_x2 + parity → jp`). Tested at the converter boundary; defects there would silently corrupt downstream data (mitigated by the diff harness in #75).
- Strata's strictly-typed schema (Int8 parity vs string `jp`) is arguably cleaner; we accept the lossy `jp` string for v0.11 because compatibility is more valuable than purity here.
- A future v1.0 schema break still has to happen if we want the cleaner encoding — this defers the decision rather than resolving it. That's deliberate: v0.11 is a data-quality release, not an API release.

### Operational

- The `_LEVEL_EXACT_MATCH_KEV` / `BuildIntegrityError` / `_ORPHAN_CEILING` / `_UNLABELED_EXCITED_CEILING` machinery from PR #57/#61/#62 becomes obsolete with clean upstream data — removed in #76 (cleanup task) once the diff harness (#75) confirms zero regression.
- The strata-data revision is pinned in `data/catalog.json` (per #68) so a future strata schema break doesn't silently propagate.

## Implementation gates

This ADR unblocks issues #69–#76. Each converter PR explicitly references this ADR's transform rules in its description. The diff harness (#75) verifies the transform produces v0.10.x-compatible output on the canonical isotope set.

## References

- Parent epic: #66 — Migrate from IAEA fetcher to Geant4 data files
- This ADR's issue: #67
- Sub-tasks: #68 (HF fetcher), #69 (ensdfstate), #70 (levels), #71 (decay), #72 (gammas), #73 (coincidences), #74 (X-ray/Auger), #75 (validation), #76 (cleanup)
- ADR-0001 — Radiation `state` API (the v0.10 design this migration preserves)
- v0.10.x rescue PRs: #57, #61, #62, #64
- Strata published dataset: https://huggingface.co/datasets/gerchowl/strata-data
- AME2020: https://www-nds.iaea.org/amdc/ame2020/
- IUPAC/CIAAW (NIST mirror): https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl
