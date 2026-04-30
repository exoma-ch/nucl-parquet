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

**Adopt option (a): preserve nucl-parquet's current schema, transform on import. Dual-carry the strata-native encoding alongside the legacy schema so a future v1.0 cleanup is a removal, not a re-introduction.**

The G4-derived input files (strata's parquet) are read by build-time converter scripts (#69–#72), which compute:

- `level_keV` ← `excitation_kev` (rename only, identical Float64 keV semantics)
- `half_life_s` ← `convert_mean_life(mean_life_ns)` (see Transform spec below)
- `jp` ← `encode_jp(spin_x2, parity)` (see Transform spec below) — kept for v0.10 compat
- **Dual-carry**: `spin_x2` (Int16) and `parity` (Int8) shipped *alongside* `jp` for structured access and lossless round-trip — recommended path forward for new consumers, mandatory for any v1.0 cleanup
- `(Z, A, state)` ← derived per-(Z,A) from ascending `level_keV` of long-lived isomers, mirroring the v0.9.0+ state convention
- AME2020 / IUPAC auxiliary tables (already shipped on `feat/g4-data-migration` per commit `0abef28`) are LEFT-JOINed into `ground_states.parquet` for mass-excess and isotopic-abundance columns

**Bonus columns are added additively, not substituted**: `magnetic_moment_jt`, `multipolarity`, `mixing_ratio`, `icc_total` get new optional columns in the existing tables. Nullable. Documented but not required.

### Transform spec (binding for #69–#72)

**Half-life conversion** (`mean_life_ns` → `half_life_s`):

| G4 input | Output `half_life_s` | Meaning |
|---|---|---|
| `> 0` (finite) | `mean_life_ns × ln(2) × 1e-9` | Standard mean-life → half-life |
| `-1` (G4 stable sentinel) | `NULL` | Stable nuclide; consistent with v0.10.x ground_states.parquet handling for stable isotopes |
| `0` (G4 prompt sentinel) | `0.0` | Prompt; same semantic as v0.10.x |
| missing / parser failure | `NULL` | Unknown |

**Crucial**: the converter must explicitly check for `-1` *before* multiplying — naively applying the formula yields `≈ -6.93e-10 s`, a negative half-life that would silently corrupt downstream queries. Test required at boundary.

**Spin/parity encoding** (`spin_x2`, `parity` → `jp`):

| G4 input | Output `jp` | Notes |
|---|---|---|
| `spin_x2=12, parity=-1` | `"6-"` | integer-spin case |
| `spin_x2=3, parity=+1` | `"3/2+"` | half-integer-spin case |
| `spin_x2=0, parity=+1` | `"0+"` | zero-spin |
| `spin_x2=99` (G4 unknown sentinel) | `NULL` | preserves "unknown" — *also* `spin_x2_raw=NULL` and `parity_raw=NULL` in the dual-carry columns |
| `parity=0` (G4 unmeasured) | spin string with no sign suffix, e.g. `"3"` | `parity_raw=NULL`, `spin_x2_raw=N` for analytics |

The dual-carry `spin_x2` / `parity` columns preserve the unknown/unmeasured distinction that the legacy `jp` string can't express. This addresses the v0.10.x `jp=""` fetcher artifact (570 rows) by construction — strata's clean G4 input never produces empty strings; G4 uses explicit sentinels which we now decode.

## Alternatives considered

### B. Adopt strata's schema verbatim (rejected)

Write `excitation_kev` / `mean_life_ns` / `spin_x2` / `parity` etc. directly into nucl-parquet's output, dropping the v0.10.x column names.

**Rejected** — most downstream consumers break in lockstep with the migration:

- The Rust crate at `clients/rs/nucl-parquet/src/meta.rs` (`DecayDb::modes`, `NuclidesDb::lookup`) reads parquet *files directly* via `parquet`/`arrow-rs` and projects fixed column names. **No mitigation possible without a coordinated release.** This is the binding constraint.
- The TypeScript SDK `clients/ts/core/{decayColumns,nuclidesColumns}` types — partially mitigable via DuckDB/Polars view aliasing on the loader side, but the SDK's static typings would still mismatch.
- The MCP server queries against the Python loader's view definitions (`GAMMA_LINES_SQL`, `IDENTIFY_GAMMA_SQL`) — fully mitigable via `CREATE VIEW nuclides AS SELECT excitation_kev AS level_keV, ... FROM read_parquet(...)`. Loader could ship a back-compat view.
- External consumers of v0.10.x (fd5, smelt, theranostics docs) reading parquet directly: same as Rust — no aliasing path.

So option (b)'s lockstep claim holds for **Rust + external Python/Polars/DuckDB consumers** (the heavy users); MCP and TS could be partially mitigated with view aliases. Even with the partial mitigation, the Rust ecosystem alone justifies rejection: a coordinated nucl-parquet + crate + downstream-Cargo-bump release is the kind of thing that warrants v1.0 framing, not a v0.11 data-quality release.

The dual-carry strategy in option (a) gives v1.0 a clean cutover path: drop `level_keV`/`half_life_s`/`jp` in v1.0, leaving the already-shipped `spin_x2`/`parity` (and any future strata-native columns) as the canonical encoding. Users get *one* breaking change (column removal), not *two* (introduction + reintroduction).

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
