# ADR 0001 — Radiation `state` API: ground-state-default function helpers

| | |
|---|---|
| **Status** | Accepted |
| **Date** | 2026-04-25 |
| **Affected versions** | v0.9.0 (data) → v0.10.0 (API) |
| **Related** | #34, #36, #37, #38, #43, #48, #49, #50, #51, #52, #53 |

## Context

`nucl-parquet` ships ENSDF-derived decay radiation data. A nuclide can have multiple parent nuclear states — a ground state and one or more isomeric states (`m`, `m2`) — each with its own half-life and gamma-line spectrum. Eu-152 is the canonical example: the 13.5-year ground state and the 9.31-hour `m` isomer emit *different* gammas at *different* intensities. Mixing them when building spectral templates causes phantom-count bugs (e.g. predicting 14.2 % at 841.63 keV for an aged Eu-152 calibration source — that line belongs to the isomer and is absent in aged sources).

Through v0.8.x the `radiation` table had no way to express which parent state emitted each line. v0.9.0 added a `state` column and a `(Z, A, state)`-keyed `nuclides` view — but the SQL helpers that consume them (`GAMMA_LINES_SQL`, `IDENTIFY_GAMMA_SQL`, `COINCIDENCE_SQL`) were "correct-by-default" only in CHANGELOG copy. In practice, queries returned mixed ground+isomer rows by default.

The decision recorded here is how the v0.10.0 API exposes the `state` column to callers.

## Decision

Adopt **function-based helpers with ground-state-as-default**, mirroring the Rust `DecayDb::modes(z, a, state)` shape:

```python
nucl_parquet.gamma_lines(db, z=63, a=152, state="", min_intensity=0.0)
nucl_parquet.identify_gamma(db, energy=1173.0, tolerance=2.0, state="")
```

Specifically:

1. **Default `state=""` (ground).** The most common workflow (aged calibration sources, NaI/HPGe spectroscopy templates, dose calibration) only ever wants ground-state lines.
2. **Isomer opt-in.** Pass `state="m"` (or `"m2"`) explicitly to query an isomeric parent.
3. **`GAMMA_LINES_SQL` / `IDENTIFY_GAMMA_SQL` constants kept** but their bodies now require a `$state` parameter. Behaviour is "ground-state-default-when-passed-`""`" rather than the v0.9.0 "all states mixed".
4. **Sibling `*_ALL_SQL` constants added** for callers who genuinely want every parent state in one query (rare, mostly for catalogue dumps and cross-state diagnostics).
5. **`COINCIDENCE_SQL` accepts a `$state` parameter** — partial fix only; see _Known limitations_ below.
6. **Pre-1.0 breakage accepted.** Callers importing `GAMMA_LINES_SQL` / `IDENTIFY_GAMMA_SQL` from v0.9.0 get a DuckDB binder error on v0.10.0 (missing `$state`). The CHANGELOG documents the migration; no `DeprecationWarning` shim shipped (tracked in #53).

## Alternatives considered

### A. Documentation-only (rejected)

Keep the v0.9.0 SQL bodies unchanged and add a prominent docstring + README note that callers must `AND state=''` for ground-state-only results. **Rejected** — leaves the bug class in place. Every downstream user who misses the doc note silently produces wrong results, and gamma-spectroscopy bugs typically manifest as 5 % intensity drift (looks like calibration error, not a library bug).

### B. New parameter on existing SQL constants only (rejected)

Add `$state` parameter to the existing constants but no Python helper. **Rejected** — string-based SQL constants don't compose well with caller-side `WHERE` clauses (the typical user pattern is `"SELECT * FROM (" + GAMMA_LINES_SQL + ") WHERE Z=63 AND A=152"`, which silently bypasses any embedded default). A function with a typed default closes the hole that strings can't.

### C. Sibling SQL constants only (rejected)

Split into `GAMMA_LINES_GROUND_SQL`, `GAMMA_LINES_ALL_SQL`, `GAMMA_LINES_ISOMER_SQL`. **Rejected** — triples the API surface and forces every cross-language client (Rust crate, TS SDK, MCP server) to mirror three names. The Rust crate already adopted `DecayDb::modes(z, a, state)` (function with state arg), so converging Python on the same shape was strictly better.

### D. Add `state` to `coincidences/*.parquet` at build time

The natural fix for `COINCIDENCE_SQL` would be to track the parent-state context in the cascade-pair extraction. **Deferred** to a follow-up — the data pipeline change is non-trivial and v0.10.0's window was constrained. Filed as #50 (Option B).

## Consequences

### Positive

- The default API call returns correct results for the dominant workflow without the user needing to remember a `state` filter. The "phantom 841.63 keV in aged Eu-152" class of bug is closed by construction.
- Cross-language consistency: Python `gamma_lines(db, z, a, state)` ↔ Rust `DecayDb::modes(z, a, state)`. Mental model converges.
- `*_ALL_SQL` exists for the catalog/diagnostic cases that legitimately want every state.

### Negative

- Pre-1.0 breaking change for any caller importing `GAMMA_LINES_SQL` / `IDENTIFY_GAMMA_SQL` directly. Accepted given the pre-1.0 status, but #53 tracks adding a `FutureWarning` shim.
- Function-based API requires DuckDB connection awareness in callers. Not a regression — `connect()` was already the entry point — but it's a slightly fatter import surface than raw SQL constants.

### Known limitations (tracked as follow-ups)

- **`COINCIDENCE_SQL` `$state` parameter is semantically broken for `state != ""`.** `coincidences/*.parquet` itself has no `state` column, so the LEFT JOIN to radiation produces NULL intensities silently for isomeric parents. Tracked as #50.
- **`build_radiation_state` assigner has two correctness bugs** that mislabel some non-isomeric excited parents as `m` (silent), and label orphan `(Z,A)` rows as `m` (invents an isomer). Tracked as #49. The XFAIL canary in `tests/test_state.py::test_radiation_state_subset_of_nuclides` watches for this — flips to PASSED when #49 + #38 land.
- **Test debt:** `COINCIDENCE_SQL` `$state`, cross-nuclide `gamma_lines(z=None, a=None)`, cross-language Python↔Rust parity, and end-to-end `build_radiation_state.build()` integration are all uncovered. Tracked as #52.
- **API polish:** `state` parameter is byte-equal (`state="M"` returns zero rows silently); `identify_gamma` lacks `z`/`a` filters for symmetry; README has no mention of the new helpers. Tracked as #53.
- **Versioning drift:** unrelated but discovered during round-2 review — MCP `serverInfo.version` is hardcoded `"0.3.8"` despite v0.10.0 release. Tracked as #51 with proposed SSoT (`env!("CARGO_PKG_VERSION")` / JSON import; release-please extra-files extended).

## Empirical data

Two rounds of fresh agent reviews informed this decision:

**Round 1 (design)** — testing methodology, gamma-spectroscopy domain, library API design. Domain reviewer surfaced the long-lived metastables (Ag-110m, Hf-178m2, Ho-166m, Lu-177m) where coincidence-summing matters most — drove the `COINCIDENCE_SQL` `$state` parameter requirement. API reviewer surfaced the function-vs-string-constant trade-off and the Rust `DecayDb::modes` precedent — drove option D over A/B/C above.

**Round 2 (implementation)** — implementation correctness, test adversary, cross-language integration. Test adversary surfaced the missing `COINCIDENCE_SQL` coverage and the XFAIL-no-floor hazard. Implementation reviewer found the assigner mislabel bugs (filed as #49). Cross-language reviewer found the version drift (filed as #51).

## References

- Issue #34 — original bug report (gamma lines from isomeric states not distinguished).
- Issue #36 — implementation tracking issue.
- PR #37 — code changes (bundled accidentally with the data-delivery fix).
- PR #43 — test coverage.
- PR #48 — release-please v0.10.0.
- `nucl_parquet/loader.py:243–490` — SQL constants + `gamma_lines()` / `identify_gamma()`.
- `nucl_parquet/build_radiation_state.py` — assigner.
- `clients/rs/nucl-parquet/src/meta.rs:230` — Rust `DecayDb::modes` signature.
- CHANGELOG.md — v0.10.0 entry.
