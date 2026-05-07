# Changelog

## [0.13.0](https://github.com/exoma-ch/nucl-parquet/compare/v0.12.0...v0.13.0) (2026-05-07)


### Features

* **em:** bremsstrahlung total cross-section (epic [#114](https://github.com/exoma-ch/nucl-parquet/issues/114)) ([#117](https://github.com/exoma-ch/nucl-parquet/issues/117)) ([1889697](https://github.com/exoma-ch/nucl-parquet/commit/1889697a6dd8b3de2df80ac05815782f4cba9f12))
* **nudex:** full ENSDF level schemes — final v0.14 deliverable (closes [#122](https://github.com/exoma-ch/nucl-parquet/issues/122)) ([#132](https://github.com/exoma-ch/nucl-parquet/issues/132)) ([ebc3591](https://github.com/exoma-ch/nucl-parquet/commit/ebc35912fba571d1b29f08a3363e11b0bcaee962))
* **nudex:** level-density parameter views (closes [#125](https://github.com/exoma-ch/nucl-parquet/issues/125)) ([#129](https://github.com/exoma-ch/nucl-parquet/issues/129)) ([826efbb](https://github.com/exoma-ch/nucl-parquet/commit/826efbb5b3e9d5a2c0b0197c839d370cf3827633))
* **nudex:** neutron-capture primary gamma spectra (epic [#115](https://github.com/exoma-ch/nucl-parquet/issues/115)) ([#116](https://github.com/exoma-ch/nucl-parquet/issues/116)) ([2c1bdce](https://github.com/exoma-ch/nucl-parquet/commit/2c1bdce46b72e1b039df3771b345c5799b1072bc))
* **nudex:** per-shell ICC factors view (closes [#123](https://github.com/exoma-ch/nucl-parquet/issues/123)) ([#127](https://github.com/exoma-ch/nucl-parquet/issues/127)) ([e346c9f](https://github.com/exoma-ch/nucl-parquet/commit/e346c9fc67571230cad156e2d75cd2d3f67993b8))
* **nudex:** photon strength functions — 6 PSF tables (closes [#124](https://github.com/exoma-ch/nucl-parquet/issues/124)) ([#130](https://github.com/exoma-ch/nucl-parquet/issues/130)) ([8312557](https://github.com/exoma-ch/nucl-parquet/commit/831255758dc4d79f245a3a3ba411439dd1df647a))


### Documentation

* **em:** reframe σ_PE per-shell decode as ETL, not workaround ([#135](https://github.com/exoma-ch/nucl-parquet/issues/135)) ([bb69fc1](https://github.com/exoma-ch/nucl-parquet/commit/bb69fc1f246d8d3774d88c81696715a9281b204e))
* surface v0.11 nuclear and v0.12 photon-matter views ([#111](https://github.com/exoma-ch/nucl-parquet/issues/111)) ([042f4d9](https://github.com/exoma-ch/nucl-parquet/commit/042f4d9a9339ccd1ae0781885636dfbaff27c135))
* surface v0.14 NUDEX views in README + connect() docstring ([#133](https://github.com/exoma-ch/nucl-parquet/issues/133)) ([92ddfef](https://github.com/exoma-ch/nucl-parquet/commit/92ddfefe6bbf4711c23b04b944b5c32eb5285c92))

## [0.12.0](https://github.com/exoma-ch/nucl-parquet/compare/v0.11.0...v0.12.0) (2026-05-05)


### Features

* **em:** atomic_relaxation + fluorescence views (closes [#100](https://github.com/exoma-ch/nucl-parquet/issues/100)) ([#101](https://github.com/exoma-ch/nucl-parquet/issues/101)) ([e04766f](https://github.com/exoma-ch/nucl-parquet/commit/e04766fc5d71a196243acfdd005d6b4ce3f452e8))
* **em:** Compton XS + scattering function + Doppler (closes [#97](https://github.com/exoma-ch/nucl-parquet/issues/97)) ([#104](https://github.com/exoma-ch/nucl-parquet/issues/104)) ([f28d725](https://github.com/exoma-ch/nucl-parquet/commit/f28d725a7864440c6fa62900608a24bb7b2485e1))
* **em:** pair + triplet production cross-sections (closes [#99](https://github.com/exoma-ch/nucl-parquet/issues/99)) ([#102](https://github.com/exoma-ch/nucl-parquet/issues/102)) ([4e322e9](https://github.com/exoma-ch/nucl-parquet/commit/4e322e9bbd99156c6b5e924674b0f90206a9c0df))
* **em:** per-shell σ_PE from EPICS2017 (closes [#105](https://github.com/exoma-ch/nucl-parquet/issues/105)) ([#108](https://github.com/exoma-ch/nucl-parquet/issues/108)) ([085d05b](https://github.com/exoma-ch/nucl-parquet/commit/085d05b01c30774f6ca1c47a77ca67df4f51d840))
* **em:** photoelectric high-Z params + angular kernels (closes [#96](https://github.com/exoma-ch/nucl-parquet/issues/96)) ([#106](https://github.com/exoma-ch/nucl-parquet/issues/106)) ([dcaa242](https://github.com/exoma-ch/nucl-parquet/commit/dcaa2420e13970075ec64ef18cc47b677c90a4a9))
* **em:** Rayleigh CDF + X-ray form factors (closes [#98](https://github.com/exoma-ch/nucl-parquet/issues/98)) ([#103](https://github.com/exoma-ch/nucl-parquet/issues/103)) ([648628b](https://github.com/exoma-ch/nucl-parquet/commit/648628b379339401230c89266d0db48184851226))

## [0.11.0](https://github.com/exoma-ch/nucl-parquet/compare/v0.10.0...v0.11.0) (2026-05-04)


### ⚠ BREAKING CHANGES

* **release:** v0.11.0 — Geant4 migration (closes epic #66) ([#91](https://github.com/exoma-ch/nucl-parquet/issues/91))

* bump release to 0.11.0 ([#93](https://github.com/exoma-ch/nucl-parquet/issues/93)) ([17be938](https://github.com/exoma-ch/nucl-parquet/commit/17be938123238b7658ed7ae71567e85319ea45dd))


### Features

* **release:** v0.11.0 — Geant4 migration (closes epic [#66](https://github.com/exoma-ch/nucl-parquet/issues/66)) ([#91](https://github.com/exoma-ch/nucl-parquet/issues/91)) ([8e3ef26](https://github.com/exoma-ch/nucl-parquet/commit/8e3ef2694b4cf9a4b03a7a76a648a69ac56ce4a5))


### Bug Fixes

* **build:** assigner stops fabricating isomer labels (closes [#49](https://github.com/exoma-ch/nucl-parquet/issues/49)) ([#57](https://github.com/exoma-ch/nucl-parquet/issues/57)) ([9892605](https://github.com/exoma-ch/nucl-parquet/commit/98926052524be026769a3aaf8b2f4d89fbe90506))
* **build:** get_state_map sources from nuclides.parquet (closes [#58](https://github.com/exoma-ch/nucl-parquet/issues/58) P1+P3) ([#61](https://github.com/exoma-ch/nucl-parquet/issues/61)) ([b2abc9d](https://github.com/exoma-ch/nucl-parquet/commit/b2abc9d77ac6f0c28f57890671efb27d00db67b7))

## [0.11.0](https://github.com/exoma-ch/nucl-parquet/compare/v0.10.1...v0.11.0) (2026-05-01)

**The Geant4 migration.** v0.11 replaces the v0.10.x IAEA-LiveChart fetcher pipeline (which had multiple silent data-quality bugs traced to a one-shot AI-written scraper that was deleted before validation) with a Geant4-derived pipeline sourced from the strata project's published Hugging Face dataset. Tracking epic: [#66](https://github.com/exoma-ch/nucl-parquet/issues/66).

### Bug classes eliminated by construction

| v0.10.x bug | Status in v0.11 | Source |
|---|---|---|
| Phantom isomers fabricated by `get_state_map`'s radiation-fallback path (~146 rows) | Gone — `nuclides.parquet` is now sourced from G4ENSDFSTATE3.0 | #69 |
| `parent_level_keV` corrupted up to 2e+215 by IAEA text-as-number parsing (525 rows) | Gone — G4 PhotonEvaporation6.1.2 is fixed-width numeric (max observed ~30 MeV) | #72 |
| `intensity_pct` > 100% on Auger/CE rows from misencoded IAEA fields (511 rows) | Gone — G4 ships per-cascade-relative intensities; documented semantics | #72 |
| `state='' AND decay_mode='IT'` rows (semantically impossible — IT requires a metastable parent) (49 rows) | Gone — G4 RadioactiveDecay6.1.2 schema enforces parent-state on IT entries | #71 |
| `(Z, A, state)` branching sums > 1.0 from duplicate IAEA rows (64 groups) | Gone — RadioactiveDecay6.1.2 has no duplicates | #71 |
| Stable isotopes shipped as `half_life_s = NULL`, conflated with "unknown / unmeasured" | Gone — stable nuclides ship `half_life_s = +inf` (IEEE-754 positive infinity, distinct from NULL) per ADR-0002 | #69 |
| Eu-152m2 (147.86 keV / 96 min) missing from catalog | Present — G4ENSDFSTATE catalogues all isomers; cascade gammas appear in `radiation/Eu.parquet` with `state='m2'` | #69, #72 |

### New / re-derived data

- **`nuclides.parquet`** — joined with AME2020 (mass excess, atomic mass, binding energy, β-decay energy with uncertainties) and IUPAC/CIAAW (isotopic composition, standard atomic weight). Both auxiliary sources are fetched via `scripts/fetch_ame2020.py` / `scripts/fetch_iupac_compositions.py`.
- **`decay_detailed.parquet`** — NEW per-transition table with `parent_ex_kev`, `daughter_ex_kev`, `q_value_kev`, `forbiddenness`. Per-shell EC fractions (`KshellEC`/`LshellEC`/`MshellEC`/`NshellEC`) preserved for downstream X-ray/Auger synthesis.
- **`radiation/{Symbol}.parquet`** — gamma rows from G4 PhotonEvaporation + X-ray + Auger rows synthesized from G4EMLOW EADL × per-shell EC/IC fractions, unioned by row with `rad_type` as discriminator.
- **`coincidences/{Symbol}.parquet`** — gamma cascade pairs derived by self-join on shared intermediate levels. ~600k pairs across 104 element files. Schema preserves v0.10.x columns (`Z, A, dataset, gamma_energy_keV, coinc_energy_keV`) alongside G4 bonus columns.

### Schema additions (additive per ADR-0002 — zero breaking changes)

Bonus columns present where source supplies them, NULL elsewhere:
- `nuclides.parquet`: `spin_x2` (Int16), `parity` (Int8), `floating_level_flag` (String), `magnetic_moment_jt` (Float64)
- `radiation/*.parquet`: `multipolarity` (Int32), `mixing_ratio` (Float32), `icc_total` (Float32), `daughter_level_keV` (Float64), `vacancy_shell` (String)
- `coincidences/*.parquet`: G4-derived bonus columns (`gamma1_intensity`, `gamma2_intensity`, `parent_level_keV`, `intermediate_level_keV`, `final_level_keV`, `pair_intensity`, `gamma1_icc_total`, `gamma2_icc_total`)

### Migration notes for downstream consumers

- **Stable-isotope half-life**: Python code that does `if half_life_s is None` (or SQL `IS NULL`) to detect stable isotopes will need to add `or math.isinf(half_life_s)` (`is_finite()` is the canonical check). The new encoding distinguishes "stable per physical observation" (`+inf`) from "genuinely unknown / unmeasured" (NULL); v0.10.x conflated both.

  ```sql
  -- v0.10 (broken — would catch unknowns too):
  SELECT * FROM nuclides WHERE half_life_s IS NULL;
  -- v0.11 (correct):
  SELECT * FROM nuclides WHERE NOT is_finite(half_life_s) AND half_life_s IS NOT NULL;
  -- "stable or known-long-lived":
  SELECT * FROM nuclides WHERE half_life_s IS NOT NULL AND (NOT is_finite(half_life_s) OR half_life_s > 1e17);
  ```

- **Per-shell EC**: `decay.parquet` now ships `decay_mode IN ('KshellEC', 'LshellEC', 'MshellEC', 'NshellEC')` separately rather than collapsing to `'EC'`. Callers searching `decay_mode == 'EC'` will find no rows after v0.11; use `decay_mode IN (...)` or sum across the per-shell rows. The aggregated total is recoverable; the v0.11 form is strictly richer.

  ```sql
  -- v0.10 (returns nothing in v0.11):
  SELECT branching FROM decay WHERE Z=43 AND A=99 AND state='m' AND decay_mode='EC';
  -- v0.11 equivalent (sum across per-shell rows):
  SELECT SUM(branching) AS ec_total FROM decay
   WHERE Z=43 AND A=99 AND state='m'
     AND decay_mode IN ('KshellEC', 'LshellEC', 'MshellEC', 'NshellEC');
  ```

- **Filing convention for radiation rows**: G4 PhotonEvaporation is keyed by the *de-exciting nucleus*, not the decaying parent. The v0.10.x "121.78 keV ground-state Eu-152 line" was a Sm-152 daughter line credited via the IAEA fetcher's parent-merging artifact; under the v0.11 convention it lives in `Sm.parquet` (Z=62), not `Eu.parquet`. Parent-keyed clinical-line queries should join through `meta/decay.parquet` on `(parent_Z, parent_A, daughter_Z, daughter_A)`.

- **⚠️ `intensity_pct` is relative-per-cascade, not 0-100% absolute** — this is the most consequential semantic change for dose-calc consumers. v0.10.x stored absolute per-decay percentages; v0.11 mirrors G4 PhotonEvaporation's relative-per-cascade-level normalization (max observed ~9000). A v0.10 dose pipeline that did `intensity_pct/100 * activity_Bq` will silently produce wrong gamma fluences post-upgrade — no error, just incorrect numbers. Normalize per `(Z, A, parent_level_keV)` before treating as a probability:

  ```sql
  -- gamma emission probability per-decay (correct v0.11 form):
  WITH cascade_totals AS (
      SELECT Z, A, parent_level_keV, SUM(intensity_pct) AS cascade_sum
        FROM radiation
       WHERE rad_type = 'gamma'
       GROUP BY Z, A, parent_level_keV
  )
  SELECT r.Z, r.A, r.energy_keV,
         r.intensity_pct / NULLIF(c.cascade_sum, 0) AS p_per_cascade
    FROM radiation r
    JOIN cascade_totals c USING (Z, A, parent_level_keV)
   WHERE r.rad_type = 'gamma' AND r.Z=63 AND r.A=152;
  ```

  See `nucl_parquet/g4/coincidences.py` module docstring for the full worked example including parent-keyed feeding fractions.

### Removed (obsolete with G4 inputs)

- `nucl_parquet/build_radiation_state.py` — IAEA-rescue assigner with the 0.5 keV threshold, `BuildIntegrityError`, `_ORPHAN_CEILING`, `_UNLABELED_EXCITED_CEILING`, `_surface_diagnostics`, `_assign_states`. All gone.
- `nucl_parquet/build_nuclides.py` — IAEA-fallback `get_state_map`. Replaced by `nucl_parquet/g4/ensdfstate.py`.
- v0.10.x rescue PR machinery: `_rescue_it_orphan_isomers` (#63), `_LEVEL_EXACT_MATCH_KEV` (#57). Bug classes are gone, code paths follow.

### Build pipeline

Single entry point: `python -m nucl_parquet.g4.build_all`. Runs all six converters in dependency order, performs the `radiation_atomic/` → `radiation/` union merge, asserts sweep invariants. Idempotent; supports `--merge-only` for partial rebuilds.

### Acceptance gate

`tests/test_g4_diff_harness.py` ships 23 canonical-isotope tests — Eu-152m + Eu-152m2, Hf-178m2, Tc-99m, Sc-44m, In-113m, Ag-108m, Ba-137m all required to be present at expected level + half-life; stable isotopes (H-1, He-4, Fe-56, Pb-208) required to ship `+inf`; Eu-152m2 cascade gammas required to be in `Sm.parquet`; the 6 v0.10.x bug classes above pinned by sweep invariants.

### References

- ADR-0001 — Radiation `state` API (the v0.10.0 design that v0.11 supersedes; a postscript section now records the migration)
- ADR-0002 — G4 migration schema decision (preserve v0.10.x schema + dual-carry; +inf for stable sentinel)
- `docs/g4-xray-auger-design.md` — synthesis flow + Integration contract for the radiation merge
- HuggingFace dataset: [`gerchowl/strata-data`](https://huggingface.co/datasets/gerchowl/strata-data) at the SHA pinned in `data/catalog.json`

### Deferred follow-ups (filed as separate issues)

- [#80](https://github.com/exoma-ch/nucl-parquet/issues/80) — per-file SHA-256 verification in catalog.json (defensive hardening)
- [#84](https://github.com/exoma-ch/nucl-parquet/issues/84) — IT-on-ground renormalization audit for No-253-class nuclides (~5 isotopes with upstream-G4-anomaly branchings)
- [#88](https://github.com/exoma-ch/nucl-parquet/issues/88) — X-ray + Auger high-Z accuracy improvements (Coster-Kronig transfer, per-shell IC partials, EC-then-IC daughter cascades)

## [0.10.0](https://github.com/exoma-ch/nucl-parquet/compare/v0.9.0...v0.10.0) (2026-04-24)


### Features

* **radiation:** `gamma_lines(db, z, a, state="", min_intensity=0.0)` and `identify_gamma(db, energy, tolerance=2.0, state="", min_intensity=0.1)` helper functions — ground-state-default, isomer opt-in. Mirrors the Rust `DecayDb::modes(z, a, state)` shape. (closes [#36](https://github.com/exoma-ch/nucl-parquet/issues/36))
* **radiation:** `GAMMA_LINES_SQL` / `IDENTIFY_GAMMA_SQL` now filter `state = $state` (default ground) — fixes the v0.9.0 "correct-by-default" claim that was aspirational. Previous mixed-state behaviour available as `GAMMA_LINES_ALL_SQL` / `IDENTIFY_GAMMA_ALL_SQL`. **Breaking (pre-1.0):** callers must pass `$state` (or switch to `gamma_lines()`) to preserve ground-only results; callers that want mixed states must switch to the `*_ALL_SQL` variants. (closes [#36](https://github.com/exoma-ch/nucl-parquet/issues/36))
* **radiation:** `COINCIDENCE_SQL` gains a `$state` parameter scoping the radiation-side intensity lookup. Previous hardcoded `state=''` JOIN silently returned NULL intensities for isomeric parents (Ag-110m, Hf-178m2, Ho-166m, Lu-177m). (closes [#36](https://github.com/exoma-ch/nucl-parquet/issues/36))
* **build:** `build_radiation_state` warns on orphan `(Z, A)` nuclides and fuzzy parent-level matches beyond 2.0 keV, and asserts uniqueness of isomeric levels at build time. (closes [#36](https://github.com/exoma-ch/nucl-parquet/issues/36))
* adopt prek for pre-commit/pre-push hooks (closes [#42](https://github.com/exoma-ch/nucl-parquet/issues/42)) ([33c9645](https://github.com/exoma-ch/nucl-parquet/commit/33c9645e6b6b6afda18ccae801d3bb85ff062ecd))
* adopt prek for pre-commit/pre-push hooks (closes [#42](https://github.com/exoma-ch/nucl-parquet/issues/42)) ([3ccc1d7](https://github.com/exoma-ch/nucl-parquet/commit/3ccc1d7518a44c0fa04bb3021614f0d8fc44b189))
* **ci:** add MCP publish jobs, lockstep MCP versions with core (closes [#39](https://github.com/exoma-ch/nucl-parquet/issues/39)) ([647a251](https://github.com/exoma-ch/nucl-parquet/commit/647a251fb5ec7f09b71352c3e6f728a1b1aeace4))
* **ci:** MCP publish jobs + lockstep versioning (closes [#39](https://github.com/exoma-ch/nucl-parquet/issues/39)) ([cfb0deb](https://github.com/exoma-ch/nucl-parquet/commit/cfb0deb7d73dd5007fda8711bbb2639f6aa998a0))


### Bug Fixes

* **tests:** `test_integrity.py::test_co60_half_life` now filters `state=''` — previous `LIMIT 1` was non-deterministic once Co-60m entries started sharing `(Z, A)` with the ground state. (closes [#36](https://github.com/exoma-ch/nucl-parquet/issues/36))
* **go:** align module path with actual directory (closes [#40](https://github.com/exoma-ch/nucl-parquet/issues/40)) ([bc6f340](https://github.com/exoma-ch/nucl-parquet/commit/bc6f340b522aa16147a24c241c9cda3a96fcd138))
* **go:** align module path with actual directory (closes [#40](https://github.com/exoma-ch/nucl-parquet/issues/40)) ([ed31931](https://github.com/exoma-ch/nucl-parquet/commit/ed3193125a1f1d03391161b06b8aa245de6512e0))
* log_log_interp handles zero energies and q=0 form factor queries ([a253fc5](https://github.com/exoma-ch/nucl-parquet/commit/a253fc5f5006aa2f32f56d0221b79f8139767424))
* repair data delivery pipeline end-to-end (closes [#35](https://github.com/exoma-ch/nucl-parquet/issues/35)) ([12eb037](https://github.com/exoma-ch/nucl-parquet/commit/12eb03788bcbfe9c38e90f16c6f7dcd88205a24a))
* repair data delivery pipeline end-to-end (closes [#35](https://github.com/exoma-ch/nucl-parquet/issues/35)) ([47c7f43](https://github.com/exoma-ch/nucl-parquet/commit/47c7f438bbb4fa8caac0bfae34fc6b0ccd0dc1b1))


### Refactoring

* **rs:** clean up pre-existing clippy warnings (closes [#41](https://github.com/exoma-ch/nucl-parquet/issues/41)) ([05d700a](https://github.com/exoma-ch/nucl-parquet/commit/05d700aa86c384e127fca172b33475e8e38fbc9b))
* **rs:** clean up pre-existing clippy warnings (closes [#41](https://github.com/exoma-ch/nucl-parquet/issues/41)) ([4abe440](https://github.com/exoma-ch/nucl-parquet/commit/4abe440e6ad87437fd63bc0942fc2fd49f00789b))

## [0.9.0] — 2026-03-27

### Added
- `nuclides.parquet`: isomeric states (Tc-99m, Sc-44m, Eu-152m, Ba-137m, etc.)
  as separate entries with own half-life, spin-parity, and decay modes (#34)
  - 699 isomeric states added alongside 3383 ground states
  - Keyed on `(Z, A, state)` where state is `""`, `"m"`, `"m2"`, etc.
- `state` column in `radiation` table scoping gamma/X-ray lines to the correct
  parent nuclear state — queries now return correct-by-default results
- `nuclides` DuckDB view as the primary nuclide lookup table
- `build_nuclides` and `build_radiation_state` build scripts

### Changed
- `GAMMA_LINES_SQL` and `IDENTIFY_GAMMA_SQL` now join on `(Z, A, state)` via
  the `nuclides` view, returning only radiation lines belonging to the queried
  nuclear state
- `COINCIDENCE_SQL` scopes radiation intensity lookups to ground state
- Dose constants backfill now uses `nuclides.parquet` (covers isomeric pure-beta
  emitters in addition to ground states)

### Deprecated
- `ground_states` DuckDB view: use `nuclides` instead. `ground_states` is now a
  filtered view (`WHERE state = ''`) for backwards compatibility and will be
  removed in v1.0.

### Migration notes
- The `radiation` table has a new `state` column. Existing queries without a
  `state` filter continue to return all parent states (backwards compatible).
- To get ground-state-only radiation (e.g. for aged calibration sources):
  `SELECT * FROM radiation WHERE Z=63 AND A=152 AND state=''`
- The `nuclides` view replaces `ground_states` for general nuclide lookups.
  `ground_states` remains as a compatibility alias filtering `state=''`.

## [0.8.1] — 2026-03-25

### Fixed
- Rust crate: enable parquet `zstd` codec (data files use zstd compression) (#32)
- Rust crate: handle `LargeStringArray` (LargeUtf8) alongside `StringArray` in all loaders (#33)
- Filter macOS `._` resource fork files in directory scans and tar extraction

## [0.8.0] — 2026-03-25

### Added
- Rust crate: `DataDir` with auto-download + cache from GitHub Releases (#31)
- Feature-gated `fetch` adds `reqwest`/`tar`/`zstd` for one-line data bootstrapping
- `DataDir::ensure()` downloads ~383 MB data archive to `~/.nucl-parquet/v{VERSION}/`
- Convenience methods: `data.photon_db()`, `data.stopping_db()`, etc.
- Data archive attached to GitHub releases as `nucl-parquet-data-v{VERSION}.tar.zst`

## [0.7.0] — 2026-03-25

### Added
- Rust crate: `SubshellPeDb` (subshell photoelectric XS from EPDL97) (#29)
- Rust crate: `XcomDb` (mass attenuation µ/ρ and µ_en/ρ for elements + compounds) (#29)
- Rust crate: `ElectronDb` (EEDL elastic, bremsstrahlung, ionization XS) (#29)

## [0.6.0] — 2026-03-24

### Added
- Neutron total/elastic XS from ENDF/B-VIII.1 (`neutron_total` view, 97 elements, 214K rows) (#26)
- Bohr energy straggling column (`straggling` [MeV² cm²/g]) in catima stopping tables (#25)
- Rust `StoppingDb::catima_straggling()` and TS `CatimaColumns.straggling` wrappers

## [0.5.0] — 2026-03-24

### Added
- Neutron KERMA coefficients computed from ENDF/B-VIII.1 cross-sections (89 elements, 208K data points)
- `kerma` DuckDB view: `SELECT * FROM kerma WHERE Z=29 AND A=63`
- `build_kerma.py` script: analytical KERMA from elastic recoil + charged-product kinematics
- Elemental KERMA is Bragg-additive for compounds — enables neutron dose in hyrr

## [0.4.0] — 2026-03-24

### Added
- Spectrum-averaged neutron cross-sections (`spectrum_xs` view) for thermal, epithermal, and fast spectra (#8)
- Rust crate: `StoppingDb`, `CrossSectionDb`, `AbundancesDb`, `DecayDb`, `DoseDb` wrappers (#19)
- TS `@nucl-parquet/core`: `stoppingColumns()`, `xsColumns()`, `catimaColumns()` for zero-copy WASM transfer (#23)
- EXFOR data restored with `data/` layout and full-library coverage
- Geant4 INCL++/ABLA07 heavy-ion fragment production cross-sections (`hi-xs-prod`)
- Deuteron/triton/helion-3 stopping power via velocity scaling from PSTAR/ASTAR

### Changed
- Repo layout: all data moved to `data/`, client SDKs moved to `clients/`
- Stopping power split from monolithic `stopping.parquet` into per-source files
- `data_dir()` now detects `data/catalog.json` instead of root-level `catalog.json`

### Fixed
- Removed 245 TALYS overflow sentinel values (1.99e38 mb) from TENDL-2025 deuteron files

## [0.3.14] — 2026-03-24

Layout migration release (data/ + clients/).

## [0.3.13] — 2026-03-24

Per-source stopping files, README and catalog updates.

## [0.3.12] — 2026-03-24

Geant4 heavy-ion production cross-sections.

## [0.3.11] — 2026-03-24

Light-ion stopping power (d/t/³He), EXFOR Li/P data.

## [0.3.10] — 2026-03-23

TENDL-2025 sentinel cleanup, theranostic document and data fixes.
