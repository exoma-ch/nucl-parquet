# Changelog

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
