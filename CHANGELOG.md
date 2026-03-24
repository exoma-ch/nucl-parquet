# Changelog

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
