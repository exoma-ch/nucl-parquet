# Changelog

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
