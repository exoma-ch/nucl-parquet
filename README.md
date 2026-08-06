# nucl-parquet

Nuclear data as Parquet files — cross-sections, stopping powers, decay data, and isotopic abundances from all major evaluated libraries. Queryable with DuckDB, Polars, Pandas, or any Arrow-compatible tool.

## Installation

```bash
pip install nucl-parquet
```

The pip package is a thin loader (~50 KB). Data files are either cloned from the git repo or downloaded from GitHub Releases:

```python
import nucl_parquet

# Download the latest data tarball to ~/.nucl-parquet/ (first time only)
nucl_parquet.download()

# Or pin a specific CalVer release
nucl_parquet.download(data_version="2026.05.11")

# Inspect which data version a checkout pins
nucl_parquet.data_version()  # → "2026.05.11"
```

Or clone the repo directly for the full dataset:

```bash
git clone https://github.com/exoma-ch/nucl-parquet.git
export NUCL_PARQUET_DATA=/path/to/nucl-parquet/data
```

### Release scheme — data on CalVer, code on semver

Data tarballs and code packages release on independent cadences. Data refreshes when upstream sources (NIST/Geant4/strata) update — tracked as `data-YYYY.MM.DD` tags on GitHub. Code releases follow conventional-commit semver via release-please as `vX.Y.Z` tags. The active data version a checkout pins lives at `data/catalog.json::data_version`.

| Release | Tag form | Trigger | Cadence |
|---|---|---|---|
| Data tarball | `data-2026.05.11` | manual push of CalVer tag matching `catalog.json::data_version` | upstream refreshes |
| Code (PyPI, crates, npm, Go) | `v0.14.0` | release-please PR merge | conventional-commit semver |

Per-package code semver (split MCP from core, etc.) is tracked in #150.

### Verifying a data release

Data releases from `data-2026.8.2` onward are signed with [minisign](https://jedisct1.github.io/minisign/); the signature ships as `nucl-parquet-data-<CalVer>.tar.zst.minisig` on the GitHub release.

```bash
just verify-release              # verify the version in data/catalog.json
just verify-release 2026.8.2     # verify a specific CalVer
```

The `data_sha256` pin and the signature answer different questions — "are these the bytes that build was tested against?" versus "did these bytes come from the nuclear-data team?". Only the second survives re-pinning, since GitHub's asset digest is only as trustworthy as the account. The public key consumers pin, the grandfathering rule for older unsigned releases, and the rotation runbook are in [docs/security/data-signing.md](docs/security/data-signing.md).

## Usage

```python
import nucl_parquet

db = nucl_parquet.connect()

# Cross-section query
db.sql("SELECT * FROM tendl_2023_iso WHERE target_A=63 AND residual_Z=30")

# Compare all libraries
db.sql("SELECT library, energy_MeV, xs_mb FROM xs WHERE target_A=63 AND residual_Z=30")

# Decay chain
db.sql(nucl_parquet.DECAY_CHAIN_SQL, params={"parent_z": 92, "parent_a": 238})

# Stopping power — light ions (NIST PSTAR/ASTAR/ESTAR; catima for ³He)
nucl_parquet.elemental_dedx(db, "p", 29, 10.0)     # protons in Cu at 10 MeV
nucl_parquet.elemental_dedx(db, "a", 29, 5.0)      # α in Cu at 5 MeV (NIST ASTAR)
nucl_parquet.elemental_dedx(db, "e", 29, 1.0)      # electrons in Cu at 1 MeV
nucl_parquet.compound_dedx(db, "p", [(29, 0.5), (30, 0.5)], 10.0)

# Stopping power — heavy ions (CatIMA, any isotope of Z=1-92)
nucl_parquet.elemental_dedx(db, "c12",  6, 12 * 100.0)   # C-12 in C at 100 MeV/u
nucl_parquet.elemental_dedx(db, "pb208", 82, 208 * 50.0)  # Pb-208 in Pb at 50 MeV/u
nucl_parquet.elemental_dedx(db, "xe132", 14, 132 * 50.0)  # Xe-132 in Si at 50 MeV/u

# Heavy-ion total reaction cross-sections (Tripathi 1997)
db.sql("SELECT * FROM hi_xs WHERE target_Z=29 ORDER BY energy_MeV")  # c12 on Cu
db.sql("""
    SELECT energy_MeV, energy_MeV/12 AS energy_MeV_u, xs_mb
    FROM hi_xs WHERE target_Z=6
""")  # c12 on C — typical carbon therapy channel

# Heavy-ion production cross-sections (Geant4 INCL++/ABLA07)
# σ(Zf, Af, E) per residual isotope — 6 projectiles × 92 targets × ~60 energies
db.sql("""
    SELECT residual_Z, residual_A, energy_MeV, xs_mb
    FROM hi_xs_prod
    WHERE target_Z=29 AND library='hi-xs-prod'
    ORDER BY residual_Z, residual_A, energy_MeV
""")  # all C-12 fragmentation products on Cu

# Nuclear structure & decay (v0.11+ — Geant4-derived from strata HF dataset)
db.sql("SELECT * FROM nuclides WHERE Z=43 AND A=99")            # Tc-99g + Tc-99m
db.sql("SELECT * FROM radiation WHERE Z=63 AND A=152 AND state=''")  # Eu-152 lines
db.sql("SELECT * FROM coincidences WHERE Z=27 AND A=60")        # Co-60 cascade pairs
db.sql("SELECT * FROM decay_detailed WHERE Z=43 AND A=99")      # per-shell EC fractions

# Photon-matter interaction (v0.12+ — Geant4 G4EMLOW8.8)
db.sql("SELECT * FROM photon_pe WHERE Z=82 AND shell=0")        # Pb K-shell σ_PE(E)
db.sql("SELECT * FROM photon_compton WHERE Z=29")               # Cu Compton σ(E)
db.sql("SELECT * FROM photon_pair WHERE Z=82 AND channel='total'")  # Pb pair σ(E)
db.sql("SELECT * FROM photon_rayleigh_cdf WHERE Z=82")          # angular sampling CDF
db.sql("SELECT * FROM atomic_relaxation WHERE Z=53 AND vacancy_shell='K'")  # I K-vacancy
db.sql("SELECT * FROM fluorescence WHERE Z=82")                 # Pb fluorescence yields

# Electron-matter interaction (v0.13+ — Geant4 G4EMLOW8.8)
db.sql("SELECT * FROM electron_brem WHERE Z=82")                # Pb bremsstrahlung σ_brem(E)

# NUDEX detailed nuclear data (v0.14+ — Geant4 G4NUDEXLIB1.0)
db.sql("SELECT * FROM nudex_levels WHERE Z=27 AND A=60")        # Co-60 full level scheme
db.sql("SELECT * FROM nudex_level_gammas WHERE Z=82 AND A=208") # Pb-208 transitions
db.sql("SELECT * FROM capture_gammas WHERE Z=27 AND A=60")      # 59Co(n,γ)60Co primaries
db.sql("SELECT * FROM icc_factors WHERE Z=82 AND shell='K'")    # Pb K-shell BrIcc
db.sql("SELECT * FROM psf_e1 WHERE Z=82 AND A=208")             # Pb-208 SMLO E1 GDR
db.sql("SELECT * FROM level_density_bfm WHERE Z=82 AND A=208")  # Pb-208 BFM params
```

### Data resolution

`connect()` finds data in this order:

1. Explicit `data_dir` argument
2. `$NUCL_PARQUET_DATA` environment variable
3. Sibling repo checkout (when running from source)
4. `~/.nucl-parquet/` (downloaded via `nucl_parquet.download()`)

## Why Parquet instead of ENDF-6?

The [ENDF-6 format](https://www.nndc.bnl.gov/endfdocs/ENDF-102/) dates from the 1960s. It was designed for Fortran on punch cards: 80-character fixed-width records, implicit column positions, and a cryptic MF/MT numbering system.

| | ENDF-6 | Parquet |
|---|---|---|
| **Format** | Fixed-width Fortran text, 80-char cards | Columnar binary, self-describing schema |
| **Parsers needed** | Specialized (NJOY, PREPRO, FUDGE, `endf` pkg) | Any language — Python, R, Julia, Rust, JS, SQL |
| **Random access** | Sequential parse from start | Predicate pushdown, skip irrelevant row groups |
| **Compression** | None (or gzip'd text) | zstd columnar compression (5-10x smaller) |
| **Cross-library comparison** | Convert each library separately first | `SELECT * FROM '*/xs/p_Cu.parquet'` |
| **Browser/WASM** | Not feasible | Works natively (DuckDB-WASM, Pyodide) |

**Size comparison** for the same data:

| Library | ENDF-6 (zipped) | Parquet (zstd) | Reduction |
|---------|-----------------|----------------|-----------|
| TENDL-2025 neutron | ~800 MB (2850 zip files) | 25 MB | **32x** |
| ENDF/B-VIII.1 (all) | ~120 MB | 4.3 MB | **28x** |
| JENDL-5 (all) | ~200 MB | 8.6 MB | **23x** |

## Libraries included

<!-- AUTO:libraries -->
| Library | Projectiles | Source |
|---------|------------|--------|
| [BROND-3.1](https://vant.ippe.ru/) | n | IPPE |
| [CENDL-3.2](http://www.nuclear.csdb.cn/) | n | CIAE |
| [ENDF/B-VIII.0 neutron (NJOY-processed)](https://github.com/openmc-data-storage/ENDF-B-VIII.0-NNDC) | n |  |
| [ENDF/B-VIII.0 neutron transport channels](https://github.com/openmc-data-storage/ENDF-B-VIII.0-NNDC) | n |  |
| [ENDF/B-VIII.1](https://www.nndc.bnl.gov/endf-b8.1/) | p, d, t, ³He, α |  |
| [EXFOR (reaction channels)](https://github.com/IAEA-NDS/exfor_master) | n, p, d, t, ³He, α |  |
| [EXFOR (residual production)](https://github.com/IAEA-NDS/exfor_master) | n, p, d, t, ³He, α |  |
| [FENDL-3.2](https://www-nds.iaea.org/fendl/) | n | IAEA |
| [HI-XS (Tripathi 1997)](https://doi.org/10.1016/S0168-583X(96)00331-X) | ar40, c12, ca40, fe56, he4, ne20, ni58, o16, p, pb208, si28, xe132 |  |
| [HI-XS Production (Geant4 INCL++/ABLA07)](https://geant4.org/) | c12, o16, ne20, si28, ar40, fe56 |  |
| [IAEA-Medical](https://www-nds.iaea.org/medical/) | p, d, ³He, α | IAEA |
| [IAEA-PD-2019](https://www-nds.iaea.org/photonuclear/) | γ | IAEA |
| [IRDFF-II](https://www-nds.iaea.org/IRDFF/) | n | IAEA |
| [JEFF-4.0](https://www.oecd-nea.org/dbdata/jeff/) | n, p | NEA |
| [JENDL-5](https://wwwndc.jaea.go.jp/jendl/j5/j5.html) | n, p, d, α | JAEA |
| [JENDL-DEU-2020](https://wwwndc.jaea.go.jp/jendl/deu/deu.html) | d | JAEA |
| [JENDL/AD-2017](https://wwwndc.jaea.go.jp/jendl/jad/jad.html) | n, p | JAEA |
| [TENDL-2023 + Aug 2024 isomeric correction](https://tendl.web.psi.ch/tendl_2023/tendl2023.html) | p, d, t, ³He, α |  |
| [TENDL-2025](https://tendl.web.psi.ch/) | n, p, d, t, ³He, α | PSI |
<!-- /AUTO:libraries -->

## Parquet schemas

**Evaluated cross-sections** (`{library}/xs/*.parquet`):

| Column | Type | Description |
|--------|------|-------------|
| target_A | Int32 | Target mass number |
| residual_Z | Int32 | Product atomic number |
| residual_A | Int32 | Product mass number |
| state | Utf8 | Isomer state: `""`, `"g"`, `"m"` |
| energy_MeV | Float64 | Projectile energy in MeV |
| xs_mb | Float64 | Cross-section in millibarn |

**EXFOR experimental** (`exfor/*.parquet`):

| Column | Type | Description |
|--------|------|-------------|
| exfor_entry | Utf8 | EXFOR accession number |
| target_Z | Int32 | Target atomic number |
| target_A | Int32 | Target mass number (0 = natural) |
| residual_Z | Int32 | Product atomic number |
| residual_A | Int32 | Product mass number |
| state | Utf8 | Isomer state |
| energy_MeV | Float64 | Projectile energy in MeV |
| energy_err_MeV | Float64 | Energy uncertainty (nullable) |
| xs_mb | Float64 | Cross-section in millibarn |
| xs_err_mb | Float64 | Cross-section uncertainty (nullable) |
| author | Utf8 | First author |
| year | Int32 | Publication year |

**Stopping powers** (`stopping/{source}.parquet` — one file per source):

| Column | Type | Description |
|--------|------|-------------|
| source | Utf8 | `PSTAR`, `ASTAR`, `ESTAR`, `dSTAR`, `tSTAR`, `catima_C12`, … |
| target_Z | Int32 | Target element Z (1–92) |
| energy_MeV | Float64 | Projectile kinetic energy (MeV, total) |
| dedx | Float64 | Mass stopping power (MeV cm²/g) |

Files: `PSTAR.parquet`, `ASTAR.parquet`, `ESTAR.parquet`, `dSTAR.parquet`, `tSTAR.parquet`, and the federated CaTiMA heavy-ion shards `catima_<Sym><A>.parquet` — one per beam isotope (≈399 files spanning every Z=1–99 isotope in the inventory, each covering all 92 targets), build-generated by `build_heavy_ions.py`. Each shard carries both `energy_MeV_u` (per-nucleon) and `energy_MeV` (total) so it is queryable as an isotope table or as a NIST-style `source`. The former single 92×92 monolith was removed in favour of the shards (#252).

PSTAR and ASTAR are reproducible from NIST CGI via `uv run python -m nucl_parquet.build_stopping`. ESTAR is rebuilt from the strata-data `em/estar_long.parquet` upstream via `uv run python -m nucl_parquet.build_em_stopping` (which also emits the richer `stopping/em/electron_stopping.parquet` and `stopping/em/density_effect_params.parquet`; see "Electron-matter tables" below). `dSTAR` and `tSTAR` are velocity-scaled from PSTAR (exact, same Z=1, just relabel energy axis) by `build_light_ions.py`. ³He has no published NIST table; α and ³He earlier shipped via wrong-by-4× files that have been removed (#137) — α now uses NIST ASTAR (ICRU-49 reference), ³He uses CaTiMA. NIST PSTAR/ASTAR only publish 25 elemental targets; for elements outside that list (e.g. Tc, Pm, Po, Rn), `elemental_dedx()` falls back to CaTiMA (Bethe-Bloch), which covers all Z=1–92.

**Heavy-ion total reaction cross-sections** (`hi-xs/xs/{proj}_{target}.parquet`):

Tripathi (1997) semi-empirical parameterization — total reaction cross-sections for all 12 projectiles against all 92 target elements.  Energy stored as total MeV for the projectile; 1–1000 MeV/u range, 60 log-spaced points.

| Column | Type | Description |
|--------|------|-------------|
| target_Z | Int32 | Target atomic number (1–92) |
| target_A | Int32 | Target mass number (most-abundant stable isotope) |
| energy_MeV | Float64 | Total projectile kinetic energy (MeV) |
| xs_mb | Float64 | Total reaction cross-section (mb) |

**Heavy-ion production cross-sections** (`hi-xs-prod/xs/{proj}_{target}.parquet`):

Geant4 11.3.2 `FTFP_INCLXX` physics list (INCL++ cascade + ABLA07 de-excitation) — per-isotope fragment production cross-sections σ(Zf,Af,E) in mb, normalized to Tripathi (1997) σ_R.  Covers C-12, O-16, Ne-20, Si-28, Ar-40, Fe-56 projectiles against all 92 target elements, ~60 log-spaced energy points from 1–1000 MeV/u.

| Column | Type | Description |
|--------|------|-------------|
| proj_Z | Int32 | Projectile atomic number |
| proj_A | Int32 | Projectile mass number |
| target_Z | Int32 | Target atomic number (1–92) |
| target_A | Int32 | Target mass number (most-abundant stable isotope) |
| residual_Z | Int32 | Fragment atomic number |
| residual_A | Int32 | Fragment mass number |
| energy_MeV | Float64 | Mean actual reaction vertex energy (MeV total) |
| xs_mb | Float64 | Production cross-section (mb) |

**Heavy-ion stopping powers** (`stopping/catima.parquet`):

Full 92×92 matrix — all projectile elements Z=1–92 against all target elements Z=1–92, computed with [CatIMA](https://github.com/hrosiak/catima). Energy stored in MeV/u; isotope-independent (divide total MeV by A to look up).

| Column | Type | Description |
|--------|------|-------------|
| proj_Z | Int32 | Projectile atomic number (1–92) |
| target_Z | Int32 | Target atomic number (1–92) |
| energy_MeV_u | Float64 | Kinetic energy per nucleon (MeV/u) |
| dedx | Float64 | Mass stopping power (MeV cm²/g) |

## Nuclear structure & decay (v0.11+, Geant4-derived)

Sourced from the [strata project's HuggingFace dataset](https://huggingface.co/datasets/gerchowl/strata-data), which republishes G4ENSDFSTATE3.0 + PhotonEvaporation6.1.2 + RadioactiveDecay6.1.2 as Parquet. Replaces the v0.10.x IAEA-LiveChart pipeline; six bug classes eliminated by construction (see [ADR-0002](docs/adr/0002-g4-migration-schema.md)). Stable isotopes ship `half_life_s = +inf` (use `is_finite(half_life_s)` to test).

| View | Source | What's in it |
|---|---|---|
| `nuclides` | `meta/ensdf/nuclides.parquet` | All known states (ground + isomers) with half-life, J^π, decay modes, AME2020 mass excess, IUPAC composition |
| `ground_states` | `nuclides WHERE state = ''` | Compatibility view |
| `decay` / `decay_detailed` | `meta/decay{,_detailed}.parquet` | Decay branches per `(Z, A, state)`. `decay_detailed` adds `parent_ex_kev`, `daughter_ex_kev`, `q_value_kev`, `forbiddenness`, and **per-shell EC fractions** (`KshellEC`/`LshellEC`/`MshellEC`/`NshellEC`) |
| `radiation` | `meta/ensdf/radiation/{Symbol}.parquet` | Per-element gamma + X-ray + Auger lines, unioned by `rad_type` discriminator |
| `coincidences` | `meta/ensdf/coincidences/{Symbol}.parquet` | Gamma cascade pairs (~600k pairs, 104 element files) |
| `summing_partners` | `meta/ensdf/summing_partners/{Symbol}.parquet` | ICC-corrected summing partners for HPGe TCS corrections |
| `emissions` | `meta/ensdf/emissions/{Symbol}.parquet` | Absolute per-decay emission intensities (NuDat-equivalent, parent-keyed) |
| `beta_spectra` | `meta/ensdf/beta_spectra/{Symbol}.parquet` | Continuous beta-decay kinetic-energy spectra (dN/dE, 200 bins) |
| `levels` | `meta/ensdf/levels/{Symbol}.parquet` | Excited-state level schemes |
| `dose_constants` | `meta/dose_constants.parquet` | Dose rate constants (Sv/h per Bq at 1 m) |

## Photon-matter interaction (v0.12+, G4EMLOW8.8)

Per-process cross-sections + sampling kernels for "photon hits material" queries. Lets users break down `xcom`'s integrated µ/ρ into the dominant processes (PE / Compton / Rayleigh / pair / atomic relaxation). All views live under `data/em/`; epic [#95](https://github.com/exoma-ch/nucl-parquet/issues/95).

| View | Process | Use case |
|---|---|---|
| `photon_pe` | Photoelectric per-shell σ | "After K-shell ionization in iodine at 33 keV…" |
| `photon_pe_high_z_params` | Analytic fit coefficients | High-Z extrapolation near edges |
| `photon_pe_angular` | Photoelectron emission angle kernel | Monte Carlo angular sampling |
| `photon_compton` | Compton σ_C(E, Z) | Bound-electron Klein-Nishina total |
| `compton_scattering_function` | S(x, Z) | Differential dσ/dΩ via S(x,Z) × Klein-Nishina |
| `compton_doppler_profiles` | Per-shell f(p) | Doppler broadening of scattered energy |
| `photon_rayleigh_cdf` | Coherent-scattering CDF | Inverse-CDF angular sampling |
| `xray_form_factor` | Anomalous f1, f2 (Henke/CXRO) | Low-energy Rayleigh corrections |
| `photon_pair` | Pair + triplet σ | Per-channel breakdown above 1.022 MeV (`channel ∈ {nuclear, triplet, total}`) |
| `atomic_relaxation` | Full vacancy cascade | Radiative + Auger transitions per shell |
| `fluorescence` | Radiative subset | K_α / K_β / L_α yields per Z |

**Worked example — "what fraction of 511 keV photons in lead Compton-scatter vs photoelectric-absorb?"**

```sql
WITH pe_total AS (
    SELECT Z, energy_MeV, SUM(sigma_b) AS sigma_pe
      FROM photon_pe
     WHERE Z = 82 AND ABS(energy_MeV - 0.511) < 1e-3
     GROUP BY Z, energy_MeV
)
SELECT 'PE' AS process, sigma_pe AS sigma_b FROM pe_total
UNION ALL
SELECT 'Compton', sigma_b FROM photon_compton
 WHERE Z = 82 AND ABS(energy_MeV - 0.511) < 1e-3;
```

## Electron-matter interaction (v0.13+, G4EMLOW8.8)

Per-process electron transport cross-sections — companion to the v0.12 photon-matter views. Epic [#114](https://github.com/exoma-ch/nucl-parquet/issues/114) tracks the full electron-matter rollout (bremsstrahlung shipped; Seltzer-Berger DCS, MSC, DPWA, ESTAR migration in progress).

| View | Process | Use case |
|---|---|---|
| `electron_brem` | Bremsstrahlung total σ_brem(Z, T) | Photon-emission rate from electron transport |

## Detailed nuclear data — NUDEX (v0.14+, G4NUDEXLIB1.0)

Where `nuclides` / `radiation` / `coincidences` ship the *summary* G4 PhotonEvaporation tables, the NUDEX views below import the **fully-detailed** ENSDF source — every known nuclear level, every measured gamma transition, full BrIcc internal-conversion tables, neutron-capture primary spectra, photon strength functions, and statistical-model level density parameters. Epic [#115](https://github.com/exoma-ch/nucl-parquet/issues/115) (5/5 sub-issues complete).

| View | Rows | What's in it |
|---|---|---|
| `nudex_levels` | 158,900 | Per-level: energy, J^π, half-life, decay modes (3,331 isotopes) |
| `nudex_level_gammas` | 245,975 | Per-transition: source/dest level, γ energy, intensity, uncertainty |
| `nudex_isotopes` | 3,331 | Per-isotope summary: level/gamma counts, mass excess, S_n |
| `capture_gammas` | 39,157 | Neutron-capture primary γ spectra (PGAA / activation analysis) |
| `capture_gammas_summary` | 982 | Per (target, daughter): reaction, S_n, multiplicity |
| `icc_factors` | 579,380 | Per-shell BrIcc factors (35 shells × 10 multipolarities × 117 Z) |
| `psf_e1` | 8,980 | **IAEA SMLO E1 (recommended modern default)** |
| `psf_gdr_lor` / `mlo` / `slo` | 145 / 178 / 180 | Experimental GDR Lorentzian / Modified-Lorentzian / Standard-Lorentzian |
| `psf_gdr_theor` | 5,986 | Goriely theoretical PSF systematics |
| `psf_photonuclear` | 1,912 | Photonuclear (γ,abs)/(γ,sn)/(γ,xn) experimental peaks |
| `level_density_bfm` | 289 | Back-shifted Fermi-gas effective parameters |
| `level_density_ctm` | 289 | Constant-temperature effective parameters |
| `level_density_params` | 3,353 | Per-nuclide T, U_cutoff, N_levels |

**Schema overlap with v0.11**: `nudex_levels` and `ensdf_levels` coexist by design. Use `ensdf_levels` for transport-aligned queries (matches G4's bundled level set); use `nudex_levels` for high-fidelity gamma spectroscopy and statistical decay calculations.

**Worked example — "predict the prompt-gamma signature of a neutron-irradiated Co-59 sample"**

```sql
SELECT energy_keV, intensity_pct
  FROM capture_gammas
 WHERE Z = 27 AND A = 60 AND variant = 'default'
 ORDER BY intensity_pct DESC
 LIMIT 10;
```

## All registered views

The complete inventory of DuckDB views registered by `nucl_parquet.connect()`. All views are declared in `catalog.json::views` — adding a new data table to the catalog makes it queryable with zero code changes across all clients (Python, TypeScript, Rust).

<!-- AUTO:views -->
| View | Path | Type |
|------|------|------|
| `abundances` | `meta/abundances.parquet` | file |
| `astar_compounds` | `stopping/compounds/ASTAR_compounds.parquet` | file |
| `atomic_relaxation` | `meta/eadl/*.parquet` | glob |
| `beta_spectra` | `meta/ensdf/beta_spectra/*.parquet` | glob |
| `capture_gammas` | `meta/capture_gammas.parquet` | file |
| `capture_gammas_summary` | `meta/capture_gammas_summary.parquet` | file |
| `catima_stopping` | `stopping/catima_*.parquet` | glob |
| `channels` | `endfb-8.0/channels/*.parquet` | glob |
| `coincidences` | `meta/ensdf/coincidences/*.parquet` | glob |
| `compound_compositions` | `meta/compound_compositions.parquet` | file |
| `compton_doppler_profiles` | `em/compton_doppler_profiles.parquet` | file |
| `compton_scattering_function` | `em/compton_scattering_function.parquet` | file |
| `decay` | `meta/decay.parquet` | file |
| `decay_detailed` | `meta/decay_detailed.parquet` | file |
| `density_effect_params` | `stopping/em/density_effect_params.parquet` | file |
| `dose_constants` | `meta/dose_constants.parquet` | file |
| `eedl_electron_xs` | `meta/eedl/*.parquet` | glob |
| `electron_brem` | `em/electron_brem.parquet` | file |
| `electron_brem_sb_dcs` | `em/electron_brem_sb_dcs.parquet` | file |
| `electron_stopping` | `stopping/em/electron_stopping.parquet` | file |
| `elements` | `meta/elements.parquet` | file |
| `emissions` | `meta/ensdf/emissions/*.parquet` | glob |
| `ensdf_gammas` | `meta/ensdf/gammas/*.parquet` | glob |
| `ensdf_levels` | `meta/ensdf/levels/*.parquet` | glob |
| `epdl_anomalous` | `meta/epdl97/anomalous/*.parquet` | glob |
| `epdl_form_factors` | `meta/epdl97/form_factors/*.parquet` | glob |
| `epdl_photon_xs` | `meta/epdl97/photon_xs/*.parquet` | glob |
| `epdl_scattering_fn` | `meta/epdl97/scattering_fn/*.parquet` | glob |
| `epdl_subshell_pe` | `meta/epdl97/subshell_pe/*.parquet` | glob |
| `ground_states` | `meta/ensdf/ground_states.parquet` | file |
| `icc_factors` | `meta/icc_factors.parquet` | file |
| `kerma` | `meta/kerma/*.parquet` | glob |
| `level_density_bfm` | `meta/level_density_bfm.parquet` | file |
| `level_density_ctm` | `meta/level_density_ctm.parquet` | file |
| `level_density_params` | `meta/level_density_params.parquet` | file |
| `neutron_total` | `meta/neutron_total/*.parquet` | glob |
| `nuclides` | `meta/ensdf/nuclides.parquet` | file |
| `nudex_general_stat` | `meta/nudex_general_stat.parquet` | file |
| `nudex_isotopes` | `meta/nudex_isotopes.parquet` | file |
| `nudex_level_gammas` | `meta/nudex_level_gammas.parquet` | file |
| `nudex_levels` | `meta/nudex_levels.parquet` | file |
| `nudex_shellcor` | `meta/nudex_shellcor.parquet` | file |
| `nudex_special_inputs` | `meta/nudex_special_inputs.parquet` | file |
| `photon_compton` | `em/photon_compton.parquet` | file |
| `photon_pair` | `em/photon_pair.parquet` | file |
| `photon_pe` | `em/photon_pe.parquet` | file |
| `photon_pe_angular` | `em/photon_pe_angular.parquet` | file |
| `photon_pe_high_z_params` | `em/photon_pe_high_z_params.parquet` | file |
| `photon_pe_total` | `em/photon_pe_total.parquet` | file |
| `photon_rayleigh_cdf` | `em/photon_rayleigh_cdf.parquet` | file |
| `psf_e1` | `meta/psf_e1.parquet` | file |
| `psf_gdr_lor` | `meta/psf_gdr_lor.parquet` | file |
| `psf_gdr_mlo` | `meta/psf_gdr_mlo.parquet` | file |
| `psf_gdr_slo` | `meta/psf_gdr_slo.parquet` | file |
| `psf_gdr_theor` | `meta/psf_gdr_theor.parquet` | file |
| `psf_photonuclear` | `meta/psf_photonuclear.parquet` | file |
| `pstar_compounds` | `stopping/compounds/PSTAR_compounds.parquet` | file |
| `radiation` | `meta/ensdf/radiation/*.parquet` | glob |
| `spectrum_xs` | `meta/spectrum_xs.parquet` | file |
| `stopping` | `stopping/*.parquet` | glob |
| `summing_partners` | `meta/ensdf/summing_partners/*.parquet` | glob |
| `xcom_compounds` | `meta/xcom_compounds.parquet` | file |
| `xcom_elements` | `meta/xcom_elements.parquet` | file |
| `xray_form_factor` | `em/xray_form_factor.parquet` | file |
| `xs` | union of all XS libraries | derived |
| `ground_states` | filtered from `nuclides` | derived |
| `eadl_transitions` | alias for `atomic_relaxation` | derived |
| `fluorescence` | filtered from `atomic_relaxation` | derived |
<!-- /AUTO:views -->

## MCP servers

Three MCP servers expose nucl-parquet data to AI assistants:

| Client | Language | Registration |
|--------|----------|-------------|
| `clients/py/nucl-parquet-mcp` | Python (DuckDB) | catalog-driven |
| `clients/ts/nucl-parquet-mcp` | TypeScript (DuckDB) | catalog-driven |
| `clients/rs/nucl-parquet-mcp` | Rust (ParquetStore) | catalog-driven |

All three read `catalog.json::views` for view registration and `catalog.json::libraries` for cross-section libraries. The Rust MCP uses the `ParquetStore` from the `nucl-parquet` client crate for zero-dependency Parquet I/O.

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run unit tests (no data needed)
uv run pytest tests/test_loader.py -v

# Run full test suite (requires data)
uv run pytest tests/ -v
```

## License & data attribution

**Code & conversion:** MIT (see [`LICENSE`](LICENSE)).

**Bundled nuclear data:** third-party — MIT does **not** apply to it. Each library
keeps its own terms and required citation; see [`ATTRIBUTION.md`](ATTRIBUTION.md)
(machine-readable in [`data/licenses.toml`](data/licenses.toml)) and the
[`NOTICE`](NOTICE). The data is provided as-is — see [`DISCLAIMER.md`](DISCLAIMER.md).
