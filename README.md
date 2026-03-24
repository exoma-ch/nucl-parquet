# nucl-parquet

Nuclear data as Parquet files — cross-sections, stopping powers, decay data, and isotopic abundances from all major evaluated libraries. Queryable with DuckDB, Polars, Pandas, or any Arrow-compatible tool.

## Installation

```bash
pip install nucl-parquet
```

The pip package is a thin loader (~50 KB). Data files are either cloned from the git repo or downloaded from GitHub Releases:

```python
import nucl_parquet

# Download data to ~/.nucl-parquet/ (first time only)
nucl_parquet.download()
```

Or clone the repo directly for the full dataset:

```bash
git clone https://github.com/exoma-ch/nucl-parquet.git
export NUCL_PARQUET_DATA=/path/to/nucl-parquet/data
```

## Usage

```python
import nucl_parquet

db = nucl_parquet.connect()

# Cross-section query
db.sql("SELECT * FROM tendl_2024 WHERE target_A=63 AND residual_Z=30")

# Compare all libraries
db.sql("SELECT library, energy_MeV, xs_mb FROM xs WHERE target_A=63 AND residual_Z=30")

# Decay chain
db.sql(nucl_parquet.DECAY_CHAIN_SQL, params={"parent_z": 92, "parent_a": 238})

# Stopping power — light ions (NIST PSTAR/ASTAR/ESTAR)
nucl_parquet.elemental_dedx(db, "p", 29, 10.0)     # protons in Cu at 10 MeV
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

| Library | Projectiles | Source |
|---------|------------|--------|
| [TENDL-2024](https://tendl.web.psi.ch/tendl_2024/tendl2024.html) | n, p, d, t, ³He, α | IAEA/PSI |
| [TENDL-2025](https://tendl.web.psi.ch/) | n, p, d, t, ³He, α | PSI |
| [ENDF/B-VIII.1](https://www.nndc.bnl.gov/endf-b8.1/) | n, p, d, t, ³He, α | NNDC/BNL |
| [JEFF-4.0](https://www.oecd-nea.org/dbdata/jeff/) | n, p | NEA |
| [JENDL-5](https://wwwndc.jaea.go.jp/jendl/j5/j5.html) | n, p, d, α | JAEA |
| [CENDL-3.2](http://www.nuclear.csdb.cn/) | n | CIAE |
| [BROND-3.1](https://vant.ippe.ru/) | n | IPPE |
| [FENDL-3.2](https://www-nds.iaea.org/fendl/) | n | IAEA |
| [EAF-2010](https://fispact.ukaea.uk/) | n | CCFE |
| [IRDFF-II](https://www-nds.iaea.org/IRDFF/) | n | IAEA |
| [IAEA-Medical](https://www-nds.iaea.org/medical/) | p, d | IAEA |
| [EXFOR](https://www-nds.iaea.org/exfor/) | n, p, d, t, ³He, α | IAEA NDS (experimental) |
| [HI-XS (Tripathi 1997)](https://doi.org/10.1016/S0168-583X(96)00331-X) | p, ⁴He, ¹²C, ¹⁶O, ²⁰Ne, ²⁸Si, ⁴⁰Ar, ⁴⁰Ca, ⁵⁶Fe, ⁵⁸Ni, ¹³²Xe, ²⁰⁸Pb | semi-empirical (Tripathi 1997) |
| HI-XS Production (Geant4 INCL++/ABLA07) | ¹²C, ¹⁶O, ²⁰Ne, ²⁸Si, ⁴⁰Ar, ⁵⁶Fe | Geant4 11.3.2 Monte Carlo |

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
| source | Utf8 | `PSTAR`, `ASTAR`, `ESTAR`, `dSTAR`, `tSTAR`, `He3STAR`, `catima_C12`, … |
| target_Z | Int32 | Target element Z (1–92) |
| energy_MeV | Float64 | Projectile kinetic energy (MeV, total) |
| dedx | Float64 | Mass stopping power (MeV cm²/g) |

Files: `PSTAR.parquet`, `ASTAR.parquet`, `ESTAR.parquet`, `dSTAR.parquet`, `tSTAR.parquet`, `He3STAR.parquet`, and `catima_{beam}.parquet` for C12/O16/Ne20/Si28/Ar40/Fe56. The full 92×92 CaTiMA matrix (MeV/u units) lives separately at `stopping/catima/catima.parquet`.

`dSTAR`, `tSTAR`, and `He3STAR` are velocity-scaled from PSTAR/ASTAR — exact for electronic stopping since Z_proj and velocity fully determine dE/dx. For elements not in the NIST table (e.g. Ra, Rn, Ac, Po, Fr, At, Tc, Pm), `elemental_dedx()` automatically falls back to CaTiMA (Bethe-Bloch), which covers all Z=1–92.

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

## Development

```bash
# Install dev dependencies
uv sync --dev

# Run unit tests (no data needed)
uv run pytest tests/test_loader.py -v

# Run full test suite (requires data)
uv run pytest tests/ -v
```

## License

MIT
