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
export NUCL_PARQUET_DATA=/path/to/nucl-parquet
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

# Stopping power
nucl_parquet.elemental_dedx(db, "p", 29, 10.0)  # protons in Cu at 10 MeV
nucl_parquet.compound_dedx(db, "p", [(29, 0.5), (30, 0.5)], 10.0)
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

**Stopping powers** (`stopping/stopping.parquet`):

| Column | Type | Description |
|--------|------|-------------|
| source | Utf8 | PSTAR, ASTAR, ICRU73, MSTAR |
| target_Z | Int32 | Target element |
| energy_MeV | Float64 | Projectile energy |
| dedx | Float64 | Stopping power (MeV cm²/g) |

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
