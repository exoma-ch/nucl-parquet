# nucl-parquet

Nuclear data as Parquet files — cross-sections, stopping powers, decay data, and isotopic abundances from all major evaluated libraries. Compressed with zstd, split for lazy loading.

## Why Parquet instead of ENDF-6?

The [ENDF-6 format](https://www.nndc.bnl.gov/endfdocs/ENDF-102/) dates from the 1960s. It was designed for Fortran on punch cards: 80-character fixed-width records, implicit column positions, and a cryptic MF/MT numbering system. This made sense when computers had kilobytes of memory and nuclear data was distributed on magnetic tape.

**Today it creates friction at every step:**

| | ENDF-6 | Parquet |
|---|---|---|
| **Format** | Fixed-width Fortran text, 80-char cards | Columnar binary, self-describing schema |
| **Parsers needed** | Specialized (NJOY, PREPRO, FUDGE, `endf` pkg) | Any language — Python, R, Julia, Rust, JS, SQL |
| **Random access** | Sequential parse from start | Predicate pushdown, skip irrelevant row groups |
| **Compression** | None (or gzip'd text) | zstd columnar compression (5-10x smaller) |
| **Cross-library comparison** | Convert each library separately first | `SELECT * FROM '*/xs/p_Cu.parquet'` |
| **Browser/WASM** | Not feasible | Works natively (DuckDB-WASM, Pyodide) |
| **Tooling** | Nuclear-specific codes only | DuckDB, Polars, Pandas, Arrow, dplyr, ... |

**Size comparison** for the same data:

| Library | ENDF-6 (zipped) | Parquet (zstd) | Reduction |
|---------|-----------------|----------------|-----------|
| TENDL-2025 neutron | ~800 MB (2850 zip files) | 25 MB | **32x** |
| ENDF/B-VIII.1 (all) | ~120 MB | 4.3 MB | **28x** |
| JENDL-5 (all) | ~200 MB | 8.6 MB | **23x** |

The ENDF-6 format forces researchers to spend time on data plumbing instead of physics. Every new tool, language, or platform requires reimplementing the same 1960s parser. Parquet eliminates that — it's the interchange format the field should have had decades ago.

### Query without conversion

```sql
-- Compare all libraries for ⁶³Cu(p,n)⁶³Zn in one query
SELECT * FROM read_parquet('*/xs/p_Cu.parquet', filename=true)
WHERE target_A = 63 AND residual_Z = 30 AND residual_A = 63
ORDER BY energy_MeV
```

```python
# Or with Polars / Pandas
import polars as pl
df = pl.read_parquet("tendl-2024/xs/p_Cu.parquet")
reaction = df.filter(
    (pl.col("target_A") == 63) & (pl.col("residual_Z") == 30)
)
```

No NJOY. No PREPRO. No Fortran compiler. Just data.

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

## Structure

```
nucl-parquet/
├── catalog.json               # registry of all libraries
├── meta/
│   ├── abundances.parquet     # natural isotopic abundances
│   ├── decay.parquet          # decay modes, half-lives, branching
│   └── elements.parquet       # Z ↔ symbol mapping
├── stopping/
│   └── stopping.parquet       # PSTAR/ASTAR/ICRU73/MSTAR
├── tendl-2024/xs/             # one directory per library
│   ├── p_Cu.parquet           # proton + Copper
│   ├── n_Fe.parquet           # neutron + Iron
│   └── ...
├── endfb-8.1/xs/
├── jeff-4.0/xs/
├── jendl-5/xs/
├── ...
└── exfor/                     # experimental data (different schema)
    ├── p_Cu.parquet
    └── ...
```

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

## Scripts

```bash
# Fetch an evaluated library from the IAEA mirror:
uv run python scripts/fetch_endf_libs.py --library endfb-8.1 --sublibrary n

# Fetch EXFOR experimental data:
uv run python scripts/fetch_exfor.py --projectile p --element Cu

# Generate SVG comparison plots:
uv run python scripts/plot_xs.py --projectile p --element Cu

# Generate Markdown catalog (like JANIS Books):
uv run python scripts/generate_catalog.py --all --plot

# Verify data integrity:
uv run python build.py --verify
```

## Data sources

All data is fetched from the [IAEA Nuclear Data Services](https://nds.iaea.org/) mirror. Evaluated libraries are parsed from ENDF-6 format. Experimental data comes from the EXFOR database via the DataExplorer API. Stopping powers from NIST PSTAR/ASTAR via libdEdx.
