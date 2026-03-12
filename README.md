# nucl-parquet

Nuclear data as Parquet files — cross-sections, stopping powers, decay data, and isotopic abundances. Compressed with zstd, split for lazy loading.

Designed as a standalone, format-agnostic data layer consumable by any language or runtime (Python, TypeScript/WASM, etc.).

## Structure

```
├── catalog.json               # registry of available libraries
├── meta/
│   ├── abundances.parquet     # natural isotopic abundances
│   ├── decay.parquet          # decay modes, half-lives, branching
│   └── elements.parquet       # Z ↔ symbol mapping
├── stopping/
│   └── stopping.parquet       # PSTAR/ASTAR/ICRU73/MSTAR
└── tendl-2024/
    ├── manifest.json          # library metadata + file inventory
    └── xs/
        ├── p_Cu.parquet       # proton + Copper
        ├── d_Fe.parquet       # deuteron + Iron
        └── ...                # one file per projectile+element
```

## Usage

```bash
# Verify data against catalog + schema:
uv run python build.py --verify

# Import all data from an external parquet directory:
uv run python build.py --import-all /path/to/parquet/

# Import cross-sections for a specific library:
uv run python build.py --import-xs /path/to/xs/ --library tendl-2024
```

## Adding a new library

1. Produce Parquet files matching the schema below (from any source)
2. Run `python build.py --import-xs /path/to/xs/ --library <name>`
3. Add entry to `catalog.json`
4. Tag release

## Parquet schema

**Cross-sections** (`xs/*.parquet`):
| Column | Type | Description |
|--------|------|-------------|
| target_A | Int32 | Target mass number |
| residual_Z | Int32 | Product atomic number |
| residual_A | Int32 | Product mass number |
| state | Utf8 | Isomer state: "", "g", "m" |
| energy_MeV | Float64 | Projectile energy |
| xs_mb | Float64 | Cross-section in millibarn |

**Stopping** (`stopping/stopping.parquet`):
| Column | Type | Description |
|--------|------|-------------|
| source | Utf8 | PSTAR, ASTAR, ICRU73, MSTAR |
| target_Z | Int32 | Target element |
| energy_MeV | Float64 | Projectile energy |
| dedx | Float64 | Stopping power (MeV cm²/g) |

## Data sources

- **TENDL-2024**: TALYS Evaluated Nuclear Data Library (IAEA/PSI)
- **PSTAR/ASTAR**: NIST stopping power tables via libdEdx
- **ICRU73**: ICRU Report 73 light-ion stopping powers
- **Decay data**: ENDF-6 format via isotopia
- **Abundances**: ENDF-6 natural isotopic abundances
