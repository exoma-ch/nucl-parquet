# Handoff — 2026-05-19 Absolute emissions (#196)

## Session summary
Implemented absolute per-decay photon emission intensities — a parent-keyed
materialized table that provides NuDat-equivalent gamma intensities.

### What changed

**Builder** (`nucl_parquet/g4/emissions.py`):
- Reads `decay_detailed.parquet`, `decay.parquet`, `nuclides.parquet`, `radiation/{Symbol}.parquet`
- Produces `emissions/{Symbol}.parquet` — filed by **parent** element symbol
- Cascade propagation: top-down level population with ICC-weighted branch fractions
- IT special case: feeding from `decay.parquet` summary + `nuclides.level_keV`
- 105 element files, ~173k rows total, builds in <2s

**Pipeline integration**:
- `build_all.py`: added step after `summing_partners`
- `loader.py`: registered `emissions` DuckDB view + `emissions()` Python helper
- `__init__.py`: exported `emissions` helper

**MCP tools** (all three servers):
- Python: `get_emissions(parent_z, parent_a, parent_state, decay_mode, energy_keV, tolerance_keV, min_intensity_pct)`
- TypeScript: same tool via DuckDB view
- Rust: same tool via local Parquet + arrow reader

**Tests**:
- Python: 11 unit tests (cascade propagation, ICC normalization, IT handling, edge cases, schema)
  + 11 @data acceptance tests (Co-60, Tc-99m, Eu-152 vs NuDat)
- TypeScript: 2 DuckDB integration tests (Co-60 absolute intensity, Eu-152 EC shell summing)
- Rust: 1 integration test (Co-60 via get_emissions tool)

### NuDat validation

| Nuclide | Gamma (keV) | Our calc | NuDat | Δ |
|---------|-------------|----------|-------|---|
| Co-60 | 1173.2 | 99.86% | 99.85% | +0.01% |
| Co-60 | 1332.5 | 99.98% | 99.98% | exact |
| Tc-99m | 140.5 | 89.04% | 89.06% | -0.02% |
| Eu-152 | 121.8 | 28.49% | 28.58% | -0.3% |
| Eu-152 | 344.3 (β⁻→Gd) | 26.58% | 26.50% | +0.3% |

### Key design decisions
1. **New table, not modifying `radiation/`**: radiation is emitter-keyed (daughter);
   absolute intensity is parent-specific. Separate table avoids row multiplication
   and breaking coincidences/summing_partners pipeline.
2. **Per-shell EC modes preserved**: Eu-152 121.8 keV gamma has separate rows for
   KshellEC, LshellEC, MshellEC, beta+. Sum matches NuDat. More granular than NuDat.
3. **ICC-weighted normalization**: G4 intensity is photon-only; total transition
   rate = intensity × (1+ICC). This is the key insight for correct branch fractions.

### Algorithm
```
For each parent (Z, A, state):
  1. Build feeding: decay_detailed → daughter level populations; IT → isomeric level
  2. Walk cascade top-down: branch_frac = intensity×(1+ICC) / Σ(intensity×(1+ICC))
  3. Absolute photon: population × branch_frac × 1/(1+ICC) × 100
```
