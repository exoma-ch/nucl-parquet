# Handoff — 2026-05-15 TCS summing_partners session

## Session summary
Implemented Sub-C (#177) — materialized `summing_partners` Parquet table with
ICC-corrected joint emission intensities for HPGe TCS corrections.

### What changed

**Builder** (`nucl_parquet/g4/summing_partners.py`):
- Reads `coincidences/{Symbol}.parquet`, produces `summing_partners/{Symbol}.parquet`
- γ-γ pairs: canonicalized (E1 ≤ E2) to dedup symmetric pairs
- X-ray/Auger ⊗ γ pairs: ICC correction only on gamma side
- NULL ICC → 0.0 (no internal conversion)
- Two derived columns: `icc_correction_factor`, `pure_emission_joint_intensity`
- 104 element files, ~9.3M rows total

**Pipeline integration**:
- `build_all.py`: added step after `mixed_coincidences`
- `loader.py`: registered `summing_partners` view via `_register_glob`
- `loader.py`: added `summing_partners()` Python helper function

**MCP tools** (all three servers):
- Python: `get_summing_partners(z, a, primary_energy_keV, tolerance_keV, emission1_rad_type)`
- TypeScript: same tool via DuckDB view
- Rust: same tool via local Parquet + arrow reader

**Rust MCP fix**: `column_value_to_json` now handles dictionary-encoded strings
via `arrow::compute::cast` fallback (was silently dropping dictionary-typed
string columns, causing filter mismatches on summing_partners data).

**Tests**:
- Python: 7 unit tests (ICC math, NULL coalescing, canonicalization, mixed pairs, schema)
  + 6 acceptance @data tests (Co-60, Eu-152, Tc-99m)
- TypeScript: 2 DuckDB integration tests (Co-60 pair, Eu-152 344 keV partners)
- Rust: 1 integration test (Co-60 via get_summing_partners tool)

### Acceptance criteria verified
- Eu-152 344 keV (Gd-152 daughter): 71 γ-γ partners (≥10 required)
- Co-60 1173/1333 keV: icc_correction_factor ≈ 0.9997 (α < 0.001 at high energy)
- Tc-99m 140 keV: min correction ~0.06 (highly converted M1+E2 transition)

### Design decisions
1. **Materialized Parquet, not compute-on-read**: consumers stay lean (SSoT pattern)
2. **One table for γ-γ and X-ray ⊗ γ**: atomic de-excitation is always prompt
   (femtoseconds), no timing edge case for HPGe
3. **Canonical ordering**: γ-γ pairs enforce E1 ≤ E2 to eliminate symmetric duplicates
