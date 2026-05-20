# Handoff — 2026-05-20 SSoT refactoring (#206)

## Session summary

Implementing the `data → client → MCP` SSoT refactoring for the Rust stack.

### Completed

**Phase 1: ParquetStore in client (#210)**
- New `clients/rs/nucl-parquet/src/store.rs` (~200 lines)
  - `ParquetStore::new(data_dir)` — generic cached Parquet→JSON reader
  - `ParquetStore::load(rel_path)` → `Arc<Vec<Value>>` (cached)
  - `ParquetStore::load_filtered(rel_path, filters)` → filtered rows
  - `ParquetStore::schema(rel_path)` → column names + types
  - `Filter::Eq`, `Filter::Near`, `Filter::Gte`
  - `parse_parquet_file()`, `column_value_to_json()` — moved from MCP
- Made `z_to_symbol()` public in `meta.rs`
- Added `serde_json`, `bytes` as runtime deps in client Cargo.toml
- Exported `ParquetStore`, `Filter`, `z_to_symbol` from lib.rs
- 5 integration tests pass (load, filter, schema, per-element, cache)

**Earlier in session:**
- #195 merged (summing_partners)
- #197 merged (absolute emissions — all 8 rad_types)
- #198 merged (COINCIDENCE_SQL state filter)
- #201 auto-merged (compound_dedx + DoseConstant source)
- #204 created (decay half-life override)
- #205 created (raw table accessors)
- data-2026.5.2 released
- #193, #196, #50 closed

### In progress

**Phase 2: Refactor Rust MCP (#207)**
- Replace 13 tool handlers' `load_parquet_rows` → `store.load_filtered()`
- Remove MCP's `load_parquet_rows`, `parse_parquet_bytes`, `column_value_to_json`, `Cache`, `Z_TO_SYMBOL`
- Remove `parquet`, `arrow`, `bytes` deps from MCP Cargo.toml
- Branch: `feat/rs-parquet-store`

### Key files
- `clients/rs/nucl-parquet/src/store.rs` — new ParquetStore (done)
- `clients/rs/nucl-parquet-mcp/src/main.rs` — MCP refactoring (in progress)
- Plan: `.claude/plans/snappy-snacking-cookie.md`

### Remaining sub-issues (#206 epic)
- #208 TS MCP: extract registerViews to client
- #209 Python MCP: fix bypasses
- #211 Z-to-symbol dedup
- #212 CI: data release triggers tests
