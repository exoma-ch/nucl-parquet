# Handoff — 2026-05-21 Catalog-driven SSoT + README automation

## Session done
- #213 merged (ParquetStore + summing_partners + TENDL rename)
- #216 merged (data bump 2026.5.4), data-2026.5.4 released
- #205 merged (raw accessors), #217 filed (README automation)

## Next: catalog-driven registration (#206 epic)

### Design
Add `views` section to `catalog.json` — maps view_name → {path, type: file|glob}.
All three clients (Py/TS/Rust) read this section to auto-register views.
New data tables become queryable with zero code changes.

```json
"views": {
  "abundances": {"path": "meta/abundances.parquet", "type": "file"},
  "radiation": {"path": "meta/ensdf/radiation", "type": "glob"},
  "emissions": {"path": "meta/ensdf/emissions", "type": "glob"},
  ...35+ entries
}
```

### Files to modify
- `data/catalog.json` — add views section
- `nucl_parquet/loader.py` — replace 35 hardcoded register calls with catalog loop
- `clients/ts/nucl-parquet-mcp/src/index.ts` — same pattern
- `clients/rs/nucl-parquet-mcp/src/main.rs` — read view paths from catalog
- `scripts/build_readme.py` — NEW: auto-generate README data sections
- `tests/test_readme_drift.py` — NEW: CI validates README

### Current state
- Python loader has 35+ hardcoded `_register_glob`/`_register_parquet` calls
- TS MCP duplicates the same 35+ calls (copy-paste of Python)
- Rust MCP ignores all of this, reads Parquet directly
- README is stale (missing emissions, summing_partners, wrong TENDL name)
