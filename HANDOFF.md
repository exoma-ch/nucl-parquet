# #121 — ESTAR + density-effect from strata (HANDOFF)

**Branch:** `feat/em-estar-from-strata` (worktree at `~/worktrees/nucl-parquet-em-estar`)
**Issue:** https://github.com/exoma-ch/nucl-parquet/issues/121
**Epic:** #114 (v0.13 electron-matter)
**Status:** in flight (compaction-safe handoff written 2026-05-13)

## Goal

Migrate ESTAR stopping-power source-of-truth from the 2024 NIST scrape to
gerchowl/strata-data, extending coverage to compounds and exposing Sternheimer
density-effect parameters as a queryable table.

## Decision made

**Replace+expand, don't ship alongside.** Three output files:
1. `data/stopping/ESTAR.parquet` — rebuilt from strata, legacy schema
   `{source, target_Z, energy_MeV, dedx}`, elements-only, drop-in replacement
   for existing consumers. dedx = total_sp_mev_cm2_g.
2. `data/stopping/electron_stopping.parquet` — NEW. Rich schema
   `{name, is_element, target_Z, energy_MeV, collision_sp, radiative_sp, total_sp, density_effect_delta}`,
   includes ~50 compounds and the collision/radiative split.
3. `data/stopping/density_effect_params.parquet` — NEW. Sternheimer per
   material `{name, I_eV, C, X0, X1, a, k, delta0, density_gcm3, state, Zeff, nElements}`.

This keeps every existing `WHERE source='ESTAR'` query working while giving
new consumers a richer table.

## Source data (HF gerchowl/strata-data, em/ subdir)

- `em/estar_basic.parquet` — 22,599 rows, energy 0.01-1000 MeV, has csda_range + radiation_yield
- `em/estar_long.parquet` — 27,137 rows, energy 0.001-... MeV (use this one — wider grid)
- `em/density_effect.parquet` — 278 rows, Sternheimer params per material

Downloaded locally to `/tmp/strata-em/em/` (one-off fetch via `hf download`).

## Implementation plan

### Step 1 — extend fetcher
- `scripts/fetch_strata_nuclear.py`: add em/{estar_basic, estar_long, density_effect}.parquet
  to the FILES list. Existing revision pin (catalog.json::strata-data-nuclear) covers both.
- Update catalog `strata-data-nuclear` name/description to note em data is also covered.

### Step 2 — build script
- `nucl_parquet/build_em_stopping.py`:
  - Reads `data/g4_raw/strata-nuclear/estar_long.parquet`
  - For elements (is_element=True), z>=1: rebuild `data/stopping/ESTAR.parquet`
    with legacy schema. `target_Z=z`, `dedx=total_sp_mev_cm2_g`.
  - Write full strata schema to `data/stopping/electron_stopping.parquet`,
    keeping elements AND compounds. For compounds, target_Z=NULL.
  - Read `data/g4_raw/strata-nuclear/density_effect.parquet`, copy to
    `data/stopping/density_effect_params.parquet` (rename columns to project conventions
    if needed).

### Step 3 — loader.py
- Register `electron_stopping` view (from electron_stopping.parquet).
- Register `density_effect_params` view.
- No change to existing `stopping` view glob — ESTAR.parquet is still in stopping/.

### Step 4 — catalog.json
- Add `shared.electron_stopping` entry with schema + coverage notes.
- Add `shared.density_effect_params` entry.
- Bump data_version to `2026.5.2`.
- Recompute data_sha256.

### Step 5 — tests
- `tests/test_em_stopping.py`:
  - schema check on all three output files
  - element coverage: 92 elements present in ESTAR.parquet and electron_stopping.parquet
  - compound coverage: known compounds (G4_WATER, G4_AIR, ...) in electron_stopping.parquet
  - sanity anchors: NIST tabulated values for e- in Cu @ 1 MeV, water @ 10 MeV
    (compare against external known values, ±1%)
  - density-effect params: water has I_eV=78.0 (per NIST), known a/k/X0/X1

### Step 6 — PR + review
- Open PR vs main with single commit
- Spawn code-reviewer subagent
- Address blockers + close #121

## Files touched (planned)

- `scripts/fetch_strata_nuclear.py` (extend FILES)
- `nucl_parquet/build_em_stopping.py` (new)
- `nucl_parquet/loader.py` (register views)
- `data/catalog.json` (entries + version bump + sha)
- `data/stopping/ESTAR.parquet` (rebuilt from strata)
- `data/stopping/electron_stopping.parquet` (new)
- `data/stopping/density_effect_params.parquet` (new)
- `tests/test_em_stopping.py` (new)

## Out of scope (for this iteration)

- Updating `build_stopping.py` NIST fetcher (the old path is still kept as fallback for PSTAR/ASTAR; ESTAR was the only one we're swapping).
- Compound-σ-per-process (#113) — separate followup; uses density-effect params from this PR.
- Seltzer-Berger DCS (#118) — deferred per scope-trim decision earlier.

## Notes for resuming after compaction

If interrupted: pick up at whichever step in /Implementation plan/ is unfinished.
The downloaded source files at `/tmp/strata-em/em/` should still be there; if
not, re-run `hf download gerchowl/strata-data --repo-type=dataset --include 'em/*.parquet' --local-dir /tmp/strata-em`.
Working tree is in `~/worktrees/nucl-parquet-em-estar` on `feat/em-estar-from-strata`.
