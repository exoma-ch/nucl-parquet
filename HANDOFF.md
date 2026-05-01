# G4 Migration Handoff (epic #66)

Live document tracking progress on the v0.10.x → v0.11.0 migration from IAEA-fetcher pipeline to Geant4-derived data via strata's HF dataset.

**Integration branch**: `feat/g4-data-migration`. Sub-PRs target this branch; squash-merge to `main` releases v0.11.0.

## Status snapshot (latest)

### ✅ Done

| Phase | Issue | PR | Commits | What landed |
|---|---|---|---|---|
| A | #67 | (direct) | adf6044, f16a678, be5b4f3 | ADR-0002: preserve schema, transform on import, dual-carry, `+inf` for stable sentinel |
| B | #68 | (direct) | f841cab, 94b2a04 | HF fetcher (`scripts/fetch_strata_nuclear.py`), 12 tests, catalog pin SHA `9a74e823…`, `huggingface-hub` dev dep |
| Aux | n/a | (direct) | 0abef28 | AME2020 + IUPAC fetchers + parquet outputs in `data/auxiliary/` |
| Loader fix | (n/a) | (direct) | 8b905e5 | catalog `path` regression introduced by #68; affects #69/#70/#71 sub-PRs until rebased |
| C | #69 | #81 ✅ merged 2026-05-01 | ensdfstate → nuclides + ground_states; AME/IUPAC LEFT JOIN; review fixes (DRY half-life expr, parity dual-carry) |
| D | #70 | #82 ✅ merged 2026-05-01 | photon_evap_levels → 117 per-element parquet files; `n.parquet` → `N.parquet` case-FS edge fix |
| E | #71 | #83 ✅ merged 2026-05-01 | radioactive_decay → decay + decay_detailed; per-shell EC preserved (NOT collapsed); IT-on-ground canary extended to detailed table |

### 🟡 In flight (loop tick 3, 2026-05-01 ~10:15 UTC)

- **#72 / PR #85 (gammas → radiation)** — rebased onto integration branch (commit 2994751); CI re-running. Merge on next tick if green. Review fixes already applied (#85 review verdict: ship-with-changes; addressed in 2053584).
- **#73 (coincidences)** — agent `a862fe9a…` active; `data/meta/ensdf/coincidences/*.parquet` regenerating, no commits yet on its branch.
- **#74 (X-ray + Auger synthesis)** — agent `ab13f63d…` 3 commits in: cfd2cfc (design memo), 3f376fe (synthesis), c0b04d1 (cascade propagation + K-edge gating). Agent still active.


### ⏳ Blocked / pending

| Phase | Issue | Blocked by | Notes |
|---|---|---|---|
| F | #72 (gammas → radiation) | #69 (#81 ✅), #70 (#82 🟡) | Can start now reading strata's parquet directly. State-assignment uses #81's nuclides.parquet output. |
| G | #73 (coincidences) | #72 | Self-join on photon_evap_gammas cascades. |
| H | #74 (X-ray + Auger) | #71 (#83 🟡) | G4EMLOW × per-shell EC fractions from decay_detailed. Heaviest physics task. |
| I | #75 (validation diff harness) | #69, #70, #71, #72, #73, #74 | Canonical-isotope diff against current data. |
| J | #76 (cleanup + v0.11.0) | #75 | Delete IAEA-rescue code; ADR amendment; CHANGELOG; release. |

### 📋 Deferred follow-ups

- **#80** — per-file SHA-256 in catalog.json (defensive hardening, non-blocking)
- **#84** — IT-on-ground renormalization audit (No-253 et al. — upstream G4 anomaly, not v0.11 regression)

## Loop-mode operating procedure

Running under `/loop 30m` (cron `df4cbf43`, fires :13 and :43 every hour). Each fire reads this file, makes the next chunk of progress, updates the table, commits.

### Per-iteration discipline

1. `git fetch origin --quiet`
2. Check sub-PRs that may now be mergeable: `gh pr view <N> --json mergeable,mergeStateStatus,statusCheckRollup`
3. Merge any CLEAN+MERGEABLE PRs (squash-delete-branch).
4. If a PR conflicts: rebase onto `feat/g4-data-migration`, resolve `nucl_parquet/g4/__init__.py` (canonical content lives in this branch — keep that version), `git push --force-with-lease`.
5. Identify next /land target from the dependency graph above.
6. Spawn /land agent in background (always remind it about the catalog `path` fix already on the branch, and the case-FS / per-shell EC / sentinel discipline).
7. Update this HANDOFF.md table with progress.
8. Commit + push the handoff update.

### When all converters merged

- Spawn /land #75 (validation diff harness) — depends on every converter
- Run the full pipeline end-to-end: fetcher → builders → assert canonical isotopes
- Document any unexpected diffs in the diff harness output
- Then /land #76 (cleanup): delete `_rescue_it_orphan_isomers`, `BuildIntegrityError`, ceilings, threshold logic. Update ADR-0001 with v0.11 postscript. CHANGELOG migration notes.
- Open final PR `feat/g4-data-migration` → `main` for v0.11.0 release-please.

## Critical pitfalls (don't relearn)

1. **G4 stable sentinel** `mean_life_ns = -1` (or `half_life_s = -1` in PhotonEvaporation/RadioactiveDecay) MUST map to `+inf`, not NULL. Boundary-test before commit.
2. **`nucl_parquet/g4/__init__.py`** is shared across the parallel converter PRs — when rebasing, prefer the integration-branch version (it lists all modules that have landed).
3. **Catalog `path` requirement**: `loader.py` and `tests/test_coverage.py` iterate `libraries.*` expecting `path`. The `strata-data-nuclear` entry doesn't have `path` (it's a HF provenance record). Fix `8b905e5` already applied — sub-branches inherit on rebase.
4. **Case-insensitive filesystems** (macOS APFS): `n.parquet` and `N.parquet` collide. Use `git mv` with explicit casing.
5. **Dual-carry contract** (ADR-0002): every nuclides/levels output ships `spin_x2` AND `parity` (Int8 nullable, all-NULL where source doesn't expose) AND `jp` string. v1.0 cleanup is column-drop, not column-reintroduction.
6. **Per-shell EC preserved**, not collapsed under `EC` — #74 needs it, downstream consumers querying `decay_mode == "EC"` get nothing post-v0.11. Documented in PR #83 description.

## Branches alive

- `feat/g4-data-migration` (integration, push target for sub-PRs)
- `fix/issue-72-photon-evap-gammas` (next /land)
- `fix/issue-74-xray-auger-synthesis` (next /land, parallel with #72)
