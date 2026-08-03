# Handoff — 2026-08-03 Licence manifest, ETH 440.4, duplicate-tree removal

Branch: `worktree-wt-session` (worktree at `.claude/worktrees/wt-session`, based on
`origin/main` @ 9a47f6d). **Committed locally, NOT pushed.**

## What landed

One commit. Work is complete and verified — no half-finished edits.

### 1. AGPL exposure fixed (the load-bearing change)

`pycatima` was in `[project].dependencies` of an **MIT** wheel. catima is
**AGPL-3.0** (confirmed via GitHub licence API; PyPI declares no licence and the
sdist ships no LICENSE file, so nothing warned us). Every `pip install
nucl-parquet` pulled AGPL code into the user's environment while `NOTICE` claimed
catima was "NOT included".

Moved to the `build` extra + dev group. `loader.py` never imported it — only
`build_heavy_ions.py` does, lazily. Verified by installing the wheel into a clean
venv and querying catima-backed stopping with pycatima absent.

### 2. Manifest-driven licensing

`data/licenses.toml` (schema v2) is the single source of truth.
`scripts/build_notices.py --write` generates `NOTICE`, the `ATTRIBUTION.md` /
`COMPLIANCE.md` AUTO sections, 21 per-library `data/*/LICENSE.txt` sidecars, and
SPDX headers on all 74 files in `nucl_parquet/` + `scripts/`. Check mode gates CI
(`scripts/ci.sh`, `just notices-check`, pre-commit). `nucl_parquet/licensing.py`
holds the logic and is deliberately stdlib-only.

Coverage enforced: **5999 data files, each claimed by exactly one entry.**

Provenance gaps closed: `endfb-8.0`, `meta/spectrum_xs`, EPICS
(`epdl97`/`eadl`/`eedl`), NUDEX/PSF/level-density, XCOM, `kerma`,
`neutron_total`. Two misattributions corrected — `stopping/ESTAR.parquet` is
Geant4-derived (not NIST) since the strata migration, and `meta.decay` comes via
Geant4's data files (not ENSDF/NNDC alone), so the Geant4 notice applies.

### 3. ETH RSETHZ 440.4

Copyright holder is now **ETH Zürich** with creators named, per Art. 27(1)(e), in
`LICENSE`, `NOTICE`, `CITATION.cff` and every SPDX header. `pyproject.toml` gained
`license-files = ["LICENSE", "NOTICE"]` so the wheel carries them.
`COMPLIANCE.md` §3 has the Art. 27(1)(a)–(e) checklist, Art. 24/34 disclosure
analysis, Art. 27(3) CLA rule. `CONTRIBUTING.md` explains the DCO-not-CLA choice.

### 4. Duplicate trees deleted

Root-level `hi-xs-prod/` (552) and `tendl-2025/` (71) were pre-refactor orphans
of `a61afc9` (data -> data/). `tendl-2025/` was byte-identical. `hi-xs-prod/`
was **not** — same names, all 552 blobs differ, because `data/hi-xs-prod/` was
re-emitted with a better schema (adds `proj_Z`/`proj_A`; the root copy encodes
the projectile only in the filename). Verified `data/` is a complete superset —
3,238,679 rows each, 0 rows missing — before deleting.

Scheme is now single-rooted: everything at `data/<library>/<kind>/`, with
`catalog.json` holding **data-dir-relative** paths so `$NUCL_PARQUET_DATA` can
point at any copy of the tree.

## Verification (all green at commit time)

- licence gate: 5999 files covered, artifacts current
- Python 27 · Rust 67 unit + 7 golden (`--include-ignored`, real data tree) ·
  TypeScript 26 (incl. golden parity vs Python fixtures) · Go ok
- ruff check + format, CI file-hygiene clean
- Known non-issue: Rust doc-test `lib.rs:36` fails only under
  `--include-ignored` (needs the `fetch` feature); plain `cargo test` skips it.

## Next steps

1. **Record the Art. 27(1)(a) release decision** — the OSS release decision
   belongs to the Forschungsgruppenleiter:in, not the creators, and is not
   recorded anywhere. Set `group_leader` + `oss_decision_date` in
   `data/licenses.toml` `[eth]`, flip `oss_decision_recorded = true`, rerun
   `just notices`. The checklist row and sign-off table update themselves.
2. Push the branch / open a PR (nothing pushed yet).
3. Optional: `COMPLIANCE.md` sign-off table still lists ETH legal / ETH transfer
   items as pending. The record is accurate and enforced, not authoritative.
