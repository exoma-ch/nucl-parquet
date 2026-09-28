#!/usr/bin/env bash
# Reproducible CI runner — the single source of truth for "what CI checks".
#
# Runs identically in two places:
#   - Locally:  nix develop -c ./scripts/ci.sh    (or `just ci`, or inside `direnv`)
#   - In CI:    .github/workflows/ci.yml invokes this in the same nix devShell
#
# Because the devShell (flake.nix) pins uv/rust/node/go/ruff AND the native libs
# (libstdc++/libz) that generic-linux wheels dynamically link against, the run is
# byte-for-byte reproducible and works on NixOS out of the box — no LD_LIBRARY_PATH
# hunting. See flake.nix.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# GitHub Actions log folding; harmless (no-op prefix) in a local terminal.
group() { echo "::group::$*"; }
endgroup() { echo "::endgroup::"; }
ok() { echo "  ✓ $*"; }

# ---------------------------------------------------------------------------
group "file hygiene (trailing whitespace / EOF / merge markers / large files)"
# Text files tracked by git, excluding vendored/generated + binary data.
mapfile -t textfiles < <(git ls-files | grep -vE '\.(parquet|lock|png|jpg|ico|woff2?)$|^data/')
hygiene_fail=0
for f in "${textfiles[@]}"; do
  [ -f "$f" ] || continue
  if grep -nE ' +$' "$f" >/dev/null; then echo "  trailing whitespace: $f"; hygiene_fail=1; fi
  if grep -nE '^(<<<<<<<|>>>>>>>|=======$)' "$f" >/dev/null; then echo "  merge marker: $f"; hygiene_fail=1; fi
  if [ -n "$(tail -c1 "$f" 2>/dev/null)" ]; then echo "  missing final newline: $f"; hygiene_fail=1; fi
done
# Guard against accidentally committing a >5 MB file outside data/.
while IFS= read -r f; do
  [ -f "$f" ] || continue
  sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f")
  if [ "$sz" -gt 5242880 ]; then echo "  file >5MB outside data/: $f ($sz bytes)"; hygiene_fail=1; fi
done < <(git ls-files | grep -vE '^data/')
[ "$hygiene_fail" -eq 0 ] || { echo "file hygiene FAILED"; exit 1; }
ok "hygiene clean"
endgroup

# ---------------------------------------------------------------------------
group "ruff (lint + format)"
ruff check nucl_parquet scripts tests
ruff format --check nucl_parquet scripts tests
ok "ruff clean"
endgroup

# ---------------------------------------------------------------------------
group "python tests"
uv sync --dev
# The whole directory, not a list of files. A list is an allowlist: a test file
# that is not named on it never runs, silently, with a green tick. 40 of 54 test
# files were in exactly that state — including tests/test_readme_drift.py, which
# CLAUDE.md promises will fail the suite if you skip the regeneration, and which
# did not run here at all. It bit two PRs on one day: #341's stale-tree gate and
# #340's ingest tests were both dead on arrival until their authors noticed by
# hand. `pyproject.toml` already sets `testpaths = ["tests"]`; explicit paths on
# the command line overrode it. See #355.
#
# `-m "not network"`, and deliberately nothing else. The `data` marker is not a
# CI filter — it exists so the suite degrades gracefully when the data tree is
# *absent*, and tests/conftest.py already skips those tests in that case. The
# data tree is committed, so deselecting them here only suppressed checks that
# had something to say: data/exfor-channels/manifest.json disagreed with its own
# parquets from #334 until #358, and nothing in CI could see it.
#
# Why each suite matters now lives in that suite's module docstring, where the
# next reader is already looking, rather than in a shell comment they will never
# open. tests/test_ci_runs_everything.py keeps the allowlist from coming back.
uv run pytest tests/ -m "not network" -v
ok "python tests passed"
endgroup

# ---------------------------------------------------------------------------
group "python MCP server (clients/py/nucl-parquet-mcp)"
# Its own distribution with its own dependencies (mcp 2.x is not in the root
# environment), so it gets its own isolated run against this checkout's
# nucl-parquet. It had no line here, and while it did not, the published server
# failed every data tool. tests/test_ci_runs_everything.py now requires one run
# per clients/py package.
(cd clients/py/nucl-parquet-mcp && uv run --isolated --no-project --with-editable ../../.. --with-editable '.[dev]' pytest -q)
ok "python MCP server tests passed"
endgroup

# ---------------------------------------------------------------------------
group "rust (fmt + clippy + test)"
# One workspace (#307) — a single lockfile and one resolution, so the two
# crates cannot disagree about a shared dependency's version.
cargo fmt --manifest-path clients/rs/Cargo.toml --all --check
cargo clippy --manifest-path clients/rs/Cargo.toml --workspace --all-targets -- -D warnings
# `fetch` is the download path consumers enable, and nothing above compiles it:
# it is off by default. A breaking reqwest/zstd bump would otherwise reach
# crates.io unbuilt. (`fetch-native-tls` needs a system OpenSSL the devShell
# does not provide, so it stays out of this line.)
cargo clippy --manifest-path clients/rs/Cargo.toml -p nucl-parquet --features fetch --all-targets -- -D warnings
cargo test --manifest-path clients/rs/Cargo.toml --workspace
# `--include-ignored` runs the 57 tests marked `#[ignore = "requires
# nucl-parquet data files"]` in `meta.rs` (plus 7 in `tests/golden.rs`) that
# assert the crate reads the shipped data correctly — Cu-64 β⁻, I-131 dose,
# Co-60 β⁻→Ni-60 cascades, identify_gamma(1173.2), the JSON goldens. Before
# #357-b none of them ran here, so four of them (`decay_cu64_beta`,
# `dose_i131_positive`, `dose_from_bytes_matches_open`,
# `radiation_emissions_ni60_has_co60_decay_gammas`) sat red on main
# unnoticed — every one was defaulting `state` to `""` (a spelling #380
# removed) and returning empty. Data is on disk in CI, so there is no
# reason not to check.
NUCL_PARQUET_DATA="$PWD/data" cargo test \
  --manifest-path clients/rs/Cargo.toml --workspace -- --include-ignored
ok "rust clean"
endgroup

# ---------------------------------------------------------------------------
group "typescript (tsc + vitest + build + attw)"
# Both TS packages. `clients/ts/nucl-parquet-mcp` was absent from this line, so
# its 25 tests and its typecheck never ran here — the same allowlist shape #355
# removed from the Python section, one directory up. It is one of the three MCP
# servers, and #348's whole point is that a claim nothing checks is weaker than
# one that can be checked; shipping the data-release fix into a package CI does
# not build would have reproduced that inside the fix.
#
# `npm run build` is what release.yml runs before `npm publish`, and it is not
# the same check as `tsc --noEmit`. #302 moved core to TypeScript 7 while tsup's
# `dts: true` still drove the compiler API TS 7 does not ship; tsc kept passing,
# and the break surfaced only at publish time, after the approval gate:
# @nucl-parquet/core 0.17.0 never reached npm. Declarations now come from
# `tsc --emitDeclarationOnly`, and this line is what proves the build builds.
for pkg in nucl-parquet nucl-parquet-mcp; do
  (cd "clients/ts/${pkg}" && npm ci && npx tsc --noEmit && npx vitest run && npm run build)
done
# A build that succeeds can still ship types consumers cannot resolve: 0.17.1
# gave `require` ESM declarations ("Masquerading as ESM" under node16).
# arethetypeswrong packs the library as npm would and checks every resolution
# mode. Core only: the MCP server is a CLI with no importable API.
(cd clients/ts/nucl-parquet && npx attw --pack .)
ok "typescript passed"
endgroup

# ---------------------------------------------------------------------------
group "go (vet + test)"
(cd clients/go/nucl-parquet && go vet ./... && go test ./...)
ok "go passed"
endgroup

echo ""
echo "✅ All CI checks passed."
