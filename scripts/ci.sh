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
group "rust (fmt + clippy + test)"
# One workspace (#307) — a single lockfile and one resolution, so the two
# crates cannot disagree about a shared dependency's version.
cargo fmt --manifest-path clients/rs/Cargo.toml --all --check
cargo clippy --manifest-path clients/rs/Cargo.toml --workspace --all-targets -- -D warnings
cargo test --manifest-path clients/rs/Cargo.toml --workspace
ok "rust clean"
endgroup

# ---------------------------------------------------------------------------
group "typescript (tsc + vitest)"
(cd clients/ts/nucl-parquet && npm ci && npx tsc --noEmit && npx vitest run)
ok "typescript passed"
endgroup

# ---------------------------------------------------------------------------
group "go (vet + test)"
(cd clients/go/nucl-parquet && go vet ./... && go test ./...)
ok "go passed"
endgroup

echo ""
echo "✅ All CI checks passed."
