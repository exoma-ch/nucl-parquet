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
# test_loader: loader/auto-wiring. test_data_release: gates data_version + data_sha256
# against the on-disk parquets (the "PR that changes data IS the data-release PR"
# guard) — must run in CI so a data re-release can't ship an inconsistent hash.
# `-m "not data and not network"` also runs the pure builder/thinning unit tests
# while skipping the ones that need the full data tree or network.
# test_release_config: gates the cross-crate version invariant that broke #281 —
# nucl-parquet-mcp declares its sibling's version for crates.io, and nothing
# enforced that the two move together.
# test_stsv: gates the Swiss StSV Annex 3 ingest (#294) — upper-bound handling,
# the H-3 chemical-form collision, footnote provenance, and the HTTP-200-HTML
# trap in the Fedlex fetch.
# test_data_signing: gates the minisign release-signing path (#289) — that the
# workflow signs unconditionally, that the signature actually verifies, and that
# tampering/replay/stripping are all rejected. Needs no data and no network.
# test_repo_layout: gates where data may live (#341) — no tracked parquet outside
# data/, and the ingest scripts' --output defaults. This list is an allowlist, so
# a gate that is not named here does not run in CI at all; #341's tests were
# written to stop a 623-file stale tree coming back and would have been silent.
# Needs only a git checkout — no data tree, no network.
uv run pytest tests/test_loader.py tests/test_data_release.py tests/test_neutron_njoy.py \
  tests/test_data_signing.py \
  tests/test_stsv.py \
  tests/test_release_config.py \
  tests/test_repo_layout.py \
  -m "not data and not network" -v
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
