# nucl-parquet task shortcuts. Run `just <recipe>` inside the nix devShell
# (direnv loads it automatically; otherwise `nix develop`).

# Run the full CI suite exactly as GitHub Actions does.
ci:
    ./scripts/ci.sh

# Fast, network-free lint (ruff) via `nix flake check`.
check:
    nix flake check

# Enter the pinned dev environment.
dev:
    nix develop

# Auto-fix formatting (ruff) before committing.
fmt:
    ruff check --fix nucl_parquet scripts tests
    ruff format nucl_parquet scripts tests

# Regenerate NOTICE, ATTRIBUTION tables, data/**/LICENSE.txt and SPDX headers
# from data/licenses.toml (the provenance manifest).
notices:
    python3 scripts/build_notices.py --write

# Verify licence provenance coverage + generated-artifact drift (what CI runs).
notices-check:
    python3 scripts/build_notices.py
