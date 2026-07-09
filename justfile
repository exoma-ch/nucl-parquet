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
