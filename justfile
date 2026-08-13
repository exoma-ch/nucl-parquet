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

# Verify a published data release's minisign signature (#289).
# Defaults to data/catalog.json::data_version; pass a CalVer to pick another.
verify-release version="":
    ./scripts/verify_data_release.sh {{ version }}

# Verify an EXTRACTED data tree against the signed content manifest (#296).
# Use when the tarball signature cannot apply: a CDR gateway repacked the
# archive, or only part of the data was carried across a diode.
verify-extracted dir version="" *flags="":
    ./scripts/verify_data_release.sh --extracted {{ dir }} {{ if version != "" { "--version " + version } else { "" } }} {{ flags }}

# Generate the data-signing keypair and install it as repo secrets.
# Run ONCE, on a trusted personal machine — never in CI. See
# docs/security/data-signing.md before rotating an existing key.
gen-signing-key:
    ./scripts/gen_signing_key.sh
