#!/usr/bin/env bash
# Build TALYS container, generate HI cross-sections, ingest into repo.
#
# Prerequisites:
#   - talys2.0.tar in this directory (free download from tendl.web.psi.ch)
#   - podman installed
#   - uv available (for ingest step)
#
# Usage:
#   cd containers/talys
#   ./run.sh [--proj c12,o16] [--zmax 92]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
HI_XS_DIR="$REPO_ROOT/hi-xs-tmp"
IMAGE="nucl-talys"
PROJ="${1:-c12,o16,ne20,si28,ar40,fe56}"
ZMAX="${2:-92}"

# --- Sanity checks ---
if [[ ! -f "$SCRIPT_DIR/talys2.0.tar" ]]; then
    echo "ERROR: talys2.0.tar not found in containers/talys/"
    echo "  Download from: https://tendl.web.psi.ch/tendl_2025/talys.html"
    exit 1
fi

# --- Build image ---
echo "==> Building $IMAGE..."
podman build -t "$IMAGE" "$SCRIPT_DIR"

# --- Run container ---
mkdir -p "$HI_XS_DIR"
echo "==> Running TALYS for projectiles: $PROJ (target Z=1..$ZMAX)"
podman run --rm \
    -v "$HI_XS_DIR:/data/hi-xs:z" \
    "$IMAGE" \
    --out /data/hi-xs \
    --proj "$PROJ" \
    --zmax "$ZMAX"

# --- Ingest into repo ---
echo "==> Ingesting output into repo..."
cd "$REPO_ROOT"
uv run python3 containers/talys/ingest.py \
    --src "$HI_XS_DIR" \
    --data-dir "$REPO_ROOT"

echo "==> Done. Review hi-xs/ and commit."
