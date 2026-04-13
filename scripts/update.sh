#!/bin/bash
# =============================================================================
# VisionAI-Flywheel — Pull latest code + restart
# Usage: bash scripts/update.sh
# =============================================================================
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Updating VisionAI-Flywheel — $(date) ==="

cd "$REPO_DIR"
git pull --ff-only
git submodule update --init --recursive

echo "Restarting services..."
bash "$REPO_DIR/scripts/stop.sh"
bash "$REPO_DIR/scripts/start.sh"

echo "=== Update complete ==="
