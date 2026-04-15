#!/bin/bash
# =============================================================================
# VisionAI-Flywheel — Stop all services
# =============================================================================
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG=/home/ubuntu/flywheel_stop.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Stopping VisionAI-Flywheel — $(date) ==="

# cosmos-transfer
cd "$REPO_DIR/services/cosmos-transfer" && docker compose down 2>/dev/null || true
echo "✓ cosmos-transfer stopped"

# kimodo-api
cd "$REPO_DIR/services/kimodo-api" && docker compose down 2>/dev/null || true
echo "✓ kimodo-api stopped"

# render-api
cd "$REPO_DIR/services/render-api" && docker compose down 2>/dev/null || true
echo "✓ render-api stopped"

# VSS stack
cd "$REPO_DIR/video-search-and-summarization/deployments"
source "$REPO_DIR/deployments/vss/env.rtxpro6000bw" 2>/dev/null || true
docker compose \
  -f compose.yml \
  -f "$REPO_DIR/deployments/vss/docker-compose.override.yml" \
  --profile bp_developer_base_2d \
  down 2>/dev/null || true
echo "✓ VSS stack stopped"

echo "=== Stack stopped — $(date) ==="
