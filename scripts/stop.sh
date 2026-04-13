#!/bin/bash
# =============================================================================
# VisionAI-Flywheel — Stop all services
# =============================================================================
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Stopping VisionAI-Flywheel — $(date) ==="

cd "$REPO_DIR/services/render-api"
docker compose down 2>/dev/null || true

cd "$REPO_DIR/video-search-and-summarization/deployments"
source "$REPO_DIR/deployments/vss/env.rtxpro6000bw" 2>/dev/null || true
docker compose \
  -f compose.yml \
  -f "$REPO_DIR/deployments/vss/docker-compose.override.yml" \
  --profile bp_developer_base_2d \
  down 2>/dev/null || true

echo "=== Stack stopped ==="
