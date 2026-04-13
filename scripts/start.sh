#!/bin/bash
# =============================================================================
# VisionAI-Flywheel — Start all services
# Called by systemd on every boot, or manually: bash scripts/start.sh
# =============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG=/home/ubuntu/flywheel_start.log
exec > >(tee -a "$LOG") 2>&1

echo "=== Starting VisionAI-Flywheel — $(date) ==="

# Load secrets if not already in env
if [ -f /home/ubuntu/.flywheel.env ]; then
  set -a; source /home/ubuntu/.flywheel.env; set +a
fi

# ── NGC login (needed after reboot) ─────────────────────────────────────────
echo "$NGC_CLI_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' --password-stdin 2>/dev/null && echo "✓ NGC login"

# ── VSS stack ────────────────────────────────────────────────────────────────
cd "$REPO_DIR/video-search-and-summarization/deployments"
set -a; source "$REPO_DIR/deployments/vss/env.rtxpro6000bw"; set +a

docker compose \
  -f compose.yml \
  -f "$REPO_DIR/deployments/vss/docker-compose.override.yml" \
  --profile bp_developer_base_2d \
  up -d
echo "✓ VSS stack started"

# ── render-api ───────────────────────────────────────────────────────────────
cd "$REPO_DIR/services/render-api"
docker compose up -d
echo "✓ render-api started"

# ── Health check loop ─────────────────────────────────────────────────────────
echo "Waiting for VSS agent to be ready..."
for i in $(seq 1 60); do
  STATUS=$(curl -s http://localhost:8000/health | python3 -c "import json,sys; print(json.load(sys.stdin).get('value',{}).get('isAlive','false'))" 2>/dev/null || echo "false")
  if [ "$STATUS" = "True" ] || [ "$STATUS" = "true" ]; then
    echo "✓ VSS agent healthy"
    break
  fi
  sleep 10
  echo "  still waiting... ($((i*10))s)"
done

echo "=== Stack ready — $(date) ==="
echo "   VSS agent  : http://localhost:8000"
echo "   render-api : http://localhost:9000"
echo "   Studio UI  : http://localhost:3000"
