#!/usr/bin/env bash
# startup.sh — bring up all VisionAI-Flywheel services (assumes already installed)
# Usage: bash scripts/startup.sh [--stop]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
source "${REPO_ROOT}/.env" 2>/dev/null || true
set -a; source "${REPO_ROOT}/deployments/vss/env.rtxpro6000bw"; set +a

ACTION=${1:-start}

VSS_DIR="/home/ubuntu/video-search-and-summarization/deployments"
VSS_COMPOSE="docker compose -f compose.yml -f ${REPO_ROOT}/deployments/vss/docker-compose.override.yml --env-file ${REPO_ROOT}/deployments/vss/env.rtxpro6000bw"
KIMODO_DIR="/home/ubuntu/kimodo"
RENDER_DIR="${REPO_ROOT}/services/render-api"

log() { echo -e "\033[1;36m[$1]\033[0m $2"; }

if [[ "$ACTION" == "--stop" ]]; then
  log "STOP" "Render API..."
  cd "$RENDER_DIR" && docker compose down 2>&1 | tail -2

  log "STOP" "Kimodo text-encoder..."
  cd "$KIMODO_DIR" && docker compose down 2>&1 | tail -2

  log "STOP" "VSS stack..."
  cd "$VSS_DIR" && $VSS_COMPOSE down 2>&1 | tail -2

  echo "All services stopped."
  exit 0
fi

# ── Fix DNS (Brev sometimes loses it on restart) ──────────────────────────
if ! ping -c1 -W2 nvcr.io >/dev/null 2>&1; then
  echo "nameserver 8.8.8.8" | tee /etc/resolv.conf 2>/dev/null || true > /dev/null
fi

# ── Render API ────────────────────────────────────────────────────────────
log "1/3" "Render API..."
cd "$RENDER_DIR" && docker compose up -d 2>&1 | tail -2

# ── Kimodo text-encoder ────────────────────────────────────────────────────
log "2/3" "Kimodo text-encoder..."
cd "$KIMODO_DIR" && docker compose up -d 2>&1 | tail -2

# ── VSS stack ─────────────────────────────────────────────────────────────
log "3/3" "VSS Blueprint stack (profile: ${BP_PROFILE})..."
cd "$VSS_DIR" && $VSS_COMPOSE up -d 2>&1 | tail -4

echo ""
echo "════════════════════════════════════"
echo " Services started!"
echo "  Render API  → http://localhost:9000"
echo "  VSS Agent   → http://localhost:8000"
echo "  VST         → http://localhost:77770"
echo "  Kimodo      → http://localhost:9550"
echo "════════════════════════════════════"
