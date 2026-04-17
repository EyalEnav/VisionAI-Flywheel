#!/bin/bash
# =============================================================================
# VisionAI-Flywheel — Container Watchdog
# Monitors critical containers and restarts them if they go down.
# Runs as a systemd service (vlm-watchdog.service).
# Logs via systemd (journalctl -u vlm-watchdog) + /home/ubuntu/flywheel_watchdog.log
# =============================================================================

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
INTERVAL=30   # seconds between health checks

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# Load secrets
[ -f /home/ubuntu/.flywheel.env ] && { set -a; source /home/ubuntu/.flywheel.env; set +a; }

log "=== Watchdog started (PID $$) ==="

# ── Container → compose dir mapping ──────────────────────────────────────────
declare -A COMPOSE_DIR
declare -A COMPOSE_ENV

COMPOSE_DIR[render-api]="$REPO_DIR/services/render-api"
COMPOSE_ENV[render-api]="RENDER_API_PORT=9001"

COMPOSE_DIR[kimodo-api]="$REPO_DIR/services/kimodo-api"
COMPOSE_ENV[kimodo-api]=""

COMPOSE_DIR[cosmos-transfer]="$REPO_DIR/services/cosmos-transfer"
COMPOSE_ENV[cosmos-transfer]="TRANSFER_GPU_DEVICE=1 HF_TOKEN=${HUGGINGFACE_TOKEN:-}"

# VSS containers — if any are down, restart the whole VSS compose
VSS_CONTAINERS=(vss-agent centralizedb-dev vss-proxy metropolis-vss-ui mdx-redis)

is_running() {
  local status
  status=$(docker inspect --format '{{.State.Status}}' "$1" 2>/dev/null)
  [ "$status" = "running" ]
}

restart_compose() {
  local name="$1"
  local dir="${COMPOSE_DIR[$name]:-}"
  local extra_env="${COMPOSE_ENV[$name]:-}"
  if [ -z "$dir" ]; then
    log "  docker start $name"
    docker start "$name" 2>&1
    return
  fi
  log "  compose up -d [$name] in $dir"
  HF_CACHE="${HF_CACHE:-/opt/dlami/nvme/hf_cache}" \
  KIMODO_OUTPUT_DIR=/home/ubuntu/kimodo_output \
  RENDER_OUTPUT_DIR=/home/ubuntu/render_output \
  HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}" \
    env $extra_env \
    docker compose -f "$dir/docker-compose.yml" up -d 2>&1
}

restart_vss() {
  log "Restarting VSS stack..."
  cd "$REPO_DIR/video-search-and-summarization/deployments" || return 1
  set -a
  source "$REPO_DIR/deployments/vss/env.rtxpro6000bw" 2>/dev/null || true
  source "$REPO_DIR/video-search-and-summarization/deployments/developer-workflow/dev-profile-base/generated.env" 2>/dev/null || true
  set +a
  docker compose \
    -f compose.yml \
    -f "$REPO_DIR/deployments/vss/docker-compose.override.yml" \
    --profile bp_developer_base_2d \
    up -d 2>&1
}

# ── Main watchdog loop ────────────────────────────────────────────────────────
while true; do
  # 1. VLM pipeline containers
  for name in render-api kimodo-api cosmos-transfer; do
    if ! is_running "$name"; then
      log "WARNING: $name is DOWN — restarting..."
      restart_compose "$name"
      log "  restart attempted for $name"
    fi
  done

  # 2. VSS stack — if any core container is down, restart the whole compose
  vss_down=0
  for name in "${VSS_CONTAINERS[@]}"; do
    if ! is_running "$name"; then
      log "WARNING: VSS container $name is DOWN"
      vss_down=1
      break
    fi
  done
  if [ "$vss_down" -eq 1 ]; then
    log "Attempting VSS stack recovery..."
    restart_vss
    log "  VSS restart attempted"
  fi

  sleep "$INTERVAL"
done
