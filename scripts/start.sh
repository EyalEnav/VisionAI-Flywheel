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

# ── Ensure Docker network exists ─────────────────────────────────────────────
docker network create vlm-net 2>/dev/null || true

# ── Shared output dirs ───────────────────────────────────────────────────────
mkdir -p /home/ubuntu/kimodo_output /home/ubuntu/render_output

# ── NGC login (needed after reboot) ─────────────────────────────────────────
echo "$NGC_CLI_API_KEY" | docker login nvcr.io \
  --username '$oauthtoken' --password-stdin 2>/dev/null && echo "✓ NGC login"

# ── 1. VSS stack (GPU0) ──────────────────────────────────────────────────────
echo "Starting VSS stack..."
cd "$REPO_DIR/video-search-and-summarization/deployments"
set -a; source "$REPO_DIR/deployments/vss/env.rtxpro6000bw"; set +a

docker compose \
  -f compose.yml \
  -f "$REPO_DIR/deployments/vss/docker-compose.override.yml" \
  --profile bp_developer_base_2d \
  up -d
echo "✓ VSS stack started"

# ── 2. render-api (GPU0, port 9001) — ghcr.io/eyalenav/render-api:latest ────
echo "Starting render-api..."
cd "$REPO_DIR/services/render-api"
RENDER_API_PORT=9001 \
HF_CACHE="${HF_CACHE:-/opt/dlami/nvme/hf_cache}" \
KIMODO_OUTPUT_DIR=/home/ubuntu/kimodo_output \
RENDER_OUTPUT_DIR=/home/ubuntu/render_output \
  docker compose up -d
echo "✓ render-api started on :9001"

# ── 3. kimodo-api (GPU1, port 9551) — ghcr.io/eyalenav/kimodo-api:latest ────
echo "Starting kimodo-api..."
cd "$REPO_DIR/services/kimodo-api"
HF_CACHE="${HF_CACHE:-/opt/dlami/nvme/hf_cache}" \
KIMODO_OUTPUT_DIR=/home/ubuntu/kimodo_output \
  docker compose up -d
echo "✓ kimodo-api started on :9551"

# ── 4. cosmos-transfer (GPU1, port 8080) ─────────────────────────────────────
echo "Starting cosmos-transfer..."
cd "$REPO_DIR/services/cosmos-transfer"
TRANSFER_GPU_DEVICE=1 \
HF_CACHE="${HF_CACHE:-/opt/dlami/nvme/hf_cache}" \
HF_TOKEN="${HUGGINGFACE_TOKEN:-}" \
  docker compose up -d
echo "✓ cosmos-transfer started on :8080"

# ── 5. vLLM (optional — GPU0 shared, port 8090) ──────────────────────────────
# Set VLLM_ENABLED=1 to start automatically; otherwise start manually with:
#   cd $REPO_DIR/services/vllm && VLLM_MODEL=Qwen/Qwen2.5-VL-2B-Instruct docker compose up -d
if [ "${VLLM_ENABLED:-0}" = "1" ]; then
  echo "Starting vLLM..."
  cd "$REPO_DIR/services/vllm"
  VLLM_GPU_DEVICE="${VLLM_GPU_DEVICE:-0}" \
  VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-VL-2B-Instruct}" \
  VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.30}" \
  HF_CACHE="${HF_CACHE:-/opt/dlami/nvme/hf_cache}" \
  HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-}" \
    docker compose up -d
  echo "✓ vLLM started on :8090 (model: ${VLLM_MODEL:-Qwen/Qwen2.5-VL-2B-Instruct})"
else
  echo "ℹ vLLM not auto-started (set VLLM_ENABLED=1 to enable)"
fi

# ── Health check: VSS ────────────────────────────────────────────────────────
echo "Waiting for VSS agent to be ready (can take 3-5 min)..."
for i in $(seq 1 60); do
  STATUS=$(curl -s http://localhost:8000/health | \
    python3 -c "import json,sys; print(json.load(sys.stdin).get('value',{}).get('isAlive','false'))" \
    2>/dev/null || echo "false")
  if [ "$STATUS" = "True" ] || [ "$STATUS" = "true" ]; then
    echo "✓ VSS agent healthy"
    break
  fi
  sleep 10
  echo "  still waiting... ($((i*10))s)"
done

echo ""
echo "=== Stack ready — $(date) ==="
echo "   VSS agent    : http://localhost:8000"
echo "   render-api   : http://localhost:9001"
echo "   kimodo-api   : http://localhost:9551"
echo "   cosmos-api   : http://localhost:8080"
echo "   Studio UI    : http://$(curl -s ifconfig.me 2>/dev/null || echo 'localhost'):9000"
