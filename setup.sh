#!/bin/bash
# =============================================================================
# VisionAI-Flywheel — One-Click Setup Script
# Deploys: VSS 3.1 + Kimodo + Cosmos-Transfer2.5 + render-api + kimodo-api
#
# Usage:
#   export NGC_CLI_API_KEY="nvapi-..."
#   export HUGGINGFACE_TOKEN="hf_..."
#   bash setup.sh
#
# Idempotent — safe to re-run on an existing machine.
# =============================================================================
set -euo pipefail
export GIT_TERMINAL_PROMPT=0

LOG=/home/ubuntu/setup.log
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo " VisionAI-Flywheel Setup — $(date)"
echo "=========================================="

# ── 0. Required secrets ──────────────────────────────────────────────────────
: "${NGC_CLI_API_KEY:?ERROR: NGC_CLI_API_KEY not set. Export it before running.}"
: "${HUGGINGFACE_TOKEN:?ERROR: HUGGINGFACE_TOKEN not set. Export it before running.}"
export HF_TOKEN="$HUGGINGFACE_TOKEN"

# Persist to .bashrc so interactive sessions and services pick them up
for var in NGC_CLI_API_KEY HUGGINGFACE_TOKEN HF_TOKEN; do
  grep -q "export $var=" ~/.bashrc 2>/dev/null || \
    echo "export $var=\"${!var}\"" >> ~/.bashrc
done

# ── 1. Storage — prefer large NVMe over root disk ────────────────────────────
NVME=/opt/dlami/nvme
mountpoint -q "$NVME" || { NVME=/home/ubuntu/data; mkdir -p "$NVME"; }
echo "✓ Storage root: $NVME"

DOCKER_DATA="$NVME/docker"
HF_CACHE="$NVME/hf_cache"
mkdir -p "$DOCKER_DATA" "$HF_CACHE"

# Configure Docker to use NVMe (skip if already done)
if ! sudo docker info 2>/dev/null | grep -q "$DOCKER_DATA"; then
  sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "data-root": "$DOCKER_DATA",
  "runtimes": {
    "nvidia": {
      "path": "/usr/bin/nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
EOF
  sudo systemctl restart docker
  sleep 5
fi
echo "✓ Docker data-root: $DOCKER_DATA"

# ── 2. NGC Docker login ──────────────────────────────────────────────────────
echo "$NGC_CLI_API_KEY" | sudo docker login nvcr.io \
  --username '$oauthtoken' --password-stdin
echo "✓ NGC Docker login OK"

# ── 3. VisionAI-Flywheel repo ───────────────────────────────────────────────
REPO_DIR=/home/ubuntu/vlm-pipeline
if [ -d "$REPO_DIR/.git" ]; then
  echo "Updating vlm-pipeline..."
  cd "$REPO_DIR" && git pull --ff-only || true
else
  echo "Cloning VisionAI-Flywheel..."
  git clone https://github.com/EyalEnav/VisionAI-Flywheel.git "$REPO_DIR"
fi
echo "✓ vlm-pipeline: $REPO_DIR"

# ── 4. VSS 3.1 — official NVIDIA-AI-Blueprints repo ────────────────────────
# Source: https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization
# Deployment: scripts/dev-profile.sh (same as Brev Launchable)
VSS_DIR=/home/ubuntu/video-search-and-summarization
VSS_REPO="https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization.git"
VSS_TAG="3.1.0"

if [ -d "$VSS_DIR/.git" ]; then
  echo "VSS exists — syncing to $VSS_TAG..."
  cd "$VSS_DIR"
  git fetch --all --prune
  git checkout "$VSS_TAG" 2>/dev/null || true
else
  echo "Cloning VSS $VSS_TAG..."
  git clone --branch "$VSS_TAG" --single-branch "$VSS_REPO" "$VSS_DIR"
fi
echo "✓ VSS 3.1: $VSS_DIR"

# ── 5. Kimodo — official nv-tlabs repo (public) ──────────────────────────────
# Source: https://github.com/nv-tlabs/kimodo
# Docker install: https://research.nvidia.com/labs/sil/projects/kimodo/docs/getting_started/installation_docker.html
KIMODO_DIR=/home/ubuntu/kimodo
KIMODO_REPO="https://github.com/nv-tlabs/kimodo.git"

if [ -d "$KIMODO_DIR/.git" ]; then
  echo "Kimodo exists — updating..."
  cd "$KIMODO_DIR" && git pull --ff-only || true
else
  echo "Cloning Kimodo..."
  git clone --depth=1 "$KIMODO_REPO" "$KIMODO_DIR"
fi

# Kimodo Docker requires kimodo-viser submodule for the demo UI
VISER_DIR="$KIMODO_DIR/kimodo-viser"
if [ ! -d "$VISER_DIR/.git" ]; then
  echo "Cloning kimodo-viser..."
  git clone --depth=1 https://github.com/nv-tlabs/kimodo-viser.git "$VISER_DIR"
fi

# Write HF token so Kimodo Docker can pull Llama text encoder
mkdir -p ~/.cache/huggingface
echo "$HUGGINGFACE_TOKEN" > ~/.cache/huggingface/token
echo "✓ Kimodo: $KIMODO_DIR"

# ── 6. Cosmos-Transfer2.5 ───────────────────────────────────────────────────
COSMOS_DIR=/home/ubuntu/cosmos-transfer2.5
if [ -d "$COSMOS_DIR/.git" ]; then
  echo "Cosmos-Transfer exists — updating..."
  cd "$COSMOS_DIR" && git pull --ff-only || true
else
  echo "Cloning Cosmos-Transfer2..."
  git clone --depth=1 --branch main https://github.com/nvidia-cosmos/cosmos-transfer2.5.git "$COSMOS_DIR"
fi
echo "✓ Cosmos-Transfer2.5: $COSMOS_DIR"

# ── 7. Symlinks into vlm-pipeline ───────────────────────────────────────────
ln -sfn "$VSS_DIR"    "$REPO_DIR/video-search-and-summarization"
ln -sfn "$KIMODO_DIR" "$REPO_DIR/kimodo"
ln -sfn "$COSMOS_DIR" "$REPO_DIR/cosmos-transfer2.5"
echo "✓ Symlinks updated"

# ── 8. Shared output dirs ────────────────────────────────────────────────────
mkdir -p /home/ubuntu/kimodo_output /home/ubuntu/render_output
echo "✓ Output dirs ready"

# ── 9. Docker network ────────────────────────────────────────────────────────
sudo docker network create vlm-net 2>/dev/null || echo "✓ vlm-net exists"

# ── 10. VSS 3.1 stack — via official dev-profile.sh ─────────────────────────
HOST_IP=$(ip route get 1.1.1.1 | awk '/src/ {for(i=1;i<=NF;i++) if($i=="src") print $(i+1)}')
EXTERNAL_IP=$(curl -s --max-time 10 ifconfig.me 2>/dev/null || echo "$HOST_IP")

echo "Starting VSS 3.1 (profile=base, hardware=RTXPRO6000BW)..."
cd "$VSS_DIR/scripts"
sudo -E bash dev-profile.sh up \
  --profile base \
  --hardware-profile RTXPRO6000BW \
  --host-ip "$HOST_IP" \
  --external-ip "$EXTERNAL_IP" \
  2>&1 | tee /home/ubuntu/vss_start.log
echo "✓ VSS stack started"

# ── 11. Kimodo — Docker build (official method) ──────────────────────────────
echo "Building Kimodo Docker image..."
cd "$KIMODO_DIR"
# Expose HF token as build-time secret via env (Docker reads from host ~/.cache)
sudo -E docker compose build 2>&1 | tee /home/ubuntu/kimodo_build.log
echo "✓ Kimodo Docker built"

# Start Kimodo text-encoder service (downloads Llama model on first run ~5min)
echo "Starting Kimodo text-encoder..."
sudo -E docker compose up -d text-encoder 2>&1
echo "✓ Kimodo text-encoder started"

# ── 12. Build render-api ─────────────────────────────────────────────────────
echo "Building render-api..."
cd "$REPO_DIR/services/render-api"
sudo docker build -t render-api:local . 2>&1 | tee /home/ubuntu/render_build.log
echo "✓ render-api built"

# ── 13. Build kimodo-api (our wrapper around Kimodo) ─────────────────────────
echo "Building kimodo-api..."
cd "$REPO_DIR"
sudo DOCKER_BUILDKIT=1 docker build \
  --build-arg HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
  -f services/kimodo-api/Dockerfile \
  -t kimodo-api:local \
  . 2>&1 | tee /home/ubuntu/kimodo_api_build.log
echo "✓ kimodo-api built"

# ── 14. Build cosmos-transfer ────────────────────────────────────────────────
echo "Building cosmos-transfer..."
cd "$REPO_DIR/services/cosmos-transfer"
bash build.sh 2>&1 | tee /home/ubuntu/cosmos_build.log
echo "✓ cosmos-transfer built"

# ── 15. Start render-api + kimodo-api + cosmos ───────────────────────────────
echo "Starting render/kimodo/cosmos services..."
cd "$REPO_DIR"
sudo -E docker compose \
  -f services/render-api/docker-compose.yml \
  -f services/kimodo-api/docker-compose.yml \
  -f services/cosmos-transfer/docker-compose.yml \
  up -d
echo "✓ All services started"

# ── 16. env file + systemd for auto-start on reboot ─────────────────────────
cat > /home/ubuntu/.flywheel.env <<EOF
NGC_CLI_API_KEY=$NGC_CLI_API_KEY
HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN
HF_TOKEN=$HUGGINGFACE_TOKEN
HF_CACHE=$HF_CACHE
EOF
chmod 600 /home/ubuntu/.flywheel.env

sudo tee /etc/systemd/system/vlm-flywheel.service > /dev/null <<'UNIT'
[Unit]
Description=VisionAI-Flywheel stack
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
User=ubuntu
WorkingDirectory=/home/ubuntu/vlm-pipeline
EnvironmentFile=/home/ubuntu/.flywheel.env
ExecStart=/home/ubuntu/vlm-pipeline/scripts/start.sh
ExecStop=/home/ubuntu/vlm-pipeline/scripts/stop.sh
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable vlm-flywheel.service
echo "✓ systemd service enabled"

echo ""
echo "=========================================="
echo " ✅ Setup complete — $(date)"
echo "   VSS UI       : http://$EXTERNAL_IP:7777"
echo "   VSS Agent    : http://$EXTERNAL_IP:8000"
echo "   render-api   : http://localhost:9001"
echo "   kimodo-api   : http://localhost:9551"
echo "   cosmos-api   : http://localhost:8080"
echo "   Kimodo demo  : http://localhost:8080 (docker compose up demo)"
echo "=========================================="
