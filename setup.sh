#!/bin/bash
# =============================================================================
# VisionAI-Flywheel — Brev Launchable Setup Script
# Runs once on first boot after the instance is provisioned.
# =============================================================================
set -euo pipefail
LOG=/home/ubuntu/setup.log
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo " VisionAI-Flywheel Setup — $(date)"
echo "=========================================="

# ── 0. Secrets ──────────────────────────────────────────────────────────────
# These must be set as Brev environment variables before launching:
#   NGC_CLI_API_KEY   — NVIDIA NGC API key  (required)
#   HUGGINGFACE_TOKEN — HuggingFace token   (required for Llama-3-8B)
: "${NGC_CLI_API_KEY:?ERROR: NGC_CLI_API_KEY env var not set}"
: "${HUGGINGFACE_TOKEN:?ERROR: HUGGINGFACE_TOKEN env var not set}"

# Persist secrets across reboots
grep -q NGC_CLI_API_KEY ~/.bashrc 2>/dev/null || {
  echo "export NGC_CLI_API_KEY=\"$NGC_CLI_API_KEY\""    >> ~/.bashrc
  echo "export HUGGINGFACE_TOKEN=\"$HUGGINGFACE_TOKEN\"" >> ~/.bashrc
}

# ── 1. Storage — use large attached NVMe ─────────────────────────────────────
NVME=/opt/dlami/nvme
if mountpoint -q "$NVME"; then
  echo "✓ NVMe mounted at $NVME"
else
  NVME=/home/ubuntu/data
  mkdir -p "$NVME"
fi

DOCKER_DATA="$NVME/docker"
HF_CACHE="$NVME/hf_cache"
mkdir -p "$DOCKER_DATA" "$HF_CACHE"

# Configure Docker to use the large disk + nvidia runtime
if ! grep -q data-root /etc/docker/daemon.json 2>/dev/null; then
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

# ── 2. Clone / update repo ───────────────────────────────────────────────────
REPO_DIR=/home/ubuntu/vlm-pipeline
if [ -d "$REPO_DIR/.git" ]; then
  echo "Updating existing repo..."
  cd "$REPO_DIR" && git pull --ff-only
else
  echo "Cloning VisionAI-Flywheel..."
  git clone https://github.com/EyalEnav/VisionAI-Flywheel.git "$REPO_DIR"
fi
cd "$REPO_DIR"
git submodule update --init --recursive
echo "✓ Repo ready at $REPO_DIR"

# ── 3. Shared directories ────────────────────────────────────────────────────
mkdir -p /home/ubuntu/kimodo_output
mkdir -p /home/ubuntu/render_output
echo "✓ Shared output dirs created"

# ── 4. Create Docker network (shared by all services) ────────────────────────
sudo docker network create vlm-net 2>/dev/null || echo "✓ vlm-net already exists"

# ── 5. NGC Docker login ──────────────────────────────────────────────────────
echo "$NGC_CLI_API_KEY" | sudo docker login nvcr.io \
  --username '$oauthtoken' --password-stdin
echo "✓ NGC Docker login OK"

# ── 6. Pull VSS images (background — takes ~10 min) ──────────────────────────
echo "Pulling VSS images in background..."
cd "$REPO_DIR/video-search-and-summarization/deployments"
set -a; source "$REPO_DIR/deployments/vss/env.rtxpro6000bw"; set +a

sudo -E docker compose \
  -f compose.yml \
  -f "$REPO_DIR/deployments/vss/docker-compose.override.yml" \
  --profile bp_developer_base_2d \
  pull 2>&1 | tee /home/ubuntu/vss_pull.log &
VSS_PULL_PID=$!

# ── 7. Build render-api ───────────────────────────────────────────────────────
echo "Building render-api..."
cd "$REPO_DIR/services/render-api"
sudo docker build -t render-api:local . 2>&1 | tee /home/ubuntu/render_build.log
echo "✓ render-api built"

# ── 8. Build kimodo-api (context = kimodo submodule) ─────────────────────────
echo "Building kimodo-api (this may take 5-10 min)..."
cd "$REPO_DIR"
sudo DOCKER_BUILDKIT=1 docker build \
  --build-arg HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
  -f services/kimodo-api/Dockerfile \
  -t kimodo-api:local \
  kimodo/ 2>&1 | tee /home/ubuntu/kimodo_build.log
echo "✓ kimodo-api built"

# ── 9. Build cosmos-transfer ──────────────────────────────────────────────────
echo "Building cosmos-transfer..."
cd "$REPO_DIR/services/cosmos-transfer"
bash build.sh 2>&1 | tee /home/ubuntu/cosmos_build.log
echo "✓ cosmos-transfer built"

# ── 10. Wait for VSS pull ─────────────────────────────────────────────────────
echo "Waiting for VSS image pull to complete..."
wait $VSS_PULL_PID && echo "✓ VSS images pulled" || echo "⚠ VSS pull had errors — check /home/ubuntu/vss_pull.log"

# ── 11. Install systemd service for auto-start on reboot ─────────────────────
sudo tee /etc/systemd/system/vlm-flywheel.service > /dev/null <<'EOF'
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
EOF

# Create env file for systemd (secrets survive reboot)
cat > /home/ubuntu/.flywheel.env <<EOF
NGC_CLI_API_KEY=$NGC_CLI_API_KEY
HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN
HF_CACHE=$HF_CACHE
HF_TOKEN=$HUGGINGFACE_TOKEN
EOF
chmod 600 /home/ubuntu/.flywheel.env

sudo systemctl daemon-reload
sudo systemctl enable vlm-flywheel.service
echo "✓ systemd service installed and enabled"

# ── 12. First start ───────────────────────────────────────────────────────────
echo "Starting full stack..."
bash /home/ubuntu/vlm-pipeline/scripts/start.sh

echo ""
echo "=========================================="
echo " ✅ Setup complete — $(date)"
echo "   render-api  : http://localhost:9001"
echo "   kimodo-api  : http://localhost:9551"
echo "   cosmos-api  : http://localhost:8080"
echo "   VSS agent   : http://localhost:8000"
echo "   Studio UI   : http://$(curl -s ifconfig.me):9000"
echo "=========================================="
