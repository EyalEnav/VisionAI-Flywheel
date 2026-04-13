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
# They are injected automatically by Brev if set in the Launchable config.

: "${NGC_CLI_API_KEY:?ERROR: NGC_CLI_API_KEY env var not set}"
: "${HUGGINGFACE_TOKEN:?ERROR: HUGGINGFACE_TOKEN env var not set}"

# Persist secrets across reboots
grep -q NGC_CLI_API_KEY ~/.bashrc 2>/dev/null || {
  echo "export NGC_CLI_API_KEY=\"$NGC_CLI_API_KEY\""   >> ~/.bashrc
  echo "export HUGGINGFACE_TOKEN=\"$HUGGINGFACE_TOKEN\"" >> ~/.bashrc
}

# ── 1. Storage — use large attached disk ─────────────────────────────────────
NVME=/opt/dlami/nvme
if mountpoint -q "$NVME"; then
  echo "✓ NVMe mounted at $NVME"
else
  # Fallback: use /data if exists, else root
  NVME=/home/ubuntu/data
  mkdir -p "$NVME"
fi

DOCKER_DATA="$NVME/docker"
mkdir -p "$DOCKER_DATA"

# Configure Docker to use the large disk
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

# Pull VSS submodule
git submodule update --init --recursive
echo "✓ Repo ready at $REPO_DIR"

# ── 3. NGC Docker login ──────────────────────────────────────────────────────
echo "$NGC_CLI_API_KEY" | sudo docker login nvcr.io \
  --username '$oauthtoken' --password-stdin
echo "✓ NGC Docker login OK"

# ── 4. Pull VSS images (background) ─────────────────────────────────────────
echo "Pulling VSS images (this takes ~10 min)..."
cd "$REPO_DIR/video-search-and-summarization/deployments"

set -a
source "$REPO_DIR/deployments/vss/env.rtxpro6000bw"
set +a

sudo -E docker compose \
  -f compose.yml \
  -f "$REPO_DIR/deployments/vss/docker-compose.override.yml" \
  --profile bp_developer_base_2d \
  pull 2>&1 | tee /home/ubuntu/vss_pull.log &

VSS_PULL_PID=$!

# ── 5. Build render-api ───────────────────────────────────────────────────────
echo "Building render-api..."
cd "$REPO_DIR/services/render-api"
sudo docker build -t render-api:local . 2>&1 | tee /home/ubuntu/render_build.log
echo "✓ render-api built"

# ── 6. Build kimodo-api ───────────────────────────────────────────────────────
echo "Building kimodo-api..."
cd "$REPO_DIR/services/kimodo-api"
sudo DOCKER_BUILDKIT=1 docker build \
  --build-arg HUGGINGFACE_TOKEN="$HUGGINGFACE_TOKEN" \
  -t kimodo-api:local . 2>&1 | tee /home/ubuntu/kimodo_build.log
echo "✓ kimodo-api built"

# ── 7. Build cosmos-transfer ──────────────────────────────────────────────────
echo "Building cosmos-transfer..."
cd "$REPO_DIR/services/cosmos-transfer"
sudo docker build -t cosmos-transfer:local . 2>&1 | tee /home/ubuntu/cosmos_build.log
echo "✓ cosmos-transfer built"

# ── 8. Wait for VSS pull ──────────────────────────────────────────────────────
echo "Waiting for VSS image pull..."
wait $VSS_PULL_PID && echo "✓ VSS images pulled" || echo "⚠ VSS pull had errors — check /home/ubuntu/vss_pull.log"

# ── 9. Install systemd service for auto-start on reboot ──────────────────────
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
EOF
chmod 600 /home/ubuntu/.flywheel.env

sudo systemctl daemon-reload
sudo systemctl enable vlm-flywheel.service
echo "✓ systemd service installed"

# ── 10. First start ───────────────────────────────────────────────────────────
echo "Starting stack..."
bash /home/ubuntu/vlm-pipeline/scripts/start.sh

echo ""
echo "=========================================="
echo " ✅ Setup complete — $(date)"
echo "   render-api : http://localhost:9000"
echo "   VSS agent  : http://localhost:8000"
echo "   Studio UI  : http://localhost:3000"
echo "=========================================="
