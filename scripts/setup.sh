#!/usr/bin/env bash
# Full setup script for a fresh Brev/Ubuntu instance with 2× RTX PRO 6000 Blackwell
# Run as: bash scripts/setup.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${REPO_ROOT}/.env" 2>/dev/null || { echo "ERROR: .env not found. cp .env.example .env and fill it."; exit 1; }

echo "═══════════════════════════════════════════"
echo " VLM Pipeline Setup"
echo " Hardware: 2× NVIDIA RTX PRO 6000 Blackwell"
echo "═══════════════════════════════════════════"

# ── 1. Submodules ────────────────────────────────────────────────────────────
echo "[1/6] Initializing git submodules..."
cd "${REPO_ROOT}"
git submodule update --init --recursive

# ── 2. NGC login ─────────────────────────────────────────────────────────────
echo "[2/6] NGC login..."
echo "${NGC_CLI_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin

# ── 3. VSS Stack ─────────────────────────────────────────────────────────────
echo "[3/6] Starting VSS Blueprint stack..."
set -a; source "${REPO_ROOT}/deployments/vss/env.rtxpro6000bw"; set +a

cd "${REPO_ROOT}/video-search-and-summarization/deployments"
docker compose \
  -f compose.yml \
  -f "${REPO_ROOT}/deployments/vss/docker-compose.override.yml" \
  --profile ${BP_PROFILE} \
  up -d

echo "Waiting for VSS Agent to become healthy..."
timeout 300 bash -c 'until curl -sf http://localhost:8000/health; do sleep 5; done'
echo "VSS Agent: OK"

# ── 4. Kimodo ────────────────────────────────────────────────────────────────
echo "[4/6] Building & starting Kimodo..."
cd "${REPO_ROOT}/kimodo"
docker compose build
docker compose up -d

echo "Waiting for Kimodo text encoder..."
timeout 180 bash -c 'until curl -sf http://localhost:9550/; do sleep 5; done'
echo "Kimodo text encoder: OK"

# ── 5. Cosmos Transfer2.5 ────────────────────────────────────────────────────
echo "[5/6] Building Cosmos Transfer2.5 Docker image..."
bash "${REPO_ROOT}/services/cosmos-transfer/build.sh"
echo "cosmos-transfer:local: built"

# ── 6. Render API ────────────────────────────────────────────────────────────
echo "[6/6] Building & starting Render API..."
cd "${REPO_ROOT}/services/render-api"
docker compose build
docker compose up -d

echo "Waiting for Render API..."
timeout 60 bash -c 'until curl -sf http://localhost:9000/health; do sleep 3; done'
echo "Render API: OK"

echo ""
echo "═══════════════════════════════════════════"
echo " All services running!"
echo ""
echo " VSS Agent:        http://localhost:8000"
echo " VST Upload:       http://localhost:77770"
echo " Cosmos-Reason2:   http://localhost:30082"
echo " Kimodo encoder:   http://localhost:9550"
echo " Render API:       http://localhost:9000"
echo "═══════════════════════════════════════════"
