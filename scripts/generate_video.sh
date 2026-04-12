#!/usr/bin/env bash
# Generate a synthetic surveillance video end-to-end
# Usage: ./scripts/generate_video.sh "prompt" [--transfer] [--face /path/face.jpg]
#
# Examples:
#   ./scripts/generate_video.sh "security guard walking patrol route"
#   ./scripts/generate_video.sh "person falling in crowd" --transfer
#   ./scripts/generate_video.sh "suspect running away" --face ~/face.jpg
set -euo pipefail

PROMPT="${1:?provide a scene prompt}"
shift

USE_TRANSFER=false
FACE_PATH=""
MOTION_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --transfer) USE_TRANSFER=true ;;
    --face)     FACE_PATH="$2"; shift ;;
    --motion)   MOTION_FILE="$2"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
  shift
done

RENDER_API="${RENDER_API_URL:-http://localhost:9000}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="./output/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo "═══════════════════════════════════════"
echo " Generating video"
echo " Prompt:   ${PROMPT}"
echo " Transfer: ${USE_TRANSFER}"
echo " Face:     ${FACE_PATH:-none}"
echo "═══════════════════════════════════════"

# ── Submit render job ────────────────────────────────────────────────────────
TEXTURE_MODE="cosmos"
[[ "${USE_TRANSFER}" == "true" ]] && TEXTURE_MODE="skeleton"
[[ -n "${FACE_PATH}" ]] && TEXTURE_MODE="faceswap"

CURL_ARGS=(
  -s -X POST "${RENDER_API}/generate"
  -F "prompt=${PROMPT}"
  -F "texture_mode=${TEXTURE_MODE}"
  -F "cosmos_prompt=${PROMPT}"
)
[[ -n "${MOTION_FILE}" ]] && CURL_ARGS+=(-F "motion_file=${MOTION_FILE}")
[[ -n "${FACE_PATH}" ]]   && CURL_ARGS+=(-F "face_image=@${FACE_PATH}")

RESPONSE=$(curl "${CURL_ARGS[@]}")
JOB_ID=$(echo "${RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "Job ID: ${JOB_ID}"

# ── Poll until done ──────────────────────────────────────────────────────────
echo -n "Rendering"
while true; do
  STATUS=$(curl -s "${RENDER_API}/jobs/${JOB_ID}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['status'], d.get('progress',0))")
  S=$(echo "${STATUS}" | awk '{print $1}')
  P=$(echo "${STATUS}" | awk '{print $2}')
  echo -ne "\r  Status: ${S} (${P}%)    "
  [[ "${S}" == "done" ]] && break
  [[ "${S}" == "error" ]] && { echo "ERROR"; curl -s "${RENDER_API}/jobs/${JOB_ID}" | python3 -m json.tool; exit 1; }
  sleep 3
done
echo ""

# ── Download render ──────────────────────────────────────────────────────────
RENDER_OUT="${OUTPUT_DIR}/render.mp4"
curl -s -o "${RENDER_OUT}" "${RENDER_API}/render/video/${JOB_ID}"
echo "Render saved: ${RENDER_OUT} ($(du -sh "${RENDER_OUT}" | cut -f1))"

# ── Cosmos Transfer (optional) ───────────────────────────────────────────────
if [[ "${USE_TRANSFER}" == "true" ]]; then
  echo "Running Cosmos Transfer2.5..."
  TRANSFER_OUT="${OUTPUT_DIR}/cosmos_render.mp4"
  bash "$(dirname "$0")/../services/cosmos-transfer/run.sh" \
    "${RENDER_OUT}" "${PROMPT}" "${TRANSFER_OUT}"
  echo "Cosmos output: ${TRANSFER_OUT}"
fi

echo ""
echo "Done! Output in ${OUTPUT_DIR}/"
