#!/usr/bin/env bash
# Run Cosmos Transfer2.5 Sim2Real on a single video
# Usage: ./run.sh <input_video.mp4> "scene prompt" <output_video.mp4>
set -euo pipefail

INPUT_VIDEO="${1:?Usage: $0 <input.mp4> <prompt> <output.mp4>}"
PROMPT="${2:?provide a scene prompt}"
OUTPUT_VIDEO="${3:-/tmp/cosmos_out.mp4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

HF_CACHE="${HF_CACHE:-${HOME}/.cache/huggingface}"
GPU="${TRANSFER_GPU_DEVICE:-1}"

# Build spec JSON
INPUT_BASENAME="$(basename "${INPUT_VIDEO}" .mp4)"
OUTPUT_DIR="$(dirname "${OUTPUT_VIDEO}")"
SPEC_FILE="/tmp/cosmos_spec_$$.json"

cat > "${SPEC_FILE}" <<EOF
{
  "name": "${INPUT_BASENAME}",
  "prompt": "${PROMPT}",
  "video_path": "${INPUT_VIDEO}",
  "edge": {"control_weight": 1.0}
}
EOF

echo "Running Cosmos Transfer2.5 on GPU ${GPU}..."
echo "  Input:  ${INPUT_VIDEO}"
echo "  Prompt: ${PROMPT}"
echo "  Output: ${OUTPUT_VIDEO}"

docker run --rm \
  --gpus "device=${GPU}" \
  -v "${REPO_ROOT}/cosmos-transfer2.5:/workspace" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "/tmp:/tmp" \
  -e "HF_TOKEN=${HF_TOKEN:-}" \
  -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}" \
  cosmos-transfer:local \
  python3 /workspace/examples/inference.py \
    -i "${SPEC_FILE}" \
    -o "${OUTPUT_DIR}" \
    --disable-guardrails \
    --offload-guardrail-models \
    control:edge

# Rename output to expected filename
COSMOS_OUT="${OUTPUT_DIR}/${INPUT_BASENAME}.mp4"
if [[ -f "${COSMOS_OUT}" && "${COSMOS_OUT}" != "${OUTPUT_VIDEO}" ]]; then
  mv "${COSMOS_OUT}" "${OUTPUT_VIDEO}"
fi

rm -f "${SPEC_FILE}"
echo "Cosmos Transfer done: ${OUTPUT_VIDEO}"
