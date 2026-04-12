#!/usr/bin/env bash
# Build the Cosmos Transfer Docker image from the submodule Dockerfile
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "Building cosmos-transfer:local from ${REPO_ROOT}/cosmos-transfer2.5/Dockerfile"

docker build \
  --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 \
  -t cosmos-transfer:local \
  "${REPO_ROOT}/cosmos-transfer2.5"

echo "Done: cosmos-transfer:local"
