#!/usr/bin/env bash
# Initialize git repo on the Brev instance and push to GitHub
# Usage: bash scripts/git_init.sh <github_repo_url>
# Example: bash scripts/git_init.sh git@github.com:yourorg/vlm-pipeline.git
set -euo pipefail

REMOTE="${1:?provide GitHub remote URL, e.g. git@github.com:yourorg/vlm-pipeline.git}"
BREV_HOST="${BREV_HOST:-ubuntu@18.116.47.77}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/kostya_ssh}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Pushing repo to ${REMOTE}..."

# Copy repo files to Brev (excluding submodule dirs — they're already there)
rsync -av --exclude='.git' \
  --exclude='kimodo/' \
  --exclude='cosmos-transfer2.5/' \
  --exclude='video-search-and-summarization/' \
  "${REPO_ROOT}/" \
  -e "ssh -i ${SSH_KEY}" \
  "${BREV_HOST}:/home/ubuntu/vlm-pipeline/"

# Init git on Brev
ssh -i "${SSH_KEY}" "${BREV_HOST}" "
cd /home/ubuntu/vlm-pipeline

# Init repo if not already
if [ ! -d .git ]; then
  git init
  git branch -m main
fi

# Add submodules (already cloned, just register them)
git submodule add https://github.com/NVlabs/Kimodo.git kimodo 2>/dev/null || true
git submodule add https://github.com/NVIDIA/Cosmos-Transfer2.git cosmos-transfer2.5 2>/dev/null || true
git submodule add https://github.com/NVIDIA-Metropolis/video-search-and-summarization.git video-search-and-summarization 2>/dev/null || true

# Create symlinks to existing clones
[ -d /home/ubuntu/kimodo ] && (rm -rf kimodo; ln -sf /home/ubuntu/kimodo kimodo)
[ -d /home/ubuntu/cosmos-transfer2.5 ] && (rm -rf cosmos-transfer2.5; ln -sf /home/ubuntu/cosmos-transfer2.5 cosmos-transfer2.5)
[ -d /home/ubuntu/video-search-and-summarization ] && (rm -rf video-search-and-summarization; ln -sf /home/ubuntu/video-search-and-summarization video-search-and-summarization)

git add -A
git commit -m 'feat: VLM synthetic data pipeline — Kimodo + Cosmos-Transfer2.5 + VSS'

git remote add origin ${REMOTE} 2>/dev/null || git remote set-url origin ${REMOTE}
git push -u origin main
echo 'Pushed!'
"
