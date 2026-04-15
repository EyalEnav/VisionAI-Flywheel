#!/bin/bash
# =============================================================================
# VisionAI-Flywheel — Installer
# Usage: bash install.sh
# =============================================================================
set -euo pipefail

curl -fsSL https://raw.githubusercontent.com/EyalEnav/VisionAI-Flywheel/main/setup.sh \
  -o /tmp/vaf_setup.sh

chmod +x /tmp/vaf_setup.sh
bash /tmp/vaf_setup.sh
rm -f /tmp/vaf_setup.sh
