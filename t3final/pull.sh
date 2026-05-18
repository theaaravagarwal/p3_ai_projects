#!/usr/bin/env bash
set -euo pipefail

REMOTE="software@100.64.0.25"
REMOTE_PATH="/home/software/Documents/utils/aarav/t3final/"

echo "Pulling generated artifacts from ${REMOTE}:${REMOTE_PATH}"
mkdir -p models runs outputs

rsync -az "${REMOTE}:${REMOTE_PATH}/models/" models/ || true
rsync -az "${REMOTE}:${REMOTE_PATH}/runs/" runs/ || true
rsync -az "${REMOTE}:${REMOTE_PATH}/outputs/" outputs/ || true

echo "Pull complete. Local source code was not overwritten."

