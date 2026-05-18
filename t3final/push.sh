#!/usr/bin/env bash
set -euo pipefail

REMOTE="software@100.64.0.25"
REMOTE_PATH="/home/software/Documents/utils/aarav/t3final/"

echo "Creating remote directory ${REMOTE}:${REMOTE_PATH}"
ssh "${REMOTE}" "mkdir -p '${REMOTE_PATH}'"

echo "Pushing source/config/docs to ${REMOTE}:${REMOTE_PATH}"
rsync -az --delete \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude "__pycache__/" \
  --exclude "data/" \
  --exclude "models/" \
  --exclude "runs/" \
  --exclude "outputs/" \
  --exclude "*.pt" \
  --exclude "*.pth" \
  --exclude "kaggle.json" \
  --exclude ".env" \
  ./ "${REMOTE}:${REMOTE_PATH}"

echo "Push complete. Generated artifacts were excluded."

