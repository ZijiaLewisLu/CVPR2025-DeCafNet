#!/usr/bin/env bash
set -euo pipefail

# Example usage for public release.
# Supported tasks: nlq, goalstep.
# Adjust the paths below to your local Ego4D checkout and checkpoint location.
#
# Example commands:
#   TASK=nlq bash run.sh
#   TASK=goalstep METADATA_DIR=/path/to/goalstep_annotations bash run.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="${TASK:-nlq}"
VIDEO_ROOT="${VIDEO_ROOT:-/path/to/ego4d/videos}"
METADATA_DIR="${METADATA_DIR:-/path/to/ego4d/annotations}"
CKPT_PATH="${CKPT_PATH:-$ROOT_DIR/egovlp.pth}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Generate all non-sidekick Ego4D artifacts for the selected task.
"$PYTHON_BIN" "$ROOT_DIR/prepare_ego4d_data.py" \
  --task "$TASK" \
  --egovlp-python "$PYTHON_BIN" \
  --resume "$CKPT_PATH" \
  --video-root "$VIDEO_ROOT" \
  --metadir "$METADATA_DIR"
