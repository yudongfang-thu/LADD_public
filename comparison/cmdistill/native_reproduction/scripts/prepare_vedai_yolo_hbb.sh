#!/usr/bin/env bash
set -euo pipefail

RESOLUTION="${1:-512}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif [[ -x /root/miniconda3/bin/python ]]; then
    PYTHON_BIN="/root/miniconda3/bin/python"
  else
    echo "No Python interpreter found. Set PYTHON=/path/to/python." >&2
    exit 1
  fi
fi

"${SCRIPT_DIR}/extract_vedai.sh" "${RESOLUTION}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_vedai_yolo_hbb.py" \
  --resolution "${RESOLUTION}" \
  --split "${VEDAI_SPLIT:-paper80_seed0}" \
  --link-mode "${VEDAI_LINK_MODE:-symlink}" \
  --overwrite
