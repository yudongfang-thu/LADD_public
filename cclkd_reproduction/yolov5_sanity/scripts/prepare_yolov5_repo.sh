#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash cclkd_reproduction/yolov5_sanity/scripts/prepare_yolov5_repo.sh

Environment:
  YOLOV5_REF    YOLOv5 git ref/tag to checkout (default: v7.0)
  YOLOV5_URL    YOLOv5 repository URL (default: https://github.com/ultralytics/yolov5.git)
  PYTHON        Python executable for environment recording (default: python3)

This script clones/checks out YOLOv5 under external/yolov5 and records the
environment. It does not run pip install automatically.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
YOLOV5_REF="${YOLOV5_REF:-v7.0}"
YOLOV5_URL="${YOLOV5_URL:-https://github.com/ultralytics/yolov5.git}"
PYTHON="${PYTHON:-python3}"
YOLOV5_DIR="$REPO_ROOT/external/yolov5"
RESULT_DIR="$REPO_ROOT/cclkd_reproduction/yolov5_sanity/results"
ENV_TXT="$RESULT_DIR/yolov5_env.txt"

mkdir -p "$REPO_ROOT/external" "$RESULT_DIR"

if [[ ! -d "$YOLOV5_DIR/.git" ]]; then
  echo "Cloning YOLOv5 into $YOLOV5_DIR"
  git clone "$YOLOV5_URL" "$YOLOV5_DIR"
else
  echo "YOLOv5 repository already exists: $YOLOV5_DIR"
fi

git -C "$YOLOV5_DIR" fetch --tags --prune
git -C "$YOLOV5_DIR" checkout "$YOLOV5_REF"

COMMIT="$(git -C "$YOLOV5_DIR" rev-parse HEAD)"
STATUS="$(git -C "$YOLOV5_DIR" status --short)"

TORCH_INFO="$("$PYTHON" - <<'PY'
try:
    import torch
    print(f"torch={torch.__version__}")
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda_version={torch.version.cuda}")
    print(f"torch_device_count={torch.cuda.device_count()}")
except Exception as exc:
    print(f"torch_import_error={exc}")
PY
)"

{
  echo "yolov5_dir=$YOLOV5_DIR"
  echo "yolov5_ref=$YOLOV5_REF"
  echo "yolov5_commit=$COMMIT"
  echo "yolov5_status=${STATUS:-clean}"
  echo "python_executable=$PYTHON"
  echo "python_version=$("$PYTHON" --version 2>&1)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "$TORCH_INFO"
  echo
  echo "requirements_note=No automatic pip install was run. If needed, inspect and install external/yolov5/requirements.txt in the intended environment."
} > "$ENV_TXT"

echo "Checked out YOLOv5 ref: $YOLOV5_REF"
echo "Commit: $COMMIT"
echo "Environment report: $ENV_TXT"
echo "Suggested dependency step, if your environment is not ready:"
echo "  $PYTHON -m pip install -r $YOLOV5_DIR/requirements.txt"
