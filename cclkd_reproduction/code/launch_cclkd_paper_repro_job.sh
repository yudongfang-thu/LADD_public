#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  cclkd_reproduction/code/launch_cclkd_paper_repro_job.sh <n|s> <seed> <gpu_id>

Required environment:
  STUDENT_DATA      SAR OGSOD HBB yaml, nc=3
  TEACHER_DATA      RGB OGSOD HBB yaml, nc=3

Optional environment:
  REPO_ROOT         repository root (auto-detected)
  PYTHON            python executable (default: python3)
  MIXUP             paper-aligned MixUp probability (default: 0.1; set ALLOW_UNVERIFIED_MIXUP=1 if unknown)
  ALLOW_UNVERIFIED_MIXUP  allow MIXUP=0 only with an explicit audit note
  CCLKD_FORMULATION paper-aligned implementation variant (default: paper)
  CCLKD_CCL_MODE paper_pair|anchor_teacher_neg (default: paper_pair)
  CCLKD_FLD_TEMPERATURE fixed FLD softmax/KL temperature (default: 1.0)
  CCLKD_FLD_TEMPERATURE_MODE fixed|patm (default: fixed)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_SIZE="${1:?missing model size n|s}"
SEED="${2:?missing seed}"
GPU_ID="${3:?missing gpu id}"
if [[ "$MODEL_SIZE" != "n" && "$MODEL_SIZE" != "s" ]]; then
  echo "CCLKD paper reproduction only allows YOLO11n or YOLO11s." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
STUDENT_DATA="${STUDENT_DATA:?set STUDENT_DATA to SAR OGSOD HBB yaml}"
TEACHER_DATA="${TEACHER_DATA:?set TEACHER_DATA to RGB OGSOD HBB yaml}"
MODEL_WEIGHTS="$REPO_ROOT/yolo11${MODEL_SIZE}.pt"
MIXUP="${MIXUP:-0.1}"
CCLKD_FORMULATION="${CCLKD_FORMULATION:-paper}"
CCLKD_CCL_MODE="${CCLKD_CCL_MODE:-paper_pair}"
if [[ "$CCLKD_FORMULATION" != "adapted" && "$CCLKD_FORMULATION" != "paper" ]]; then
  echo "CCLKD_FORMULATION must be adapted or paper, got: $CCLKD_FORMULATION" >&2
  exit 1
fi
if [[ "$CCLKD_CCL_MODE" != "paper_pair" && "$CCLKD_CCL_MODE" != "anchor_teacher_neg" ]]; then
  echo "CCLKD_CCL_MODE must be paper_pair or anchor_teacher_neg, got: $CCLKD_CCL_MODE" >&2
  exit 1
fi
if [[ ! -f "$MODEL_WEIGHTS" ]]; then
  echo "Missing YOLO pretrain checkpoint: $MODEL_WEIGHTS" >&2
  exit 1
fi
ALLOW_FLAG=()
if [[ "${ALLOW_UNVERIFIED_MIXUP:-0}" == "1" ]]; then
  ALLOW_FLAG+=(--allow-unverified-mixup)
fi

"$PYTHON" "$SCRIPT_DIR/check_cclkd_repro_protocol.py" \
  --model-size "$MODEL_SIZE" \
  --student-data "$STUDENT_DATA" \
  --teacher-data "$TEACHER_DATA" \
  --epochs 400 \
  --batch 32 \
  --imgsz 256 \
  --optimizer SGD \
  --lr0 0.01 \
  --mosaic 1.0 \
  --mixup "$MIXUP" \
  --online-trainer \
  "${ALLOW_FLAG[@]}"

RUN_NAME="cclkd_paper_repro_yolo11${MODEL_SIZE}_${CCLKD_FORMULATION}_s${SEED}_400ep_online"
mkdir -p "$REPO_ROOT/logs/cclkd_reproduction"

exec "$PYTHON" "$SCRIPT_DIR/train_cclkd_online_hbb.py" \
  --model-size "$MODEL_SIZE" \
  --data "$STUDENT_DATA" \
  --teacher-data "$TEACHER_DATA" \
  --imgsz 256 \
  --epochs 400 \
  --batch 32 \
  --workers 8 \
  --device "$GPU_ID" \
  --project "$REPO_ROOT/runs_public/ogsod/hbb/cclkd_reproduction" \
  --name "$RUN_NAME" \
  --optimizer SGD \
  --lr0 0.01 \
  --mosaic 1.0 \
  --mixup "$MIXUP" \
  --close-mosaic 10 \
  --kd-weight 1.0 \
  --lld-weight 1.0 \
  --fld-weight 1.0 \
  --rld-weight 1.0 \
  --ccl-weight 1.0 \
  --cclkd-formulation "$CCLKD_FORMULATION" \
  --cclkd-ccl-mode "$CCLKD_CCL_MODE" \
  --cclkd-fld-temperature "${CCLKD_FLD_TEMPERATURE:-1.0}" \
  --cclkd-fld-temperature-mode "${CCLKD_FLD_TEMPERATURE_MODE:-fixed}" \
  --cclkd-roi-grid-size "${CCLKD_ROI_GRID_SIZE:-3}" \
  --seed "$SEED"
