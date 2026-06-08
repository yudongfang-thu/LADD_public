#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  cclkd_reproduction/code/launch_cclkd_n_ablation_job.sh <ablation> <seed> <gpu_id>

Ablations, matching CCLKD Table 12:
  lld            LLD only, fixed T=1
  lld_fld        LLD + FLD, fixed T=1
  lld_fld_rld    LLD + FLD + RLD, fixed T=1
  atkd           LLD + FLD + RLD + PATM, no CCL
  ccl_only       CCL only
  full           LLD + FLD + RLD + PATM + CCL, paper full setting
  full_ccl05     Same as full, but CCL weight 0.5 for diagnosing older runs

Required environment:
  STUDENT_DATA      SAR OGSOD HBB yaml, nc=3
  TEACHER_DATA      RGB OGSOD HBB yaml, nc=3

Optional environment:
  REPO_ROOT         repository root (auto-detected)
  PYTHON            python executable (default: python3)
  EPOCHS            default 400
  BATCH             default 32
  MIXUP             default 0.1
  CCLKD_CCL_MODE    paper_pair|anchor_teacher_neg (default: paper_pair)
  PROJECT_DIR       default runs_public/ogsod/hbb/cclkd_reproduction_ablation/yolo11n
  EXIST_OK          pass --exist-ok when set to 1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ABLATION="${1:?missing ablation}"
SEED="${2:?missing seed}"
GPU_ID="${3:?missing gpu id}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
STUDENT_DATA="${STUDENT_DATA:?set STUDENT_DATA to SAR OGSOD HBB yaml}"
TEACHER_DATA="${TEACHER_DATA:?set TEACHER_DATA to RGB OGSOD HBB yaml}"
MODEL_WEIGHTS="$REPO_ROOT/yolo11n.pt"
EPOCHS="${EPOCHS:-400}"
BATCH="${BATCH:-32}"
MIXUP="${MIXUP:-0.1}"
CCLKD_CCL_MODE="${CCLKD_CCL_MODE:-paper_pair}"
PROJECT_DIR="${PROJECT_DIR:-$REPO_ROOT/runs_public/ogsod/hbb/cclkd_reproduction_ablation/yolo11n}"

if [[ "$CCLKD_CCL_MODE" != "paper_pair" && "$CCLKD_CCL_MODE" != "anchor_teacher_neg" ]]; then
  echo "CCLKD_CCL_MODE must be paper_pair or anchor_teacher_neg, got: $CCLKD_CCL_MODE" >&2
  exit 1
fi
if [[ ! -f "$MODEL_WEIGHTS" ]]; then
  echo "Missing YOLO pretrain checkpoint: $MODEL_WEIGHTS" >&2
  exit 1
fi

LLD=0
FLD=0
RLD=0
CCL=0
TMIN=1.0
TMAX=1.0

case "$ABLATION" in
  lld)
    LLD=1
    ;;
  lld_fld)
    LLD=1; FLD=1
    ;;
  lld_fld_rld)
    LLD=1; FLD=1; RLD=1
    ;;
  atkd)
    LLD=1; FLD=1; RLD=1; TMIN=0.5; TMAX=5.0
    ;;
  ccl_only)
    CCL=1
    ;;
  full)
    LLD=1; FLD=1; RLD=1; CCL=1; TMIN=0.5; TMAX=5.0
    ;;
  full_ccl05)
    LLD=1; FLD=1; RLD=1; CCL=0.5; TMIN=0.5; TMAX=5.0
    ;;
  *)
    echo "Unknown ablation: $ABLATION" >&2
    usage >&2
    exit 2
    ;;
esac

"$PYTHON" "$SCRIPT_DIR/check_cclkd_repro_protocol.py" \
  --model-size n \
  --student-data "$STUDENT_DATA" \
  --teacher-data "$TEACHER_DATA" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz 256 \
  --optimizer SGD \
  --lr0 0.01 \
  --mosaic 1.0 \
  --mixup "$MIXUP" \
  --online-trainer

RUN_NAME="cclkd_ablation_yolo11n_${ABLATION}_s${SEED}_${EPOCHS}ep"
EXIST_ARGS=()
if [[ "${EXIST_OK:-0}" == "1" ]]; then
  EXIST_ARGS+=(--exist-ok)
fi

exec "$PYTHON" "$SCRIPT_DIR/train_cclkd_online_hbb.py" \
  --model-size n \
  --data "$STUDENT_DATA" \
  --teacher-data "$TEACHER_DATA" \
  --imgsz 256 \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --workers 8 \
  --device "$GPU_ID" \
  --project "$PROJECT_DIR" \
  --name "$RUN_NAME" \
  "${EXIST_ARGS[@]}" \
  --optimizer SGD \
  --lr0 0.01 \
  --mosaic 1.0 \
  --mixup "$MIXUP" \
  --close-mosaic 10 \
  --teacher-det-weight 1.0 \
  --kd-weight 1.0 \
  --lld-weight "$LLD" \
  --fld-weight "$FLD" \
  --rld-weight "$RLD" \
  --ccl-weight "$CCL" \
  --cclkd-temperature-min "$TMIN" \
  --cclkd-temperature-max "$TMAX" \
  --cclkd-formulation paper \
  --cclkd-ccl-mode "$CCLKD_CCL_MODE" \
  --seed "$SEED"
