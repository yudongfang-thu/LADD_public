#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  comparison/code/launch_formal_online_cclkd_ablation_job.sh <n|s> <ablation> <seed> <gpu_id>

Runs one CCLKD ablation under the LADD formal OGSOD HBB baseline protocol,
while keeping the CCLKD loss formulation aligned to the paper.

Ablations:
  lld            LLD only, fixed T=1
  lld_fld        LLD + FLD, fixed T=1
  lld_fld_rld    LLD + FLD + RLD, fixed T=1
  atkd           LLD + FLD + RLD + PATM, no CCL
  ccl_only       CCL only
  full           LLD + FLD + RLD + PATM + CCL

Default protocol:
  imgsz=256, epochs=400, batch n/s=64, cos-lr, mosaic=0.0,
  mixup=0.0, close-mosaic=0, deterministic=True.

Optional:
  EPOCHS=400
  BATCH_SIZE=64
  CCLKD_FORMULATION=paper
  EXIST_OK=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SIZE="${1:-}"
ABLATION="${2:-}"
SEED="${3:-}"
GPU_ID="${4:-}"

if [[ ! "$SIZE" =~ ^(n|s)$ ]]; then
  echo "CCLKD formal ablation currently allows YOLO11n or YOLO11s only." >&2
  usage >&2
  exit 1
fi
if [[ -z "$ABLATION" || ! "$SEED" =~ ^[0-9]+$ || -z "$GPU_ID" ]]; then
  usage >&2
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
  *)
    echo "Unknown ablation: $ABLATION" >&2
    usage >&2
    exit 1
    ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
if [[ -d /root/miniconda3/bin ]]; then
  export PATH="/root/miniconda3/bin:${PATH}"
fi

PYTHON="${PYTHON:-python3}"
BASE_ROOT="runs_public/ogsod/hbb/formal_nomosaic_20260528"
PRETRAIN="yolo11${SIZE}.pt"
SAR_DATA="configs/datasets/ogsod_hbb_sar.yaml"
RGB_DATA="configs/datasets/ogsod_hbb_rgb.yaml"
EPOCHS="${EPOCHS:-400}"
case "$SIZE" in
  n|s) DEFAULT_BATCH=64 ;;
esac
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH}"
CCLKD_FORMULATION="${CCLKD_FORMULATION:-paper}"

if [[ "$CCLKD_FORMULATION" != "paper" ]]; then
  echo "Formal CCLKD ablations should use CCLKD_FORMULATION=paper, got: $CCLKD_FORMULATION" >&2
  exit 1
fi
if [[ ! -f "$PRETRAIN" ]]; then
  echo "Missing YOLO pretrain checkpoint: $PRETRAIN" >&2
  exit 1
fi
if [[ ! -f "$SAR_DATA" ]]; then
  echo "Missing SAR dataset yaml: $SAR_DATA" >&2
  exit 1
fi
if [[ ! -f "$RGB_DATA" ]]; then
  echo "Missing RGB dataset yaml: $RGB_DATA" >&2
  exit 1
fi

RUN_TAG="formal_nomosaic_yolo11${SIZE}_cclkd_paper_${ABLATION}_s${SEED}${RUN_TAG_SUFFIX:-}"
PROJECT_DIR="${BASE_ROOT}/comparisons/online_cclkd/yolo11${SIZE}/paper_ablation"
LOG_DIR="logs/formal_nomosaic_20260528/comparisons/online_cclkd_paper_ablation"
OUTER_LOG="${LOG_DIR}/${RUN_TAG}_gpu${GPU_ID}.outer.log"
PID_PATH="${LOG_DIR}/${RUN_TAG}_gpu${GPU_ID}.pid"
RUN_NAME="online_cclkd_hbb_ogsod11${SIZE}_from_yolo_${RUN_TAG}_e${EPOCHS}_b${BATCH_SIZE}_s${SEED}_gpu${GPU_ID}"
mkdir -p "$PROJECT_DIR" "$LOG_DIR"

if [[ -e "${PROJECT_DIR}/${RUN_NAME}" && "${EXIST_OK:-0}" != "1" ]]; then
  echo "Run directory already exists: ${PROJECT_DIR}/${RUN_NAME}" >&2
  echo "Set EXIST_OK=1 only if intentional." >&2
  exit 1
fi

cmd=(
  "$PYTHON" cclkd_reproduction/code/train_cclkd_online_hbb.py
  --model-size "$SIZE"
  --model "$PRETRAIN"
  --teacher-weights "$PRETRAIN"
  --data "$SAR_DATA"
  --teacher-data "$RGB_DATA"
  --imgsz 256
  --epochs "$EPOCHS"
  --batch "$BATCH_SIZE"
  --workers "${WORKERS:-8}"
  --device "$GPU_ID"
  --patience "$EPOCHS"
  --project "$PROJECT_DIR"
  --name "$RUN_NAME"
  --teacher-det-weight "${CCLKD_TEACHER_DET_WEIGHT:-1.0}"
  --kd-weight "${CCLKD_KD_WEIGHT:-1.0}"
  --lld-weight "$LLD"
  --fld-weight "$FLD"
  --rld-weight "$RLD"
  --ccl-weight "$CCL"
  --cclkd-temperature-min "$TMIN"
  --cclkd-temperature-max "$TMAX"
  --cclkd-formulation "$CCLKD_FORMULATION"
  --cclkd-roi-grid-size "${CCLKD_ROI_GRID_SIZE:-3}"
  --optimizer "${OPTIMIZER:-auto}"
  --lr0 "${LR0:-0.01}"
  --lrf "${LRF:-0.01}"
  --cos-lr
  --mosaic 0.0
  --mixup 0.0
  --cutmix 0.0
  --close-mosaic 0
  --degrees 0.0
  --perspective 0.0
  --translate 0.1
  --scale 0.5
  --fliplr 0.5
  --flipud 0.0
  --hsv-h 0.0
  --hsv-s 0.0
  --hsv-v 0.0
  --erasing 0.0
  --seed "$SEED"
  --deterministic
  --save-period "${SAVE_PERIOD:-100}"
)

if [[ "${EXIST_OK:-0}" == "1" ]]; then
  cmd+=(--exist-ok)
fi

echo "[$(date '+%F %T')] Launching formal CCLKD paper ablation=${ABLATION} yolo11${SIZE} seed=${SEED} gpu=${GPU_ID}"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, not launching."
  exit 0
fi

nohup env PYTHONUNBUFFERED=1 "${cmd[@]}" > "$OUTER_LOG" 2>&1 &
pid=$!
echo "$pid" > "$PID_PATH"
echo "Launched pid=${pid}; log=${OUTER_LOG}; pid_file=${PID_PATH}"
