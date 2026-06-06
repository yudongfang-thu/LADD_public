#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  comparison/code/launch_formal_online_cclkd_job.sh <n|s> <seed> <gpu_id>

Runs CCLKD as an online teacher-student controlled comparison under the LADD
formal OGSOD HBB protocol:
  student init = yolo11<size>.pt
  teacher init = yolo11<size>.pt
  teacher input = paired RGB
  student input = SAR
  inference/eval = SAR-only student
  data/augmentation = formal no-mosaic comparison settings
  epochs default = 800; convergence is judged by best checkpoint, not train length.

This is NOT the 400ep paper reproduction launcher. Use
  cclkd_reproduction/code/launch_cclkd_paper_repro_job.sh
for the CCLKD paper-protocol reproduction.

Optional:
  RUN_TAG_SUFFIX=_r1
  EPOCHS=800
  BATCH_SIZE=64
  CCLKD_KD_WEIGHT=1.0
  CCLKD_CCL_WEIGHT=1.0
  EXIST_OK=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SIZE="${1:-}"
SEED="${2:-}"
GPU_ID="${3:-}"

if [[ ! "$SIZE" =~ ^(n|s)$ ]]; then
  echo "CCLKD online comparison currently allows YOLO11n or YOLO11s only." >&2
  usage >&2
  exit 1
fi
if [[ ! "$SEED" =~ ^[0-9]+$ || -z "$GPU_ID" ]]; then
  usage >&2
  exit 1
fi

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
EPOCHS="${EPOCHS:-800}"
case "$SIZE" in
  n|s) DEFAULT_BATCH=64 ;;
esac
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH}"

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

RUN_TAG="formal_nomosaic_yolo11${SIZE}_cclkd_online_from_yolo_s${SEED}${RUN_TAG_SUFFIX:-}"
PROJECT_DIR="${BASE_ROOT}/comparisons/online_cclkd/yolo11${SIZE}/cclkd"
LOG_DIR="logs/formal_nomosaic_20260528/comparisons/online_cclkd"
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
  --project "$PROJECT_DIR"
  --name "$RUN_NAME"
  --teacher-det-weight "${CCLKD_TEACHER_DET_WEIGHT:-1.0}"
  --kd-weight "${CCLKD_KD_WEIGHT:-1.0}"
  --lld-weight "${CCLKD_LLD_WEIGHT:-1.0}"
  --fld-weight "${CCLKD_FLD_WEIGHT:-1.0}"
  --rld-weight "${CCLKD_RLD_WEIGHT:-1.0}"
  --ccl-weight "${CCLKD_CCL_WEIGHT:-1.0}"
  --optimizer "${OPTIMIZER:-SGD}"
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
  --save-period "${SAVE_PERIOD:-100}"
)

if [[ "${EXIST_OK:-0}" == "1" ]]; then
  cmd+=(--exist-ok)
fi

echo "[$(date '+%F %T')] Launching formal online CCLKD yolo11${SIZE} seed=${SEED} gpu=${GPU_ID}"
echo "pretrain=${PRETRAIN}"
echo "sar_data=${SAR_DATA}"
echo "rgb_data=${RGB_DATA}"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, not launching."
  exit 0
fi

nohup "${cmd[@]}" > "$OUTER_LOG" 2>&1 &
pid=$!
echo "$pid" > "$PID_PATH"
echo "Launched pid=${pid}; log=${OUTER_LOG}; pid_file=${PID_PATH}"
