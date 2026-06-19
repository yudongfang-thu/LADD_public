#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash baseline/scripts/run_formal_baseline.sh <sar|rgb> <n|s|m|l|x> <seed> [gpu_id]

Formal OGSOD HBB baseline protocol:
  imgsz=256, epochs=800, cos_lr=True, lr0=lrf=0.01,
  default Ultralytics Albumentations kept.

Protocols:
  PROTOCOL=nomosaic    mosaic=0.0, close_mosaic=0
  PROTOCOL=mosaic100   mosaic=1.0, close_mosaic=700
                       (800 epochs with mosaic enabled for the first 100 epochs)

Fixed batch by model size:
  n:64, s:64, m:32, l:32, x:16

Examples:
  bash baseline/scripts/run_formal_baseline.sh sar n 0 2
  PROTOCOL=nomosaic bash baseline/scripts/run_formal_baseline.sh rgb m 0 5

Set DRY_RUN=1 to print the command without launching.
Set EXIST_OK=1 only when intentionally reusing an existing run directory.
Optional overrides: DATA_CFG, MODEL, RUN_NAME, PROJECT, LOG_DIR, EPOCHS, BATCH_SIZE.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODALITY="${1:-}"
SIZE="${2:-}"
SEED="${3:-}"
GPU_ID="${4:-${GPU_ID:-}}"

if [[ "$MODALITY" != "sar" && "$MODALITY" != "rgb" ]]; then
  usage >&2
  exit 1
fi
if [[ ! "$SIZE" =~ ^(n|s|m|l|x)$ ]]; then
  usage >&2
  exit 1
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  usage >&2
  exit 1
fi
if [[ -z "$GPU_ID" ]]; then
  echo "Missing gpu_id. Pass it as the fourth argument or set GPU_ID." >&2
  exit 1
fi

case "$SIZE" in
  n|s) BATCH_SIZE_DEFAULT=64 ;;
  m|l) BATCH_SIZE_DEFAULT=32 ;;
  x) BATCH_SIZE_DEFAULT=16 ;;
esac
BATCH_SIZE="${BATCH_SIZE:-$BATCH_SIZE_DEFAULT}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
if [[ -d /root/miniconda3/bin ]]; then
  export PATH="/root/miniconda3/bin:${PATH}"
fi

PROTOCOL="${PROTOCOL:-nomosaic}"
case "$PROTOCOL" in
  nomosaic|formal_nomosaic)
    PROTOCOL_KEY="nomosaic"
    MOSAIC_VALUE="0.0"
    CLOSE_MOSAIC_VALUE="0"
    PROJECT_DEFAULT="runs_public/ogsod/hbb/formal_nomosaic_20260528/baselines/${MODALITY}"
    LOG_DIR_DEFAULT="logs/formal_nomosaic_20260528/baselines"
    RUN_NAME_DEFAULT="${MODALITY}_yolo11${SIZE}_hbb_800ep_cos_nomosaic_albu_b${BATCH_SIZE}_s${SEED}"
    ;;
  mosaic100|mosaic_first100_close700)
    PROTOCOL_KEY="mosaic100"
    MOSAIC_VALUE="1.0"
    CLOSE_MOSAIC_VALUE="700"
    PROJECT_DEFAULT="runs_public/ogsod/hbb/baseline_controls/mosaic_baselines_20260615"
    LOG_DIR_DEFAULT="logs/baseline_controls/mosaic_baselines_20260615"
    RUN_NAME_DEFAULT="${MODALITY}_yolo11${SIZE}_hbb_mosaicE800_closeAt100_s${SEED}_gpu${GPU_ID}_b${BATCH_SIZE}"
    ;;
  *)
    echo "Unknown PROTOCOL=${PROTOCOL}. Use nomosaic or mosaic100." >&2
    exit 1
    ;;
esac

if [[ "${PAPER_RUN:-0}" == "1" && "$PROTOCOL_KEY" != "mosaic100" ]]; then
  echo "PAPER_RUN=1 requires PROTOCOL=mosaic100; got ${PROTOCOL}." >&2
  exit 2
fi

MODEL="${MODEL:-yolo11${SIZE}.pt}"
DATA="${DATA_CFG:-shared/configs/datasets_public/ogsod1_${MODALITY}_detect.yaml}"
PROJECT="${PROJECT:-$PROJECT_DEFAULT}"
LOG_DIR="${LOG_DIR:-$LOG_DIR_DEFAULT}"
META_DIR="${LOG_DIR}/metadata"
RUN_NAME="${RUN_NAME:-$RUN_NAME_DEFAULT}"
LOG_PATH="${LOG_DIR}/${RUN_NAME}_gpu${GPU_ID}.log"
PID_PATH="${LOG_DIR}/${RUN_NAME}_gpu${GPU_ID}.pid"
META_PATH="${META_DIR}/${RUN_NAME}_gpu${GPU_ID}.env"
RUN_DIR="${PROJECT}/${RUN_NAME}"
EPOCHS_VALUE="${EPOCHS:-800}"
PATIENCE_VALUE="${PATIENCE:-$EPOCHS_VALUE}"
SAVE_PERIOD_VALUE="${SAVE_PERIOD:-100}"

mkdir -p "$PROJECT" "$LOG_DIR" "$META_DIR"

if [[ ! -f "$MODEL" ]]; then
  echo "Model weights not found locally: ${ROOT_DIR}/${MODEL}; Ultralytics will try to download ${MODEL}." >&2
fi
if [[ ! -f "$DATA" ]]; then
  echo "Missing dataset YAML: ${ROOT_DIR}/${DATA}" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" && "${EXIST_OK:-0}" != "1" ]]; then
  echo "Run directory already exists: ${ROOT_DIR}/${RUN_DIR}" >&2
  echo "Set EXIST_OK=1 only if this is intentional." >&2
  exit 1
fi

cmd=(
  python3 baseline/code/train_ogsod_baseline.py
  --task hbb
  --model "$MODEL"
  --data "$DATA"
  --imgsz 256
  --epochs "$EPOCHS_VALUE"
  --batch "$BATCH_SIZE"
  --workers 8
  --device "$GPU_ID"
  --patience "$PATIENCE_VALUE"
  --project "$PROJECT"
  --name "$RUN_NAME"
  --lr0 0.01
  --lrf 0.01
  --cos-lr
  --mosaic "$MOSAIC_VALUE"
  --close-mosaic "$CLOSE_MOSAIC_VALUE"
  --translate 0.1
  --scale 0.5
  --fliplr 0.5
  --flipud 0.0
  --degrees 0.0
  --perspective 0.0
  --hsv-h 0.0
  --hsv-s 0.0
  --hsv-v 0.0
  --mixup 0.0
  --cutmix 0.0
  --erasing 0.0
  --save-period "$SAVE_PERIOD_VALUE"
  --seed "$SEED"
  --deterministic
)

if [[ "${EXIST_OK:-0}" == "1" ]]; then
  cmd+=(--exist-ok)
fi

{
  printf 'RUN_NAME=%q\n' "$RUN_NAME"
  printf 'PAPER_RUN=%q\n' "${PAPER_RUN:-0}"
  printf 'PAPER_PROTOCOL_ID=%q\n' "${PAPER_PROTOCOL_ID:-}"
  printf 'paper_protocol_id=%q\n' "${PAPER_PROTOCOL_ID:-}"
  printf 'protocol_id=%q\n' "${PAPER_PROTOCOL_ID:-}"
  printf 'PROTOCOL=%q\n' "$PROTOCOL_KEY"
  printf 'MODALITY=%q\n' "$MODALITY"
  printf 'SIZE=%q\n' "$SIZE"
  printf 'SEED=%q\n' "$SEED"
  printf 'GPU_ID=%q\n' "$GPU_ID"
  printf 'BATCH_SIZE=%q\n' "$BATCH_SIZE"
  printf 'EPOCHS=%q\n' "$EPOCHS_VALUE"
  printf 'PATIENCE=%q\n' "$PATIENCE_VALUE"
  printf 'MODEL=%q\n' "$MODEL"
  printf 'DATA=%q\n' "$DATA"
  printf 'PROJECT=%q\n' "$PROJECT"
  printf 'LOG_PATH=%q\n' "$LOG_PATH"
  printf 'MOSAIC=%q\n' "$MOSAIC_VALUE"
  printf 'CLOSE_MOSAIC=%q\n' "$CLOSE_MOSAIC_VALUE"
  printf 'CMD='
  printf '%q ' "${cmd[@]}"
  printf '\n'
} > "$META_PATH"

echo "[$(date '+%F %T')] Prepared ${PROTOCOL_KEY} baseline: ${RUN_NAME}"
echo "gpu=${GPU_ID} batch=${BATCH_SIZE} epochs=${EPOCHS_VALUE} mosaic=${MOSAIC_VALUE} close_mosaic=${CLOSE_MOSAIC_VALUE} log=${LOG_PATH}"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, not launching."
  exit 0
fi

nohup env PYTHONUNBUFFERED=1 "${cmd[@]}" > "$LOG_PATH" 2>&1 &
pid=$!
echo "$pid" > "$PID_PATH"
echo "Launched pid=${pid}; pid_file=${PID_PATH}"
