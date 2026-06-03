#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/ogsod_public/formal_nomosaic_20260528/run_formal_baseline.sh <sar|rgb> <n|s|m|l|x> <seed> [gpu_id]

Formal OGSOD HBB baseline protocol, 2026-05-28:
  imgsz=256, epochs=800, cos_lr=True, lr0=lrf=0.01,
  mosaic=0.0 for the full run, default Ultralytics Albumentations kept.

Fixed batch by model size:
  n:64, s:64, m:32, l:32, x:16

Examples:
  scripts/ogsod_public/formal_nomosaic_20260528/run_formal_baseline.sh sar n 0 2
  GPU_ID=5 scripts/ogsod_public/formal_nomosaic_20260528/run_formal_baseline.sh rgb m 42

Set DRY_RUN=1 to print the command without launching.
Set EXIST_OK=1 only when intentionally reusing an existing run directory.
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
  n|s) BATCH_SIZE=64 ;;
  m|l) BATCH_SIZE=32 ;;
  x) BATCH_SIZE=16 ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"
if [[ -d /root/miniconda3/bin ]]; then
  export PATH="/root/miniconda3/bin:${PATH}"
fi

MODEL="yolo11${SIZE}.pt"
DATA="configs/datasets/ogsod_hbb_${MODALITY}.yaml"
PROJECT="runs_public/ogsod/hbb/formal_nomosaic_20260528/baselines/${MODALITY}"
LOG_DIR="logs/formal_nomosaic_20260528/baselines"
META_DIR="${LOG_DIR}/metadata"
RUN_NAME="${MODALITY}_yolo11${SIZE}_hbb_800ep_cos_nomosaic_albu_b${BATCH_SIZE}_s${SEED}"
LOG_PATH="${LOG_DIR}/${RUN_NAME}_gpu${GPU_ID}.log"
PID_PATH="${LOG_DIR}/${RUN_NAME}_gpu${GPU_ID}.pid"
META_PATH="${META_DIR}/${RUN_NAME}_gpu${GPU_ID}.env"
RUN_DIR="${PROJECT}/${RUN_NAME}"

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
  python3 tools/train_ogsod_baseline.py
  --task hbb
  --model "$MODEL"
  --data "$DATA"
  --imgsz 256
  --epochs 800
  --batch "$BATCH_SIZE"
  --workers 8
  --device "$GPU_ID"
  --patience 800
  --project "$PROJECT"
  --name "$RUN_NAME"
  --lr0 0.01
  --lrf 0.01
  --cos-lr
  --mosaic 0.0
  --close-mosaic 0
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
  --save-period 100
  --seed "$SEED"
  --deterministic
)

if [[ "${EXIST_OK:-0}" == "1" ]]; then
  cmd+=(--exist-ok)
fi

{
  printf 'RUN_NAME=%q\n' "$RUN_NAME"
  printf 'MODALITY=%q\n' "$MODALITY"
  printf 'SIZE=%q\n' "$SIZE"
  printf 'SEED=%q\n' "$SEED"
  printf 'GPU_ID=%q\n' "$GPU_ID"
  printf 'BATCH_SIZE=%q\n' "$BATCH_SIZE"
  printf 'MODEL=%q\n' "$MODEL"
  printf 'DATA=%q\n' "$DATA"
  printf 'PROJECT=%q\n' "$PROJECT"
  printf 'LOG_PATH=%q\n' "$LOG_PATH"
  printf 'CMD='
  printf '%q ' "${cmd[@]}"
  printf '\n'
} > "$META_PATH"

echo "[$(date '+%F %T')] Prepared formal baseline: ${RUN_NAME}"
echo "gpu=${GPU_ID} batch=${BATCH_SIZE} log=${LOG_PATH}"
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
