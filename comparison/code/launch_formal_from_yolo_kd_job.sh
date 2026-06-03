#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/ogsod_public/formal_nomosaic_20260528/launch_formal_from_yolo_kd_job.sh <fgd|mgd|ld|crosskd|c2kd|mmanet|hallucidet> <n|s|m|l|x> <seed> <gpu_id>

Runs a from-YOLO-pretrain KD comparison under the formal OGSOD HBB protocol:
  student init = yolo11<size>.pt
  teacher = same-seed RGB formal baseline best.pt
  data/augmentation = formal no-mosaic baseline settings
  epochs default = 800; convergence is judged by the best checkpoint, not by train length.

Optional:
  RUN_TAG_SUFFIX=_r1
  EPOCHS_B=800
  PROFILE_KD_WEIGHT=1.0
  EXIST_OK=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

METHOD="${1:-}"
SIZE="${2:-}"
SEED="${3:-}"
GPU_ID="${4:-}"

case "$METHOD" in
  fgd|mgd|ld|crosskd|c2kd|mmanet|hallucidet) ;;
  *) usage >&2; exit 1 ;;
esac
if [[ ! "$SIZE" =~ ^(n|s|m|l|x)$ ]]; then
  usage >&2
  exit 1
fi
if [[ ! "$SEED" =~ ^[0-9]+$ || -z "$GPU_ID" ]]; then
  usage >&2
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

BASE_ROOT="runs_public/ogsod/hbb/formal_nomosaic_20260528"
PRETRAIN="yolo11${SIZE}.pt"
RGB_RUN="rgb_yolo11${SIZE}_hbb_800ep_cos_nomosaic_albu_b${BATCH_SIZE}_s${SEED}"
RGB_TEACHER="${BASE_ROOT}/baselines/rgb/${RGB_RUN}/weights/best.pt"

if [[ ! -f "$PRETRAIN" ]]; then
  echo "Missing YOLO pretrain checkpoint: $PRETRAIN" >&2
  exit 1
fi
if [[ ! -f "$RGB_TEACHER" ]]; then
  echo "Missing formal RGB teacher best.pt: $RGB_TEACHER" >&2
  exit 1
fi

RUN_TAG="formal_nomosaic_yolo11${SIZE}_${METHOD}_from_yolo_s${SEED}${RUN_TAG_SUFFIX:-}"
PROJECT_DIR="${BASE_ROOT}/comparisons/from_yolo_pretrain/yolo11${SIZE}/${METHOD}"
LOG_DIR="logs/formal_nomosaic_20260528/comparisons/from_yolo_pretrain"
PHASE_LOG_DIR="${LOG_DIR}/${RUN_TAG}_gpu${GPU_ID}"
OUTER_LOG="${LOG_DIR}/${RUN_TAG}_gpu${GPU_ID}.outer.log"
PID_PATH="${LOG_DIR}/${RUN_TAG}_gpu${GPU_ID}.pid"
mkdir -p "$PROJECT_DIR" "$PHASE_LOG_DIR" "$LOG_DIR"

RUN_NAME="transfer_${METHOD}_hbb_ogsod11${SIZE}_from_yolo_${RUN_TAG}_b_e${EPOCHS_B:-800}_b${BATCH_SIZE}_s${SEED}_gpu${GPU_ID}"
if [[ -e "${PROJECT_DIR}/${RUN_NAME}" && "${EXIST_OK:-0}" != "1" ]]; then
  echo "Run directory already exists: ${PROJECT_DIR}/${RUN_NAME}" >&2
  echo "Set EXIST_OK=1 only if intentional." >&2
  exit 1
fi

cmd=(
  env
  "MODEL=${PRETRAIN}"
  "SAR_BASELINE=${PRETRAIN}"
  "RGB_TEACHER=${RGB_TEACHER}"
  "DATA_CFG=configs/datasets/ogsod_hbb_sar.yaml"
  "TEACHER_DATA_CFG=configs/datasets/ogsod_hbb_rgb.yaml"
  "GPU_ID=${GPU_ID}"
  "SEED=${SEED}"
  "BATCH_SIZE=${BATCH_SIZE}"
  "WORKERS=8"
  "IMGSZ=256"
  "EPOCHS=${EPOCHS_B:-800}"
  "PATIENCE=${PATIENCE_B:-800}"
  "PHASE_MIN_EPOCHS=${EPOCHS_B:-800}"
  "PROJECT_DIR=${PROJECT_DIR}"
  "LOG_DIR=${PHASE_LOG_DIR}"
  "RUN_NAME=${RUN_NAME}"
  "COMPARISON_KD_PROFILE=${METHOD}"
  "PROFILE_KD_WEIGHT=${PROFILE_KD_WEIGHT:-1.0}"
  "PROFILE_KD_REPLACE_BASE=${PROFILE_KD_REPLACE_BASE:-1}"
  "STUDENT_BRANCH_MODE=raw"
  "TEACHER_FEATURE_MODE=raw"
  "USE_MASK=0"
  "USE_FG_MASK_FOR_REACH=0"
  "USE_FG_MASK_FOR_REC=0"
  "LAMBDA_REACH=0.0"
  "LAMBDA_REC=0.0"
  "LAMBDA_SEP=0.0"
  "LAMBDA_TASKL=0.0"
  "ALPHA_S_REC=0.0"
  "ALPHA_SEP=0.0"
  "RESIDUAL_AUX_MODE=none"
  "LAMBDA_RESIDUAL_AUX=0.0"
  "TEACHER_PRIVATE_AUX_MODE=none"
  "LAMBDA_TEACHER_PRIVATE_AUX=0.0"
  "COS_LR=1"
  "LR0=${B_LR0:-0.01}"
  "LRF=${B_LRF:-0.01}"
  "MOSAIC=0.0"
  "CLOSE_AT_EPOCH=${EPOCHS_B:-800}"
  "MIXUP=0.0"
  "CUTMIX=0.0"
  "DEGREES=0.0"
  "PERSPECTIVE=0.0"
  "TRANSLATE=0.1"
  "SCALE=0.5"
  "FLIPLR=0.5"
  "FLIPUD=0.0"
  "HSV_H=0.0"
  "HSV_S=0.0"
  "HSV_V=0.0"
  "ERASING=0.0"
  "SAVE_PERIOD=100"
  "EXIST_OK=${EXIST_OK:-0}"
)

case "$METHOD" in
  fgd)
    cmd+=(
      "FGD_BG_WEIGHT=${FGD_BG_WEIGHT:-0.25}"
      "FGD_RELATION_WEIGHT=${FGD_RELATION_WEIGHT:-0.1}"
    )
    ;;
  mgd)
    cmd+=("MGD_MASK_RATIO=${MGD_MASK_RATIO:-0.5}")
    ;;
  ld)
    cmd+=("CROSSKD_TEMPERATURE=${CROSSKD_TEMPERATURE:-2.0}")
    ;;
  crosskd)
    cmd+=(
      "CROSSKD_TEMPERATURE=${CROSSKD_TEMPERATURE:-2.0}"
      "CROSSKD_PRED_WEIGHT=${CROSSKD_PRED_WEIGHT:-1.0}"
      "CROSSKD_FEAT_WEIGHT=${CROSSKD_FEAT_WEIGHT:-0.25}"
      "CROSSKD_TEACHER_CONF_THRESHOLD=${CROSSKD_TEACHER_CONF_THRESHOLD:-0.25}"
    )
    ;;
  c2kd)
    cmd+=(
      "C2KD_SELECTION_THRESHOLD=${C2KD_SELECTION_THRESHOLD:-0.25}"
      "C2KD_TEACHER_CONF_THRESHOLD=${C2KD_TEACHER_CONF_THRESHOLD:-0.3}"
    )
    ;;
  mmanet)
    cmd+=(
      "MMANET_RELATION_MARGIN=${MMANET_RELATION_MARGIN:-0.2}"
      "MMANET_MAX_TOKENS=${MMANET_MAX_TOKENS:-512}"
    )
    ;;
  hallucidet)
    cmd+=(
      "HALLUCIDET_BG_WEIGHT=${HALLUCIDET_BG_WEIGHT:-0.05}"
      "HALLUCIDET_RESPONSE_WEIGHT=${HALLUCIDET_RESPONSE_WEIGHT:-0.5}"
      "HALLUCIDET_MARGIN_WEIGHT=${HALLUCIDET_MARGIN_WEIGHT:-0.1}"
      "HALLUCIDET_MARGIN=${HALLUCIDET_MARGIN:-0.2}"
    )
    ;;
esac

cmd+=(scripts/ogsod_public/run_ladd_phase.sh hbb b "$RUN_TAG")

echo "[$(date '+%F %T')] Launching formal from-YOLO KD ${METHOD} yolo11${SIZE} seed=${SEED} gpu=${GPU_ID}"
echo "student_pretrain=${PRETRAIN}"
echo "rgb_teacher=${RGB_TEACHER}"
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
