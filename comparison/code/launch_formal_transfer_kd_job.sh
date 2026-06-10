#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  comparison/code/launch_formal_transfer_kd_job.sh <fgd|ld|hallucidet_style> <n|s|m|l|x> <seed> <gpu_id>

Runs a B-only transferred KD comparison under the formal OGSOD HBB protocol:
  OGSOD-1.0 HBB, imgsz=256, epochs=800, full no-mosaic, default Albumentations.

These are method-style transferred baselines, not official reproductions:
  fgd    - generic detector KD, foreground/background feature weighting + relation
  ld     - generic detector KD, YOLO DFL localization-distribution KL
  hallucidet_style - task-driven privileged-modality feature/response/margin baseline

Optional:
  RUN_TAG_SUFFIX=_debug
  EPOCHS_B=20
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
  fgd|ld|hallucidet_style) ;;
  hallucidet)
    echo "Use hallucidet_style for the current feature/response/margin portable baseline. Strict HalluciDet is not implemented in this launcher." >&2
    exit 2
    ;;
  cclkd)
    echo "CCLKD is an online teacher-student method; use cclkd_reproduction/ and the future online launcher." >&2
    exit 2
    ;;
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

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
if [[ -d /root/miniconda3/bin ]]; then
  export PATH="/root/miniconda3/bin:${PATH}"
fi

BASE_ROOT="runs_public/ogsod/hbb/formal_nomosaic_20260528"
SAR_RUN="sar_yolo11${SIZE}_hbb_800ep_cos_nomosaic_albu_b${BATCH_SIZE}_s${SEED}"
RGB_RUN="rgb_yolo11${SIZE}_hbb_800ep_cos_nomosaic_albu_b${BATCH_SIZE}_s${SEED}"
SAR_BASELINE="${BASE_ROOT}/baselines/sar/${SAR_RUN}/weights/best.pt"
RGB_TEACHER="${BASE_ROOT}/baselines/rgb/${RGB_RUN}/weights/best.pt"

if [[ "${DRY_RUN:-0}" != "1" && ! -f "$SAR_BASELINE" ]]; then
  echo "Missing formal SAR baseline best.pt: $SAR_BASELINE" >&2
  exit 1
fi
if [[ "${DRY_RUN:-0}" != "1" && ! -f "$RGB_TEACHER" ]]; then
  echo "Missing formal RGB teacher best.pt: $RGB_TEACHER" >&2
  exit 1
fi

COMPARISON_IMPL_VERSION="${COMPARISON_IMPL_VERSION:-v2_20260610}"
RUN_TAG="formal_nomosaic_yolo11${SIZE}_${METHOD}_${COMPARISON_IMPL_VERSION}_transfer_s${SEED}${RUN_TAG_SUFFIX:-}"
PROJECT_DIR="${BASE_ROOT}/comparisons/transferred_kd/yolo11${SIZE}/${METHOD}"
LOG_DIR="logs/formal_nomosaic_20260528/comparisons/transferred_kd"
PHASE_LOG_DIR="${LOG_DIR}/${RUN_TAG}_gpu${GPU_ID}"
OUTER_LOG="${LOG_DIR}/${RUN_TAG}_gpu${GPU_ID}.outer.log"
PID_PATH="${LOG_DIR}/${RUN_TAG}_gpu${GPU_ID}.pid"
mkdir -p "$PROJECT_DIR" "$PHASE_LOG_DIR" "$LOG_DIR"

RUN_NAME="transfer_${METHOD}_hbb_ogsod11${SIZE}_${RUN_TAG}_b_e${EPOCHS_B:-800}_b${BATCH_SIZE}_s${SEED}_gpu${GPU_ID}"
if [[ -e "${PROJECT_DIR}/${RUN_NAME}" && "${EXIST_OK:-0}" != "1" ]]; then
  echo "Run directory already exists: ${PROJECT_DIR}/${RUN_NAME}" >&2
  echo "Set EXIST_OK=1 only if intentional." >&2
  exit 1
fi

cmd=(
  env
  "MODEL=${SAR_BASELINE}"
  "SAR_BASELINE=${SAR_BASELINE}"
  "RGB_TEACHER=${RGB_TEACHER}"
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
      "FGD_ALPHA=${FGD_ALPHA:-0.001}"
      "FGD_BETA=${FGD_BETA:-0.0005}"
      "FGD_GAMMA=${FGD_GAMMA:-0.001}"
      "FGD_LAMBDA=${FGD_LAMBDA:-${FGD_RELATION_WEIGHT:-0.0}}"
      "FGD_TEMPERATURE=${FGD_TEMPERATURE:-0.5}"
      "FGD_MASK_MODE=${FGD_MASK_MODE:-gt_box}"
      "FGD_BG_NORM=${FGD_BG_NORM:-1}"
    )
    ;;
  ld)
    cmd+=(
      "LD_TEMPERATURE=${LD_TEMPERATURE:-10.0}"
      "LD_USE_VLR=${LD_USE_VLR:-1}"
      "LD_QUALITY_POWER=${LD_QUALITY_POWER:-1.0}"
      "LD_MIN_VLR_WEIGHT=${LD_MIN_VLR_WEIGHT:-0.0}"
      "LD_VLR_TOPK=${LD_VLR_TOPK:-0}"
      "LD_VLR_WEIGHT=${LD_VLR_WEIGHT:-0.25}"
      "LD_MAIN_WEIGHT=${LD_MAIN_WEIGHT:-0.25}"
      "LD_ALLOW_EMPTY_VLR=${LD_ALLOW_EMPTY_VLR:-1}"
    )
    ;;
  hallucidet_style)
    cmd+=(
      "HALLUCIDET_BG_WEIGHT=${HALLUCIDET_BG_WEIGHT:-0.05}"
      "HALLUCIDET_RESPONSE_WEIGHT=${HALLUCIDET_RESPONSE_WEIGHT:-0.5}"
      "HALLUCIDET_MARGIN_WEIGHT=${HALLUCIDET_MARGIN_WEIGHT:-0.1}"
      "HALLUCIDET_MARGIN=${HALLUCIDET_MARGIN:-0.2}"
    )
    ;;
esac

cmd+=(scripts/ogsod_public/run_ladd_phase.sh hbb b "$RUN_TAG")

echo "[$(date '+%F %T')] Launching formal transferred KD ${METHOD} yolo11${SIZE} seed=${SEED} gpu=${GPU_ID}"
echo "sar_baseline=${SAR_BASELINE}"
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
