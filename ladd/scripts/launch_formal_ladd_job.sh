#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/ogsod_public/formal_nomosaic_20260528/launch_formal_ladd_job.sh <original|cap2> <n|s|m|l|x> <seed> <gpu_id>

Formal OGSOD HBB LADD protocol, 2026-05-28:
  A1=10, A2=50, B=800, imgsz=256, cos_lr in B,
  mosaic=0.0 for every phase, default Albumentations kept.

The launcher uses formal baseline checkpoints:
  runs_public/ogsod/hbb/formal_nomosaic_20260528/baselines/{sar,rgb}/...

Variants:
  original: RANK_D_NEG_CAP=4.0
  cap2:     RANK_D_NEG_CAP=2.0

Optional:
  RUN_TAG_SUFFIX=_a2lr1e3  # append a suffix to avoid overwriting prior runs
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

VARIANT="${1:-}"
SIZE="${2:-}"
SEED="${3:-}"
GPU_ID="${4:-}"

case "$VARIANT" in
  original) RANK_D_NEG_CAP=4.0 ;;
  cap2) RANK_D_NEG_CAP=2.0 ;;
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

BASE_ROOT="runs_public/ogsod/hbb/formal_nomosaic_20260528"
SAR_RUN="sar_yolo11${SIZE}_hbb_800ep_cos_nomosaic_albu_b${BATCH_SIZE}_s${SEED}"
RGB_RUN="rgb_yolo11${SIZE}_hbb_800ep_cos_nomosaic_albu_b${BATCH_SIZE}_s${SEED}"
SAR_BASELINE="${BASE_ROOT}/baselines/sar/${SAR_RUN}/weights/best.pt"
RGB_TEACHER="${BASE_ROOT}/baselines/rgb/${RGB_RUN}/weights/best.pt"

if [[ ! -f "$SAR_BASELINE" ]]; then
  echo "Missing formal SAR baseline best.pt: $SAR_BASELINE" >&2
  exit 1
fi
if [[ ! -f "$RGB_TEACHER" ]]; then
  echo "Missing formal RGB teacher best.pt: $RGB_TEACHER" >&2
  exit 1
fi

RUN_TAG="formal_nomosaic_yolo11${SIZE}_${VARIANT}_s${SEED}${RUN_TAG_SUFFIX:-}"
PROJECT_DIR="${BASE_ROOT}/ladd/yolo11${SIZE}/${VARIANT}"
CHAIN_LOG_DIR="logs/formal_nomosaic_20260528/ladd/${RUN_TAG}_gpu${GPU_ID}"
LOG_DIR="logs/formal_nomosaic_20260528/ladd"
OUTER_LOG="${LOG_DIR}/${RUN_TAG}_gpu${GPU_ID}.outer.log"
PID_PATH="${LOG_DIR}/${RUN_TAG}_gpu${GPU_ID}.pid"
mkdir -p "$PROJECT_DIR" "$CHAIN_LOG_DIR" "$LOG_DIR"

if [[ -e "${PROJECT_DIR}/ladd_hbb_ogsod11n_${RUN_TAG}_b_e800_b${BATCH_SIZE}_s${SEED}_gpu${GPU_ID}" && "${EXIST_OK:-0}" != "1" ]]; then
  echo "A likely final B run directory already exists under $PROJECT_DIR" >&2
  echo "Set EXIST_OK=1 only if intentional." >&2
  exit 1
fi

cmd=(
  env
  "SAR_BASELINE=${SAR_BASELINE}"
  "RGB_TEACHER=${RGB_TEACHER}"
  "GPU_ID=${GPU_ID}"
  "SEED=${SEED}"
  "BATCH_SIZE=${BATCH_SIZE}"
  "WORKERS=8"
  "IMGSZ=256"
  "EPOCHS_A1=10"
  "EPOCHS_A2=50"
  "EPOCHS_B=800"
  "PATIENCE_A=200"
  "PATIENCE_B=800"
  "PROJECT_DIR=${PROJECT_DIR}"
  "CHAIN_LOG_DIR=${CHAIN_LOG_DIR}"
  "MOSAIC=0.0"
  "A_CLOSE_MOSAIC=0"
  "B_CLOSE_AT_EPOCH=800"
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
  "USE_FG_MASK_FOR_REACH=1"
  "USE_FG_MASK_FOR_REC=0"
  "RANK_D_NEG_CAP=${RANK_D_NEG_CAP}"
  "LAMBDA_ANTI_COLLAPSE=0.0"
  "ANTI_COLLAPSE_FLOOR=0.0"
  "EXIST_OK=${EXIST_OK:-0}"
)

for optional_env in \
  A1_OPTIMIZER A1_LR0 A1_LRF A1_COS_LR A1_WARMUP_EPOCHS A1_WARMUP_BIAS_LR A1_WARMUP_MOMENTUM A1_DET_LOSS_SCALE \
  A2_OPTIMIZER A2_LR0 A2_LRF A2_COS_LR A2_WARMUP_EPOCHS A2_WARMUP_BIAS_LR A2_WARMUP_MOMENTUM A2_DET_LOSS_SCALE \
  B_OPTIMIZER B_LR0 B_LRF B_WARMUP_EPOCHS B_WARMUP_BIAS_LR B_WARMUP_MOMENTUM; do
  if [[ -n "${!optional_env:-}" ]]; then
    cmd+=("${optional_env}=${!optional_env}")
  fi
done

cmd+=(
  scripts/ogsod_public/run_hbb_ladd_converged_chain.sh
  "$RUN_TAG"
)

echo "[$(date '+%F %T')] Launching formal LADD ${VARIANT} yolo11${SIZE} seed=${SEED} gpu=${GPU_ID}"
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
