#!/usr/bin/env bash
set -euo pipefail

# 90-server CoLD-only queue.
# This script assumes it is run from the isolated CoLD workspace:
#   /mnt/dataY/ydf/projects/LADD_cold_v5p0_20260603

WORKDIR="${WORKDIR:-/mnt/dataY/ydf/projects/LADD_cold_v5p0_20260603}"
cd "$WORKDIR"

RUN_ROOT="${RUN_ROOT:-$WORKDIR/cold_anchor}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
PROJECT="${PROJECT:-$RUN_ROOT/runs/ogsod_cold_offline_terms}"
YOLOV5_WORKDIR="${YOLOV5_WORKDIR:-/mnt/dataY/ydf/projects/yolov5_cold_v5p0}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
DATA_YAML="${DATA_YAML:-/mnt/dataY/ydf/projects/LADD_ref/configs/datasets/ogsod_hbb_sar.yaml}"
TEACHER_DATA_YAML="${TEACHER_DATA_YAML:-/mnt/dataY/ydf/projects/LADD_ref/configs/datasets/ogsod_hbb_rgb.yaml}"
STUDENT_WEIGHTS="${STUDENT_WEIGHTS:-/mnt/dataY/ydf/projects/LADD_ref/yolov5x.pt}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$RUN_ROOT/weights/rgb_teacher_yolov5x_v5p0_coco_mixup010_e100_117_best.pt}"

EPOCHS="${EPOCHS:-50}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-64}"
RUN_TAG="${RUN_TAG:-90_$(date +%Y%m%d_%H%M%S)}"
TERMS="${TERMS:-ncld tcld}"

TEMPERATURE="${TEMPERATURE:-20.0}"
ALPHA_NON_TARGET="${ALPHA_NON_TARGET:-2.0}"
CANDIDATE_TOPK="${CANDIDATE_TOPK:-1000}"
CANDIDATE_MIN_CONF="${CANDIDATE_MIN_CONF:-0.001}"
CANDIDATE_IOU_WEIGHT_FLOOR="${CANDIDATE_IOU_WEIGHT_FLOOR:-0.0}"
COLD_IWM_MODE="${COLD_IWM_MODE:-none}"
MAX_BATCHES="${MAX_BATCHES:-}"

QUEUE_LOG="${QUEUE_LOG:-$LOG_DIR/cold_offline_terms_serial_e${EPOCHS}_b${BATCH_SIZE}_${RUN_TAG}_90_queue.log}"
mkdir -p "$LOG_DIR" "$PROJECT"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_dir "$YOLOV5_WORKDIR" "YOLOv5 v5.0 workdir"
require_file "$DATA_YAML" "SAR dataset YAML"
require_file "$TEACHER_DATA_YAML" "RGB dataset YAML"
require_file "$STUDENT_WEIGHTS" "student init weights"
require_file "$TEACHER_WEIGHTS" "offline RGB teacher weights"
require_file "$RUN_ROOT/train_cold_v5p0_hbb.py" "CoLD training script"
require_file "$WORKDIR/scripts/ogsod_public/cold_baseline_repro_20260528/run_cold_v5p0_hbb.sh" "CoLD launcher"

run_one() {
  local term="$1"
  local run_name="cold_v5p0_offline_rgbteacher_e100_${term}_t${TEMPERATURE}_l1_topk${CANDIDATE_TOPK}_noiwm_e${EPOCHS}_b${BATCH_SIZE}_${RUN_TAG}_90"
  local log_file="$LOG_DIR/${run_name}.log"
  echo "[$(date +%F_%T)] START term=${term} run=${run_name}" | tee -a "$QUEUE_LOG"
  echo "teacher_weights=${TEACHER_WEIGHTS}" | tee -a "$QUEUE_LOG"
  RUN_ROOT="$RUN_ROOT" \
  YOLOV5_WORKDIR="$YOLOV5_WORKDIR" \
  PYTHON_BIN="$PYTHON_BIN" \
  DATA_YAML="$DATA_YAML" \
  TEACHER_DATA_YAML="$TEACHER_DATA_YAML" \
  PROJECT="$PROJECT" \
  RUN_NAME="$run_name" \
  GPU_ID="$GPU_ID" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  EFFECTIVE_BATCH_SIZE="$EFFECTIVE_BATCH_SIZE" \
  STUDENT_WEIGHTS="$STUDENT_WEIGHTS" \
  TEACHER_WEIGHTS="$TEACHER_WEIGHTS" \
  TEACHER_DET_WEIGHT=0.0 \
  COLD_LOSS_MODE=candidate \
  COLD_TERMS="$term" \
  COLD_IWM_MODE="$COLD_IWM_MODE" \
  ASSERT_NONNEGATIVE_COLD=1 \
  TEMPERATURE="$TEMPERATURE" \
  ALPHA_NON_TARGET="$ALPHA_NON_TARGET" \
  LAMBDA_CLS_COLD=0.0 \
  LAMBDA_LOC_COLD=1.0 \
  CANDIDATE_TOPK="$CANDIDATE_TOPK" \
  CANDIDATE_MIN_CONF="$CANDIDATE_MIN_CONF" \
  CANDIDATE_IOU_WEIGHT_FLOOR="$CANDIDATE_IOU_WEIGHT_FLOOR" \
  MAX_BATCHES="$MAX_BATCHES" \
  bash "$WORKDIR/scripts/ogsod_public/cold_baseline_repro_20260528/run_cold_v5p0_hbb.sh" 2>&1 | tee "$log_file"
  echo "[$(date +%F_%T)] DONE term=${term} run=${run_name}" | tee -a "$QUEUE_LOG"
}

echo "[$(date +%F_%T)] CoLD 90 serial queue start" | tee -a "$QUEUE_LOG"
echo "workdir=${WORKDIR}" | tee -a "$QUEUE_LOG"
echo "gpu=${GPU_ID} epochs=${EPOCHS} batch=${BATCH_SIZE} terms=${TERMS}" | tee -a "$QUEUE_LOG"

for term in $TERMS; do
  run_one "$term"
done

echo "[$(date +%F_%T)] ALL DONE" | tee -a "$QUEUE_LOG"
