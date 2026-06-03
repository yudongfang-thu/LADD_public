#!/usr/bin/env bash
set -euo pipefail

cd /home/xmu/djd/ladd

RUN_ROOT="${RUN_ROOT:-/home/xmu/djd/ladd/cold_anchor}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
PROJECT="${PROJECT:-$RUN_ROOT/runs/ogsod_cold_offline_terms}"
EPOCHS="${EPOCHS:-50}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-64}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-$RUN_ROOT/runs/ogsod_rgb_teacher/cold_anchor_rgb_yolov5x_v5p0_coco_coco_mixup010_e100_b64_5880ada_20260602_102140/weights/best.pt}"
QUEUE_LOG="${QUEUE_LOG:-$LOG_DIR/cold_offline_terms_serial_e${EPOCHS}_b${BATCH_SIZE}_${RUN_TAG}_5880ada_queue.log}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="/home/xmu/djd/ladd/.deps/python${PYTHONPATH:+:$PYTHONPATH}"

run_one() {
  local term="$1"
  local run_name="cold_v5p0_offline_rgbteacher_e100_${term}_t20_l1_topk1000_noiwm_e${EPOCHS}_b${BATCH_SIZE}_${RUN_TAG}_5880ada"
  local log_file="$LOG_DIR/${run_name}.log"
  echo "[$(date +%F_%T)] START term=${term} run=${run_name}" | tee -a "$QUEUE_LOG"
  echo "teacher_weights=${TEACHER_WEIGHTS}" | tee -a "$QUEUE_LOG"
  RUN_ROOT="$RUN_ROOT" \
  YOLOV5_WORKDIR="${YOLOV5_WORKDIR:-$RUN_ROOT/yolov5_v5p0}" \
  PYTHON_BIN="${PYTHON_BIN:-/home/xmu/miniconda3/envs/cold/bin/python}" \
  DATA_YAML="${DATA_YAML:-/home/xmu/djd/ladd/datasets/ogsod_hbb_sar.yaml}" \
  TEACHER_DATA_YAML="${TEACHER_DATA_YAML:-/home/xmu/djd/ladd/datasets/ogsod_hbb_rgb.yaml}" \
  PROJECT="$PROJECT" \
  RUN_NAME="$run_name" \
  GPU_ID="$GPU_ID" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  EFFECTIVE_BATCH_SIZE="$EFFECTIVE_BATCH_SIZE" \
  STUDENT_WEIGHTS="${STUDENT_WEIGHTS:-yolov5x.pt}" \
  TEACHER_WEIGHTS="$TEACHER_WEIGHTS" \
  TEACHER_DET_WEIGHT=0.0 \
  COLD_LOSS_MODE=candidate \
  COLD_TERMS="$term" \
  COLD_IWM_MODE=none \
  ASSERT_NONNEGATIVE_COLD=1 \
  TEMPERATURE=20.0 \
  ALPHA_NON_TARGET=2.0 \
  LAMBDA_CLS_COLD=0.0 \
  LAMBDA_LOC_COLD=1.0 \
  CANDIDATE_TOPK=1000 \
  CANDIDATE_MIN_CONF=0.001 \
  CANDIDATE_IOU_WEIGHT_FLOOR=0.0 \
  bash scripts/ogsod_public/cold_baseline_repro_20260528/run_cold_v5p0_hbb.sh 2>&1 | tee "$log_file"
  echo "[$(date +%F_%T)] DONE term=${term} run=${run_name}" | tee -a "$QUEUE_LOG"
}

run_one ncld
run_one tcld

echo "[$(date +%F_%T)] ALL DONE" | tee -a "$QUEUE_LOG"
