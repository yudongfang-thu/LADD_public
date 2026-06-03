#!/usr/bin/env bash
set -euo pipefail

cd /home/xmu/djd/ladd

RUN_ROOT="${RUN_ROOT:-/home/xmu/djd/ladd/cold_anchor}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
PROJECT="${PROJECT:-$RUN_ROOT/runs/ogsod_cold_terms}"
EPOCHS="${EPOCHS:-50}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-64}"
QUEUE_LOG="${QUEUE_LOG:-$LOG_DIR/cold_terms_line_e${EPOCHS}_5880ada_queue.log}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="/home/xmu/djd/ladd/.deps/python${PYTHONPATH:+:$PYTHONPATH}"

run_one() {
  local term="$1"
  local run_name="cold_v5p0_terms_${term}_t20_l1_topk1000_noiwm_e${EPOCHS}_b${BATCH_SIZE}_5880ada"
  local log_file="$LOG_DIR/${run_name}.log"
  echo "[$(date +%F_%T)] START term=${term} run=${run_name}" | tee -a "$QUEUE_LOG"
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
  COLD_LOSS_MODE=candidate \
  COLD_TERMS="$term" \
  COLD_IWM_MODE=none \
  ASSERT_NONNEGATIVE_COLD=1 \
  TEMPERATURE=20.0 \
  ALPHA_NON_TARGET=2.0 \
  LAMBDA_LOC_COLD=1.0 \
  CANDIDATE_TOPK=1000 \
  CANDIDATE_MIN_CONF=0.001 \
  CANDIDATE_IOU_WEIGHT_FLOOR=0.0 \
  bash scripts/ogsod_public/cold_baseline_repro_20260528/run_cold_v5p0_hbb.sh 2>&1 | tee "$log_file"
  echo "[$(date +%F_%T)] DONE term=${term} run=${run_name}" | tee -a "$QUEUE_LOG"
}

run_one tcld
run_one ncld
run_one both

echo "[$(date +%F_%T)] ALL DONE" | tee -a "$QUEUE_LOG"
