#!/usr/bin/env bash
set -euo pipefail

cd /home/xmu/djd/ladd

RUN_ROOT="${RUN_ROOT:-/home/xmu/djd/ladd/cold_anchor}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-64}"
GPU_ID="${GPU_ID:-0}"
QUEUE_LOG="${QUEUE_LOG:-$LOG_DIR/cold_four_point_2parallel_e${EPOCHS}_b${BATCH_SIZE}_5880ada_queue.log}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="/home/xmu/djd/ladd/.deps/python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

run_term() {
  local term="$1"
  local run_name="cold_v5p0_terms_${term}_t20_l1_topk1000_noiwm_e${EPOCHS}_b${BATCH_SIZE}_5880ada"
  local log_file="$LOG_DIR/${run_name}.log"
  echo "[$(date +%F_%T)] START term=${term} run=${run_name}" | tee -a "$QUEUE_LOG"
  RUN_ROOT="$RUN_ROOT" \
  YOLOV5_WORKDIR="${YOLOV5_WORKDIR:-$RUN_ROOT/yolov5_v5p0}" \
  PYTHON_BIN="${PYTHON_BIN:-/home/xmu/miniconda3/envs/cold/bin/python}" \
  DATA_YAML="${DATA_YAML:-/home/xmu/djd/ladd/datasets/ogsod_hbb_sar.yaml}" \
  TEACHER_DATA_YAML="${TEACHER_DATA_YAML:-/home/xmu/djd/ladd/datasets/ogsod_hbb_rgb.yaml}" \
  PROJECT="${PROJECT:-$RUN_ROOT/runs/ogsod_cold_terms}" \
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
  bash scripts/ogsod_public/cold_baseline_repro_20260528/run_cold_v5p0_hbb.sh \
    2>&1 | tee "$log_file"
  echo "[$(date +%F_%T)] DONE term=${term} run=${run_name}" | tee -a "$QUEUE_LOG"
}

run_nocold() {
  echo "[$(date +%F_%T)] START no-CoLD" | tee -a "$QUEUE_LOG"
  RUN_ROOT="$RUN_ROOT" \
  LOG_DIR="$LOG_DIR" \
  PROJECT="${PROJECT:-$RUN_ROOT/runs/ogsod_cold_terms}" \
  GPU_ID="$GPU_ID" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  bash scripts/ogsod_public/cold_baseline_repro_20260528/launch_yolov5_v5p0_baseline_5880ada.sh \
    2>&1 | tee -a "$QUEUE_LOG"
  echo "[$(date +%F_%T)] DONE no-CoLD" | tee -a "$QUEUE_LOG"
}

wait_wave() {
  local name="$1"
  local pids=("${@:2}")
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" != "0" ]]; then
    echo "[$(date +%F_%T)] WAVE FAILED ${name}" | tee -a "$QUEUE_LOG"
    exit "$status"
  fi
  echo "[$(date +%F_%T)] WAVE DONE ${name}" | tee -a "$QUEUE_LOG"
}

wait_for_log() {
  local label="$1"
  local file="$2"
  local pattern="$3"
  local timeout="${4:-900}"
  local elapsed=0
  while (( elapsed < timeout )); do
    if [[ -f "$file" ]] && grep -q "$pattern" "$file"; then
      echo "[$(date +%F_%T)] READY ${label}: ${pattern}" | tee -a "$QUEUE_LOG"
      return 0
    fi
    sleep 10
    elapsed=$((elapsed + 10))
  done
  echo "[$(date +%F_%T)] READY TIMEOUT ${label}: ${pattern}" | tee -a "$QUEUE_LOG"
  return 0
}

echo "[$(date +%F_%T)] FOUR-POINT 2-PARALLEL START epochs=${EPOCHS} batch=${BATCH_SIZE}" | tee -a "$QUEUE_LOG"

run_term tcld &
pid_a=$!
wait_for_log "tcld" "$LOG_DIR/cold_v5p0_terms_tcld_t20_l1_topk1000_noiwm_e${EPOCHS}_b${BATCH_SIZE}_5880ada.log" "Starting native YOLOv5-v5.0"
run_term ncld &
pid_b=$!
wait_wave "tcld+ncld" "$pid_a" "$pid_b"

run_nocold &
pid_c=$!
wait_for_log "no-CoLD" "$LOG_DIR/cold_anchor_sar_yolov5x_v5p0_coco_nocold_coco_mixup010_e${EPOCHS}_b${BATCH_SIZE}_5880ada.log" "Image sizes"
run_term both &
pid_d=$!
wait_wave "nocold+both" "$pid_c" "$pid_d"

echo "[$(date +%F_%T)] FOUR-POINT 50ep ALL DONE" | tee -a "$QUEUE_LOG"

if [[ "${ENABLE_400EP:-0}" == "1" ]]; then
  echo "[$(date +%F_%T)] === WAVE 3 (400ep) START ===" | tee -a "$QUEUE_LOG"
  EPOCHS=400 BATCH_SIZE=64 EFFECTIVE_BATCH_SIZE=64
  QUEUE_LOG="$LOG_DIR/cold_four_point_2parallel_e400_b64_5880ada_queue.log"
  run_nocold &
  pid_e=$!
  wait_for_log "no-CoLD-400" "$LOG_DIR/cold_anchor_sar_yolov5x_v5p0_coco_nocold_coco_mixup010_e400_b64_5880ada.log" "Image sizes"
  run_term both &
  pid_f=$!
  wait_wave "400ep_nocold+both" "$pid_e" "$pid_f"
  echo "[$(date +%F_%T)] WAVE 3 (400ep) ALL DONE" | tee -a "$QUEUE_LOG"
fi

echo "[$(date +%F_%T)] FOUR-POINT ALL DONE" | tee -a "$QUEUE_LOG"
