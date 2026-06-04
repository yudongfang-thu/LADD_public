#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/home/xmu/djd/ladd/cold_anchor"
LOG_DIR="$RUN_ROOT/logs"
PROJECT="$RUN_ROOT/runs/ogsod_cold_cpm_ablation"
EPOCHS=50
GPU_ID=0
BATCH_SIZE=64
QUEUE_LOG="$LOG_DIR/cpm_ablation_e${EPOCHS}_queue.log"
mkdir -p "$LOG_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/xmu/miniconda3/bin/python3}"
# fallback: find python3
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(which python3)"
fi

run_one() {
  local term="$1"
  local lambda_loc="$2"
  local run_name
  local cold_terms="$term"
  if [[ "$term" == "baseline" ]]; then
    run_name="cold_v5p0_cpm_ablation_baseline_nocold_e${EPOCHS}_b${BATCH_SIZE}"
    cold_terms="both"  # dummy; loss multiplied by 0
  else
    run_name="cold_v5p0_cpm_ablation_${term}_t1_l${lambda_loc}_topk1000_noiwm_e${EPOCHS}_b${BATCH_SIZE}"
  fi
  local log_file="$LOG_DIR/${run_name}.log"
  echo "[$(date +%F_%T)] START term=${term} run=${run_name}" | tee -a "$QUEUE_LOG"

  RUN_ROOT="$RUN_ROOT" \
  YOLOV5_WORKDIR="$RUN_ROOT/yolov5_v5p0" \
  PYTHON_BIN="$PYTHON_BIN" \
  DATA_YAML="/home/xmu/djd/ladd/datasets/ogsod_hbb_sar.yaml" \
  TEACHER_DATA_YAML="/home/xmu/djd/ladd/datasets/ogsod_hbb_rgb.yaml" \
  PROJECT="$PROJECT" \
  RUN_NAME="$run_name" \
  GPU_ID="$GPU_ID" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  EFFECTIVE_BATCH_SIZE="$BATCH_SIZE" \
  COLD_LOSS_MODE=candidate \
  COLD_TERMS="$cold_terms" \
  COLD_IWM_MODE=none \
  TEACHER_DET_WEIGHT=0.0 \
  LAMBDA_CLS_COLD=0.0 \
  LAMBDA_LOC_COLD="$lambda_loc" \
  TEMPERATURE=1.0 \
  ALPHA_NON_TARGET=2.0 \
  CANDIDATE_TOPK=1000 \
  CANDIDATE_MIN_CONF=0.001 \
  CANDIDATE_IOU_WEIGHT_FLOOR=0.0 \
  ASSERT_NONNEGATIVE_COLD=1 \
  bash "$RUN_ROOT/run_cold_v5p0_hbb.sh" 2>&1 | tee "$log_file"
  echo "[$(date +%F_%T)] DONE term=${term} run=${run_name}" | tee -a "$QUEUE_LOG"
}

echo "[$(date +%F_%T)] ===== CPM ablation queue: baseline -> tcld -> ncld -> both =====" | tee "$QUEUE_LOG"
echo "Server: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)" | tee -a "$QUEUE_LOG"
echo "Python: $PYTHON_BIN" | tee -a "$QUEUE_LOG"

# Step 1: baseline (no CoLD)
run_one baseline 0.0

# Step 2: TCLD only
run_one tcld 1.0

# Step 3: NCLD only
run_one ncld 1.0

# Step 4: TCLD + NCLD
run_one both 1.0

echo "[$(date +%F_%T)] ALL DONE" | tee -a "$QUEUE_LOG"
