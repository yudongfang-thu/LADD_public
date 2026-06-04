#!/usr/bin/env bash
set -euo pipefail

cd /home/xmu/djd/ladd

TERM_SET="${1:-ncld}"
EPOCHS_ARG="${2:-1}"
MAX_BATCHES_ARG="${3:-3}"

if [[ "$TERM_SET" != "tcld" && "$TERM_SET" != "ncld" && "$TERM_SET" != "both" ]]; then
  echo "Usage: $0 <tcld|ncld|both> [epochs] [max_batches]" >&2
  exit 2
fi

RUN_ROOT="${RUN_ROOT:-/home/xmu/djd/ladd/cold_anchor}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="/home/xmu/djd/ladd/.deps/python${PYTHONPATH:+:$PYTHONPATH}"

RUN_NAME="${RUN_NAME:-cold_v5p0_terms_${TERM_SET}_t20_l1_topk1000_noiwm_e${EPOCHS_ARG}_mb${MAX_BATCHES_ARG}_5880ada}"
LOG="${LOG:-$LOG_DIR/${RUN_NAME}.log}"

RUN_ROOT="$RUN_ROOT" \
YOLOV5_WORKDIR="${YOLOV5_WORKDIR:-$RUN_ROOT/yolov5_v5p0}" \
PYTHON_BIN="${PYTHON_BIN:-/home/xmu/miniconda3/envs/cold/bin/python}" \
DATA_YAML="${DATA_YAML:-/home/xmu/djd/ladd/datasets/ogsod_hbb_sar.yaml}" \
TEACHER_DATA_YAML="${TEACHER_DATA_YAML:-/home/xmu/djd/ladd/datasets/ogsod_hbb_rgb.yaml}" \
PROJECT="${PROJECT:-$RUN_ROOT/runs/ogsod_cold_terms}" \
RUN_NAME="$RUN_NAME" \
GPU_ID="${GPU_ID:-0}" \
EPOCHS="$EPOCHS_ARG" \
BATCH_SIZE="${BATCH_SIZE:-8}" \
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-8}" \
COLD_LOSS_MODE="${COLD_LOSS_MODE:-candidate}" \
COLD_TERMS="$TERM_SET" \
COLD_IWM_MODE="${COLD_IWM_MODE:-none}" \
ASSERT_NONNEGATIVE_COLD="${ASSERT_NONNEGATIVE_COLD:-1}" \
TEMPERATURE="${TEMPERATURE:-20.0}" \
ALPHA_NON_TARGET="${ALPHA_NON_TARGET:-2.0}" \
LAMBDA_LOC_COLD="${LAMBDA_LOC_COLD:-1.0}" \
CANDIDATE_TOPK="${CANDIDATE_TOPK:-1000}" \
CANDIDATE_MIN_CONF="${CANDIDATE_MIN_CONF:-0.001}" \
CANDIDATE_IOU_WEIGHT_FLOOR="${CANDIDATE_IOU_WEIGHT_FLOOR:-0.0}" \
MAX_BATCHES="$MAX_BATCHES_ARG" \
bash scripts/ogsod_public/cold_baseline_repro_20260528/run_cold_v5p0_hbb.sh 2>&1 | tee "$LOG"
