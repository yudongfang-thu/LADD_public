#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/LADD

RUN_NAME="${RUN_NAME:-cold_v5p0_yolov5x_coco_mixup010_candidate_topk200_t1_loccold50_iwmfloor02_4090d}"
LOG="${LOG:-/root/autodl-tmp/cold_anchor/logs/${RUN_NAME}.log}"

RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/cold_anchor}" \
YOLOV5_WORKDIR="${YOLOV5_WORKDIR:-/root/autodl-tmp/cold_anchor/yolov5_v5p0}" \
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}" \
PROJECT="${PROJECT:-/root/autodl-tmp/cold_anchor/runs/ogsod_cold_repro}" \
RUN_NAME="$RUN_NAME" \
GPU_ID="${GPU_ID:-0}" \
EPOCHS="${EPOCHS:-400}" \
BATCH_SIZE="${BATCH_SIZE:-64}" \
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-64}" \
COLD_LOSS_MODE="${COLD_LOSS_MODE:-candidate}" \
TEMPERATURE="${TEMPERATURE:-1.0}" \
LAMBDA_LOC_COLD="${LAMBDA_LOC_COLD:-50.0}" \
CANDIDATE_TOPK="${CANDIDATE_TOPK:-200}" \
CANDIDATE_MIN_CONF="${CANDIDATE_MIN_CONF:-0.001}" \
CANDIDATE_IOU_WEIGHT_FLOOR="${CANDIDATE_IOU_WEIGHT_FLOOR:-0.2}" \
bash scripts/ogsod_public/cold_baseline_repro_20260528/run_cold_v5p0_hbb.sh 2>&1 | tee "$LOG"
