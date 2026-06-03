#!/usr/bin/env bash
set -euo pipefail

cd /home/xmu/djd/ladd

RUN_ROOT="${RUN_ROOT:-/home/xmu/djd/ladd/cold_anchor}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
PROJECT="${PROJECT:-$RUN_ROOT/runs/ogsod_cold_terms}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
RUN_SUFFIX="${RUN_SUFFIX:-nocold_coco_mixup010_e${EPOCHS}_b${BATCH_SIZE}_5880ada}"
LOG="${LOG:-$LOG_DIR/cold_anchor_sar_yolov5x_v5p0_coco_${RUN_SUFFIX}.log}"
mkdir -p "$LOG_DIR"
export PYTHONPATH="/home/xmu/djd/ladd/.deps/python${PYTHONPATH:+:$PYTHONPATH}"

YOLOV5_WORKDIR="${YOLOV5_WORKDIR:-$RUN_ROOT/yolov5_v5p0}" \
PROJECT="$PROJECT" \
PYTHON_BIN="${PYTHON_BIN:-/home/xmu/miniconda3/envs/cold/bin/python}" \
RUN_SUFFIX="$RUN_SUFFIX" \
MIXUP="${MIXUP:-0.1}" \
EPOCHS="$EPOCHS" \
bash scripts/ogsod_public/cold_baseline_repro_20260528/run_yolov5_v5p0_baseline.sh \
  "${DATA_YAML:-/home/xmu/djd/ladd/datasets/ogsod_hbb_sar.yaml}" \
  coco \
  "${GPU_ID:-0}" 2>&1 | tee "$LOG"
