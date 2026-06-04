#!/usr/bin/env bash
set -euo pipefail

DATA_YAML="${DATA_YAML:-/mnt/dataY/ydf/projects/LADD_ref/configs/datasets/ogsod_hbb_sar.yaml}"
INIT="${INIT:-scratch}"
SCRIPT_PATH="${SCRIPT_PATH:-/mnt/dataY/ydf/projects/LADD_og/scripts/ogsod_public/cold_baseline_repro_20260528/run_yolov5_v5p0_baseline.sh}"
YOLOV5_WORKDIR="${YOLOV5_WORKDIR:-/mnt/dataY/ydf/projects/yolov5_cold_v5p0}"
PROJECT="${PROJECT:-/mnt/dataY/ydf/projects/LADD_ref/runs_public/ogsod/hbb/cold_anchor_repro_20260528}"
RUN_SUFFIX="${RUN_SUFFIX:-s0_mixup010}"
MIXUP="${MIXUP:-0.1}"
MAX_MEM_MB="${MAX_MEM_MB:-2000}"
MAX_UTIL="${MAX_UTIL:-15}"
POLL_SECONDS="${POLL_SECONDS:-300}"

echo "[$(date '+%F %T')] waiting for free GPU"
echo "data=${DATA_YAML}"
echo "init=${INIT} mixup=${MIXUP} project=${PROJECT}"
echo "free criterion: memory <= ${MAX_MEM_MB} MiB and util <= ${MAX_UTIL}%"

while true; do
  selected="$(
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      awk -F',' -v max_mem="$MAX_MEM_MB" -v max_util="$MAX_UTIL" '
        {
          gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
          if ($2 <= max_mem && $3 <= max_util) { print $1; exit }
        }'
  )"
  if [[ -n "$selected" ]]; then
    echo "[$(date '+%F %T')] selected GPU ${selected}"
    export YOLOV5_WORKDIR PROJECT RUN_SUFFIX MIXUP
    exec "$SCRIPT_PATH" "$DATA_YAML" "$INIT" "$selected"
  fi
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  sleep "$POLL_SECONDS"
done
