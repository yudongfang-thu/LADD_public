#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/LADD_public}"
cd "${REPO_ROOT}"

TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-64}"
IMGSZ="${IMGSZ:-640}"
SEED="${SEED:-0}"
WORKERS="${WORKERS:-8}"
GPU_FREE_MIN_MB="${GPU_FREE_MIN_MB:-15000}"

PROBE_NAME="${PROBE_NAME:-native_noalign_nokd}"
STUDENT_MODALITY="${STUDENT_MODALITY:-rgb}"
TEACHER_MODALITY="${TEACHER_MODALITY:-ir}"
ALIGNED_NO_GEO="${ALIGNED_NO_GEO:-0}"
KEEP_COLOR_AUG="${KEEP_COLOR_AUG:-0}"
PAIRED_SYNC_GEO="${PAIRED_SYNC_GEO:-0}"
FEATURE_WEIGHT="${FEATURE_WEIGHT:-0.0}"
RELATION_WEIGHT="${RELATION_WEIGHT:-0.0}"
LOGIT_WEIGHT="${LOGIT_WEIGHT:-0.0}"
FEATURE_LAYERS="${FEATURE_LAYERS:-shallow_deep}"
RELATION_LAYER="${RELATION_LAYER:-deepest}"
FEATURE_ADAPT="${FEATURE_ADAPT:-1}"
RAW_OUTPUT_KD="${RAW_OUTPUT_KD:-0}"
KD_WARMUP_EPOCHS="${KD_WARMUP_EPOCHS:-0.0}"
KD_GAIN="${KD_GAIN:-1.0}"

WATCH_POLL_SECONDS="${WATCH_POLL_SECONDS:-300}"
WATCH_MIN_EPOCH="${WATCH_MIN_EPOCH:-120}"
WATCH_WINDOW="${WATCH_WINDOW:-35}"
WATCH_MIN_BEST_MAP50="${WATCH_MIN_BEST_MAP50:-0.62}"
WATCH_MIN_IMPROVEMENT="${WATCH_MIN_IMPROVEMENT:-0.008}"

LOG_DIR="${REPO_ROOT}/comparison/cmdistill/native_reproduction/logs/vedai_yolov5_cmdistill_alignment_probe/${TS}"
PROJECT="${REPO_ROOT}/comparison/cmdistill/native_reproduction/runs/vedai_yolov5_cmdistill_alignment_probe/paper80_seed0"
NAME="vedai512_${STUDENT_MODALITY}_${PROBE_NAME}_yolov5s_e${EPOCHS}_b${BATCH}_img${IMGSZ}_s${SEED}_${TS}"
SCREEN_NAME="${SCREEN_NAME:-cmdi_align_${PROBE_NAME}_${TS}}"
WATCH_SCREEN="${WATCH_SCREEN:-cmdi_align_watch_${PROBE_NAME}_${TS}}"
RUN_LOG="${LOG_DIR}/${PROBE_NAME}.log"
WATCH_LOG="${LOG_DIR}/${PROBE_NAME}_watch.log"
MASTER_LOG="${LOG_DIR}/master.log"
RESULTS_CSV="${PROJECT}/${NAME}/results.csv"

mkdir -p "${LOG_DIR}" "${PROJECT}"

wait_for_gpu_cmd="
while true; do
  free_mb=\$(nvidia-smi -i '${DEVICE}' --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  used_mb=\$(nvidia-smi -i '${DEVICE}' --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  util=\$(nvidia-smi -i '${DEVICE}' --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  echo \"[\$(date '+%F %T')] gpu${DEVICE}: used=\${used_mb}MiB free=\${free_mb}MiB util=\${util}%\" | tee -a '${MASTER_LOG}'
  if (( free_mb >= ${GPU_FREE_MIN_MB} )); then
    break
  fi
  sleep 120
done
"

screen -dmS "${SCREEN_NAME}" bash -lc "
set -euo pipefail
cd '${REPO_ROOT}'
${wait_for_gpu_cmd}
echo \"[\$(date '+%F %T')] start ${PROBE_NAME} name=${NAME}\" | tee -a '${MASTER_LOG}'
PYTHON=/root/miniconda3/bin/python \
YOLOV5_DIR=/root/autodl-tmp/yolov5-v6.2 \
STUDENT_MODALITY='${STUDENT_MODALITY}' \
TEACHER_MODALITY='${TEACHER_MODALITY}' \
STUDENT_WEIGHTS=/root/autodl-tmp/yolov5-v6.2/yolov5s.pt \
EPOCHS='${EPOCHS}' \
BATCH='${BATCH}' \
IMGSZ='${IMGSZ}' \
WORKERS='${WORKERS}' \
DEVICE='${DEVICE}' \
SEED='${SEED}' \
PROJECT='${PROJECT}' \
NAME='${NAME}' \
RUN_TAG='${PROBE_NAME}_${TS}' \
ALIGNED_NO_GEO='${ALIGNED_NO_GEO}' \
KEEP_COLOR_AUG='${KEEP_COLOR_AUG}' \
PAIRED_SYNC_GEO='${PAIRED_SYNC_GEO}' \
VAL_INTERVAL=1 \
PATIENCE='${EPOCHS}' \
FEATURE_WEIGHT='${FEATURE_WEIGHT}' \
RELATION_WEIGHT='${RELATION_WEIGHT}' \
LOGIT_WEIGHT='${LOGIT_WEIGHT}' \
FEATURE_LAYERS='${FEATURE_LAYERS}' \
RELATION_LAYER='${RELATION_LAYER}' \
FEATURE_ADAPT='${FEATURE_ADAPT}' \
RAW_OUTPUT_KD='${RAW_OUTPUT_KD}' \
KD_WARMUP_EPOCHS='${KD_WARMUP_EPOCHS}' \
KD_GAIN='${KD_GAIN}' \
bash comparison/cmdistill/native_reproduction/scripts/run_vedai_yolov5_cmdistill_native.sh \
  2>&1 | tee '${RUN_LOG}'
status=\${PIPESTATUS[0]}
echo \"[\$(date '+%F %T')] done ${PROBE_NAME} status=\${status}\" | tee -a '${MASTER_LOG}'
exit \${status}
"

screen -dmS "${WATCH_SCREEN}" bash -lc "
set -euo pipefail
cd '${REPO_ROOT}'
/root/miniconda3/bin/python comparison/cmdistill/native_reproduction/scripts/monitor_vedai_cmdistill_probe.py \
  --results '${RESULTS_CSV}' \
  --target-screen '${SCREEN_NAME}' \
  --target-pattern '${NAME}' \
  --log '${WATCH_LOG}' \
  --poll-seconds '${WATCH_POLL_SECONDS}' \
  --min-epoch '${WATCH_MIN_EPOCH}' \
  --window '${WATCH_WINDOW}' \
  --min-best-map50 '${WATCH_MIN_BEST_MAP50}' \
  --min-improvement '${WATCH_MIN_IMPROVEMENT}'
"

echo "SCREEN=${SCREEN_NAME}"
echo "WATCH_SCREEN=${WATCH_SCREEN}"
echo "LOG_DIR=${LOG_DIR}"
echo "MASTER_LOG=${MASTER_LOG}"
echo "WATCH_LOG=${WATCH_LOG}"
echo "RESULTS_CSV=${RESULTS_CSV}"
