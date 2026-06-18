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
GPU_FREE_MIN_MB="${GPU_FREE_MIN_MB:-21000}"

LOG_DIR="${REPO_ROOT}/comparison/cmdistill/native_reproduction/logs/vedai_yolov5_cmdistill_table3_ablation/${TS}"
PROJECT="${REPO_ROOT}/comparison/cmdistill/native_reproduction/runs/vedai_yolov5_cmdistill_table3_ablation/paper80_seed0"
mkdir -p "${LOG_DIR}" "${PROJECT}"

SCREEN_NAME="${SCREEN_NAME:-cmdi_table3_ablation_${TS}}"
MASTER_LOG="${LOG_DIR}/master.log"

screen -dmS "${SCREEN_NAME}" bash -lc "
set -euo pipefail
cd '${REPO_ROOT}'

wait_for_gpu() {
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
}

run_job() {
  local job=\"\$1\"
  local feature_w=\"\$2\"
  local relation_w=\"\$3\"
  local logit_w=\"\$4\"
  local log_file='${LOG_DIR}'/\"\${job}.log\"

  wait_for_gpu
  echo \"[\$(date '+%F %T')] start \${job} fw=\${feature_w} rw=\${relation_w} lw=\${logit_w}\" | tee -a '${MASTER_LOG}'
  PYTHON=/root/miniconda3/bin/python \
  YOLOV5_DIR=/root/autodl-tmp/yolov5-v6.2 \
  STUDENT_MODALITY=rgb \
  TEACHER_MODALITY=ir \
  STUDENT_WEIGHTS=/root/autodl-tmp/yolov5-v6.2/yolov5s.pt \
  EPOCHS='${EPOCHS}' \
  BATCH='${BATCH}' \
  IMGSZ='${IMGSZ}' \
  WORKERS='${WORKERS}' \
  DEVICE='${DEVICE}' \
  SEED='${SEED}' \
  PROJECT='${PROJECT}' \
  NAME=vedai512_rgb_ir_table3_\"\${job}\"_yolov5s_e'${EPOCHS}'_b'${BATCH}'_img'${IMGSZ}'_s'${SEED}'_${TS} \
  RUN_TAG=table3_\"\${job}\"_${TS} \
  ALIGNED_NO_GEO=1 \
  KEEP_COLOR_AUG=0 \
  VAL_INTERVAL=1 \
  PATIENCE='${EPOCHS}' \
  FEATURE_WEIGHT=\"\${feature_w}\" \
  RELATION_WEIGHT=\"\${relation_w}\" \
  LOGIT_WEIGHT=\"\${logit_w}\" \
  FEATURE_LAYERS=shallow_deep \
  RELATION_LAYER=deepest \
  FEATURE_ADAPT=1 \
  RAW_OUTPUT_KD=0 \
  KD_WARMUP_EPOCHS=0.0 \
  bash comparison/cmdistill/native_reproduction/scripts/run_vedai_yolov5_cmdistill_native.sh \
    2>&1 | tee \"\${log_file}\"
  echo \"[\$(date '+%F %T')] done \${job}\" | tee -a '${MASTER_LOG}'
}

run_job no_kd 0.0 0.0 0.0
run_job log_only 0.0 0.0 1.0
run_job feature_only 1.0 0.0 0.0
run_job relation_only 0.0 1.0 0.0
run_job all 1.0 1.0 1.0

echo \"[\$(date '+%F %T')] all table3 ablations finished\" | tee -a '${MASTER_LOG}'
"

echo "SCREEN=${SCREEN_NAME}"
echo "LOG_DIR=${LOG_DIR}"
echo "MASTER_LOG=${MASTER_LOG}"
