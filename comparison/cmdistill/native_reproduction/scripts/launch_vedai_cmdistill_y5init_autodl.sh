#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/LADD_public}"
cd "${REPO_ROOT}"

TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${REPO_ROOT}/comparison/cmdistill/native_reproduction/logs/vedai_yolov5_cmdistill_native_formal"
mkdir -p "${LOG_DIR}"

SCREEN_NAME="${SCREEN_NAME:-cmdi_rgb_ir_y5init_e300_${TS}}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${TS}_rgb_ir_e300_b64_yolov5sinit_map50best_aligned_nogeo.log}"

screen -dmS "${SCREEN_NAME}" bash -lc "
PYTHON=/root/miniconda3/bin/python \
YOLOV5_DIR=/root/autodl-tmp/yolov5-v6.2 \
STUDENT_MODALITY=rgb \
TEACHER_MODALITY=ir \
STUDENT_WEIGHTS=/root/autodl-tmp/yolov5-v6.2/yolov5s.pt \
EPOCHS=300 \
BATCH=64 \
IMGSZ=640 \
WORKERS=8 \
DEVICE=0 \
SEED=0 \
RUN_TAG=rgb_ir_e300_b64_yolov5sinit_map50best_aligned_nogeo_${TS} \
ALIGNED_NO_GEO=1 \
VAL_INTERVAL=1 \
PATIENCE=300 \
FEATURE_WEIGHT=1.0 \
RELATION_WEIGHT=1.0 \
LOGIT_WEIGHT=1.0 \
KD_WARMUP_EPOCHS=0.0 \
bash comparison/cmdistill/native_reproduction/scripts/run_vedai_yolov5_cmdistill_native.sh \
2>&1 | tee ${LOG_FILE}
"

echo "SCREEN=${SCREEN_NAME}"
echo "LOG=${LOG_FILE}"
