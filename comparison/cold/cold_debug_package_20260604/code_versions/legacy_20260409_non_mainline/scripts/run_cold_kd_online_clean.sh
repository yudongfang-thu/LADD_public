#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
GPU_ID="${GPU_ID:-7}"
GPU_WAIT_MAX_USED="${GPU_WAIT_MAX_USED:-2000}"
TEACHER_DET_WEIGHT="${TEACHER_DET_WEIGHT:-0.1}"

PROJECT_DIR="runs/cold_kd_online"
LOG_DIR="logs/cold_kd_online"
mkdir -p "$LOG_DIR"

RUN_NAME="sar_cold_kd_online_clean_t20_a2_l10_tdet${TEACHER_DET_WEIGHT//./p}_e300_b64_gpu${GPU_ID}_${RUN_TAG}"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

wait_for_gpu() {
  local gpu_id="$1"
  while true; do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((gpu_id + 1))p" | tr -d ' ')"
    if [[ -n "$used" && "$used" -lt "$GPU_WAIT_MAX_USED" ]]; then
      echo "[$(date '+%F %T')] GPU ${gpu_id} available with ${used} MiB used" | tee -a "$LOG_FILE"
      break
    fi
    echo "[$(date '+%F %T')] Waiting for GPU ${gpu_id} (currently ${used:-unknown} MiB used, threshold ${GPU_WAIT_MAX_USED})" | tee -a "$LOG_FILE"
    sleep 180
  done
}

wait_for_gpu "$GPU_ID"

PYTHONUNBUFFERED=1 python3 tools/train_cold_kd_online.py \
  --model yolo11n-obb.pt \
  --data configs/datasets/sixiang_sar_obb.yaml \
  --teacher-data configs/datasets/sixiang_rgb_obb.yaml \
  --teacher-weights runs/yolo11_obb/rgb_yolo11n_obb_clean_e300_b64_gpu4/weights/best.pt \
  --lambda-kd 1.0 \
  --lambda-cls-cold 1.0 \
  --lambda-loc-cold 1.0 \
  --alpha-non-target 2.0 \
  --temperature 20.0 \
  --teacher-det-weight "$TEACHER_DET_WEIGHT" \
  --kd-region positive \
  --imgsz 512 \
  --epochs 300 \
  --batch 64 \
  --workers 8 \
  --device "$GPU_ID" \
  --patience 80 \
  --fraction 1.0 \
  --project "$PROJECT_DIR" \
  --name "$RUN_NAME" \
  --mosaic 0 \
  --mixup 0 \
  --degrees 0 \
  --perspective 0 \
  --translate 0 \
  --scale 0 \
  --fliplr 0 \
  --flipud 0 \
  --hsv-h 0 \
  --hsv-s 0 \
  --hsv-v 0 \
  --erasing 0 \
  --close-mosaic 0 \
  2>&1 | tee -a "$LOG_FILE"
