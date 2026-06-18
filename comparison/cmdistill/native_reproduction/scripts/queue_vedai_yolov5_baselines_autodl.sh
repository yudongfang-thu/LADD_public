#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPRO_DIR}/logs/vedai_yolov5_queue"
mkdir -p "${LOG_DIR}"

GPU_ID="${DEVICE:-0}"
MAX_USED_MB="${MAX_USED_MB:-2000}"
MAX_UTIL="${MAX_UTIL:-20}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MODALITIES="${MODALITIES:-ir rgb}"
PYTHON="${PYTHON:-/root/miniconda3/bin/python}"
export PYTHON DEVICE="${GPU_ID}"

gpu_state() {
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "${GPU_ID}" |
    awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1" "$2}'
}

echo "[queue] waiting for GPU ${GPU_ID}: memory.used < ${MAX_USED_MB} MiB and util < ${MAX_UTIL}%"
while true; do
  read -r used util < <(gpu_state)
  now="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[${now}] gpu_used=${used}MiB gpu_util=${util}%"
  if [[ "${used}" -lt "${MAX_USED_MB}" && "${util}" -lt "${MAX_UTIL}" ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

bash "${SCRIPT_DIR}/setup_yolov5_v62.sh"
bash "${SCRIPT_DIR}/prepare_vedai_yolo_hbb.sh" "${VEDAI_RESOLUTION:-512}"

for modality in ${MODALITIES}; do
  run_log="${LOG_DIR}/$(date '+%Y%m%d_%H%M%S')_${modality}_baseline.log"
  echo "[run] ${modality}, log=${run_log}"
  bash "${SCRIPT_DIR}/run_vedai_yolov5_baseline.sh" "${modality}" 2>&1 | tee "${run_log}"
done

echo "[done] VEDAI YOLOv5 baselines complete"
