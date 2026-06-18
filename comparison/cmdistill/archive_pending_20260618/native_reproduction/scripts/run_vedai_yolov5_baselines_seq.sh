#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${REPRO_DIR}/logs/vedai_yolov5_seq_b64"
mkdir -p "${LOG_DIR}"

MODALITIES="${MODALITIES:-ir rgb}"
RUN_TAG="${RUN_TAG:-seq_$(date +%Y%m%d_%H%M%S)}"
BATCH="${BATCH:-64}"

echo "[seq] start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "[seq] modalities=${MODALITIES}"
echo "[seq] batch=${BATCH}"
echo "[seq] run_tag=${RUN_TAG}"

for modality in ${MODALITIES}; do
  run_log="${LOG_DIR}/$(date '+%Y%m%d_%H%M%S')_${modality}_baseline_b${BATCH}_${RUN_TAG}.log"
  echo "[seq] start modality=${modality} log=${run_log}"
  BATCH="${BATCH}" RUN_TAG="${RUN_TAG}" bash "${SCRIPT_DIR}/run_vedai_yolov5_baseline.sh" "${modality}" 2>&1 | tee "${run_log}"
  echo "[seq] done modality=${modality} time=$(date '+%Y-%m-%d %H:%M:%S')"
done

echo "[seq] complete=$(date '+%Y-%m-%d %H:%M:%S')"
