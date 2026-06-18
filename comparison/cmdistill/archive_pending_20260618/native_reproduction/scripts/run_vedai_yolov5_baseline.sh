#!/usr/bin/env bash
set -euo pipefail

MODALITY="${1:-rgb}"
case "${MODALITY}" in
  rgb|ir) ;;
  *)
    echo "Usage: $0 [rgb|ir]" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${CMDISTILL_NATIVE_DATA_ROOT:-${REPRO_DIR}/data}"
RESOLUTION="${VEDAI_RESOLUTION:-512}"
SPLIT="${VEDAI_SPLIT:-paper80_seed0}"
DATASET_ROOT="${DATA_ROOT}/processed/VEDAI${RESOLUTION}_paper8_hbb_${SPLIT}"
DATA_YAML="${DATASET_ROOT}/configs/vedai${RESOLUTION}_${MODALITY}_hbb.yaml"

PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif [[ -x /root/miniconda3/bin/python ]]; then
    PYTHON_BIN="/root/miniconda3/bin/python"
  else
    echo "No Python interpreter found. Set PYTHON=/path/to/python." >&2
    exit 1
  fi
fi

YOLOV5_DIR="${YOLOV5_DIR:-/root/autodl-tmp/yolov5-v6.2}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-${EPOCHS}}"
BATCH="${BATCH:-64}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
SEED="${SEED:-0}"
PROJECT="${PROJECT:-${REPRO_DIR}/runs/vedai_yolov5_baseline/${SPLIT}/${MODALITY}}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
NAME="${NAME:-vedai${RESOLUTION}_${MODALITY}_yolov5s_e${EPOCHS}_b${BATCH}_img${IMGSZ}_s${SEED}_${RUN_TAG}}"

if [[ ! -f "${DATA_YAML}" ]]; then
  echo "[missing] ${DATA_YAML}" >&2
  echo "Run scripts/prepare_vedai_yolo_hbb.sh ${RESOLUTION} first." >&2
  exit 1
fi
if [[ ! -f "${YOLOV5_DIR}/train.py" ]]; then
  echo "[missing] ${YOLOV5_DIR}/train.py" >&2
  echo "Run scripts/setup_yolov5_v62.sh first." >&2
  exit 1
fi

mkdir -p "${PROJECT}"

cd "${YOLOV5_DIR}"
# YOLOv5 v6.2 official checkpoints are trusted legacy pickled checkpoints;
# PyTorch >=2.6 otherwise defaults torch.load() to weights_only=True.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "[run] VEDAI ${MODALITY} YOLOv5s baseline"
echo "data=${DATA_YAML}"
echo "project=${PROJECT}"
echo "name=${NAME}"
echo "patience=${PATIENCE}"

"${PYTHON_BIN}" train.py \
  --img "${IMGSZ}" \
  --batch "${BATCH}" \
  --epochs "${EPOCHS}" \
  --data "${DATA_YAML}" \
  --weights "${YOLOV5_DIR}/yolov5s.pt" \
  --project "${PROJECT}" \
  --name "${NAME}" \
  --device "${DEVICE}" \
  --workers "${WORKERS}" \
  --optimizer SGD \
  --cos-lr \
  --seed "${SEED}" \
  --patience "${PATIENCE}"
