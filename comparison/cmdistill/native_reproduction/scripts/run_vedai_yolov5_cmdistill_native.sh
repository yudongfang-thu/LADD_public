#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${REPRO_DIR}/../../.." && pwd)"

DATA_ROOT="${CMDISTILL_NATIVE_DATA_ROOT:-${REPRO_DIR}/data}"
RESOLUTION="${VEDAI_RESOLUTION:-512}"
SPLIT="${VEDAI_SPLIT:-paper80_seed0}"
DATASET_ROOT="${DATA_ROOT}/processed/VEDAI${RESOLUTION}_paper8_hbb_${SPLIT}"
STUDENT_MODALITY="${STUDENT_MODALITY:-rgb}"
TEACHER_MODALITY="${TEACHER_MODALITY:-ir}"
case "${STUDENT_MODALITY}:${TEACHER_MODALITY}" in
  rgb:ir|ir:rgb) ;;
  *)
    echo "Use STUDENT_MODALITY/TEACHER_MODALITY as rgb:ir or ir:rgb." >&2
    exit 2
    ;;
esac
STUDENT_DATA_YAML="${DATASET_ROOT}/configs/vedai${RESOLUTION}_${STUDENT_MODALITY}_hbb.yaml"

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
PROJECT="${PROJECT:-${REPRO_DIR}/runs/vedai_yolov5_cmdistill_native/${SPLIT}}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
NAME="${NAME:-vedai${RESOLUTION}_${STUDENT_MODALITY}_${TEACHER_MODALITY}_cmdistill_yolov5s_e${EPOCHS}_b${BATCH}_img${IMGSZ}_s${SEED}_${RUN_TAG}}"

DEFAULT_RGB_BEST="${REPRO_DIR}/runs/vedai_yolov5_baseline/${SPLIT}/rgb/vedai${RESOLUTION}_rgb_yolov5s_e300_b64_img${IMGSZ}_s${SEED}_table1_rgb_b64_20260618_124005/weights/best.pt"
DEFAULT_IR_BEST="${REPRO_DIR}/runs/vedai_yolov5_baseline/${SPLIT}/ir/vedai${RESOLUTION}_ir_yolov5s_e300_b64_img${IMGSZ}_s${SEED}_seq_b64_20260618_114037/weights/best.pt"
DEFAULT_STUDENT_WEIGHTS="${YOLOV5_DIR}/yolov5s.pt"
if [[ "${STUDENT_MODALITY}" == "rgb" ]]; then
  DEFAULT_TEACHER_WEIGHTS="${DEFAULT_IR_BEST}"
else
  DEFAULT_TEACHER_WEIGHTS="${DEFAULT_RGB_BEST}"
fi
STUDENT_WEIGHTS="${STUDENT_WEIGHTS:-${DEFAULT_STUDENT_WEIGHTS}}"
TEACHER_WEIGHTS="${TEACHER_WEIGHTS:-${DEFAULT_TEACHER_WEIGHTS}}"

FEATURE_WEIGHT="${FEATURE_WEIGHT:-1.0}"
RELATION_WEIGHT="${RELATION_WEIGHT:-1.0}"
LOGIT_WEIGHT="${LOGIT_WEIGHT:-1.0}"
FEATURE_LAYERS="${FEATURE_LAYERS:-shallow_deep}"
RELATION_LAYER="${RELATION_LAYER:-deepest}"
FEATURE_ADAPT="${FEATURE_ADAPT:-1}"
RAW_OUTPUT_KD="${RAW_OUTPUT_KD:-0}"
TEMPERATURE="${TEMPERATURE:-4.0}"
MAX_TOKENS="${MAX_TOKENS:-256}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.05}"
OUTPUT_TOPK="${OUTPUT_TOPK:-128}"
KD_WARMUP_EPOCHS="${KD_WARMUP_EPOCHS:-0.0}"
KD_GAIN="${KD_GAIN:-1.0}"
ALIGNED_NO_GEO="${ALIGNED_NO_GEO:-1}"
KEEP_COLOR_AUG="${KEEP_COLOR_AUG:-0}"
VAL_INTERVAL="${VAL_INTERVAL:-1}"

if [[ ! -f "${STUDENT_DATA_YAML}" ]]; then
  echo "[missing] ${STUDENT_DATA_YAML}" >&2
  exit 1
fi
if [[ ! -f "${STUDENT_WEIGHTS}" ]]; then
  echo "[missing] student weights: ${STUDENT_WEIGHTS}" >&2
  exit 1
fi
if [[ ! -f "${TEACHER_WEIGHTS}" ]]; then
  echo "[missing] teacher weights: ${TEACHER_WEIGHTS}" >&2
  exit 1
fi
if [[ ! -f "${YOLOV5_DIR}/train.py" ]]; then
  echo "[missing] ${YOLOV5_DIR}/train.py" >&2
  exit 1
fi

mkdir -p "${PROJECT}"
cd "${REPO_ROOT}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS_OVERRIDE:-1}"

CMD=(
  "${PYTHON_BIN}" "${REPRO_DIR}/scripts/train_vedai_yolov5_cmdistill_native.py"
  --yolov5-dir "${YOLOV5_DIR}"
  --data "${STUDENT_DATA_YAML}"
  --weights "${STUDENT_WEIGHTS}"
  --teacher-weights "${TEACHER_WEIGHTS}"
  --student-token "/images/${STUDENT_MODALITY}/"
  --teacher-token "/images/${TEACHER_MODALITY}/"
  --hyp "${YOLOV5_DIR}/data/hyps/hyp.scratch-low.yaml"
  --img "${IMGSZ}"
  --batch-size "${BATCH}"
  --epochs "${EPOCHS}"
  --project "${PROJECT}"
  --name "${NAME}"
  --device "${DEVICE}"
  --workers "${WORKERS}"
  --optimizer SGD
  --cos-lr
  --seed "${SEED}"
  --patience "${PATIENCE}"
  --val-interval "${VAL_INTERVAL}"
  --feature-weight "${FEATURE_WEIGHT}"
  --relation-weight "${RELATION_WEIGHT}"
  --logit-weight "${LOGIT_WEIGHT}"
  --feature-layers "${FEATURE_LAYERS}"
  --relation-layer "${RELATION_LAYER}"
  --temperature "${TEMPERATURE}"
  --max-tokens "${MAX_TOKENS}"
  --min-confidence "${MIN_CONFIDENCE}"
  --output-topk "${OUTPUT_TOPK}"
  --kd-warmup-epochs "${KD_WARMUP_EPOCHS}"
  --kd-gain "${KD_GAIN}"
)
if [[ "${ALIGNED_NO_GEO}" == "1" ]]; then
  CMD+=(--aligned-no-geo)
fi
if [[ "${KEEP_COLOR_AUG}" == "1" ]]; then
  CMD+=(--keep-color-aug)
fi
if [[ "${FEATURE_ADAPT}" != "1" ]]; then
  CMD+=(--no-feature-adapt)
fi
if [[ "${RAW_OUTPUT_KD}" == "1" ]]; then
  CMD+=(--raw-output-kd)
fi

echo "[run] VEDAI YOLOv5s CMDistill native"
echo "student_modality=${STUDENT_MODALITY}"
echo "teacher_modality=${TEACHER_MODALITY}"
echo "data=${STUDENT_DATA_YAML}"
echo "student=${STUDENT_WEIGHTS}"
echo "teacher=${TEACHER_WEIGHTS}"
echo "project=${PROJECT}"
echo "name=${NAME}"
echo "aligned_no_geo=${ALIGNED_NO_GEO}"
echo "feature_layers=${FEATURE_LAYERS}"
echo "relation_layer=${RELATION_LAYER}"
echo "feature_adapt=${FEATURE_ADAPT}"
echo "raw_output_kd=${RAW_OUTPUT_KD}"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
