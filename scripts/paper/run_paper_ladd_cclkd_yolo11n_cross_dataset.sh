#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/paper/run_paper_ladd_cclkd_yolo11n_cross_dataset.sh <vedai|dronevehicle> <seed> <gpu_id>

Runs LADD Probe-A / clean A1B under the CCLKD YOLO11n cross-dataset protocol:
  model=yolo11n, imgsz=512, A1=10, B=200, batch=16, SGD lr=0.01,
  momentum=0.937, mosaic=1.0, close_mosaic=10, mixup=0.1.

Dataset directions:
  vedai        RGB/visible teacher -> IR student
  dronevehicle IR teacher -> RGB/visible student

Required checkpoints:
  STUDENT_BASELINE=/path/to/student/best.pt
  TEACHER_BASELINE=/path/to/teacher/best.pt

If not set, the launcher tries to find the latest matching baseline under:
  runs_public/cross_dataset/cclkd_yolo11n/<dataset>/baselines/

Useful overrides:
  STUDENT_DATA_CFG=/path/to/student.yaml
  TEACHER_DATA_CFG=/path/to/teacher.yaml
  DRY_RUN=1
  EXIST_OK=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DATASET="${1:-}"
SEED="${2:-}"
GPU_ID="${3:-${GPU_ID:-}}"

if [[ "$DATASET" != "vedai" && "$DATASET" != "dronevehicle" ]]; then
  usage >&2
  exit 1
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  usage >&2
  exit 1
fi
if [[ -z "$GPU_ID" ]]; then
  echo "Missing gpu_id. Pass it as the third argument or set GPU_ID." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

find_latest_weight() {
  local pattern="$1"
  local -a candidates=()
  local candidate
  shopt -s nullglob
  for candidate in $pattern; do
    candidates+=("$candidate")
  done
  shopt -u nullglob
  if (( ${#candidates[@]} == 0 )); then
    return 1
  fi
  ls -t "${candidates[@]}" 2>/dev/null | head -n 1
}

PROTOCOL_ID="${PAPER_PROTOCOL_ID:-cclkd_yolo11n_cross_dataset_20260619}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"

case "$DATASET" in
  vedai)
    STUDENT_MODALITY="ir"
    TEACHER_MODALITY="rgb"
    TRANSFER_DIRECTION="visible_to_infrared"
    DEFAULT_STUDENT_DATA="comparison/cmdistill/native_reproduction/data/processed/VEDAI512_paper8_hbb_paper80_seed0/configs/vedai512_ir_hbb.yaml"
    DEFAULT_TEACHER_DATA="comparison/cmdistill/native_reproduction/data/processed/VEDAI512_paper8_hbb_paper80_seed0/configs/vedai512_rgb_hbb.yaml"
    ;;
  dronevehicle)
    STUDENT_MODALITY="rgb"
    TEACHER_MODALITY="ir"
    TRANSFER_DIRECTION="infrared_to_visible"
    DEFAULT_STUDENT_DATA="comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb/configs/dronevehicle_rgb_hbb.yaml"
    DEFAULT_TEACHER_DATA="comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb/configs/dronevehicle_ir_hbb.yaml"
    ;;
esac

STUDENT_DATA_CFG="${STUDENT_DATA_CFG:-$DEFAULT_STUDENT_DATA}"
TEACHER_DATA_CFG="${TEACHER_DATA_CFG:-$DEFAULT_TEACHER_DATA}"

if [[ -z "${STUDENT_BASELINE:-}" ]]; then
  STUDENT_BASELINE="$(find_latest_weight "runs_public/cross_dataset/cclkd_yolo11n/${DATASET}/baselines/student_${STUDENT_MODALITY}/${DATASET}_student_${STUDENT_MODALITY}_yolo11n_cclkdproto_e200_b16_img512_s${SEED}_*/weights/best.pt" || true)"
fi
if [[ -z "${TEACHER_BASELINE:-}" ]]; then
  TEACHER_BASELINE="$(find_latest_weight "runs_public/cross_dataset/cclkd_yolo11n/${DATASET}/baselines/teacher_${TEACHER_MODALITY}/${DATASET}_teacher_${TEACHER_MODALITY}_yolo11n_cclkdproto_e200_b16_img512_s${SEED}_*/weights/best.pt" || true)"
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  for item in \
    "$STUDENT_BASELINE:student baseline checkpoint" \
    "$TEACHER_BASELINE:teacher baseline checkpoint" \
    "$STUDENT_DATA_CFG:student dataset YAML" \
    "$TEACHER_DATA_CFG:teacher dataset YAML"; do
    path="${item%%:*}"
    label="${item#*:}"
    if [[ -z "$path" || ! -f "$path" ]]; then
      echo "Missing ${label}: ${path:-<empty>}" >&2
      exit 1
    fi
  done
fi

RUN_TAG="${RUN_TAG:-cclkdproto_${DATASET}_ladd_probea_yolo11n_${STUDENT_MODALITY}_student_${TEACHER_MODALITY}_teacher_s${SEED}_${DATE_TAG}}"
PROJECT_DIR="${PROJECT_DIR:-runs_public/cross_dataset/cclkd_yolo11n/${DATASET}/ladd_probea}"
CHAIN_LOG_DIR="${CHAIN_LOG_DIR:-logs/cross_dataset/cclkd_yolo11n/${DATASET}/ladd_probea/${RUN_TAG}_gpu${GPU_ID}}"
META_PATH="${CHAIN_LOG_DIR}/cross_dataset_meta.env"
mkdir -p "$CHAIN_LOG_DIR"

{
  printf 'paper_protocol_id=%q\n' "$PROTOCOL_ID"
  printf 'protocol_id=%q\n' "$PROTOCOL_ID"
  printf 'scope=%q\n' "cross_dataset_cclkd_yolo11n_ladd"
  printf 'dataset=%q\n' "$DATASET"
  printf 'transfer_direction=%q\n' "$TRANSFER_DIRECTION"
  printf 'student_modality=%q\n' "$STUDENT_MODALITY"
  printf 'teacher_modality=%q\n' "$TEACHER_MODALITY"
  printf 'inference_modality=%q\n' "$STUDENT_MODALITY"
  printf 'seed=%q\n' "$SEED"
  printf 'gpu_id=%q\n' "$GPU_ID"
  printf 'model_size=%q\n' "n"
  printf 'method=%q\n' "ladd_probea"
  printf 'ladd_mode=%q\n' "dynamic_probe"
  printf 'phase_chain=%q\n' "A1->B"
  printf 'student_baseline=%q\n' "$STUDENT_BASELINE"
  printf 'teacher_baseline=%q\n' "$TEACHER_BASELINE"
  printf 'student_data_cfg=%q\n' "$STUDENT_DATA_CFG"
  printf 'teacher_data_cfg=%q\n' "$TEACHER_DATA_CFG"
  printf 'run_tag=%q\n' "$RUN_TAG"
  printf 'project_dir=%q\n' "$PROJECT_DIR"
  printf 'chain_log_dir=%q\n' "$CHAIN_LOG_DIR"
  printf 'imgsz=%q\n' "512"
  printf 'epochs_a1=%q\n' "10"
  printf 'epochs_b=%q\n' "200"
  printf 'batch=%q\n' "16"
  printf 'optimizer=%q\n' "SGD"
  printf 'lr0=%q\n' "0.01"
  printf 'momentum=%q\n' "0.937"
  printf 'weight_decay=%q\n' "0.0005"
  printf 'mosaic=%q\n' "1.0"
  printf 'close_mosaic=%q\n' "10"
  printf 'mixup=%q\n' "0.1"
  printf 'git_commit=%q\n' "$(git rev-parse HEAD 2>/dev/null || echo unknown)"
} > "$META_PATH"

echo "[$(date '+%F %T')] Prepared CCLKD-protocol YOLO11n LADD ${DATASET}"
echo "direction=${TRANSFER_DIRECTION} student=${STUDENT_MODALITY} teacher=${TEACHER_MODALITY}"
echo "student_baseline=${STUDENT_BASELINE:-<not-found>}"
echo "teacher_baseline=${TEACHER_BASELINE:-<not-found>}"
echo "meta=${META_PATH}"

env \
  "PAPER_PROTOCOL_ID=${PROTOCOL_ID}" \
  "LADD_A1B_MODE=dynamic_probe" \
  "DATASET_TAG=${DATASET}_yolo11n" \
  "RUN_TAG=${RUN_TAG}" \
  "SAR_BASELINE=${STUDENT_BASELINE}" \
  "RGB_TEACHER=${TEACHER_BASELINE}" \
  "DATA_CFG=${STUDENT_DATA_CFG}" \
  "TEACHER_DATA_CFG=${TEACHER_DATA_CFG}" \
  "PROJECT_DIR=${PROJECT_DIR}" \
  "CHAIN_LOG_DIR=${CHAIN_LOG_DIR}" \
  "EPOCHS_A1=10" \
  "EPOCHS_B=200" \
  "PATIENCE_A=200" \
  "PATIENCE_B=200" \
  "BATCH_SIZE=16" \
  "WORKERS=${WORKERS:-8}" \
  "IMGSZ=512" \
  "A1_MOSAIC=1.0" \
  "A1_CLOSE_MOSAIC=0" \
  "B_MOSAIC=1.0" \
  "B_CLOSE_MOSAIC=10" \
  "MIXUP=0.1" \
  "CUTMIX=0.0" \
  "DEGREES=0.0" \
  "PERSPECTIVE=0.0" \
  "TRANSLATE=0.1" \
  "SCALE=0.5" \
  "FLIPLR=0.5" \
  "FLIPUD=0.0" \
  "HSV_H=0.0" \
  "HSV_S=0.0" \
  "HSV_V=0.0" \
  "ERASING=0.0" \
  "A1_OPTIMIZER=SGD" \
  "A1_LR0=0.01" \
  "A1_LRF=0.01" \
  "MOMENTUM=0.937" \
  "WEIGHT_DECAY=0.0005" \
  "A1_COS_LR=1" \
  "A1_WARMUP_EPOCHS=3.0" \
  "A1_WARMUP_BIAS_LR=0.1" \
  "B_OPTIMIZER=SGD" \
  "B_LR0=0.01" \
  "B_LRF=0.01" \
  "B_COS_LR=1" \
  "B_WARMUP_EPOCHS=3.0" \
  "B_WARMUP_BIAS_LR=0.1" \
  "SAVE_PERIOD=50" \
  "RANK_D_NEG_CAP=2.0" \
  "EXIST_OK=${EXIST_OK:-0}" \
  "DRY_RUN=${DRY_RUN:-0}" \
  bash ladd/scripts/launch_ladd_clean_a1b_job.sh n "$SEED" "$GPU_ID"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

if [[ -f "${CHAIN_LOG_DIR}/run_meta_clean_a1b.env" ]]; then
  cat "${CHAIN_LOG_DIR}/run_meta_clean_a1b.env" >> "$META_PATH"
fi
