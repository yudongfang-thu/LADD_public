#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/paper/run_paper_cclkd_yolo11n_cross_baseline.sh <vedai|dronevehicle> <student|teacher> <seed> <gpu_id>

Runs one YOLO11n single-modality baseline under the CCLKD cross-dataset protocol:
  imgsz=512, epochs=200, batch=16, SGD lr=0.01, momentum=0.937,
  mosaic=1.0, close_mosaic=10, mixup=0.1.

Dataset directions:
  vedai        RGB/visible teacher -> IR student
  dronevehicle IR teacher -> RGB/visible student

Useful overrides:
  STUDENT_DATA_CFG=/path/to/student.yaml
  TEACHER_DATA_CFG=/path/to/teacher.yaml
  MODEL=yolo11n.pt
  DRY_RUN=1
  EXIST_OK=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DATASET="${1:-}"
ROLE="${2:-}"
SEED="${3:-}"
GPU_ID="${4:-${GPU_ID:-}}"

if [[ "$DATASET" != "vedai" && "$DATASET" != "dronevehicle" ]]; then
  usage >&2
  exit 1
fi
if [[ "$ROLE" != "student" && "$ROLE" != "teacher" ]]; then
  usage >&2
  exit 1
fi
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  usage >&2
  exit 1
fi
if [[ -z "$GPU_ID" ]]; then
  echo "Missing gpu_id. Pass it as the fourth argument or set GPU_ID." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PROTOCOL_ID="${PAPER_PROTOCOL_ID:-cclkd_yolo11n_cross_dataset_20260619}"
DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-yolo11n.pt}"
IMGSZ_VALUE="${IMGSZ:-512}"
EPOCHS_VALUE="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PATIENCE_VALUE="${PATIENCE:-$EPOCHS_VALUE}"
SAVE_PERIOD_VALUE="${SAVE_PERIOD:-50}"
WORKERS_VALUE="${WORKERS:-8}"

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
if [[ "$ROLE" == "student" ]]; then
  DATA_CFG="$STUDENT_DATA_CFG"
  MODALITY="$STUDENT_MODALITY"
else
  DATA_CFG="$TEACHER_DATA_CFG"
  MODALITY="$TEACHER_MODALITY"
fi

PROJECT="${PROJECT:-runs_public/cross_dataset/cclkd_yolo11n/${DATASET}/baselines/${ROLE}_${MODALITY}}"
LOG_DIR="${LOG_DIR:-logs/cross_dataset/cclkd_yolo11n/${DATASET}/baselines}"
META_DIR="${LOG_DIR}/metadata"
RUN_NAME="${RUN_NAME:-${DATASET}_${ROLE}_${MODALITY}_yolo11n_cclkdproto_e${EPOCHS_VALUE}_b${BATCH_SIZE}_img${IMGSZ_VALUE}_s${SEED}_${DATE_TAG}}"
RUN_DIR="${PROJECT}/${RUN_NAME}"
LOG_PATH="${LOG_DIR}/${RUN_NAME}_gpu${GPU_ID}.log"
PID_PATH="${LOG_DIR}/${RUN_NAME}_gpu${GPU_ID}.pid"
META_PATH="${META_DIR}/${RUN_NAME}_gpu${GPU_ID}.env"

mkdir -p "$PROJECT" "$LOG_DIR" "$META_DIR"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  if [[ ! -f "$DATA_CFG" ]]; then
    echo "Missing ${DATASET} ${ROLE} dataset YAML: ${ROOT_DIR}/${DATA_CFG}" >&2
    exit 1
  fi
  if [[ -e "$RUN_DIR" && "${EXIST_OK:-0}" != "1" ]]; then
    echo "Run directory already exists: ${ROOT_DIR}/${RUN_DIR}" >&2
    echo "Set EXIST_OK=1 only if this is intentional." >&2
    exit 1
  fi
fi

cmd=(
  python3 baseline/code/train_ogsod_baseline.py
  --task hbb
  --model "$MODEL"
  --data "$DATA_CFG"
  --imgsz "$IMGSZ_VALUE"
  --epochs "$EPOCHS_VALUE"
  --batch "$BATCH_SIZE"
  --workers "$WORKERS_VALUE"
  --device "$GPU_ID"
  --patience "$PATIENCE_VALUE"
  --project "$PROJECT"
  --name "$RUN_NAME"
  --optimizer SGD
  --lr0 0.01
  --lrf 0.01
  --momentum 0.937
  --weight-decay 0.0005
  --cos-lr
  --mosaic 1.0
  --close-mosaic 10
  --mixup 0.1
  --cutmix 0.0
  --degrees 0.0
  --perspective 0.0
  --translate 0.1
  --scale 0.5
  --fliplr 0.5
  --flipud 0.0
  --hsv-h 0.0
  --hsv-s 0.0
  --hsv-v 0.0
  --erasing 0.0
  --save-period "$SAVE_PERIOD_VALUE"
  --seed "$SEED"
  --deterministic
)

if [[ "${EXIST_OK:-0}" == "1" ]]; then
  cmd+=(--exist-ok)
fi

{
  printf 'paper_protocol_id=%q\n' "$PROTOCOL_ID"
  printf 'protocol_id=%q\n' "$PROTOCOL_ID"
  printf 'scope=%q\n' "cross_dataset_cclkd_yolo11n"
  printf 'dataset=%q\n' "$DATASET"
  printf 'transfer_direction=%q\n' "$TRANSFER_DIRECTION"
  printf 'role=%q\n' "$ROLE"
  printf 'modality=%q\n' "$MODALITY"
  printf 'student_modality=%q\n' "$STUDENT_MODALITY"
  printf 'teacher_modality=%q\n' "$TEACHER_MODALITY"
  printf 'seed=%q\n' "$SEED"
  printf 'gpu_id=%q\n' "$GPU_ID"
  printf 'model=%q\n' "$MODEL"
  printf 'data_cfg=%q\n' "$DATA_CFG"
  printf 'student_data_cfg=%q\n' "$STUDENT_DATA_CFG"
  printf 'teacher_data_cfg=%q\n' "$TEACHER_DATA_CFG"
  printf 'run_name=%q\n' "$RUN_NAME"
  printf 'run_dir=%q\n' "$RUN_DIR"
  printf 'log_path=%q\n' "$LOG_PATH"
  printf 'imgsz=%q\n' "$IMGSZ_VALUE"
  printf 'epochs=%q\n' "$EPOCHS_VALUE"
  printf 'batch=%q\n' "$BATCH_SIZE"
  printf 'optimizer=%q\n' "SGD"
  printf 'lr0=%q\n' "0.01"
  printf 'momentum=%q\n' "0.937"
  printf 'weight_decay=%q\n' "0.0005"
  printf 'mosaic=%q\n' "1.0"
  printf 'close_mosaic=%q\n' "10"
  printf 'mixup=%q\n' "0.1"
  printf 'git_commit=%q\n' "$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  printf 'cmd='
  printf '%q ' "${cmd[@]}"
  printf '\n'
} > "$META_PATH"

echo "[$(date '+%F %T')] Prepared CCLKD-protocol YOLO11n ${DATASET} ${ROLE} baseline"
echo "direction=${TRANSFER_DIRECTION} modality=${MODALITY} run=${RUN_DIR}"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, not launching."
  exit 0
fi

nohup env PYTHONUNBUFFERED=1 "${cmd[@]}" > "$LOG_PATH" 2>&1 &
pid=$!
echo "$pid" > "$PID_PATH"
echo "Launched pid=${pid}; log=${LOG_PATH}; pid_file=${PID_PATH}"
