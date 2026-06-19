#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/paper/run_paper_cclkd_online.sh <n|s> <seed> <gpu_id>

Optional paper-facing CCLKD online comparison wrapper. This is not the CCLKD
paper-protocol reproduction launcher and not a frozen-teacher CCLKD profile.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paper_common.sh"
cd "$PAPER_REPO_ROOT"

SIZE="${1:-}"
SEED="${2:-}"
GPU_ID="${3:-${GPU_ID:-}}"
[[ "$SIZE" =~ ^(n|s)$ ]] || paper_die "CCLKD online paper wrapper allows n|s, got: ${SIZE:-<empty>}"
paper_require_seed "$SEED"
[[ -n "$GPU_ID" ]] || paper_die "Missing gpu_id."
paper_check_strict_git
paper_require_file "$PAPER_SAR_DATA_CFG" "SAR paper dataset YAML"
paper_require_file "$PAPER_RGB_DATA_CFG" "RGB paper dataset YAML"

BATCH_SIZE="$(paper_batch_for_size "$SIZE")"
RUN_TAG="paper_ogsod_hbb_mosaic100_cclkd_online_yolo11${SIZE}_e${PAPER_EPOCHS}_b${BATCH_SIZE}_s${SEED}${RUN_TAG_SUFFIX:-}"
PROJECT_DIR="${PAPER_RUN_ROOT}/comparisons/cclkd_online/yolo11${SIZE}/seed${SEED}"
LOG_DIR="${PAPER_LOG_ROOT}/comparisons/cclkd_online/yolo11${SIZE}/seed${SEED}/${RUN_TAG}"
RUN_DIR="${PROJECT_DIR}/${RUN_TAG}"
META_PATH="${LOG_DIR}/paper_run_meta.env"
OUTER_LOG="${LOG_DIR}/outer.log"
PID_PATH="${LOG_DIR}/pid.txt"

cmd=(
  python3 cclkd_reproduction/code/train_cclkd_online_hbb.py
  --model-size "$SIZE"
  --data "$PAPER_SAR_DATA_CFG"
  --teacher-data "$PAPER_RGB_DATA_CFG"
  --imgsz "$PAPER_IMGSZ"
  --epochs "$PAPER_EPOCHS"
  --batch "$BATCH_SIZE"
  --workers "$PAPER_WORKERS"
  --device "$GPU_ID"
  --patience "$PAPER_EPOCHS"
  --project "$PROJECT_DIR"
  --name "$RUN_TAG"
  --teacher-det-weight "${CCLKD_TEACHER_DET_WEIGHT:-1.0}"
  --kd-weight "${CCLKD_KD_WEIGHT:-1.0}"
  --lld-weight "${CCLKD_LLD_WEIGHT:-1.0}"
  --fld-weight "${CCLKD_FLD_WEIGHT:-1.0}"
  --rld-weight "${CCLKD_RLD_WEIGHT:-1.0}"
  --ccl-weight "${CCLKD_CCL_WEIGHT:-1.0}"
  --cclkd-formulation "${CCLKD_FORMULATION:-paper}"
  --cclkd-ccl-mode "${CCLKD_CCL_MODE:-paper_pair}"
  --cclkd-rld-mode "${CCLKD_RLD_MODE:-paper_instance}"
  --optimizer "$PAPER_OPTIMIZER"
  --lr0 "$PAPER_LR0"
  --lrf "$PAPER_LRF"
  --cos-lr
  --mosaic "$PAPER_MOSAIC"
  --close-mosaic "$PAPER_CLOSE_MOSAIC"
  --mixup "$PAPER_MIXUP"
  --cutmix "$PAPER_CUTMIX"
  --degrees "$PAPER_DEGREES"
  --perspective "$PAPER_PERSPECTIVE"
  --translate "$PAPER_TRANSLATE"
  --scale "$PAPER_SCALE"
  --fliplr "$PAPER_FLIPLR"
  --flipud "$PAPER_FLIPUD"
  --hsv-h "$PAPER_HSV_H"
  --hsv-s "$PAPER_HSV_S"
  --hsv-v "$PAPER_HSV_V"
  --erasing "$PAPER_ERASING"
  --deterministic
  --seed "$SEED"
  --save-period "$PAPER_SAVE_PERIOD"
)

if [[ "${EXIST_OK:-0}" == "1" ]]; then
  cmd+=(--exist-ok)
fi

paper_write_meta_common "$META_PATH" "cclkd_online" "CCLKD online comparison" "$SIZE" "$SEED" "$GPU_ID" "$BATCH_SIZE" "$RUN_TAG" "$PROJECT_DIR" "$RUN_DIR" "$(paper_command_text "${cmd[@]}")"
{
  printf 'phase_chain=%q\n' "online"
  printf 'student_modality=%q\n' "SAR"
  printf 'teacher_modality=%q\n' "RGB"
  printf 'inference_modality=%q\n' "SAR"
  printf 'protocol_gate_status=%q\n' "optional"
} >> "$META_PATH"

echo "[$(date '+%F %T')] Prepared optional paper CCLKD online yolo11${SIZE} seed=${SEED}"
echo "meta=${META_PATH}"
paper_print_command "${cmd[@]}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, not launching."
  exit 0
fi

if [[ -e "$RUN_DIR" && "${EXIST_OK:-0}" != "1" ]]; then
  paper_die "Run directory already exists: ${RUN_DIR}. Set EXIST_OK=1 only if intentional."
fi
mkdir -p "$LOG_DIR"
nohup "${cmd[@]}" > "$OUTER_LOG" 2>&1 &
pid=$!
echo "$pid" > "$PID_PATH"
echo "Launched pid=${pid}; log=${OUTER_LOG}; pid_file=${PID_PATH}"
