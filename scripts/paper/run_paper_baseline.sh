#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/paper/run_paper_baseline.sh <sar|rgb> <n|s|m|l|x> <seed> <gpu_id>

Runs the paper OGSOD HBB mosaic100 SAR/RGB baseline.
Set DRY_RUN=1 to print the command without launching.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paper_common.sh"
cd "$PAPER_REPO_ROOT"

MODALITY="${1:-}"
SIZE="${2:-}"
SEED="${3:-}"
GPU_ID="${4:-${GPU_ID:-}}"

[[ "$MODALITY" == "sar" || "$MODALITY" == "rgb" ]] || { usage >&2; exit 1; }
paper_require_size "$SIZE"
paper_require_seed "$SEED"
[[ -n "$GPU_ID" ]] || paper_die "Missing gpu_id."
paper_check_strict_git

if [[ -n "${PROTOCOL:-}" && "$PROTOCOL" != "mosaic100" && "$PROTOCOL" != "mosaic_first100_close700" ]]; then
  paper_die "Paper baseline wrapper requires PROTOCOL=mosaic100; got ${PROTOCOL}."
fi

BATCH_SIZE="$(paper_batch_for_size "$SIZE")"
DATA_CFG="$PAPER_SAR_DATA_CFG"
if [[ "$MODALITY" == "rgb" ]]; then
  DATA_CFG="$PAPER_RGB_DATA_CFG"
fi
paper_require_file "$DATA_CFG" "${MODALITY} paper dataset YAML"

RUN_TAG="paper_ogsod_hbb_mosaic100_${MODALITY}_yolo11${SIZE}_e${PAPER_EPOCHS}_b${BATCH_SIZE}_s${SEED}${RUN_TAG_SUFFIX:-}"
PROJECT_DIR="${PAPER_RUN_ROOT}/baselines/${MODALITY}/yolo11${SIZE}/seed${SEED}"
LOG_DIR="${PAPER_LOG_ROOT}/baselines/${MODALITY}/yolo11${SIZE}/seed${SEED}/${RUN_TAG}"
RUN_DIR="${PROJECT_DIR}/${RUN_TAG}"
META_PATH="${LOG_DIR}/paper_run_meta.env"

cmd=(
  env
  "PAPER_RUN=1"
  "PAPER_PROTOCOL_ID=${PAPER_PROTOCOL_ID}"
  "PROTOCOL=mosaic100"
  "DATA_CFG=${DATA_CFG}"
  "PROJECT=${PROJECT_DIR}"
  "LOG_DIR=${LOG_DIR}"
  "RUN_NAME=${RUN_TAG}"
  "MODEL=yolo11${SIZE}.pt"
  "EPOCHS=${PAPER_EPOCHS}"
  "PATIENCE=${PAPER_EPOCHS}"
  "BATCH_SIZE=${BATCH_SIZE}"
  "SAVE_PERIOD=${PAPER_SAVE_PERIOD}"
  "EXIST_OK=${EXIST_OK:-0}"
  "DRY_RUN=${DRY_RUN:-0}"
  bash baseline/scripts/run_formal_baseline.sh "$MODALITY" "$SIZE" "$SEED" "$GPU_ID"
)

paper_write_meta_common "$META_PATH" "${MODALITY}_baseline" "$([[ "$MODALITY" == "sar" ]] && printf 'SAR baseline' || printf 'RGB teacher')" "$SIZE" "$SEED" "$GPU_ID" "$BATCH_SIZE" "$RUN_TAG" "$PROJECT_DIR" "$RUN_DIR" "$(paper_command_text "${cmd[@]}")"
{
  printf 'student_modality=%q\n' "$([[ "$MODALITY" == "sar" ]] && printf 'SAR' || printf 'RGB')"
  printf 'teacher_modality=%q\n' "none"
  printf 'inference_modality=%q\n' "$([[ "$MODALITY" == "sar" ]] && printf 'SAR' || printf 'RGB')"
  printf 'phase_chain=%q\n' "baseline"
} >> "$META_PATH"

echo "[$(date '+%F %T')] Prepared paper ${MODALITY} baseline yolo11${SIZE} seed=${SEED}"
echo "meta=${META_PATH}"
paper_print_command "${cmd[@]}"
exec "${cmd[@]}"
