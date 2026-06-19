#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/paper/run_paper_ladd_probea.sh <n|s|m|l|x> <seed> <gpu_id>

Runs paper LADD Probe-A / clean_a1b_dynprobe under mosaic100.
Requires matching paper SAR/RGB baseline checkpoints unless DRY_RUN=1.
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
paper_require_size "$SIZE"
paper_require_seed "$SEED"
[[ -n "$GPU_ID" ]] || paper_die "Missing gpu_id."
paper_check_strict_git
paper_require_file "$PAPER_SAR_DATA_CFG" "SAR paper dataset YAML"
paper_require_file "$PAPER_RGB_DATA_CFG" "RGB paper dataset YAML"

if [[ -n "${LADD_A1B_MODE:-}" && "$LADD_A1B_MODE" != "dynamic_probe" && "$LADD_A1B_MODE" != "dyn_probe" && "$LADD_A1B_MODE" != "probe" ]]; then
  paper_die "Paper LADD Probe-A wrapper requires LADD_A1B_MODE=dynamic_probe; got ${LADD_A1B_MODE}."
fi

BATCH_SIZE="$(paper_batch_for_size "$SIZE")"
SAR_BASELINE="${SAR_BASELINE:-$(paper_find_baseline sar "$SIZE" "$SEED" || true)}"
RGB_TEACHER="${RGB_TEACHER:-$(paper_find_baseline rgb "$SIZE" "$SEED" || true)}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  SAR_BASELINE="${SAR_BASELINE:-<paper_sar_yolo11${SIZE}_seed${SEED}_best.pt>}"
  RGB_TEACHER="${RGB_TEACHER:-<paper_rgb_yolo11${SIZE}_seed${SEED}_best.pt>}"
else
  paper_require_file "$SAR_BASELINE" "paper SAR baseline checkpoint"
  paper_require_file "$RGB_TEACHER" "paper RGB teacher checkpoint"
fi

RUN_TAG="paper_clean_a1b_dynprobe_mosaic100_yolo11${SIZE}_s${SEED}${RUN_TAG_SUFFIX:-}"
PROJECT_DIR="${PAPER_RUN_ROOT}/ladd_probea/yolo11${SIZE}/seed${SEED}"
CHAIN_LOG_DIR="${PAPER_LOG_ROOT}/ladd_probea/yolo11${SIZE}/seed${SEED}/${RUN_TAG}"
META_PATH="${CHAIN_LOG_DIR}/paper_run_meta.env"
RUN_DIR="${PROJECT_DIR}/ladd_clean_a1b_dynprobe_ogsod11${SIZE}_${RUN_TAG}_b_e${PAPER_B_EPOCHS}_b${BATCH_SIZE}_s${SEED}_gpu${GPU_ID}"

cmd=(
  env
  "PAPER_RUN=1"
  "PAPER_PROTOCOL_ID=${PAPER_PROTOCOL_ID}"
  "LADD_A1B_MODE=dynamic_probe"
  "RANK_D_NEG_CAP=2.0"
  "SAR_BASELINE=${SAR_BASELINE}"
  "RGB_TEACHER=${RGB_TEACHER}"
  "DATA_CFG=${PAPER_SAR_DATA_CFG}"
  "TEACHER_DATA_CFG=${PAPER_RGB_DATA_CFG}"
  "PROJECT_DIR=${PROJECT_DIR}"
  "CHAIN_LOG_DIR=${CHAIN_LOG_DIR}"
  "RUN_TAG=${RUN_TAG}"
  "EPOCHS_A1=${PAPER_A1_EPOCHS}"
  "EPOCHS_B=${PAPER_B_EPOCHS}"
  "PATIENCE_B=${PAPER_B_EPOCHS}"
  "BATCH_SIZE=${BATCH_SIZE}"
  "A1_MOSAIC=${PAPER_A1_MOSAIC}"
  "A1_CLOSE_MOSAIC=${PAPER_A1_CLOSE_MOSAIC}"
  "B_MOSAIC=${PAPER_B_MOSAIC}"
  "B_CLOSE_MOSAIC=${PAPER_B_CLOSE_MOSAIC}"
  "MIXUP=${PAPER_MIXUP}"
  "CUTMIX=${PAPER_CUTMIX}"
  "DEGREES=${PAPER_DEGREES}"
  "PERSPECTIVE=${PAPER_PERSPECTIVE}"
  "TRANSLATE=${PAPER_TRANSLATE}"
  "SCALE=${PAPER_SCALE}"
  "FLIPLR=${PAPER_FLIPLR}"
  "FLIPUD=${PAPER_FLIPUD}"
  "HSV_H=${PAPER_HSV_H}"
  "HSV_S=${PAPER_HSV_S}"
  "HSV_V=${PAPER_HSV_V}"
  "ERASING=${PAPER_ERASING}"
  "A1_OPTIMIZER=${PAPER_OPTIMIZER}"
  "A1_LR0=${PAPER_LR0}"
  "A1_LRF=${PAPER_LRF}"
  "A1_COS_LR=1"
  "A1_WARMUP_EPOCHS=${PAPER_WARMUP_EPOCHS}"
  "A1_WARMUP_BIAS_LR=${PAPER_WARMUP_BIAS_LR}"
  "B_OPTIMIZER=${PAPER_OPTIMIZER}"
  "B_LR0=${PAPER_LR0}"
  "B_LRF=${PAPER_LRF}"
  "B_COS_LR=1"
  "B_WARMUP_EPOCHS=${PAPER_WARMUP_EPOCHS}"
  "B_WARMUP_BIAS_LR=${PAPER_WARMUP_BIAS_LR}"
  "SAVE_PERIOD=${PAPER_SAVE_PERIOD}"
  "EXIST_OK=${EXIST_OK:-0}"
  "DRY_RUN=${DRY_RUN:-0}"
  bash ladd/scripts/launch_ladd_clean_a1b_job.sh "$SIZE" "$SEED" "$GPU_ID"
)

paper_write_meta_common "$META_PATH" "ladd_probea" "LADD Probe-A / LADD-clean A1B, ours" "$SIZE" "$SEED" "$GPU_ID" "$BATCH_SIZE" "$RUN_TAG" "$PROJECT_DIR" "$RUN_DIR" "$(paper_command_text "${cmd[@]}")"
{
  printf 'phase_chain=%q\n' "A1->B"
  printf 'ladd_a1b_mode=%q\n' "dynamic_probe"
  printf 'sar_baseline=%q\n' "$SAR_BASELINE"
  printf 'rgb_teacher=%q\n' "$RGB_TEACHER"
  printf 'student_modality=%q\n' "SAR"
  printf 'teacher_modality=%q\n' "RGB"
  printf 'inference_modality=%q\n' "SAR"
} >> "$META_PATH"

echo "[$(date '+%F %T')] Prepared paper LADD Probe-A yolo11${SIZE} seed=${SEED}"
echo "sar_baseline=${SAR_BASELINE}"
echo "rgb_teacher=${RGB_TEACHER}"
echo "meta=${META_PATH}"
paper_print_command "${cmd[@]}"
exec "${cmd[@]}"
