#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/paper/run_paper_hallucidet.sh <n|s|m> <seed> <gpu_id>

Paper-facing HalluciDet-YOLO official-style U-Net adaptation wrapper. This is
not an official HalluciDet reproduction and not the removed hallucidet_style KD
profile.

The current standalone trainer lacks close_mosaic support, so real launches
are P1-gated. Set ALLOW_P1_HALLUCIDET=1 to launch intentionally; dry-run is
always allowed.
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
[[ "$SIZE" =~ ^(n|s|m)$ ]] || paper_die "HalluciDet paper wrapper allows n|s|m, got: ${SIZE:-<empty>}"
paper_require_seed "$SEED"
[[ -n "$GPU_ID" ]] || paper_die "Missing gpu_id."
paper_check_strict_git
paper_require_file "$PAPER_SAR_DATA_CFG" "SAR paper dataset YAML"
paper_require_file "$PAPER_RGB_DATA_CFG" "RGB paper dataset YAML"

BATCH_SIZE="$(paper_batch_for_size "$SIZE")"
RGB_TEACHER="${RGB_TEACHER:-$(paper_find_baseline rgb "$SIZE" "$SEED" || true)}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  RGB_TEACHER="${RGB_TEACHER:-<paper_rgb_yolo11${SIZE}_seed${SEED}_best.pt>}"
elif [[ "${ALLOW_P1_HALLUCIDET:-0}" != "1" ]]; then
  paper_die "HalluciDet trainer lacks close_mosaic support; set ALLOW_P1_HALLUCIDET=1 for an intentional P1 run."
else
  paper_require_file "$RGB_TEACHER" "paper RGB teacher checkpoint"
fi

RUN_TAG="paper_ogsod_hbb_mosaic100_hallucidet_official_unet_yolo11${SIZE}_e${PAPER_EPOCHS}_b${BATCH_SIZE}_s${SEED}${RUN_TAG_SUFFIX:-}"
PROJECT_DIR="${PAPER_RUN_ROOT}/comparisons/hallucidet_yolo/yolo11${SIZE}/seed${SEED}"
LOG_DIR="${PAPER_LOG_ROOT}/comparisons/hallucidet_yolo/yolo11${SIZE}/seed${SEED}/${RUN_TAG}"
RUN_DIR="${PROJECT_DIR}/${RUN_TAG}"
META_PATH="${LOG_DIR}/paper_run_meta.env"
OUTER_LOG="${LOG_DIR}/outer.log"
PID_PATH="${LOG_DIR}/pid.txt"

cmd=(
  python3 comparison/hallucidet/train_hallucidet.py
  --data "$PAPER_SAR_DATA_CFG"
  --teacher-data "$PAPER_RGB_DATA_CFG"
  --teacher-weights "$RGB_TEACHER"
  --imgsz "$PAPER_IMGSZ"
  --epochs "$PAPER_EPOCHS"
  --batch "$BATCH_SIZE"
  --project "$PROJECT_DIR"
  --name "$RUN_TAG"
  --device "$GPU_ID"
  --workers "$PAPER_WORKERS"
  --seed "$SEED"
  --deterministic
  --mosaic "$PAPER_MOSAIC"
  --hallucination-input-mode replicate3
  --encoder-name resnet34
  --encoder-weights imagenet
  --save-period "$PAPER_SAVE_PERIOD"
)

paper_write_meta_common "$META_PATH" "hallucidet_yolo" "HalluciDet-YOLO official-style U-Net adaptation" "$SIZE" "$SEED" "$GPU_ID" "$BATCH_SIZE" "$RUN_TAG" "$PROJECT_DIR" "$RUN_DIR" "$(paper_command_text "${cmd[@]}")"
{
  printf 'phase_chain=%q\n' "standalone"
  printf 'comparison_impl_version=%q\n' "locked_hallucidet_yolo_official_unet_b64_20260618"
  printf 'hallucination_arch=%q\n' "official_unet"
  printf 'hallucination_input_mode=%q\n' "replicate3"
  printf 'encoder_name=%q\n' "resnet34"
  printf 'encoder_weights=%q\n' "imagenet"
  printf 'rgb_teacher=%q\n' "$RGB_TEACHER"
  printf 'student_modality=%q\n' "SAR"
  printf 'teacher_modality=%q\n' "RGB"
  printf 'inference_modality=%q\n' "SAR"
  printf 'protocol_gate_status=%q\n' "p1_not_main_table_until_close_mosaic_supported"
} >> "$META_PATH"

echo "[$(date '+%F %T')] Prepared paper HalluciDet-YOLO official-style U-Net yolo11${SIZE} seed=${SEED}"
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
