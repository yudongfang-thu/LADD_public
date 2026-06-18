#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/paper/run_paper_comparison_kd.sh <fgd|ld|cmdistill> <n|s|m|l|x> <seed> <gpu_id>

Runs frozen-teacher KD comparisons under the paper OGSOD HBB mosaic100 gate.
Requires matching paper SAR/RGB baseline checkpoints unless DRY_RUN=1.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paper_common.sh"
cd "$PAPER_REPO_ROOT"

METHOD="${1:-}"
SIZE="${2:-}"
SEED="${3:-}"
GPU_ID="${4:-${GPU_ID:-}}"
case "$METHOD" in
  fgd|ld|cmdistill) ;;
  *) usage >&2; exit 1 ;;
esac
paper_require_size "$SIZE"
paper_require_seed "$SEED"
[[ -n "$GPU_ID" ]] || paper_die "Missing gpu_id."
paper_check_strict_git
paper_require_file "$PAPER_SAR_DATA_CFG" "SAR paper dataset YAML"
paper_require_file "$PAPER_RGB_DATA_CFG" "RGB paper dataset YAML"

if [[ -n "${PROTOCOL:-}" && "$PROTOCOL" != "mosaic100" && "$PROTOCOL" != "mosaic_first100_close700" ]]; then
  paper_die "Paper comparison wrapper requires PROTOCOL=mosaic100; got ${PROTOCOL}."
fi
INIT_TYPE="${INIT_TYPE:-transferred_kd}"
if [[ "$INIT_TYPE" != "transferred_kd" && "$INIT_TYPE" != "from_yolo_pretrain" ]]; then
  paper_die "INIT_TYPE must be transferred_kd or from_yolo_pretrain; got ${INIT_TYPE}."
fi

if [[ "$METHOD" == "cmdistill" && "${KD_CALIBRATION_MODE:-affine}" != "affine" ]]; then
  paper_die "CMDistill paper gate requires KD_CALIBRATION_MODE=affine."
fi

BATCH_SIZE="$(paper_batch_for_size "$SIZE")"
RGB_TEACHER="${RGB_TEACHER:-$(paper_find_baseline rgb "$SIZE" "$SEED" || true)}"
if [[ "$INIT_TYPE" == "transferred_kd" ]]; then
  SAR_BASELINE="${SAR_BASELINE:-$(paper_find_baseline sar "$SIZE" "$SEED" || true)}"
else
  SAR_BASELINE="${MODEL:-yolo11${SIZE}.pt}"
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  if [[ "$INIT_TYPE" == "transferred_kd" ]]; then
    SAR_BASELINE="${SAR_BASELINE:-<paper_sar_yolo11${SIZE}_seed${SEED}_best.pt>}"
  fi
  RGB_TEACHER="${RGB_TEACHER:-<paper_rgb_yolo11${SIZE}_seed${SEED}_best.pt>}"
else
  if [[ "$INIT_TYPE" == "transferred_kd" ]]; then
    paper_require_file "$SAR_BASELINE" "paper SAR baseline checkpoint"
  else
    paper_require_file "$SAR_BASELINE" "YOLO student pretrain checkpoint"
  fi
  paper_require_file "$RGB_TEACHER" "paper RGB teacher checkpoint"
fi

METHOD_LABEL="$METHOD"
case "$METHOD" in
  fgd) METHOD_LABEL="FGD-style / FGD-YOLO adaptation" ;;
  ld) METHOD_LABEL="LD" ;;
  cmdistill) METHOD_LABEL="CMDistill-style / paper-aligned adaptation" ;;
esac

RUN_TAG="paper_ogsod_hbb_mosaic100_${METHOD}_yolo11${SIZE}_e${PAPER_EPOCHS}_b${BATCH_SIZE}_s${SEED}${RUN_TAG_SUFFIX:-}"
PROJECT_DIR="${PAPER_RUN_ROOT}/comparisons/${METHOD}/yolo11${SIZE}/seed${SEED}"
LOG_DIR="${PAPER_LOG_ROOT}/comparisons/${METHOD}/yolo11${SIZE}/seed${SEED}/${RUN_TAG}"
PHASE_LOG_DIR="${LOG_DIR}/phase_b"
RUN_NAME="paper_${METHOD}_hbb_ogsod11${SIZE}_mosaic100_b_e${PAPER_EPOCHS}_b${BATCH_SIZE}_s${SEED}_gpu${GPU_ID}"
RUN_DIR="${PROJECT_DIR}/${RUN_NAME}"
OUTER_LOG="${LOG_DIR}/outer.log"
PID_PATH="${LOG_DIR}/pid.txt"
META_PATH="${LOG_DIR}/paper_run_meta.env"

cmd=(
  env
  "PAPER_RUN=1"
  "PAPER_PROTOCOL_ID=${PAPER_PROTOCOL_ID}"
  "PROTOCOL=mosaic100"
  "INIT_TYPE=${INIT_TYPE}"
  "MODEL=${SAR_BASELINE}"
  "SAR_BASELINE=${SAR_BASELINE}"
  "RGB_TEACHER=${RGB_TEACHER}"
  "DATA_CFG=${PAPER_SAR_DATA_CFG}"
  "TEACHER_DATA_CFG=${PAPER_RGB_DATA_CFG}"
  "GPU_ID=${GPU_ID}"
  "SEED=${SEED}"
  "BATCH_SIZE=${BATCH_SIZE}"
  "WORKERS=${PAPER_WORKERS}"
  "IMGSZ=${PAPER_IMGSZ}"
  "EPOCHS=${PAPER_EPOCHS}"
  "PATIENCE=${PAPER_EPOCHS}"
  "PHASE_MIN_EPOCHS=${PAPER_EPOCHS}"
  "PROJECT_DIR=${PROJECT_DIR}"
  "LOG_DIR=${PHASE_LOG_DIR}"
  "RUN_NAME=${RUN_NAME}"
  "COMPARISON_KD_PROFILE=${METHOD}"
  "PROFILE_KD_WEIGHT=${PROFILE_KD_WEIGHT:-1.0}"
  "PROFILE_KD_REPLACE_BASE=1"
  "STUDENT_BRANCH_MODE=raw"
  "TEACHER_FEATURE_MODE=raw"
  "USE_MASK=0"
  "USE_FG_MASK_FOR_REACH=0"
  "USE_FG_MASK_FOR_REC=0"
  "LAMBDA_REACH=0.0"
  "LAMBDA_REC=0.0"
  "LAMBDA_TASKL=0.0"
  "ALPHA_S_REC=0.0"
  "COS_LR=1"
  "OPTIMIZER=${PAPER_OPTIMIZER}"
  "LR0=${PAPER_LR0}"
  "LRF=${PAPER_LRF}"
  "WARMUP_EPOCHS=${PAPER_WARMUP_EPOCHS}"
  "WARMUP_BIAS_LR=${PAPER_WARMUP_BIAS_LR}"
  "MOSAIC=${PAPER_MOSAIC}"
  "CLOSE_MOSAIC=${PAPER_CLOSE_MOSAIC}"
  "CLOSE_AT_EPOCH="
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
  "SAVE_PERIOD=${PAPER_SAVE_PERIOD}"
  "EXIST_OK=${EXIST_OK:-0}"
)

case "$METHOD" in
  fgd)
    cmd+=(
      "COMPARISON_IMPL_VERSION=locked_fgd_yolo_gtbox_attention_20260618"
    )
    ;;
  ld)
    cmd+=(
      "COMPARISON_IMPL_VERSION=locked_ld_yolo_dfl_vlr_20260618"
    )
    ;;
  cmdistill)
    cmd+=(
      "KD_CALIBRATION_MODE=affine"
      "CMDISTILL_FEATURE_WEIGHT=${CMDISTILL_FEATURE_WEIGHT:-1.0}"
      "CMDISTILL_RELATION_WEIGHT=${CMDISTILL_RELATION_WEIGHT:-1.0}"
      "CMDISTILL_LOGIT_WEIGHT=${CMDISTILL_LOGIT_WEIGHT:-1.0}"
      "CMDISTILL_TEMPERATURE=${CMDISTILL_TEMPERATURE:-4.0}"
      "CMDISTILL_MAX_TOKENS=${CMDISTILL_MAX_TOKENS:-512}"
      "CMDISTILL_MIN_CONFIDENCE=${CMDISTILL_MIN_CONFIDENCE:-0.05}"
    )
    ;;
esac

cmd+=(bash ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh hbb b "$RUN_TAG")

paper_write_meta_common "$META_PATH" "$METHOD" "$METHOD_LABEL" "$SIZE" "$SEED" "$GPU_ID" "$BATCH_SIZE" "$RUN_TAG" "$PROJECT_DIR" "$RUN_DIR" "$(paper_command_text "${cmd[@]}")"
{
  printf 'phase_chain=%q\n' "B-only"
  printf 'init_type=%q\n' "$INIT_TYPE"
  printf 'sar_baseline=%q\n' "$SAR_BASELINE"
  printf 'rgb_teacher=%q\n' "$RGB_TEACHER"
  printf 'student_modality=%q\n' "SAR"
  printf 'teacher_modality=%q\n' "RGB"
  printf 'inference_modality=%q\n' "SAR"
  printf 'profile_kd_replace_base=%q\n' "1"
  printf 'student_branch_mode=%q\n' "raw"
  printf 'teacher_feature_mode=%q\n' "raw"
  printf 'kd_calibration_mode=%q\n' "$([[ "$METHOD" == "cmdistill" ]] && printf 'affine' || printf 'none')"
  if [[ "$METHOD" == "fgd" ]]; then
    printf 'comparison_impl_version=%q\n' "locked_fgd_yolo_gtbox_attention_20260618"
    printf 'fgd_alpha=%q\n' "0.0001"
    printf 'fgd_beta=%q\n' "0.00005"
    printf 'fgd_gamma=%q\n' "0.001"
    printf 'fgd_temperature=%q\n' "0.5"
    printf 'fgd_mask_mode=%q\n' "gt_box"
    printf 'fgd_relation=%q\n' "removed"
  elif [[ "$METHOD" == "ld" ]]; then
    printf 'comparison_impl_version=%q\n' "locked_ld_yolo_dfl_vlr_20260618"
    printf 'ld_temperature=%q\n' "10.0"
    printf 'ld_main_weight=%q\n' "0.25"
    printf 'ld_vlr_weight=%q\n' "0.25"
    printf 'ld_vlr_candidate=%q\n' "teacher_confidence_x_teacher_box_gt_iou"
  fi
} >> "$META_PATH"

echo "[$(date '+%F %T')] Prepared paper comparison ${METHOD} yolo11${SIZE} seed=${SEED}"
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
