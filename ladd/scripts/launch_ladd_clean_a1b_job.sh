#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash ladd/scripts/launch_ladd_clean_a1b_job.sh <n|s|m|l|x> <seed> <gpu_id>

LADD-clean / LADD-A1B launcher:
  - runs A1, then directly runs B from A1 weights/best.pt
  - does not run A2
  - keeps A1 reconstruction/reach/taskL and B detector/KD/student reconstruction
  - uses the cleaned LADD loss surface without sep/private/residual/debug auxiliary losses
  - supports LADD_A1B_MODE=static or dynamic

Modes:
  static   B freezes teacher decomposition; B loss = det + KD + student rec
  dynamic  B keeps teacher decomposition/reach/taskL active via --ladd-b-a2-core

Required checkpoints:
  SAR_BASELINE=/path/to/sar/best.pt
  RGB_TEACHER=/path/to/rgb/best.pt

If SAR_BASELINE/RGB_TEACHER are not set, the launcher tries to find matching
weights under:
  runs_public/ogsod/hbb/baseline_controls/mosaic_baselines_20260615/

Defaults:
  A1 epochs=10
  B epochs=800
  B mosaic=1.0, close_mosaic=700
  rank_d_neg_cap=2.0
  batch n/s=64, m/l=32, x=16

Useful overrides:
  DRY_RUN=1
  RUN_TAG_SUFFIX=_try1
  EXP_SUFFIX=try1
  LADD_A1B_MODE=dynamic
  EPOCHS_A1=1 EPOCHS_B=1
  PROJECT_DIR=/path/to/project
  CHAIN_LOG_DIR=/path/to/logs
  EXIST_OK=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SIZE="${1:-}"
SEED="${2:-}"
GPU_ID="${3:-${GPU_ID:-}}"

if [[ ! "$SIZE" =~ ^(n|s|m|l|x)$ ]]; then
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

LADD_A1B_MODE="${LADD_A1B_MODE:-static}"
case "$LADD_A1B_MODE" in
  static)
    MODE_TAG="clean_a1b"
    MODE_PROJECT_KEY="ladd_clean_a1b"
    LADD_B_A2_CORE_VALUE="0"
    ;;
  dynamic|dyn)
    LADD_A1B_MODE="dynamic"
    MODE_TAG="clean_a1b_dyn"
    MODE_PROJECT_KEY="ladd_clean_a1b_dynamic"
    LADD_B_A2_CORE_VALUE="1"
    ;;
  *)
    echo "Unknown LADD_A1B_MODE=${LADD_A1B_MODE}. Use static or dynamic." >&2
    exit 1
    ;;
esac

case "$SIZE" in
  n|s) BATCH_SIZE_DEFAULT=64 ;;
  m|l) BATCH_SIZE_DEFAULT=32 ;;
  x) BATCH_SIZE_DEFAULT=16 ;;
esac
BATCH_SIZE="${BATCH_SIZE:-$BATCH_SIZE_DEFAULT}"

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

if [[ -z "${SAR_BASELINE:-}" ]]; then
  if [[ -n "${SAR_RUN_DIR:-}" ]]; then
    SAR_BASELINE="${SAR_RUN_DIR%/}/weights/best.pt"
  else
    SAR_BASELINE="$(find_latest_weight "runs_public/ogsod/hbb/baseline_controls/mosaic_baselines_20260615/sar_yolo11${SIZE}_hbb_mosaicE800_closeAt100_s${SEED}_*/weights/best.pt" || true)"
  fi
fi
if [[ -z "${RGB_TEACHER:-}" ]]; then
  if [[ -n "${RGB_RUN_DIR:-}" ]]; then
    RGB_TEACHER="${RGB_RUN_DIR%/}/weights/best.pt"
  else
    RGB_TEACHER="$(find_latest_weight "runs_public/ogsod/hbb/baseline_controls/mosaic_baselines_20260615/rgb_yolo11${SIZE}_hbb_mosaicE800_closeAt100_s${SEED}_*/weights/best.pt" || true)"
  fi
fi

DATA_CFG="${DATA_CFG:-shared/configs/datasets_public/ogsod1_sar_detect.yaml}"
TEACHER_DATA_CFG="${TEACHER_DATA_CFG:-shared/configs/datasets_public/ogsod1_rgb_detect.yaml}"
PHASE_SCRIPT="${LADD_PHASE_SCRIPT:-ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return
  fi
  if [[ -z "$path" || ! -f "$path" ]]; then
    echo "Missing ${label}: ${path:-<empty>}" >&2
    exit 1
  fi
}

require_file "$SAR_BASELINE" "SAR baseline checkpoint"
require_file "$RGB_TEACHER" "RGB teacher checkpoint"
require_file "$DATA_CFG" "student dataset YAML"
require_file "$TEACHER_DATA_CFG" "teacher dataset YAML"
require_file "$PHASE_SCRIPT" "phase launcher"

if [[ -n "${EXP_SUFFIX:-}" && -z "${RUN_TAG_SUFFIX:-}" ]]; then
  RUN_TAG_SUFFIX="_${EXP_SUFFIX}"
fi

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-${MODE_TAG}_yolo11${SIZE}_cap2_s${SEED}_mosaic_first100_close700${RUN_TAG_SUFFIX:-}_${DATE_TAG}}"
PROJECT_DIR="${PROJECT_DIR:-runs_public/ogsod/hbb/${MODE_PROJECT_KEY}/mosaic_first100_close700/yolo11${SIZE}/cap2}"
CHAIN_LOG_DIR="${CHAIN_LOG_DIR:-logs/${MODE_PROJECT_KEY}/mosaic_first100_close700/${RUN_TAG}_gpu${GPU_ID}}"
META_PATH="${CHAIN_LOG_DIR}/run_meta_clean_a1b.env"

EPOCHS_A1_VALUE="${EPOCHS_A1:-10}"
EPOCHS_B_VALUE="${EPOCHS_B:-${EPOCHS:-800}}"
PATIENCE_A_VALUE="${PATIENCE_A:-200}"
PATIENCE_B_VALUE="${PATIENCE_B:-$EPOCHS_B_VALUE}"
SAVE_PERIOD_VALUE="${SAVE_PERIOD:-100}"

A1_RUN_NAME="ladd_${MODE_TAG}_ogsod11${SIZE}_${RUN_TAG}_a1_e${EPOCHS_A1_VALUE}_b${BATCH_SIZE}_s${SEED}_gpu${GPU_ID}"
B_RUN_NAME="ladd_${MODE_TAG}_ogsod11${SIZE}_${RUN_TAG}_b_e${EPOCHS_B_VALUE}_b${BATCH_SIZE}_s${SEED}_gpu${GPU_ID}"
A1_LOG_DIR="${CHAIN_LOG_DIR}/a1"
B_LOG_DIR="${CHAIN_LOG_DIR}/b"

mkdir -p "$PROJECT_DIR" "$CHAIN_LOG_DIR"

RANK_D_NEG_CAP_VALUE="${RANK_D_NEG_CAP:-2.0}"
LAMBDA_REC_VALUE="${LAMBDA_REC:-0.1}"
LAMBDA_TASKL_VALUE="${LAMBDA_TASKL:-1.0}"
ALPHA_KD_VALUE="${ALPHA_KD:-1.0}"
ALPHA_S_REC_VALUE="${ALPHA_S_REC:-0.1}"
LAMBDA_REACH_VALUE="${LAMBDA_REACH:-1.0}"
LAMBDA_MATCH_INNER_VALUE="${LAMBDA_MATCH_INNER:-1.0}"
LAMBDA_RANK_INNER_VALUE="${LAMBDA_RANK_INNER:-1.0}"
USE_MASK_VALUE="${USE_MASK:-1}"
USE_FG_MASK_FOR_REACH_VALUE="${USE_FG_MASK_FOR_REACH:-1}"
USE_FG_MASK_FOR_REC_VALUE="${USE_FG_MASK_FOR_REC:-0}"
LADD_B_DET_ONLY_VALUE="0"
LADD_A2_DET_ONLY_VALUE="0"

A1_MOSAIC_VALUE="${A1_MOSAIC:-1.0}"
A1_CLOSE_MOSAIC_VALUE="${A1_CLOSE_MOSAIC:-0}"
B_MOSAIC_VALUE="${B_MOSAIC:-1.0}"
B_CLOSE_MOSAIC_VALUE="${B_CLOSE_MOSAIC:-700}"
MIXUP_VALUE="${MIXUP:-0.0}"
CUTMIX_VALUE="${CUTMIX:-0.0}"
DEGREES_VALUE="${DEGREES:-0.0}"
PERSPECTIVE_VALUE="${PERSPECTIVE:-0.0}"
TRANSLATE_VALUE="${TRANSLATE:-0.1}"
SCALE_VALUE="${SCALE:-0.5}"
FLIPLR_VALUE="${FLIPLR:-0.5}"
FLIPUD_VALUE="${FLIPUD:-0.0}"
HSV_H_VALUE="${HSV_H:-0.0}"
HSV_S_VALUE="${HSV_S:-0.0}"
HSV_V_VALUE="${HSV_V:-0.0}"
ERASING_VALUE="${ERASING:-0.0}"

A1_OPTIMIZER_VALUE="${A1_OPTIMIZER:-auto}"
A1_LR0_VALUE="${A1_LR0:-0.01}"
A1_LRF_VALUE="${A1_LRF:-0.01}"
A1_COS_LR_VALUE="${A1_COS_LR:-1}"
A1_WARMUP_EPOCHS_VALUE="${A1_WARMUP_EPOCHS:-3.0}"
A1_WARMUP_BIAS_LR_VALUE="${A1_WARMUP_BIAS_LR:-0.1}"
B_OPTIMIZER_VALUE="${B_OPTIMIZER:-auto}"
B_LR0_VALUE="${B_LR0:-0.01}"
B_LRF_VALUE="${B_LRF:-0.01}"
B_COS_LR_VALUE="${B_COS_LR:-1}"
B_WARMUP_EPOCHS_VALUE="${B_WARMUP_EPOCHS:-3.0}"
B_WARMUP_BIAS_LR_VALUE="${B_WARMUP_BIAS_LR:-0.1}"

write_meta() {
  {
    printf 'run_tag=%q\n' "$RUN_TAG"
    printf 'ladd_a1b_mode=%q\n' "$LADD_A1B_MODE"
    printf 'size=%q\n' "$SIZE"
    printf 'seed=%q\n' "$SEED"
    printf 'gpu_id=%q\n' "$GPU_ID"
    printf 'batch_size=%q\n' "$BATCH_SIZE"
    printf 'sar_baseline=%q\n' "$SAR_BASELINE"
    printf 'rgb_teacher=%q\n' "$RGB_TEACHER"
    printf 'data_cfg=%q\n' "$DATA_CFG"
    printf 'teacher_data_cfg=%q\n' "$TEACHER_DATA_CFG"
    printf 'phase_script=%q\n' "$PHASE_SCRIPT"
    printf 'project_dir=%q\n' "$PROJECT_DIR"
    printf 'chain_log_dir=%q\n' "$CHAIN_LOG_DIR"
    printf 'a1_run_name=%q\n' "$A1_RUN_NAME"
    printf 'b_run_name=%q\n' "$B_RUN_NAME"
    printf 'epochs_a1=%q\n' "$EPOCHS_A1_VALUE"
    printf 'epochs_b=%q\n' "$EPOCHS_B_VALUE"
    printf 'patience_a=%q\n' "$PATIENCE_A_VALUE"
    printf 'patience_b=%q\n' "$PATIENCE_B_VALUE"
    printf 'save_period=%q\n' "$SAVE_PERIOD_VALUE"
    printf 'workers=%q\n' "${WORKERS:-8}"
    printf 'imgsz=%q\n' "${IMGSZ:-256}"
    printf 'rank_d_neg_cap=%q\n' "$RANK_D_NEG_CAP_VALUE"
    printf 'lambda_rec=%q\n' "$LAMBDA_REC_VALUE"
    printf 'lambda_taskL=%q\n' "$LAMBDA_TASKL_VALUE"
    printf 'alpha_kd=%q\n' "$ALPHA_KD_VALUE"
    printf 'alpha_s_rec=%q\n' "$ALPHA_S_REC_VALUE"
    printf 'lambda_reach=%q\n' "$LAMBDA_REACH_VALUE"
    printf 'lambda_match_inner=%q\n' "$LAMBDA_MATCH_INNER_VALUE"
    printf 'lambda_rank_inner=%q\n' "$LAMBDA_RANK_INNER_VALUE"
    printf 'use_mask=%q\n' "$USE_MASK_VALUE"
    printf 'use_fg_mask_for_reach=%q\n' "$USE_FG_MASK_FOR_REACH_VALUE"
    printf 'use_fg_mask_for_rec=%q\n' "$USE_FG_MASK_FOR_REC_VALUE"
    printf 'ladd_b_a2_core=%q\n' "$LADD_B_A2_CORE_VALUE"
    printf 'ladd_b_det_only=%q\n' "$LADD_B_DET_ONLY_VALUE"
    printf 'ladd_a2_det_only=%q\n' "$LADD_A2_DET_ONLY_VALUE"
    printf 'comparison_kd_profile=%q\n' "none"
    printf 'profile_kd_weight=%q\n' "0.0"
    printf 'profile_kd_replace_base=%q\n' "0"
    printf 'a1_mosaic=%q\n' "$A1_MOSAIC_VALUE"
    printf 'a1_close_mosaic=%q\n' "$A1_CLOSE_MOSAIC_VALUE"
    printf 'b_mosaic=%q\n' "$B_MOSAIC_VALUE"
    printf 'b_close_mosaic=%q\n' "$B_CLOSE_MOSAIC_VALUE"
    printf 'mixup=%q\n' "$MIXUP_VALUE"
    printf 'cutmix=%q\n' "$CUTMIX_VALUE"
    printf 'degrees=%q\n' "$DEGREES_VALUE"
    printf 'perspective=%q\n' "$PERSPECTIVE_VALUE"
    printf 'translate=%q\n' "$TRANSLATE_VALUE"
    printf 'scale=%q\n' "$SCALE_VALUE"
    printf 'fliplr=%q\n' "$FLIPLR_VALUE"
    printf 'flipud=%q\n' "$FLIPUD_VALUE"
    printf 'hsv_h=%q\n' "$HSV_H_VALUE"
    printf 'hsv_s=%q\n' "$HSV_S_VALUE"
    printf 'hsv_v=%q\n' "$HSV_V_VALUE"
    printf 'erasing=%q\n' "$ERASING_VALUE"
    printf 'optimizer_a1=%q\n' "$A1_OPTIMIZER_VALUE"
    printf 'lr0_a1=%q\n' "$A1_LR0_VALUE"
    printf 'lrf_a1=%q\n' "$A1_LRF_VALUE"
    printf 'cos_lr_a1=%q\n' "$A1_COS_LR_VALUE"
    printf 'warmup_epochs_a1=%q\n' "$A1_WARMUP_EPOCHS_VALUE"
    printf 'warmup_bias_lr_a1=%q\n' "$A1_WARMUP_BIAS_LR_VALUE"
    printf 'optimizer_b=%q\n' "$B_OPTIMIZER_VALUE"
    printf 'lr0_b=%q\n' "$B_LR0_VALUE"
    printf 'lrf_b=%q\n' "$B_LRF_VALUE"
    printf 'cos_lr_b=%q\n' "$B_COS_LR_VALUE"
    printf 'warmup_epochs_b=%q\n' "$B_WARMUP_EPOCHS_VALUE"
    printf 'warmup_bias_lr_b=%q\n' "$B_WARMUP_BIAS_LR_VALUE"
    printf 'git_commit=%q\n' "$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  } > "$META_PATH"
}

run_phase() {
  local phase="$1"
  local model="$2"
  local run_name="$3"
  local log_dir="$4"
  local epochs="$5"
  local patience="$6"
  local det_scale="$7"
  local mosaic="$8"
  local close_mosaic="$9"
  local optimizer="${10}"
  local lr0="${11}"
  local lrf="${12}"
  local cos_lr="${13}"
  local warmup_epochs="${14}"
  local warmup_bias_lr="${15}"

  local -a cmd=(
    env
    "SAR_BASELINE=${SAR_BASELINE}"
    "RGB_TEACHER=${RGB_TEACHER}"
    "MODEL=${model}"
    "DATA_CFG=${DATA_CFG}"
    "TEACHER_DATA_CFG=${TEACHER_DATA_CFG}"
    "GPU_ID=${GPU_ID}"
    "SEED=${SEED}"
    "BATCH_SIZE=${BATCH_SIZE}"
    "WORKERS=${WORKERS:-8}"
    "IMGSZ=${IMGSZ:-256}"
    "EPOCHS=${epochs}"
    "PATIENCE=${patience}"
    "DET_LOSS_SCALE=${det_scale}"
    "PROJECT_DIR=${PROJECT_DIR}"
    "RUN_NAME=${run_name}"
    "LOG_DIR=${log_dir}"
    "MOSAIC=${mosaic}"
    "CLOSE_MOSAIC=${close_mosaic}"
    "MIXUP=${MIXUP_VALUE}"
    "CUTMIX=${CUTMIX_VALUE}"
    "DEGREES=${DEGREES_VALUE}"
    "PERSPECTIVE=${PERSPECTIVE_VALUE}"
    "TRANSLATE=${TRANSLATE_VALUE}"
    "SCALE=${SCALE_VALUE}"
    "FLIPLR=${FLIPLR_VALUE}"
    "FLIPUD=${FLIPUD_VALUE}"
    "HSV_H=${HSV_H_VALUE}"
    "HSV_S=${HSV_S_VALUE}"
    "HSV_V=${HSV_V_VALUE}"
    "ERASING=${ERASING_VALUE}"
    "OPTIMIZER=${optimizer}"
    "LR0=${lr0}"
    "LRF=${lrf}"
    "COS_LR=${cos_lr}"
    "WARMUP_EPOCHS=${warmup_epochs}"
    "WARMUP_BIAS_LR=${warmup_bias_lr}"
    "SAVE_PERIOD=${SAVE_PERIOD_VALUE}"
    "RANK_D_NEG_CAP=${RANK_D_NEG_CAP_VALUE}"
    "LAMBDA_REC=${LAMBDA_REC_VALUE}"
    "LAMBDA_TASKL=${LAMBDA_TASKL_VALUE}"
    "ALPHA_KD=${ALPHA_KD_VALUE}"
    "ALPHA_S_REC=${ALPHA_S_REC_VALUE}"
    "LAMBDA_REACH=${LAMBDA_REACH_VALUE}"
    "LAMBDA_MATCH_INNER=${LAMBDA_MATCH_INNER_VALUE}"
    "LAMBDA_RANK_INNER=${LAMBDA_RANK_INNER_VALUE}"
    "USE_MASK=${USE_MASK_VALUE}"
    "USE_FG_MASK_FOR_REACH=${USE_FG_MASK_FOR_REACH_VALUE}"
    "USE_FG_MASK_FOR_REC=${USE_FG_MASK_FOR_REC_VALUE}"
    "LADD_B_A2_CORE=${LADD_B_A2_CORE_VALUE}"
    "LADD_B_DET_ONLY=${LADD_B_DET_ONLY_VALUE}"
    "LADD_A2_DET_ONLY=${LADD_A2_DET_ONLY_VALUE}"
    "COMPARISON_KD_PROFILE=none"
    "PROFILE_KD_WEIGHT=0.0"
    "PROFILE_KD_REPLACE_BASE=0"
    "EXIST_OK=${EXIST_OK:-0}"
    "PYTHON_BIN=${PYTHON_BIN:-python3}"
    bash "$PHASE_SCRIPT" hbb "$phase" "$RUN_TAG"
  )

  echo
  echo "[$(date '+%F %T')] Prepared LADD-clean phase ${phase}: ${run_name}"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  "${cmd[@]}"
}

write_meta

echo "[$(date '+%F %T')] LADD-clean/A1B mode=${LADD_A1B_MODE} yolo11${SIZE} seed=${SEED} gpu=${GPU_ID}"
echo "sar_baseline=${SAR_BASELINE:-<not-found>}"
echo "rgb_teacher=${RGB_TEACHER:-<not-found>}"
echo "meta=${META_PATH}"

if [[ -e "${PROJECT_DIR}/${A1_RUN_NAME}" && "${EXIST_OK:-0}" != "1" && "${DRY_RUN:-0}" != "1" ]]; then
  echo "A1 run directory already exists: ${PROJECT_DIR}/${A1_RUN_NAME}" >&2
  echo "Set EXIST_OK=1 only if intentional." >&2
  exit 1
fi
if [[ -e "${PROJECT_DIR}/${B_RUN_NAME}" && "${EXIST_OK:-0}" != "1" && "${DRY_RUN:-0}" != "1" ]]; then
  echo "B run directory already exists: ${PROJECT_DIR}/${B_RUN_NAME}" >&2
  echo "Set EXIST_OK=1 only if intentional." >&2
  exit 1
fi

run_phase \
  a1 \
  "$SAR_BASELINE" \
  "$A1_RUN_NAME" \
  "$A1_LOG_DIR" \
  "$EPOCHS_A1_VALUE" \
  "$PATIENCE_A_VALUE" \
  "${A1_DET_LOSS_SCALE:-0.0}" \
  "$A1_MOSAIC_VALUE" \
  "$A1_CLOSE_MOSAIC_VALUE" \
  "$A1_OPTIMIZER_VALUE" \
  "$A1_LR0_VALUE" \
  "$A1_LRF_VALUE" \
  "$A1_COS_LR_VALUE" \
  "$A1_WARMUP_EPOCHS_VALUE" \
  "$A1_WARMUP_BIAS_LR_VALUE"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  A1_BEST="<A1 actual_run_dir>/weights/best.pt"
else
  A1_ACTUAL_RUN_DIR="$(cat "${A1_LOG_DIR}/actual_run_dir.txt")"
  A1_BEST="${A1_ACTUAL_RUN_DIR}/weights/best.pt"
  require_file "$A1_BEST" "A1 best checkpoint"
  {
    printf 'a1_actual_run_dir=%q\n' "$A1_ACTUAL_RUN_DIR"
    printf 'a1_best=%q\n' "$A1_BEST"
  } >> "$META_PATH"
fi

run_phase \
  b \
  "$A1_BEST" \
  "$B_RUN_NAME" \
  "$B_LOG_DIR" \
  "$EPOCHS_B_VALUE" \
  "$PATIENCE_B_VALUE" \
  "${B_DET_LOSS_SCALE:-1.0}" \
  "$B_MOSAIC_VALUE" \
  "$B_CLOSE_MOSAIC_VALUE" \
  "$B_OPTIMIZER_VALUE" \
  "$B_LR0_VALUE" \
  "$B_LRF_VALUE" \
  "$B_COS_LR_VALUE" \
  "$B_WARMUP_EPOCHS_VALUE" \
  "$B_WARMUP_BIAS_LR_VALUE"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo
  echo "DRY_RUN=1, not launching A1/B."
  exit 0
fi

B_ACTUAL_RUN_DIR="$(cat "${B_LOG_DIR}/actual_run_dir.txt")"
{
  printf 'b_actual_run_dir=%q\n' "$B_ACTUAL_RUN_DIR"
  printf 'completed_at=%q\n' "$(date '+%F %T')"
} >> "$META_PATH"

echo
echo "[$(date '+%F %T')] LADD-clean/A1B finished."
echo "A1 best: ${A1_BEST}"
echo "B run:   ${B_ACTUAL_RUN_DIR}"
echo "Meta:    ${META_PATH}"
