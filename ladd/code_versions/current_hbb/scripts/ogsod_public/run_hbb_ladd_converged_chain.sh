#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/ogsod_public/run_hbb_ladd_converged_chain.sh [run_tag]

Runs the 2026-05-24 OGSOD HBB LADD chain on top of the converged
YOLO11n baseline protocol:
  A1=10 -> A2=50 -> B=800

Required:
  SAR_BASELINE=/path/to/converged_sar_best.pt
  RGB_TEACHER=/path/to/converged_rgb_best.pt

Defaults:
  imgsz=256, batch=64, seed=0, fg reach enabled.
  A2 and B use the formal stability guard:
    optimizer=MuSGD, lr0=0.001, lrf=0.01, warmup_epochs=0,
    warmup_bias_lr=0.001.
  B phase uses cos_lr=True.

Common overrides:
  GPU_ID, SEED, PROJECT_DIR, CHAIN_LOG_DIR, EPOCHS_A1, EPOCHS_A2, EPOCHS_B,
  BATCH_SIZE, WORKERS, PATIENCE_B, B_CLOSE_MOSAIC, B_CLOSE_AT_EPOCH,
  RANK_D_NEG_CAP, LAMBDA_ANTI_COLLAPSE,
  ANTI_COLLAPSE_FLOOR, A1_/A2_/B_ optimizer overrides, B_FREEZE_BN_AFTER_EPOCH,
  LADD_DIAG_LOG_BN, LADD_DIAG_LOG_GRAD, LADD_GRAD_CLIP_NORM,
  LADD_ASSERT_PHASE_FREEZE, LADD_DIAG_LOG_EVERY, LADD_CHAIN_PHASES, EXIST_OK=1,
  LADD_KD_DECAY_MODE, LADD_KD_DECAY_START_EPOCH, LADD_KD_DECAY_END_EPOCH,
  LADD_KD_FINAL_MULT, LADD_KD_STOP_AFTER_EPOCH,
  LADD_B_LOSS_WARMUP_MODE, LADD_B_LOSS_WARMUP_START_EPOCH,
  LADD_B_LOSS_WARMUP_END_EPOCH, LADD_B_LOSS_WARMUP_FINAL_MULT,
  LADD_B_LOSS_WARMUP_SCOPE, LADD_B_A2_CORE, LADD_B_DET_ONLY, LADD_A2_DET_ONLY,
  START_MODEL, VALIDATE_BEFORE_TRAIN, B_DETECTOR_SOURCE, B_DECOMP_SOURCE,
  B_SPLIT_LOAD_STRICT, B_RESET_STUDENT_FROM_SCRATCH, B_LOAD_STUDENT_SPLIT,
  B_LOAD_STUDENT_REACHABILITY, B_LOAD_STUDENT_AUX
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

RUN_TAG="${1:-converged800_ladd_$(date +%Y%m%d_%H%M%S)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ROOT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  ROOT_DIR="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
fi
cd "$ROOT_DIR"

: "${SAR_BASELINE:?Set SAR_BASELINE to the converged SAR baseline checkpoint.}"
: "${RGB_TEACHER:?Set RGB_TEACHER to the converged RGB teacher checkpoint.}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
MODEL_SIZE="${MODEL_SIZE:-n}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WORKERS="${WORKERS:-8}"
IMGSZ="${IMGSZ:-256}"
EPOCHS_A1="${EPOCHS_A1:-10}"
EPOCHS_A2="${EPOCHS_A2:-50}"
EPOCHS_B="${EPOCHS_B:-800}"
PATIENCE_A="${PATIENCE_A:-200}"
PATIENCE_B="${PATIENCE_B:-80}"
PROJECT_DIR="${PROJECT_DIR:-runs_public/ogsod/hbb/ladd_converged_20260524}"
CHAIN_LOG_DIR="${CHAIN_LOG_DIR:-logs/ogsod_public/hbb_${RUN_TAG}_a1_a2_b_converged_chain}"
LADD_CHAIN_PHASES="${LADD_CHAIN_PHASES:-a1,a2,b}"
GIT_COMMIT="${GIT_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
SERVER_TAG="${SERVER_TAG:-unknown_server}"

if [[ -n "${B_CLOSE_MOSAIC:-}" && -n "${B_CLOSE_AT_EPOCH:-}" ]]; then
  echo "Both B_CLOSE_MOSAIC and B_CLOSE_AT_EPOCH are set; B_CLOSE_MOSAIC=${B_CLOSE_MOSAIC} takes precedence." >&2
fi

mkdir -p "$CHAIN_LOG_DIR"
manifest="${CHAIN_LOG_DIR}/manifest.txt"
{
  echo "git_commit=${GIT_COMMIT}"
  echo "server_tag=${SERVER_TAG}"
  echo "python_bin=${PYTHON_BIN:-python3}"
  echo "task=hbb"
  echo "chain=${LADD_CHAIN_PHASES}"
  echo "run_tag=${RUN_TAG}"
  echo "start_model=${START_MODEL:-}"
  echo "sar_baseline=${SAR_BASELINE}"
  echo "rgb_teacher=${RGB_TEACHER}"
  echo "seed=${SEED}"
  echo "gpu_id=${GPU_ID}"
  echo "imgsz=${IMGSZ}"
  echo "batch_size=${BATCH_SIZE}"
  echo "epochs_a1=${EPOCHS_A1}"
  echo "epochs_a2=${EPOCHS_A2}"
  echo "epochs_b=${EPOCHS_B}"
  echo "a2_schedule=optimizer=${A2_OPTIMIZER:-MuSGD}, lr0=${A2_LR0:-0.001}, lrf=${A2_LRF:-0.01}, warmup_epochs=${A2_WARMUP_EPOCHS:-0}, warmup_bias_lr=${A2_WARMUP_BIAS_LR:-0.001}"
  echo "b_schedule=optimizer=${B_OPTIMIZER:-MuSGD}, cos_lr=${B_COS_LR:-1}, lr0=${B_LR0:-0.001}, lrf=${B_LRF:-0.01}, warmup_epochs=${B_WARMUP_EPOCHS:-0}, warmup_bias_lr=${B_WARMUP_BIAS_LR:-0.001}, close_mosaic=${B_CLOSE_MOSAIC:-}, close_at_epoch=${B_CLOSE_AT_EPOCH:-${EPOCHS_B}}"
  echo "b_close_mosaic=${B_CLOSE_MOSAIC:-}"
  echo "b_close_at_epoch=${B_CLOSE_AT_EPOCH:-${EPOCHS_B}}"
  echo "alpha_kd=${ALPHA_KD:-1.0}"
  echo "alpha_s_rec=${ALPHA_S_REC:-0.1}"
  echo "alpha_sep=${ALPHA_SEP:-0.05}"
  echo "lambda_reach=${LAMBDA_REACH:-1.0}"
  echo "lambda_match_inner=${LAMBDA_MATCH_INNER:-1.0}"
  echo "lambda_rank_inner=${LAMBDA_RANK_INNER:-1.0}"
  echo "lambda_residual_aux=${LAMBDA_RESIDUAL_AUX:-0.25}"
  echo "residual_aux_mode=${RESIDUAL_AUX_MODE:-energy}"
  echo "b_freeze_bn_after_epoch=${B_FREEZE_BN_AFTER_EPOCH:--1}"
  echo "ladd_diag_log_bn=${LADD_DIAG_LOG_BN:-1}"
  echo "ladd_diag_log_grad=${LADD_DIAG_LOG_GRAD:-0}"
  echo "ladd_grad_clip_norm=${LADD_GRAD_CLIP_NORM:-0.0}"
  echo "ladd_assert_phase_freeze=${LADD_ASSERT_PHASE_FREEZE:-0}"
  echo "ladd_diag_log_every=${LADD_DIAG_LOG_EVERY:-1}"
  echo "ladd_kd_decay_mode=${LADD_KD_DECAY_MODE:-none}"
  echo "ladd_kd_decay_start_epoch=${LADD_KD_DECAY_START_EPOCH:--1}"
  echo "ladd_kd_decay_end_epoch=${LADD_KD_DECAY_END_EPOCH:--1}"
  echo "ladd_kd_final_mult=${LADD_KD_FINAL_MULT:-1.0}"
  echo "ladd_kd_stop_after_epoch=${LADD_KD_STOP_AFTER_EPOCH:--1}"
  echo "ladd_b_loss_warmup_mode=${LADD_B_LOSS_WARMUP_MODE:-none}"
  echo "ladd_b_loss_warmup_start_epoch=${LADD_B_LOSS_WARMUP_START_EPOCH:--1}"
  echo "ladd_b_loss_warmup_end_epoch=${LADD_B_LOSS_WARMUP_END_EPOCH:--1}"
  echo "ladd_b_loss_warmup_final_mult=${LADD_B_LOSS_WARMUP_FINAL_MULT:-1.0}"
  echo "ladd_b_loss_warmup_scope=${LADD_B_LOSS_WARMUP_SCOPE:-core}"
  echo "ladd_b_a2_core=${LADD_B_A2_CORE:-0}"
  echo "ladd_b_det_only=${LADD_B_DET_ONLY:-0}"
  echo "ladd_a2_det_only=${LADD_A2_DET_ONLY:-0}"
  echo "validate_before_train=${VALIDATE_BEFORE_TRAIN:-0}"
  echo "b_detector_source=${B_DETECTOR_SOURCE:-}"
  echo "b_decomp_source=${B_DECOMP_SOURCE:-}"
  echo "b_split_load_strict=${B_SPLIT_LOAD_STRICT:-0}"
  echo "b_reset_student_from_scratch=${B_RESET_STUDENT_FROM_SCRATCH:-0}"
  echo "b_load_student_split=${B_LOAD_STUDENT_SPLIT:-0}"
  echo "b_load_student_reachability=${B_LOAD_STUDENT_REACHABILITY:-1}"
  echo "b_load_student_aux=${B_LOAD_STUDENT_AUX:-0}"
  echo "rank_d_neg_cap=${RANK_D_NEG_CAP:-4.0}"
  echo "lambda_anti_collapse=${LAMBDA_ANTI_COLLAPSE:-0.0}"
  echo "anti_collapse_floor=${ANTI_COLLAPSE_FLOOR:-0.0}"
} > "$manifest"

current_model="${START_MODEL:-$SAR_BASELINE}"

run_phase() {
  local phase="$1"
  local epochs="$2"
  local patience="$3"
  local phase_log_dir="${CHAIN_LOG_DIR}/${phase}"
  local run_name="ladd_hbb_ogsod11${MODEL_SIZE}_${RUN_TAG}_${phase}_e${epochs}_b${BATCH_SIZE}_s${SEED}_gpu${GPU_ID}"
  local phase_prefix=""

  echo "[$(date '+%F %T')] phase=${phase} model=${current_model}" | tee -a "${CHAIN_LOG_DIR}/chain.log"
  local env_args=(
    MODEL="$current_model"
    SAR_BASELINE="$SAR_BASELINE"
    RGB_TEACHER="$RGB_TEACHER"
    GPU_ID="$GPU_ID"
    SEED="$SEED"
    IMGSZ="$IMGSZ"
    BATCH_SIZE="$BATCH_SIZE"
    WORKERS="$WORKERS"
    EPOCHS="$epochs"
    PATIENCE="$patience"
    PHASE_MIN_EPOCHS="$epochs"
    PROJECT_DIR="$PROJECT_DIR"
    LOG_DIR="$phase_log_dir"
    RUN_NAME="$run_name"
    PYTHON_BIN="${PYTHON_BIN:-python3}"
    USE_FG_MASK_FOR_REACH="${USE_FG_MASK_FOR_REACH:-1}"
    USE_FG_MASK_FOR_REC="${USE_FG_MASK_FOR_REC:-0}"
    MOSAIC="${MOSAIC:-1.0}"
    MIXUP="${MIXUP:-0.0}"
    CUTMIX="${CUTMIX:-0.0}"
    DEGREES="${DEGREES:-0.0}"
    PERSPECTIVE="${PERSPECTIVE:-0.0}"
    TRANSLATE="${TRANSLATE:-0.1}"
    SCALE="${SCALE:-0.5}"
    FLIPLR="${FLIPLR:-0.5}"
    FLIPUD="${FLIPUD:-0.0}"
    HSV_H="${HSV_H:-0.0}"
    HSV_S="${HSV_S:-0.0}"
    HSV_V="${HSV_V:-0.0}"
    ERASING="${ERASING:-0.0}"
    RANK_D_NEG_CAP="${RANK_D_NEG_CAP:-4.0}"
    LAMBDA_ANTI_COLLAPSE="${LAMBDA_ANTI_COLLAPSE:-0.0}"
    ANTI_COLLAPSE_FLOOR="${ANTI_COLLAPSE_FLOOR:-0.0}"
    ALPHA_KD="${ALPHA_KD:-1.0}"
    ALPHA_S_REC="${ALPHA_S_REC:-0.1}"
    ALPHA_SEP="${ALPHA_SEP:-0.05}"
    LAMBDA_REACH="${LAMBDA_REACH:-1.0}"
    LAMBDA_MATCH_INNER="${LAMBDA_MATCH_INNER:-1.0}"
    LAMBDA_RANK_INNER="${LAMBDA_RANK_INNER:-1.0}"
    LAMBDA_RESIDUAL_AUX="${LAMBDA_RESIDUAL_AUX:-0.25}"
    RESIDUAL_AUX_MODE="${RESIDUAL_AUX_MODE:-energy}"
    EXIST_OK="${EXIST_OK:-0}"
  )

  if [[ "$phase" == "b" ]]; then
    env_args+=(
      COS_LR="${B_COS_LR:-1}"
      OPTIMIZER="${B_OPTIMIZER:-MuSGD}"
      LR0="${B_LR0:-0.001}"
      LRF="${B_LRF:-0.01}"
      WARMUP_EPOCHS="${B_WARMUP_EPOCHS:-0}"
      WARMUP_BIAS_LR="${B_WARMUP_BIAS_LR:-0.001}"
      SAVE_PERIOD="${SAVE_PERIOD:-100}"
      FREEZE_BN_AFTER_EPOCH="${B_FREEZE_BN_AFTER_EPOCH:--1}"
    )
    if [[ -n "${B_CLOSE_MOSAIC:-}" ]]; then
      env_args+=(CLOSE_MOSAIC="${B_CLOSE_MOSAIC}")
    else
      env_args+=(CLOSE_AT_EPOCH="${B_CLOSE_AT_EPOCH:-${EPOCHS_B}}")
    fi
    phase_prefix="B"
  else
    env_args+=(CLOSE_MOSAIC="${A_CLOSE_MOSAIC:-10}")
    if [[ "$phase" == "a1" ]]; then
      phase_prefix="A1"
    else
      phase_prefix="A2"
      env_args+=(
        OPTIMIZER="${A2_OPTIMIZER:-MuSGD}"
        LR0="${A2_LR0:-0.001}"
        LRF="${A2_LRF:-0.01}"
        WARMUP_EPOCHS="${A2_WARMUP_EPOCHS:-0}"
        WARMUP_BIAS_LR="${A2_WARMUP_BIAS_LR:-0.001}"
        DET_LOSS_SCALE="${A2_DET_LOSS_SCALE:-1.0}"
      )
    fi
  fi

  local phase_lr0_var="${phase_prefix}_LR0"
  local phase_lrf_var="${phase_prefix}_LRF"
  local phase_cos_var="${phase_prefix}_COS_LR"
  local phase_optimizer_var="${phase_prefix}_OPTIMIZER"
  local phase_warmup_epochs_var="${phase_prefix}_WARMUP_EPOCHS"
  local phase_warmup_bias_lr_var="${phase_prefix}_WARMUP_BIAS_LR"
  local phase_warmup_momentum_var="${phase_prefix}_WARMUP_MOMENTUM"
  local phase_det_loss_scale_var="${phase_prefix}_DET_LOSS_SCALE"
  local phase_freeze_bn_stats_var="${phase_prefix}_FREEZE_BN_STATS"
  [[ -n "${!phase_lr0_var:-}" ]] && env_args+=(LR0="${!phase_lr0_var}")
  [[ -n "${!phase_lrf_var:-}" ]] && env_args+=(LRF="${!phase_lrf_var}")
  [[ -n "${!phase_cos_var:-}" ]] && env_args+=(COS_LR="${!phase_cos_var}")
  [[ -n "${!phase_optimizer_var:-}" ]] && env_args+=(OPTIMIZER="${!phase_optimizer_var}")
  [[ -n "${!phase_warmup_epochs_var:-}" ]] && env_args+=(WARMUP_EPOCHS="${!phase_warmup_epochs_var}")
  [[ -n "${!phase_warmup_bias_lr_var:-}" ]] && env_args+=(WARMUP_BIAS_LR="${!phase_warmup_bias_lr_var}")
  [[ -n "${!phase_warmup_momentum_var:-}" ]] && env_args+=(WARMUP_MOMENTUM="${!phase_warmup_momentum_var}")
  [[ -n "${!phase_det_loss_scale_var:-}" ]] && env_args+=(DET_LOSS_SCALE="${!phase_det_loss_scale_var}")
  [[ -n "${!phase_freeze_bn_stats_var:-}" ]] && env_args+=(FREEZE_BN_STATS="${!phase_freeze_bn_stats_var}")
  env_args+=(
    LADD_DIAG_LOG_BN="${LADD_DIAG_LOG_BN:-1}"
    LADD_DIAG_LOG_GRAD="${LADD_DIAG_LOG_GRAD:-0}"
    LADD_GRAD_CLIP_NORM="${LADD_GRAD_CLIP_NORM:-0.0}"
    LADD_ASSERT_PHASE_FREEZE="${LADD_ASSERT_PHASE_FREEZE:-0}"
    LADD_DIAG_LOG_EVERY="${LADD_DIAG_LOG_EVERY:-1}"
    LADD_KD_DECAY_MODE="${LADD_KD_DECAY_MODE:-none}"
    LADD_KD_DECAY_START_EPOCH="${LADD_KD_DECAY_START_EPOCH:--1}"
    LADD_KD_DECAY_END_EPOCH="${LADD_KD_DECAY_END_EPOCH:--1}"
    LADD_KD_FINAL_MULT="${LADD_KD_FINAL_MULT:-1.0}"
    LADD_KD_STOP_AFTER_EPOCH="${LADD_KD_STOP_AFTER_EPOCH:--1}"
    LADD_B_LOSS_WARMUP_MODE="${LADD_B_LOSS_WARMUP_MODE:-none}"
    LADD_B_LOSS_WARMUP_START_EPOCH="${LADD_B_LOSS_WARMUP_START_EPOCH:--1}"
    LADD_B_LOSS_WARMUP_END_EPOCH="${LADD_B_LOSS_WARMUP_END_EPOCH:--1}"
    LADD_B_LOSS_WARMUP_FINAL_MULT="${LADD_B_LOSS_WARMUP_FINAL_MULT:-1.0}"
    LADD_B_LOSS_WARMUP_SCOPE="${LADD_B_LOSS_WARMUP_SCOPE:-core}"
    LADD_B_A2_CORE="${LADD_B_A2_CORE:-0}"
    LADD_B_DET_ONLY="${LADD_B_DET_ONLY:-0}"
    LADD_A2_DET_ONLY="${LADD_A2_DET_ONLY:-0}"
    VALIDATE_BEFORE_TRAIN="${VALIDATE_BEFORE_TRAIN:-0}"
  )
  if [[ "$phase" == "b" ]]; then
    env_args+=(
      B_DETECTOR_SOURCE="${B_DETECTOR_SOURCE:-}"
      B_DECOMP_SOURCE="${B_DECOMP_SOURCE:-}"
      B_SPLIT_LOAD_STRICT="${B_SPLIT_LOAD_STRICT:-0}"
      B_RESET_STUDENT_FROM_SCRATCH="${B_RESET_STUDENT_FROM_SCRATCH:-0}"
      B_LOAD_STUDENT_SPLIT="${B_LOAD_STUDENT_SPLIT:-0}"
      B_LOAD_STUDENT_REACHABILITY="${B_LOAD_STUDENT_REACHABILITY:-1}"
      B_LOAD_STUDENT_AUX="${B_LOAD_STUDENT_AUX:-0}"
    )
  fi

  env "${env_args[@]}" ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh hbb "$phase" "$RUN_TAG"
  local actual_run_dir
  actual_run_dir="$(cat "${phase_log_dir}/actual_run_dir.txt")"
  echo "${phase}=${actual_run_dir}" >> "$manifest"
  local next_model="${actual_run_dir}/weights/best.pt"
  if [[ ! -f "$next_model" ]]; then
    echo "Missing best checkpoint after phase ${phase}: ${next_model}" >&2
    exit 1
  fi
  current_model="$next_model"
}

IFS=',' read -r -a requested_phases <<< "$LADD_CHAIN_PHASES"
first_phase="${requested_phases[0]}"
if [[ "$first_phase" != "a1" && -z "${START_MODEL:-}" ]]; then
  if [[ "$first_phase" == "b" && -n "${B_DETECTOR_SOURCE:-}" && -n "${B_DECOMP_SOURCE:-}" ]]; then
    :
  else
    echo "LADD_CHAIN_PHASES starts from ${first_phase}, but START_MODEL is not set." >&2
    echo "Only B split-load may omit START_MODEL, and it requires both B_DETECTOR_SOURCE and B_DECOMP_SOURCE." >&2
    exit 2
  fi
fi
for phase in "${requested_phases[@]}"; do
  case "$phase" in
    a1) run_phase a1 "$EPOCHS_A1" "$PATIENCE_A" ;;
    a2) run_phase a2 "$EPOCHS_A2" "$PATIENCE_A" ;;
    b) run_phase b "$EPOCHS_B" "$PATIENCE_B" ;;
    *)
      echo "Unsupported LADD_CHAIN_PHASES entry: ${phase}. Valid sequence entries: a1,a2,b" >&2
      exit 2
      ;;
  esac
done

echo "[$(date '+%F %T')] Chain complete. Final model: ${current_model}" | tee -a "${CHAIN_LOG_DIR}/chain.log"
printf '%s\n' "$current_model" > "${CHAIN_LOG_DIR}/final_model.txt"
