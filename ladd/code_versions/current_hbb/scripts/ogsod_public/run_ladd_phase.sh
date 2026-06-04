#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/ogsod_public/run_ladd_phase.sh <hbb|obb> <a1|a2|b|c> [run_tag]

Notes:
  b1/b2 are available only for the HBB experimental entry.

Required:
  SAR_BASELINE=/path/to/sar/best.pt
  RGB_TEACHER=/path/to/rgb/best.pt

Usually set by prepare/baseline scripts:
  OGSOD_PREP_ROOT=data/ogsod_public_prepared

Phase chaining:
  phase a1 defaults MODEL=$SAR_BASELINE.
  later phases require MODEL=/path/to/previous/phase/weights/best.pt unless provided by run_ladd_chain.sh.

Detector schedule overrides:
  LR0, LRF, COS_LR=1, OPTIMIZER, WARMUP_EPOCHS, WARMUP_BIAS_LR,
  WARMUP_MOMENTUM, SAVE_PERIOD, CLOSE_MOSAIC, CLOSE_AT_EPOCH,
  MOSAIC, MIXUP, CUTMIX, HSV_H, HSV_S, HSV_V, ERASING

Reach anti-collapse overrides:
  RANK_D_NEG_CAP, LAMBDA_ANTI_COLLAPSE, ANTI_COLLAPSE_FLOOR

Note:
  Ultralytics close_mosaic means "number of final epochs with mosaic off".
  If CLOSE_AT_EPOCH is set, this launcher converts it to
  close_mosaic=max(EPOCHS-CLOSE_AT_EPOCH, 0) for the current phase.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TASK="${1:-}"
PHASE="${2:-}"
RUN_TAG="${3:-$(date +%Y%m%d_%H%M%S)}"
if [[ "$TASK" != "hbb" && "$TASK" != "obb" ]]; then
  usage >&2
  exit 1
fi
case "$PHASE" in
  a1|a2|b|c|b1|b2) ;;
  *) usage >&2; exit 1 ;;
esac
if [[ "$TASK" == "obb" && ( "$PHASE" == "b1" || "$PHASE" == "b2" ) ]]; then
  echo "OBB LADD currently supports phases a1/a2/b/c. b1/b2 are HBB-only experimental phases." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

case "$PHASE" in
  a1)
    DEFAULT_EPOCHS=50
    DEFAULT_PATIENCE=200
    DEFAULT_DET_LOSS_SCALE=0.0
    DEFAULT_REACH_TARGET_MODE="coupled"
    DEFAULT_KD_TARGET_MODE="detach"
    ;;
  a2)
    DEFAULT_EPOCHS=100
    DEFAULT_PATIENCE=200
    DEFAULT_DET_LOSS_SCALE=1.0
    DEFAULT_REACH_TARGET_MODE="coupled"
    DEFAULT_KD_TARGET_MODE="detach"
    ;;
  b)
    DEFAULT_EPOCHS=200
    DEFAULT_PATIENCE=200
    DEFAULT_DET_LOSS_SCALE=1.0
    DEFAULT_REACH_TARGET_MODE="detach"
    DEFAULT_KD_TARGET_MODE="detach"
    ;;
  c)
    DEFAULT_EPOCHS=150
    DEFAULT_PATIENCE=50
    DEFAULT_DET_LOSS_SCALE=1.0
    DEFAULT_REACH_TARGET_MODE="detach"
    DEFAULT_KD_TARGET_MODE="detach"
    ;;
  b1)
    DEFAULT_EPOCHS=50
    DEFAULT_PATIENCE=200
    DEFAULT_DET_LOSS_SCALE=0.0
    DEFAULT_REACH_TARGET_MODE="detach"
    DEFAULT_KD_TARGET_MODE="detach"
    ;;
  b2)
    DEFAULT_EPOCHS=300
    DEFAULT_PATIENCE=50
    DEFAULT_DET_LOSS_SCALE=1.0
    DEFAULT_REACH_TARGET_MODE="detach"
    DEFAULT_KD_TARGET_MODE="detach"
    ;;
esac

OGSOD_PREP_ROOT="${OGSOD_PREP_ROOT:-data/ogsod_public_prepared}"
DATA_CFG="${DATA_CFG:-${OGSOD_PREP_ROOT}/yamls/ogsod_${TASK}_sar.yaml}"
TEACHER_DATA_CFG="${TEACHER_DATA_CFG:-${OGSOD_PREP_ROOT}/yamls/ogsod_${TASK}_rgb.yaml}"
PROJECT_DIR="${PROJECT_DIR:-runs_public/ogsod/${TASK}/ladd}"
LOG_DIR="${LOG_DIR:-logs/ogsod_public/${TASK}_${RUN_TAG}_${PHASE}}"
MASTER_LOG="${LOG_DIR}/master.log"
MANIFEST="${LOG_DIR}/manifest.txt"
TOOL="tools/train_teacher_student_decomposition_kd_phase_nrrl_teacher_u_aux.py"
if [[ "$TASK" == "hbb" ]]; then
  TOOL="tools/train_ladd_hbb.py"
fi

GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-8}"
IMGSZ="${IMGSZ:-512}"
FRACTION="${FRACTION:-1.0}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
PATIENCE="${PATIENCE:-$DEFAULT_PATIENCE}"
PHASE_DETECT_MODE="${PHASE_DETECT_MODE:-raw}"
DET_LOSS_SCALE="${DET_LOSS_SCALE:-$DEFAULT_DET_LOSS_SCALE}"
REACH_TARGET_MODE="${REACH_TARGET_MODE:-$DEFAULT_REACH_TARGET_MODE}"
KD_TARGET_MODE="${KD_TARGET_MODE:-$DEFAULT_KD_TARGET_MODE}"

LAMBDA_REACH="${LAMBDA_REACH:-1.0}"
LAMBDA_MATCH_INNER="${LAMBDA_MATCH_INNER:-1.0}"
LAMBDA_RANK_INNER="${LAMBDA_RANK_INNER:-1.0}"
DELTA="${DELTA:-0.2}"
LAMBDA_REC="${LAMBDA_REC:-0.1}"
LAMBDA_SEP="${LAMBDA_SEP:-0.05}"
LAMBDA_TASKL="${LAMBDA_TASKL:-1.0}"
ALPHA_KD="${ALPHA_KD:-1.0}"
ALPHA_S_REC="${ALPHA_S_REC:-0.1}"
ALPHA_SEP="${ALPHA_SEP:-0.05}"
RESIDUAL_AUX_MODE="${RESIDUAL_AUX_MODE:-energy}"
LAMBDA_RESIDUAL_AUX="${LAMBDA_RESIDUAL_AUX:-0.25}"
TEACHER_PRIVATE_AUX_MODE="${TEACHER_PRIVATE_AUX_MODE:-energy}"
LAMBDA_TEACHER_PRIVATE_AUX="${LAMBDA_TEACHER_PRIVATE_AUX:-0.25}"
STUDENT_BRANCH_MODE="${STUDENT_BRANCH_MODE:-split}"
TEACHER_FEATURE_MODE="${TEACHER_FEATURE_MODE:-decomposed}"
REACH_INPUT_MODE="${REACH_INPUT_MODE:-adapter}"
KD_WEIGHT_MODE="${KD_WEIGHT_MODE:-none}"
KD_AGGREGATION_MODE="${KD_AGGREGATION_MODE:-token}"
USE_MASK="${USE_MASK:-1}"
USE_FG_MASK_FOR_REACH="${USE_FG_MASK_FOR_REACH:-1}"
USE_FG_MASK_FOR_REC="${USE_FG_MASK_FOR_REC:-0}"
STRICT_BATCH_SIZE="${STRICT_BATCH_SIZE:-0}"
VALIDATE_BEFORE_TRAIN="${VALIDATE_BEFORE_TRAIN:-0}"
EXIST_OK="${EXIST_OK:-0}"

resolve_close_mosaic() {
  if [[ -n "${CLOSE_MOSAIC:-}" ]]; then
    printf '%s\n' "$CLOSE_MOSAIC"
    return
  fi
  if [[ -n "${CLOSE_AT_EPOCH:-}" ]]; then
    if (( CLOSE_AT_EPOCH < 0 || CLOSE_AT_EPOCH > EPOCHS )); then
      echo "CLOSE_AT_EPOCH must be in [0, EPOCHS]; got ${CLOSE_AT_EPOCH} with EPOCHS=${EPOCHS}" >&2
      exit 1
    fi
    printf '%s\n' "$((EPOCHS - CLOSE_AT_EPOCH))"
    return
  fi
  printf '%s\n' 10
}

RESOLVED_CLOSE_MOSAIC="$(resolve_close_mosaic)"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

resolve_actual_run_dir() {
  python3 - "$PROJECT_DIR" "$RUN_NAME" "$RUN_DIR" <<'PY'
import sys
from pathlib import Path

project_dir = Path(sys.argv[1]).resolve()
run_name = sys.argv[2]
expected = Path(sys.argv[3]).resolve()
candidates = []
if expected.exists():
    candidates.append(expected)
if project_dir.is_dir():
    for path in sorted(project_dir.glob(f"{run_name}*")):
        if path.is_dir() and path.resolve() not in candidates:
            candidates.append(path.resolve())

def key(path: Path):
    results = 1 if (path / "results.csv").is_file() else 0
    args = 1 if (path / "args.yaml").is_file() else 0
    return (results, args, path.stat().st_mtime if path.exists() else 0.0)

print(max(candidates, key=key) if candidates else expected)
PY
}

: "${SAR_BASELINE:?Set SAR_BASELINE to the clean SAR baseline checkpoint.}"
: "${RGB_TEACHER:?Set RGB_TEACHER to the clean RGB teacher checkpoint.}"
if [[ -z "${MODEL:-}" ]]; then
  if [[ "$PHASE" == "a1" ]]; then
    MODEL="$SAR_BASELINE"
  else
    echo "MODEL is required for phase ${PHASE}; pass previous phase best.pt." >&2
    exit 1
  fi
fi

require_file "$MODEL" "input model"
require_file "$SAR_BASELINE" "SAR baseline"
require_file "$RGB_TEACHER" "RGB teacher"
require_file "$DATA_CFG" "student dataset YAML"
require_file "$TEACHER_DATA_CFG" "teacher dataset YAML"

RUN_NAME="${RUN_NAME:-ogsod_${TASK}_ladd_${RUN_TAG}_${PHASE}_e${EPOCHS}_b${BATCH_SIZE}_s${SEED}_gpu${GPU_ID}}"
RUN_DIR="${PROJECT_DIR}/${RUN_NAME}"
STAGE_LOG="${LOG_DIR}/${PHASE}.log"
mkdir -p "$LOG_DIR"

{
  echo "task=${TASK}"
  echo "phase=${PHASE}"
  echo "run_tag=${RUN_TAG}"
  echo "model=${MODEL}"
  echo "sar_baseline=${SAR_BASELINE}"
  echo "rgb_teacher=${RGB_TEACHER}"
  echo "data_cfg=${DATA_CFG}"
  echo "teacher_data_cfg=${TEACHER_DATA_CFG}"
  echo "project_dir=${PROJECT_DIR}"
  echo "run_name=${RUN_NAME}"
  echo "seed=${SEED}"
  echo "epochs=${EPOCHS}"
  echo "batch_size=${BATCH_SIZE}"
  echo "cos_lr=${COS_LR:-0}"
  echo "lr0=${LR0:-}"
  echo "lrf=${LRF:-}"
  echo "optimizer=${OPTIMIZER:-}"
  echo "warmup_epochs=${WARMUP_EPOCHS:-}"
  echo "warmup_bias_lr=${WARMUP_BIAS_LR:-}"
  echo "warmup_momentum=${WARMUP_MOMENTUM:-}"
  echo "freeze_bn_stats=${FREEZE_BN_STATS:-0}"
  echo "close_at_epoch=${CLOSE_AT_EPOCH:-}"
  echo "close_mosaic=${RESOLVED_CLOSE_MOSAIC}"
  echo "student_branch_mode=${STUDENT_BRANCH_MODE}"
  echo "alpha_s_rec=${ALPHA_S_REC}"
  echo "comparison_kd_profile=${COMPARISON_KD_PROFILE:-none}"
  echo "profile_kd_weight=${PROFILE_KD_WEIGHT:-1.0}"
  echo "profile_kd_replace_base=${PROFILE_KD_REPLACE_BASE:-0}"
  echo "cclkd_base_temperature=${CCLKD_BASE_TEMPERATURE:-2.0}"
  echo "cclkd_contrastive_temperature=${CCLKD_CONTRASTIVE_TEMPERATURE:-0.1}"
  echo "cclkd_feat_weight=${CCLKD_FEAT_WEIGHT:-1.0}"
  echo "cclkd_contrast_weight=${CCLKD_CONTRAST_WEIGHT:-0.5}"
  echo "cclkd_bg_weight=${CCLKD_BG_WEIGHT:-0.1}"
  echo "cclkd_min_confidence=${CCLKD_MIN_CONFIDENCE:-0.1}"
  echo "hallucidet_bg_weight=${HALLUCIDET_BG_WEIGHT:-0.05}"
  echo "hallucidet_response_weight=${HALLUCIDET_RESPONSE_WEIGHT:-0.5}"
  echo "hallucidet_margin_weight=${HALLUCIDET_MARGIN_WEIGHT:-0.1}"
  echo "hallucidet_margin=${HALLUCIDET_MARGIN:-0.2}"
} > "$MANIFEST"

cmd=(
  python3 "$TOOL"
  --phase "$PHASE"
  --model "$MODEL"
  --data "$DATA_CFG"
  --teacher-data "$TEACHER_DATA_CFG"
  --teacher-weights "$RGB_TEACHER"
  --imgsz "$IMGSZ"
  --epochs "$EPOCHS"
  --batch "$BATCH_SIZE"
  --workers "$WORKERS"
  --device "$GPU_ID"
  --patience "$PATIENCE"
  --fraction "$FRACTION"
  --project "$PROJECT_DIR"
  --name "$RUN_NAME"
  --phase-detect-mode "$PHASE_DETECT_MODE"
  --det-loss-scale "$DET_LOSS_SCALE"
  --phase-stop-metric "${PHASE_STOP_METRIC:-default}"
  --reach-target-mode "$REACH_TARGET_MODE"
  --kd-target-mode "$KD_TARGET_MODE"
  --lambda-rec "$LAMBDA_REC"
  --lambda-sep "$LAMBDA_SEP"
  --lambda-taskL "$LAMBDA_TASKL"
  --alpha-kd "$ALPHA_KD"
  --alpha-s-rec "$ALPHA_S_REC"
  --alpha-sep "$ALPHA_SEP"
  --lambda-reach "$LAMBDA_REACH"
  --lambda-match-inner "$LAMBDA_MATCH_INNER"
  --lambda-rank-inner "$LAMBDA_RANK_INNER"
  --delta "$DELTA"
  --rank-d-neg-cap "${RANK_D_NEG_CAP:-4.0}"
  --lambda-anti-collapse "${LAMBDA_ANTI_COLLAPSE:-0.0}"
  --anti-collapse-floor "${ANTI_COLLAPSE_FLOOR:-0.0}"
  --reach-input-mode "$REACH_INPUT_MODE"
  --student-detect-mode "${STUDENT_DETECT_MODE:-raw}"
  --student-branch-mode "$STUDENT_BRANCH_MODE"
  --teacher-feature-mode "$TEACHER_FEATURE_MODE"
  --kd-weight-mode "$KD_WEIGHT_MODE"
  --kd-aggregation-mode "$KD_AGGREGATION_MODE"
  --residual-aux-mode "$RESIDUAL_AUX_MODE"
  --lambda-residual-aux "$LAMBDA_RESIDUAL_AUX"
  --teacher-private-aux-mode "$TEACHER_PRIVATE_AUX_MODE"
  --lambda-teacher-private-aux "$LAMBDA_TEACHER_PRIVATE_AUX"
  --mosaic "${MOSAIC:-1.0}"
  --mixup "${MIXUP:-0.0}"
  --cutmix "${CUTMIX:-0.0}"
  --degrees "${DEGREES:-0.0}"
  --perspective "${PERSPECTIVE:-0.0}"
  --translate "${TRANSLATE:-0.1}"
  --scale "${SCALE:-0.5}"
  --fliplr "${FLIPLR:-0.5}"
  --flipud "${FLIPUD:-0.0}"
  --hsv-h "${HSV_H:-0}"
  --hsv-s "${HSV_S:-0}"
  --hsv-v "${HSV_V:-0}"
  --erasing "${ERASING:-0}"
  --close-mosaic "$RESOLVED_CLOSE_MOSAIC"
  --seed "$SEED"
  --deterministic
)

if [[ "$TASK" == "hbb" ]]; then
  cmd+=(
    --comparison-kd-profile "${COMPARISON_KD_PROFILE:-none}"
    --profile-kd-weight "${PROFILE_KD_WEIGHT:-1.0}"
    --fgd-bg-weight "${FGD_BG_WEIGHT:-0.25}"
    --fgd-relation-weight "${FGD_RELATION_WEIGHT:-0.1}"
    --fgd-temperature "${FGD_TEMPERATURE:-0.5}"
    --mgd-mask-ratio "${MGD_MASK_RATIO:-0.5}"
    --ld-temperature "${LD_TEMPERATURE:-10.0}"
    --crosskd-temperature "${CROSSKD_TEMPERATURE:-2.0}"
    --crosskd-pred-weight "${CROSSKD_PRED_WEIGHT:-1.0}"
    --crosskd-feat-weight "${CROSSKD_FEAT_WEIGHT:-0.25}"
    --crosskd-teacher-conf-threshold "${CROSSKD_TEACHER_CONF_THRESHOLD:-0.25}"
    --cclkd-base-temperature "${CCLKD_BASE_TEMPERATURE:-2.0}"
    --cclkd-contrastive-temperature "${CCLKD_CONTRASTIVE_TEMPERATURE:-0.1}"
    --cclkd-feat-weight "${CCLKD_FEAT_WEIGHT:-1.0}"
    --cclkd-logit-weight "${CCLKD_LOGIT_WEIGHT:-1.0}"
    --cclkd-contrast-weight "${CCLKD_CONTRAST_WEIGHT:-0.5}"
    --cclkd-bg-weight "${CCLKD_BG_WEIGHT:-0.1}"
    --cclkd-min-confidence "${CCLKD_MIN_CONFIDENCE:-0.1}"
    --cclkd-max-tokens "${CCLKD_MAX_TOKENS:-512}"
    --c2kd-selection-threshold "${C2KD_SELECTION_THRESHOLD:-0.25}"
    --c2kd-teacher-conf-threshold "${C2KD_TEACHER_CONF_THRESHOLD:-0.3}"
    --mmanet-relation-margin "${MMANET_RELATION_MARGIN:-0.2}"
    --mmanet-max-tokens "${MMANET_MAX_TOKENS:-512}"
    --hallucidet-bg-weight "${HALLUCIDET_BG_WEIGHT:-0.05}"
    --hallucidet-response-weight "${HALLUCIDET_RESPONSE_WEIGHT:-0.5}"
    --hallucidet-margin-weight "${HALLUCIDET_MARGIN_WEIGHT:-0.1}"
    --hallucidet-margin "${HALLUCIDET_MARGIN:-0.2}"
  )
  if [[ "${PROFILE_KD_REPLACE_BASE:-0}" == "1" ]]; then
    cmd+=(--profile-kd-replace-base)
  fi
fi

if [[ -n "${LR0:-}" ]]; then
  cmd+=(--lr0 "$LR0")
fi
if [[ -n "${LRF:-}" ]]; then
  cmd+=(--lrf "$LRF")
fi
if [[ "${COS_LR:-0}" == "1" ]]; then
  cmd+=(--cos-lr)
fi
if [[ -n "${OPTIMIZER:-}" ]]; then
  cmd+=(--optimizer "$OPTIMIZER")
fi
if [[ -n "${WARMUP_EPOCHS:-}" ]]; then
  cmd+=(--warmup-epochs "$WARMUP_EPOCHS")
fi
if [[ -n "${WARMUP_BIAS_LR:-}" ]]; then
  cmd+=(--warmup-bias-lr "$WARMUP_BIAS_LR")
fi
if [[ -n "${WARMUP_MOMENTUM:-}" ]]; then
  cmd+=(--warmup-momentum "$WARMUP_MOMENTUM")
fi
if [[ -n "${SAVE_PERIOD:-}" ]]; then
  cmd+=(--save-period "$SAVE_PERIOD")
fi
if [[ -n "${PHASE_MIN_EPOCHS:-}" ]]; then
  cmd+=(--phase-min-epochs "$PHASE_MIN_EPOCHS")
fi
if [[ "$USE_MASK" == "1" ]]; then
  cmd+=(--use-mask)
fi
if [[ "$USE_FG_MASK_FOR_REACH" == "1" ]]; then
  cmd+=(--use-fg-mask-for-reach)
fi
if [[ "$USE_FG_MASK_FOR_REC" == "1" ]]; then
  cmd+=(--use-fg-mask-for-rec)
fi
if [[ "${TASK_LOSS_FG_ONLY:-0}" == "1" ]]; then
  cmd+=(--task-loss-fg-only)
fi
if [[ "$STRICT_BATCH_SIZE" == "1" ]]; then
  cmd+=(--strict-batch-size)
fi
if [[ "$VALIDATE_BEFORE_TRAIN" == "1" ]]; then
  cmd+=(--validate-before-train)
fi
if [[ "${FREEZE_BN_STATS:-0}" == "1" ]]; then
  cmd+=(--freeze-bn-stats)
fi
if [[ "$EXIST_OK" == "1" ]]; then
  cmd+=(--exist-ok)
fi
if [[ "${FORCE_STUDENT_REC:-0}" == "1" && "$TASK" == "hbb" ]]; then
  cmd+=(--force-student-rec)
fi

echo "[$(date '+%F %T')] Launching OGSOD ${TASK} LADD phase ${PHASE}: ${RUN_NAME}" | tee "$MASTER_LOG"
{
  echo "[$(date '+%F %T')] command:"
  printf ' %q' "${cmd[@]}"
  printf '\n'
} > "$STAGE_LOG"

if PYTHONUNBUFFERED=1 "${cmd[@]}" >> "$STAGE_LOG" 2>&1; then
  echo "[$(date '+%F %T')] Phase ${PHASE} finished" | tee -a "$MASTER_LOG"
else
  status=$?
  echo "[$(date '+%F %T')] Phase ${PHASE} failed with code ${status}" | tee -a "$MASTER_LOG"
  tail -n 80 "$STAGE_LOG" | tee -a "$MASTER_LOG"
  exit "$status"
fi

ACTUAL_RUN_DIR="$(resolve_actual_run_dir)"
printf '%s\n' "$ACTUAL_RUN_DIR" > "${LOG_DIR}/actual_run_dir.txt"
if [[ -f "${ACTUAL_RUN_DIR}/results.csv" ]]; then
  python3 tools/summarize_tskd_results.py "$ACTUAL_RUN_DIR" | tee -a "$MASTER_LOG" || true
fi
echo "[$(date '+%F %T')] Run directory: ${ACTUAL_RUN_DIR}" | tee -a "$MASTER_LOG"
