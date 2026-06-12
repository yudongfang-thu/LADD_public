#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  LAUNCH=1 DRY_RUN=0 bash cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_cclkd_full.sh \
    <batch> <seed> <gpu_id> <tag>

Example:
  LAUNCH=1 DRY_RUN=0 CCLKD_YOLOV5_MODE=paper_full bash cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_cclkd_full.sh \
    64 0 1 cclkd_full_yolov5x_b64_s0

Environment:
  CCLKD_YOLOV5_MODE    det_only_same_trainer|two_branch_no_kd|raw_proxy_full|paper_atkd_only|paper_ccl_only|paper_full|current_full (default: paper_full)
  CCLKD_YOLOV5_COS_LR  1 to append --cos-lr, 0 otherwise (default: 0)
  CCLKD_YOLOV5_KD_WEIGHT           outer KD scale (default: 1.0)
  CCLKD_YOLOV5_ATKD_WEIGHT         optional ATKD weight override
  CCLKD_YOLOV5_CCL_WEIGHT          optional CCL weight override
  CCLKD_YOLOV5_KD_WARMUP_EPOCHS    KD warmup epochs (default: 3)
  CCLKD_YOLOV5_TEACHER_DET_WEIGHT  teacher detection loss weight (default: 1.0)
  ALLOW_RAW_PROXY      1 to allow raw_proxy_full/current_full (default: 0)
  SMOKE_EPOCHS         override epochs for smoke-only runs
  MAX_TRAIN_BATCHES    limit train batches per epoch, -1 disables (default: -1)
  SKIP_VAL             1 to skip validation for smoke-only runs (default: 0)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 4 ]]; then
  usage >&2
  exit 2
fi

BATCH="$1"
SEED="$2"
GPU_ID="$3"
TAG="$4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
DRY_RUN="${DRY_RUN:-1}"
LAUNCH="${LAUNCH:-0}"
WAIT_FOR_GPU="${WAIT_FOR_GPU:-0}"
MIN_FREE_MB="${MIN_FREE_MB:-22000}"
POLL_SECONDS="${POLL_SECONDS:-120}"
CCLKD_YOLOV5_MODE="${CCLKD_YOLOV5_MODE:-paper_full}"
CCLKD_YOLOV5_COS_LR="${CCLKD_YOLOV5_COS_LR:-0}"
CCLKD_YOLOV5_KD_WEIGHT="${CCLKD_YOLOV5_KD_WEIGHT:-1.0}"
CCLKD_YOLOV5_ATKD_WEIGHT="${CCLKD_YOLOV5_ATKD_WEIGHT:-}"
CCLKD_YOLOV5_CCL_WEIGHT="${CCLKD_YOLOV5_CCL_WEIGHT:-}"
CCLKD_YOLOV5_KD_WARMUP_EPOCHS="${CCLKD_YOLOV5_KD_WARMUP_EPOCHS:-3}"
CCLKD_YOLOV5_TEACHER_DET_WEIGHT="${CCLKD_YOLOV5_TEACHER_DET_WEIGHT:-1.0}"
ALLOW_RAW_PROXY="${ALLOW_RAW_PROXY:-0}"
EPOCHS="${SMOKE_EPOCHS:-400}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:--1}"
SKIP_VAL="${SKIP_VAL:-0}"
case "$CCLKD_YOLOV5_MODE" in
  det_only_same_trainer|two_branch_no_kd|raw_proxy_full|paper_atkd_only|paper_ccl_only|paper_full|current_full) ;;
  *)
    echo "Invalid CCLKD_YOLOV5_MODE: $CCLKD_YOLOV5_MODE" >&2
    exit 2
    ;;
esac
case "$CCLKD_YOLOV5_COS_LR" in 0|1) ;; *) echo "Invalid CCLKD_YOLOV5_COS_LR: $CCLKD_YOLOV5_COS_LR" >&2; exit 2 ;; esac
case "$SKIP_VAL" in 0|1) ;; *) echo "Invalid SKIP_VAL: $SKIP_VAL" >&2; exit 2 ;; esac
case "$ALLOW_RAW_PROXY" in 0|1) ;; *) echo "Invalid ALLOW_RAW_PROXY: $ALLOW_RAW_PROXY" >&2; exit 2 ;; esac

if [[ "$CCLKD_YOLOV5_MODE" == "current_full" || "$CCLKD_YOLOV5_MODE" == "raw_proxy_full" ]]; then
  if [[ "$ALLOW_RAW_PROXY" != "1" ]]; then
    echo "ERROR: $CCLKD_YOLOV5_MODE is a legacy raw proxy mode and is blocked by default." >&2
    echo "Set ALLOW_RAW_PROXY=1 only for historical debugging. Use paper_full for CCLKD reproduction." >&2
    exit 2
  fi
  echo "WARNING: $CCLKD_YOLOV5_MODE is legacy raw_proxy_full and is not a verified CCLKD implementation." >&2
fi

SANITY_DIR="$REPO_ROOT/cclkd_reproduction/yolov5_sanity"
YOLOV5_DIR="$REPO_ROOT/external/yolov5"
PROJECT="$SANITY_DIR/results/runs"
RUN_NAME="yolov5x_${CCLKD_YOLOV5_MODE}_b${BATCH}_s${SEED}_${TAG}"
RUN_DIR="$PROJECT/$RUN_NAME"
HYP="$SANITY_DIR/configs/hyp_cold_ogsod.yaml"
SAR_DATA="${SAR_DATA:-$REPO_ROOT/configs/datasets/ogsod_hbb_sar.yaml}"
RGB_DATA="${RGB_DATA:-$REPO_ROOT/configs/datasets/ogsod_hbb_rgb.yaml}"
WEIGHTS="$YOLOV5_DIR/yolov5x.pt"
TRAIN_SCRIPT="$SANITY_DIR/code/train_yolov5_cclkd_full.py"

CMD=(
  "$PYTHON" "$TRAIN_SCRIPT"
  --img 256
  --epochs "$EPOCHS"
  --batch-size "$BATCH"
  --data "$SAR_DATA"
  --teacher-data "$RGB_DATA"
  --hyp "$HYP"
  --device "$GPU_ID"
  --project "$PROJECT"
  --name "$RUN_NAME"
  --weights "$WEIGHTS"
  --teacher-weights "$WEIGHTS"
  --optimizer SGD
  --patience 400
  --workers 4
  --seed "$SEED"
  --save-period 100
  --teacher-det-weight "$CCLKD_YOLOV5_TEACHER_DET_WEIGHT"
  --kd-weight "$CCLKD_YOLOV5_KD_WEIGHT"
  --kd-warmup-epochs "$CCLKD_YOLOV5_KD_WARMUP_EPOCHS"
  --mode "$CCLKD_YOLOV5_MODE"
  --max-train-batches "$MAX_TRAIN_BATCHES"
  --exist-ok
)

if [[ "$CCLKD_YOLOV5_COS_LR" == "1" ]]; then
  CMD+=(--cos-lr)
fi
if [[ -n "$CCLKD_YOLOV5_ATKD_WEIGHT" ]]; then
  CMD+=(--atkd-weight "$CCLKD_YOLOV5_ATKD_WEIGHT")
fi
if [[ -n "$CCLKD_YOLOV5_CCL_WEIGHT" ]]; then
  CMD+=(--ccl-weight "$CCLKD_YOLOV5_CCL_WEIGHT")
fi
if [[ "$ALLOW_RAW_PROXY" == "1" ]]; then
  CMD+=(--allow-raw-proxy)
fi
if [[ "$SKIP_VAL" == "1" ]]; then
  CMD+=(--skip-val)
fi

print_command() {
  printf 'cd %q && ' "$REPO_ROOT"
  printf '%q ' "${CMD[@]}"
  printf '\n'
}

if [[ "$DRY_RUN" == "1" || "$LAUNCH" != "1" ]]; then
  print_command
  exit 0
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Missing train script: $TRAIN_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$WEIGHTS" ]]; then
  echo "Missing YOLOv5x weights: $WEIGHTS" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
{
  echo "tag=$TAG"
  echo "model=yolov5x"
  echo "experiment=online_cclkd_full"
  echo "mode=$CCLKD_YOLOV5_MODE"
  echo "student_modality=sar"
  echo "teacher_modality=rgb"
  echo "batch=$BATCH"
  echo "seed=$SEED"
  echo "gpu_id=$GPU_ID"
  echo "epochs=$EPOCHS"
  echo "cos_lr=$CCLKD_YOLOV5_COS_LR"
  echo "kd_weight=$CCLKD_YOLOV5_KD_WEIGHT"
  echo "atkd_weight=$CCLKD_YOLOV5_ATKD_WEIGHT"
  echo "ccl_weight=$CCLKD_YOLOV5_CCL_WEIGHT"
  echo "kd_warmup_epochs=$CCLKD_YOLOV5_KD_WARMUP_EPOCHS"
  echo "teacher_det_weight=$CCLKD_YOLOV5_TEACHER_DET_WEIGHT"
  echo "allow_raw_proxy=$ALLOW_RAW_PROXY"
  echo "max_train_batches=$MAX_TRAIN_BATCHES"
  echo "skip_val=$SKIP_VAL"
  echo "wait_for_gpu=$WAIT_FOR_GPU"
  echo "min_free_mb=$MIN_FREE_MB"
  echo "note=YOLOv5-adapted CCLKD audit launcher; current_full is not a verified CCLKD reproduction."
} > "$RUN_DIR/run_meta.txt"
{
  printf 'cd %q\n' "$REPO_ROOT"
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "$RUN_DIR/command.sh"
chmod +x "$RUN_DIR/command.sh"

(
  cd "$REPO_ROOT"
  nohup bash -c '
    set -euo pipefail
    if [[ "'"$WAIT_FOR_GPU"'" == "1" ]]; then
      while true; do
        free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "'"$GPU_ID"'" | awk "{print \$1}")
        echo "$(date +%F\ %T) gpu'"$GPU_ID"'_free_mb=${free_mb}, need>='"$MIN_FREE_MB"'"
        if [[ "$free_mb" =~ ^[0-9]+$ ]] && (( free_mb >= '"$MIN_FREE_MB"' )); then
          break
        fi
        sleep '"$POLL_SECONDS"'
      done
    fi
    export YOLOv5_AUTOINSTALL=false
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    '"$(printf '%q ' "${CMD[@]}")"'
  ' > "$RUN_DIR/nohup.log" 2>&1 &
  echo $! > "$RUN_DIR/pid.txt"
)

echo "Started YOLOv5x CCLKD audit launcher: $RUN_NAME"
echo "PID: $(cat "$RUN_DIR/pid.txt")"
echo "Log: $RUN_DIR/nohup.log"
