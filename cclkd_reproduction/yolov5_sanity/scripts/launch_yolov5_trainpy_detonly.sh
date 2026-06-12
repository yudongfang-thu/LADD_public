#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_trainpy_detonly.sh \
    <batch> <seed> <gpu_id> <tag>

Example:
  DRY_RUN=1 SMOKE_EPOCHS=80 bash cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_trainpy_detonly.sh \
    32 0 0 trainpy_detonly_b32_e80

Environment:
  DRY_RUN      Print command only by default (default: 1)
  LAUNCH       Set LAUNCH=1 and DRY_RUN=0 to start one nohup job (default: 0)
  SMOKE_EPOCHS Epoch count override (default: 80)
  SAR_DATA     SAR YOLO yaml (default: configs/datasets/ogsod_hbb_sar.yaml)
  PYTHON       Python executable (default: python3)
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
MODE="${CCLKD_YOLOV5_MODE:-det_only_same_trainpy}"
EPOCHS="${SMOKE_EPOCHS:-80}"

if [[ "$MODE" != "det_only_same_trainpy" ]]; then
  echo "This first-stage launcher only supports mode=det_only_same_trainpy, got: $MODE" >&2
  exit 2
fi

SANITY_DIR="$REPO_ROOT/cclkd_reproduction/yolov5_sanity"
TRAIN_SCRIPT="$SANITY_DIR/code/train_yolov5_cclkd_from_trainpy.py"
PROJECT="$SANITY_DIR/results/runs"
RUN_NAME="yolov5x_trainpy_detonly_b${BATCH}_s${SEED}_${TAG}"
RUN_DIR="$PROJECT/$RUN_NAME"

DATA="${SAR_DATA:-configs/datasets/ogsod_hbb_sar.yaml}"
HYP="cclkd_reproduction/yolov5_sanity/configs/hyp_cold_ogsod.yaml"
WEIGHTS="external/yolov5/yolov5x.pt"

CMD=(
  "$PYTHON" "$TRAIN_SCRIPT"
  --mode "$MODE"
  --img 256
  --epochs "$EPOCHS"
  --batch-size "$BATCH"
  --data "$DATA"
  --hyp "$HYP"
  --weights "$WEIGHTS"
  --optimizer SGD
  --seed "$SEED"
  --device "$GPU_ID"
  --project "$PROJECT"
  --name "$RUN_NAME"
  --patience 400
  --workers 4
  --save-period 100
  --exist-ok
)

print_command() {
  printf 'cd %q\n' "$REPO_ROOT"
  printf 'YOLOv5_AUTOINSTALL=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 '
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
if [[ ! -f "$REPO_ROOT/$DATA" && ! -f "$DATA" ]]; then
  echo "Missing data yaml: $DATA" >&2
  exit 1
fi
if [[ ! -f "$REPO_ROOT/$HYP" && ! -f "$HYP" ]]; then
  echo "Missing hyp yaml: $HYP" >&2
  exit 1
fi
if [[ ! -f "$REPO_ROOT/$WEIGHTS" && ! -f "$WEIGHTS" ]]; then
  echo "Missing weights: $WEIGHTS" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
{
  echo "tag=$TAG"
  echo "run_name=$RUN_NAME"
  echo "model=yolov5x"
  echo "experiment=trainpy_detonly_alignment"
  echo "mode=$MODE"
  echo "student_modality=sar"
  echo "teacher_modality=none"
  echo "kd=none"
  echo "batch=$BATCH"
  echo "seed=$SEED"
  echo "gpu_id=$GPU_ID"
  echo "epochs=$EPOCHS"
  echo "img=256"
  echo "optimizer=SGD"
  echo "patience=400"
  echo "workers=4"
  echo "weights=$WEIGHTS"
  echo "data=$DATA"
  echo "hyp=$HYP"
  echo "target_standard_trainpy_ap50=0.57056"
  echo "target_standard_trainpy_ap=0.30964"
  echo "acceptance_max_ap_gap=0.02"
  echo "note=Minimal-intrusion copy of official YOLOv5 train.py. No teacher, no KD, no paired dataloader."
} > "$RUN_DIR/run_meta.txt"

{
  print_command
} > "$RUN_DIR/command.sh"
chmod +x "$RUN_DIR/command.sh"

(
  cd "$REPO_ROOT"
  YOLOv5_AUTOINSTALL=false TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    nohup "${CMD[@]}" > "$RUN_DIR/nohup.log" 2>&1 &
  echo $! > "$RUN_DIR/pid.txt"
)

echo "Started YOLOv5x train.py det-only alignment run: $RUN_NAME"
echo "Run directory: $RUN_DIR"
echo "PID: $(cat "$RUN_DIR/pid.txt")"
echo "Log: $RUN_DIR/nohup.log"
