#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_ogsod_baseline.sh \
    <sar|rgb> <x|x6> <pretrained|scratch> <batch> <seed> <gpu_id> <tag>

Example:
  DRY_RUN=1 bash cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_ogsod_baseline.sh \
    sar x pretrained 32 0 0 cclkd_gate_sar_x_b32_pretrained

Environment:
  DRY_RUN      Print command only by default (default: 1)
  LAUNCH       Set LAUNCH=1 and DRY_RUN=0 to start one nohup job (default: 0)
  SAR_DATA     SAR OGSOD YOLO yaml (default: configs/datasets/ogsod_hbb_sar.yaml if present)
  RGB_DATA     RGB OGSOD YOLO yaml (default: configs/datasets/ogsod_hbb_rgb.yaml if present)
  YOLOV5_REF   Metadata label recorded in run_meta.txt (default: v7.0)
  PYTHON       Python executable (default: python3)

Safety:
  This launcher starts at most one job. The matrix script is dry-run oriented.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -ne 7 ]]; then
  usage >&2
  exit 2
fi

MODALITY="$1"
MODEL_SIZE="$2"
INIT="$3"
BATCH="$4"
SEED="$5"
GPU_ID="$6"
TAG="$7"

if [[ "$MODALITY" != "sar" && "$MODALITY" != "rgb" ]]; then
  echo "modality must be sar or rgb, got: $MODALITY" >&2
  exit 2
fi
if [[ "$MODEL_SIZE" != "x" && "$MODEL_SIZE" != "x6" ]]; then
  echo "model must be x or x6, got: $MODEL_SIZE" >&2
  exit 2
fi
if [[ "$INIT" != "pretrained" && "$INIT" != "scratch" ]]; then
  echo "init must be pretrained or scratch, got: $INIT" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
DRY_RUN="${DRY_RUN:-1}"
LAUNCH="${LAUNCH:-0}"
YOLOV5_REF="${YOLOV5_REF:-v7.0}"
YOLOV5_DIR="$REPO_ROOT/external/yolov5"
SANITY_DIR="$REPO_ROOT/cclkd_reproduction/yolov5_sanity"
PROJECT="$SANITY_DIR/results/runs"
RUN_NAME="yolov5_${MODALITY}_${MODEL_SIZE}_${INIT}_b${BATCH}_s${SEED}_${TAG}"
RUN_DIR="$PROJECT/$RUN_NAME"
HYP="$SANITY_DIR/configs/hyp_cold_ogsod.yaml"
DEFAULT_SAR_DATA="$REPO_ROOT/configs/datasets/ogsod_hbb_sar.yaml"
DEFAULT_RGB_DATA="$REPO_ROOT/configs/datasets/ogsod_hbb_rgb.yaml"
if [[ ! -f "$DEFAULT_SAR_DATA" ]]; then
  DEFAULT_SAR_DATA="$REPO_ROOT/shared/configs/datasets_public/ogsod1_sar_detect.yaml"
fi
if [[ ! -f "$DEFAULT_RGB_DATA" ]]; then
  DEFAULT_RGB_DATA="$REPO_ROOT/shared/configs/datasets_public/ogsod1_rgb_detect.yaml"
fi
SAR_DATA="${SAR_DATA:-$DEFAULT_SAR_DATA}"
RGB_DATA="${RGB_DATA:-$DEFAULT_RGB_DATA}"

if [[ ! -d "$YOLOV5_DIR/.git" ]]; then
  echo "Missing YOLOv5 repository: $YOLOV5_DIR" >&2
  echo "Run first: bash $SANITY_DIR/scripts/prepare_yolov5_repo.sh" >&2
  exit 1
fi

if [[ "$MODALITY" == "sar" ]]; then
  DATA_YAML="$SAR_DATA"
else
  DATA_YAML="$RGB_DATA"
fi

if [[ ! -f "$DATA_YAML" ]]; then
  echo "Missing data yaml: $DATA_YAML" >&2
  exit 1
fi
if [[ ! -f "$HYP" ]]; then
  echo "Missing hyp yaml: $HYP" >&2
  exit 1
fi

CFG="models/yolov5${MODEL_SIZE}.yaml"
WEIGHTS="yolov5${MODEL_SIZE}.pt"
if [[ "$INIT" == "scratch" ]]; then
  WEIGHTS=""
fi

TRAIN_HELP="$("$PYTHON" "$YOLOV5_DIR/train.py" --help 2>&1 || true)"
has_train_flag() {
  grep -Eq "(^|[[:space:]])$1([,[:space:]]|$)" <<<"$TRAIN_HELP"
}

OPTIONAL_ARGS=()
SUPPORTS_OPTIMIZER=0
SUPPORTS_SEED=0
SUPPORTS_EXIST_OK=0
if has_train_flag "--optimizer"; then
  OPTIONAL_ARGS+=(--optimizer SGD)
  SUPPORTS_OPTIMIZER=1
fi
if has_train_flag "--seed"; then
  OPTIONAL_ARGS+=(--seed "$SEED")
  SUPPORTS_SEED=1
fi
if has_train_flag "--exist-ok"; then
  OPTIONAL_ARGS+=(--exist-ok)
  SUPPORTS_EXIST_OK=1
fi

CMD=(
  "$PYTHON" train.py
  --img 256
  --epochs 400
  --batch-size "$BATCH"
  --data "$DATA_YAML"
  --hyp "$HYP"
  --device "$GPU_ID"
  --project "$PROJECT"
  --name "$RUN_NAME"
  --weights "$WEIGHTS"
)

if [[ "$INIT" == "scratch" ]]; then
  CMD+=(--cfg "$CFG")
fi
CMD+=("${OPTIONAL_ARGS[@]}")

print_command() {
  printf 'cd %q && ' "$YOLOV5_DIR"
  printf '%q ' "${CMD[@]}"
  printf '\n'
}

if [[ "$DRY_RUN" == "1" || "$LAUNCH" != "1" ]]; then
  if [[ "$MODEL_SIZE" == "x6" ]]; then
    echo "[diagnostic only] yolov5x6 is not part of the primary CCLKD reproduction gate."
  fi
  print_command
  exit 0
fi

mkdir -p "$RUN_DIR"

MODEL_INFO_TXT="$RUN_DIR/model_info.txt"
PARAMS_M="unknown"
if (
  cd "$YOLOV5_DIR"
  "$PYTHON" - "$CFG" "$MODEL_INFO_TXT" <<'PY'
import sys
from pathlib import Path

cfg = sys.argv[1]
out = Path(sys.argv[2])
try:
    from models.yolo import Model
    model = Model(cfg, ch=3, nc=3)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    out.write_text(f"cfg={cfg}\nnc=3\nparams_m={params:.3f}\n", encoding="utf-8")
    print(f"{params:.3f}")
except Exception as exc:
    out.write_text(f"cfg={cfg}\nnc=3\nmodel_info_error={exc}\n", encoding="utf-8")
    raise
PY
); then
  PARAMS_M="$(awk -F= '/^params_m=/{print $2}' "$MODEL_INFO_TXT")"
else
  echo "Warning: model information check failed. See $MODEL_INFO_TXT" >&2
fi

if [[ "$PARAMS_M" != "unknown" ]]; then
  "$PYTHON" - "$MODEL_SIZE" "$PARAMS_M" <<'PY'
import sys
model, params = sys.argv[1], float(sys.argv[2])
if model == "x" and not (85.0 <= params <= 88.0):
    print(f"Warning: YOLOv5x nc=3 params_m={params:.3f}, expected about 86M.", file=sys.stderr)
elif model == "x6" and not (135.0 <= params <= 145.0):
    print(f"Warning: YOLOv5x6 nc=3 params_m={params:.3f}, expected about 140M.", file=sys.stderr)
PY
fi

SANITY_ARGS=(--data "$DATA_YAML" --output-dir "$SANITY_DIR/results")
if [[ "$MODALITY" == "sar" && -f "$RGB_DATA" ]]; then
  SANITY_ARGS+=(--teacher-data "$RGB_DATA")
fi
"$PYTHON" "$SANITY_DIR/tools/check_yolov5_ogsod_dataset.py" "${SANITY_ARGS[@]}"
cp "$SANITY_DIR"/results/dataset_sanity_*.json "$RUN_DIR"/ 2>/dev/null || true
cp "$SANITY_DIR"/results/dataset_sanity_*.md "$RUN_DIR"/ 2>/dev/null || true

YOLOV5_COMMIT="$(git -C "$YOLOV5_DIR" rev-parse HEAD)"
{
  echo "tag=$TAG"
  echo "modality=$MODALITY"
  echo "model=$MODEL_SIZE"
  echo "init=$INIT"
  echo "batch=$BATCH"
  echo "seed=$SEED"
  echo "gpu_id=$GPU_ID"
  echo "yolov5_ref=$YOLOV5_REF"
  echo "yolov5_commit=$YOLOV5_COMMIT"
  echo "params_m=$PARAMS_M"
  echo "class_name_order=bridge,harbor,storage_tank"
  echo "target_ap50=80.9"
  echo "target_ap=46.3"
  echo "pass_threshold_ap50=78"
  echo "pass_threshold_ap=44"
  echo "supports_optimizer=$SUPPORTS_OPTIMIZER"
  echo "supports_seed=$SUPPORTS_SEED"
  echo "supports_exist_ok=$SUPPORTS_EXIST_OK"
  if [[ "$MODEL_SIZE" == "x6" ]]; then
    echo "diagnostic_only=1"
    echo "diagnostic_note=yolov5x6 is only for the 139.99M parameter inconsistency, not the primary CCLKD reproduction gate."
  else
    echo "diagnostic_only=0"
  fi
} > "$RUN_DIR/run_meta.txt"

{
  printf 'cd %q\n' "$YOLOV5_DIR"
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "$RUN_DIR/command.sh"
chmod +x "$RUN_DIR/command.sh"

(
  cd "$YOLOV5_DIR"
  nohup "${CMD[@]}" > "$RUN_DIR/nohup.log" 2>&1 &
  echo $! > "$RUN_DIR/pid.txt"
)

echo "Started YOLOv5 sanity run: $RUN_NAME"
echo "Run directory: $RUN_DIR"
echo "PID: $(cat "$RUN_DIR/pid.txt")"
echo "Log: $RUN_DIR/nohup.log"
