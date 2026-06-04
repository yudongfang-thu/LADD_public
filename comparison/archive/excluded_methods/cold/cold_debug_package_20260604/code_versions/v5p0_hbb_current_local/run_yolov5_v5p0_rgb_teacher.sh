#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/ogsod_public/cold_baseline_repro_20260528/run_yolov5_v5p0_rgb_teacher.sh <ogsod_hbb_rgb.yaml> <scratch|coco> [gpu_id]

Runs an OGSOD RGB/optical YOLOv5x v5.0 teacher for offline CoLD:
  imgsz=256, batch=64 by default, SGD-style YOLOv5 v5.0 hyp with configurable MixUp.

Environment overrides:
  YOLOV5_WORKDIR  Directory for the v5.0 clone. Default: ../yolov5_cold_v5p0
  PROJECT         YOLOv5 project path. Default: runs/ogsod_rgb_teacher
  RUN_SUFFIX      Extra suffix for the run name.
  MIXUP           MixUp probability written into hyp. Default: 0.1
  EPOCHS          Number of training epochs. Default: 100
  BATCH_SIZE      Batch size. Default: 64
  PYTHON_BIN      Python executable. Default: python3
  INSTALL_REQS=1  Run pip install -r requirements.txt after clone.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DATA_YAML="${1:-}"
INIT="${2:-}"
GPU_ID="${3:-${GPU_ID:-0}}"

if [[ -z "$DATA_YAML" || ! -f "$DATA_YAML" ]]; then
  echo "Missing dataset YAML: ${DATA_YAML:-<empty>}" >&2
  usage >&2
  exit 1
fi
if [[ "$INIT" != "scratch" && "$INIT" != "coco" ]]; then
  echo "init must be one of: scratch, coco" >&2
  usage >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEFAULT_WORKDIR="$(cd "$ROOT_DIR/.." && pwd)/yolov5_cold_v5p0"
YOLOV5_WORKDIR="${YOLOV5_WORKDIR:-$DEFAULT_WORKDIR}"
PROJECT="${PROJECT:-runs/ogsod_rgb_teacher}"
RUN_SUFFIX="${RUN_SUFFIX:-s0}"
MIXUP="${MIXUP:-0.1}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "$YOLOV5_WORKDIR/.git" ]]; then
  mkdir -p "$(dirname "$YOLOV5_WORKDIR")"
  git clone --depth 1 --branch v5.0 https://github.com/ultralytics/yolov5.git "$YOLOV5_WORKDIR"
fi

cd "$YOLOV5_WORKDIR"

if [[ "${INSTALL_REQS:-0}" == "1" ]]; then
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

"$PYTHON_BIN" - <<'PY'
from pathlib import Path

patches = {
    "utils/datasets.py": [
        ("torch.load(cache_path), True  # load", "torch.load(cache_path, weights_only=False), True  # load"),
        ("np.int)", "int)"),
        ("dtype=np.int)", "dtype=int)"),
        ("astype(np.int)", "astype(int)"),
    ],
    "utils/general.py": [
        ("astype(np.int)", "astype(int)"),
    ],
    "models/experimental.py": [
        (
            "torch.load(attempt_download(w), map_location=map_location)",
            "torch.load(attempt_download(w), map_location=map_location, weights_only=False)",
        ),
    ],
    "train.py": [
        (
            "torch.load(weights).get('wandb_id')",
            "torch.load(weights, weights_only=False).get('wandb_id')",
        ),
        (
            "torch.load(weights, map_location=device)  # load checkpoint",
            "torch.load(weights, map_location=device, weights_only=False)  # load checkpoint",
        ),
    ],
    "utils/metrics.py": [
        ("np.trapz(", "np.trapezoid("),
    ],
}

for rel, replacements in patches.items():
    path = Path(rel)
    if not path.exists():
        continue
    text = path.read_text()
    for old, new in replacements:
        text = text.replace(old, new)
    path.write_text(text)
PY

mkdir -p data
"$PYTHON_BIN" - "$DATA_YAML" data/ogsod_hbb_rgb.yaml <<'PY'
from pathlib import Path
import sys
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
cfg = yaml.safe_load(src.read_text())
base = Path(cfg.get("path", src.parent)).expanduser()

def absolutize(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return str(path)

out = {
    "train": absolutize(cfg["train"]),
    "val": absolutize(cfg["val"]),
    "test": absolutize(cfg.get("test", cfg["val"])),
    "nc": int(cfg["nc"]),
    "names": cfg["names"],
}
dst.write_text(yaml.safe_dump(out, sort_keys=False))
PY

cp data/hyp.scratch.yaml data/hyp.cold_paper.yaml
perl -0pi -e "s/^mixup:.*/mixup: ${MIXUP}  # CoLD anchor setting/m" data/hyp.cold_paper.yaml

if [[ "$INIT" == "scratch" ]]; then
  WEIGHTS=""
else
  WEIGHTS="yolov5x.pt"
fi

RUN_NAME="cold_anchor_rgb_yolov5x_v5p0_${INIT}_${RUN_SUFFIX}"

echo "[$(date '+%F %T')] Launching ${RUN_NAME}"
echo "workdir=${YOLOV5_WORKDIR}"
echo "data=${DATA_YAML}"
echo "hyp=data/hyp.cold_paper.yaml mixup=${MIXUP}"
echo "epochs=${EPOCHS} batch=${BATCH_SIZE} gpu=${GPU_ID}"

"$PYTHON_BIN" train.py \
  --img-size 256 \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --data data/ogsod_hbb_rgb.yaml \
  --cfg models/yolov5x.yaml \
  --weights "$WEIGHTS" \
  --hyp data/hyp.cold_paper.yaml \
  --device "$GPU_ID" \
  --project "$PROJECT" \
  --name "$RUN_NAME"
