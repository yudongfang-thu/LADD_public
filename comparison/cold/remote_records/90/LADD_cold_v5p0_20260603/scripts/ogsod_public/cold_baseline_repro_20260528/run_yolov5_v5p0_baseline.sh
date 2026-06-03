#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/ogsod_public/cold_baseline_repro_20260528/run_yolov5_v5p0_baseline.sh <ogsod_hbb_sar.yaml> <scratch|coco> [gpu_id]

Runs the CoLD-paper anchor baseline only:
  OGSOD-1.0 HBB, SAR-only, native Ultralytics YOLOv5 v5.0, yolov5x, imgsz=256,
  batch=64, epochs=400, SGD-style v5.0 default hyp.

Environment overrides:
  YOLOV5_WORKDIR  Directory for the v5.0 clone. Default: ../yolov5_cold_v5p0
  PROJECT         YOLOv5 project path. Default: runs/ogsod_cold_anchor
  RUN_SUFFIX      Extra suffix for the run name.
  MIXUP           MixUp probability written into hyp. Default: 0.1
  EPOCHS          Number of training epochs. Default: 400
  PYTHON_BIN      Python executable. Default: python3
  INSTALL_REQS=1  Run pip install -r requirements.txt after clone.

Examples:
  scripts/ogsod_public/cold_baseline_repro_20260528/run_yolov5_v5p0_baseline.sh \
    /mnt/dataY/ydf/projects/LADD_og/data/ogsod_public_prepared/yamls/ogsod_hbb_sar.yaml scratch 0

  YOLOV5_WORKDIR=/mnt/dataY/ydf/projects/yolov5_cold_v5p0 \
  scripts/ogsod_public/cold_baseline_repro_20260528/run_yolov5_v5p0_baseline.sh \
    /path/to/ogsod_hbb_sar.yaml coco 1
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
PROJECT="${PROJECT:-runs/ogsod_cold_anchor}"
RUN_SUFFIX="${RUN_SUFFIX:-s0}"
MIXUP="${MIXUP:-0.1}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "$YOLOV5_WORKDIR/.git" ]]; then
  mkdir -p "$(dirname "$YOLOV5_WORKDIR")"
  git clone --depth 1 --branch v5.0 https://github.com/ultralytics/yolov5.git "$YOLOV5_WORKDIR"
fi

cd "$YOLOV5_WORKDIR"

if [[ "${INSTALL_REQS:-0}" == "1" ]]; then
  python3 -m pip install -r requirements.txt
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
for rel, reps in patches.items():
    p = Path(rel)
    if not p.exists():
        continue
    s = p.read_text()
    for old, new in reps:
        s = s.replace(old, new)
    p.write_text(s)
PY

mkdir -p data
"$PYTHON_BIN" - "$DATA_YAML" data/ogsod_hbb_sar.yaml <<'PY'
from pathlib import Path
import sys
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
cfg = yaml.safe_load(src.read_text())
base = Path(cfg.get("path", src.parent)).expanduser()

def absolutize(value: str) -> str:
    p = Path(value)
    if not p.is_absolute():
        p = base / p
    return str(p)

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
perl -0pi -e "s/^mixup:.*/mixup: ${MIXUP}  # CoLD paper states Mosaic + Mixup; exact probability is unspecified/m" data/hyp.cold_paper.yaml

if [[ "$INIT" == "scratch" ]]; then
  WEIGHTS=""
else
  WEIGHTS="yolov5x.pt"
fi

RUN_NAME="cold_anchor_sar_yolov5x_v5p0_${INIT}_${RUN_SUFFIX}"

echo "[$(date '+%F %T')] Launching ${RUN_NAME}"
echo "workdir=${YOLOV5_WORKDIR}"
echo "data=${DATA_YAML}"
echo "hyp=data/hyp.cold_paper.yaml mixup=${MIXUP}"

"$PYTHON_BIN" train.py \
  --img-size 256 \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --data data/ogsod_hbb_sar.yaml \
  --cfg models/yolov5x.yaml \
  --weights "$WEIGHTS" \
  --hyp data/hyp.cold_paper.yaml \
  --device "$GPU_ID" \
  --project "$PROJECT" \
  --name "$RUN_NAME"
