#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/home/xmu/djd/ladd/cold_anchor}"
YOLOV5_WORKDIR="${YOLOV5_WORKDIR:-$RUN_ROOT/yolov5_v5p0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_YAML="${DATA_YAML:-/home/xmu/djd/ladd/datasets/ogsod_hbb_sar.yaml}"
TEACHER_DATA_YAML="${TEACHER_DATA_YAML:-/home/xmu/djd/ladd/datasets/ogsod_hbb_rgb.yaml}"
PROJECT="${PROJECT:-$RUN_ROOT/runs/ogsod_cold_repro}"
RUN_NAME="${RUN_NAME:-cold_v5p0_yolov5x_coco_mixup010_cpm_iwm_b32_acc64}"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-64}"
MIXUP="${MIXUP:-0.1}"
LAMBDA_CLS_COLD="${LAMBDA_CLS_COLD:-0.0}"
LAMBDA_LOC_COLD="${LAMBDA_LOC_COLD:-1.0}"
TEACHER_DET_WEIGHT="${TEACHER_DET_WEIGHT:-1.0}"
COLD_LOSS_MODE="${COLD_LOSS_MODE:-matched}"
COLD_TERMS="${COLD_TERMS:-both}"
COLD_IWM_MODE="${COLD_IWM_MODE:-mean}"
ASSERT_NONNEGATIVE_COLD="${ASSERT_NONNEGATIVE_COLD:-0}"
TEMPERATURE="${TEMPERATURE:-20.0}"
ALPHA_NON_TARGET="${ALPHA_NON_TARGET:-2.0}"
CANDIDATE_TOPK="${CANDIDATE_TOPK:-1000}"
CANDIDATE_MIN_CONF="${CANDIDATE_MIN_CONF:-0.001}"
CANDIDATE_IOU_WEIGHT_FLOOR="${CANDIDATE_IOU_WEIGHT_FLOOR:-0.0}"
MAX_BATCHES="${MAX_BATCHES:-}"
EXTRA_ARGS=()
if [[ -n "$MAX_BATCHES" ]]; then
  EXTRA_ARGS+=(--max-batches "$MAX_BATCHES")
fi
if [[ "$ASSERT_NONNEGATIVE_COLD" == "1" ]]; then
  EXTRA_ARGS+=(--assert-nonnegative-cold)
fi

cd "$YOLOV5_WORKDIR"

"$PYTHON_BIN" - <<'PY'
from pathlib import Path

patches = {
    "utils/datasets.py": [
        ("torch.load(cache_path), True", "torch.load(cache_path, weights_only=False), True"),
        ("np.int)", "int)"),
        ("dtype=np.int)", "dtype=int)"),
        ("astype(np.int)", "astype(int)"),
    ],
    "utils/loss.py": [
        (
            "gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)",
            "gj.clamp_(0, int(gain[3] - 1)), gi.clamp_(0, int(gain[2] - 1))",
        ),
    ],
    "models/experimental.py": [
        (
            "torch.load(attempt_download(w), map_location=map_location)",
            "torch.load(attempt_download(w), map_location=map_location, weights_only=False)",
        ),
    ],
    "utils/general.py": [
        (
            "torch.load(f, map_location=torch.device('cpu'))",
            "torch.load(f, map_location=torch.device('cpu'), weights_only=False)",
        ),
        ("astype(np.int)", "astype(int)"),
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

def absolutize(value):
    p = Path(value)
    return str(p if p.is_absolute() else base / p)

out = {
    "train": absolutize(cfg["train"]),
    "val": absolutize(cfg["val"]),
    "test": absolutize(cfg.get("test", cfg["val"])),
    "nc": int(cfg["nc"]),
    "names": cfg["names"],
}
dst.write_text(yaml.safe_dump(out, sort_keys=False))
PY
"$PYTHON_BIN" - "$TEACHER_DATA_YAML" data/ogsod_hbb_rgb.yaml <<'PY'
from pathlib import Path
import sys
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
cfg = yaml.safe_load(src.read_text())
base = Path(cfg.get("path", src.parent)).expanduser()

def absolutize(value):
    p = Path(value)
    return str(p if p.is_absolute() else base / p)

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
perl -0pi -e "s/^mixup:.*/mixup: ${MIXUP}/m" data/hyp.cold_paper.yaml

"$PYTHON_BIN" "$RUN_ROOT/train_cold_v5p0_hbb.py" \
  --yolov5-root "$YOLOV5_WORKDIR" \
  --data data/ogsod_hbb_sar.yaml \
  --teacher-data data/ogsod_hbb_rgb.yaml \
  --cfg models/yolov5x.yaml \
  --weights yolov5x.pt \
  --teacher-weights yolov5x.pt \
  --hyp data/hyp.cold_paper.yaml \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --effective-batch-size "$EFFECTIVE_BATCH_SIZE" \
  --img-size 256 256 \
  --device "$GPU_ID" \
  --workers 8 \
  --project "$PROJECT" \
  --name "$RUN_NAME" \
  --lambda-cls-cold "$LAMBDA_CLS_COLD" \
  --lambda-loc-cold "$LAMBDA_LOC_COLD" \
  --teacher-det-weight "$TEACHER_DET_WEIGHT" \
  --cold-loss-mode "$COLD_LOSS_MODE" \
  --cold-terms "$COLD_TERMS" \
  --cold-iwm-mode "$COLD_IWM_MODE" \
  --temperature "$TEMPERATURE" \
  --alpha-non-target "$ALPHA_NON_TARGET" \
  --candidate-topk "$CANDIDATE_TOPK" \
  --candidate-min-conf "$CANDIDATE_MIN_CONF" \
  --candidate-iou-weight-floor "$CANDIDATE_IOU_WEIGHT_FLOOR" \
  "${EXTRA_ARGS[@]}"
