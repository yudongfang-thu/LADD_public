#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/cold_anchor}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DATA_YAML="${DATA_YAML:-/root/autodl-tmp/datasets/ogsod_hbb_sar.yaml}"
PROJECT="${PROJECT:-$RUN_ROOT/runs/ogsod_cold_anchor}"
EPOCHS="${EPOCHS:-400}"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="$RUN_ROOT/logs"
QUEUE_LOG="$LOG_DIR/queued_baselines_5090d.log"

mkdir -p "$LOG_DIR"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$QUEUE_LOG"
}

active_train_count() {
  pgrep -af "train.py .*ogsod_cold_anchor" | grep -v queued_baselines_5090d | wc -l | tr -d ' '
}

wait_for_no_train() {
  local n
  while true; do
    n="$(active_train_count || true)"
    if [[ "$n" == "0" ]]; then
      log "No active YOLOv5 training process."
      return 0
    fi
    log "Waiting for active training processes to finish: $n"
    sleep 300
  done
}

wait_for_wave_to_start() {
  local timeout="${1:-300}"
  local start now n
  start="$(date +%s)"
  while true; do
    n="$(active_train_count || true)"
    if [[ "$n" != "0" ]]; then
      log "Wave started; active train process count: $n"
      return 0
    fi
    now="$(date +%s)"
    if (( now - start > timeout )); then
      log "WARNING: wave did not start within ${timeout}s; continuing to next guard."
      return 0
    fi
    sleep 10
  done
}

patch_common_compat() {
  local workdir="$1"
  "$PYTHON_BIN" - "$workdir" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
for rel in ["utils/datasets.py", "utils/general.py"]:
    p = root / rel
    if p.exists():
        s = p.read_text()
        s = s.replace("np.int)", "int)")
        s = s.replace("dtype=np.int)", "dtype=int)")
        s = s.replace("astype(np.int)", "astype(int)")
        if rel == "utils/datasets.py":
            s = s.replace("torch.load(cache_path), True", "torch.load(cache_path, weights_only=False), True")
        p.write_text(s)

p = root / "utils/loss.py"
if p.exists():
    s = p.read_text()
    s = s.replace(
        "gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)",
        "gj.clamp_(0, int(gain[3] - 1)), gi.clamp_(0, int(gain[2] - 1))",
    )
    p.write_text(s)

for rel in ["train.py", "models/experimental.py", "utils/general.py"]:
    p = root / rel
    if not p.exists():
        continue
    s = p.read_text()
    s = s.replace("torch.load(weights)", "torch.load(weights, weights_only=False)")
    s = s.replace("torch.load(weights, map_location=device)", "torch.load(weights, map_location=device, weights_only=False)")
    s = s.replace("torch.load(w, map_location=map_location)", "torch.load(w, map_location=map_location, weights_only=False)")
    s = s.replace("torch.load(attempt_download(w), map_location=map_location)", "torch.load(attempt_download(w), map_location=map_location, weights_only=False)")
    p.write_text(s)
PY
}

start_v5_job() {
  local init="$1"
  local mixup="$2"
  local suffix="$3"
  local screen_name="q_${suffix}"
  local log_file="$LOG_DIR/${suffix}.log"

  log "Starting v5.0 job: init=$init mixup=$mixup suffix=$suffix"
  screen -S "$screen_name" -dm bash -lc \
    "cd '$RUN_ROOT' && PYTHON_BIN='$PYTHON_BIN' YOLOV5_WORKDIR='$RUN_ROOT/yolov5_v5p0' PROJECT='$PROJECT' RUN_SUFFIX='$suffix' MIXUP='$mixup' EPOCHS='$EPOCHS' ./run_yolov5_v5p0_baseline.sh '$DATA_YAML' '$init' '$GPU_ID' 2>&1 | tee '$log_file'"
}

prepare_v6() {
  local workdir="$RUN_ROOT/yolov5_v6p0"
  if [[ ! -d "$workdir/.git" ]]; then
    git clone --branch v6.0 --depth 1 https://github.com/ultralytics/yolov5.git "$workdir"
  fi
  patch_common_compat "$workdir"
  mkdir -p "$workdir/data" "$workdir/data/hyps"
  "$PYTHON_BIN" - "$DATA_YAML" "$workdir/data/ogsod_hbb_sar.yaml" <<'PY'
from pathlib import Path
import sys
import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
cfg = yaml.safe_load(src.read_text())
base = Path(cfg.get("path", src.parent)).expanduser()

def absolutize(value):
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
}

write_v6_hyp() {
  local workdir="$RUN_ROOT/yolov5_v6p0"
  local mixup="$1"
  local src="$workdir/data/hyps/hyp.scratch-low.yaml"
  [[ -f "$src" ]] || src="$workdir/data/hyp.scratch.yaml"
  cp "$src" "$workdir/data/hyps/hyp.cold_paper.yaml"
  perl -0pi -e "s/^mixup:.*/mixup: ${mixup}/m" "$workdir/data/hyps/hyp.cold_paper.yaml"
}

start_v6_job() {
  local init="$1"
  local mixup="$2"
  local suffix="$3"
  local workdir="$RUN_ROOT/yolov5_v6p0"
  local screen_name="q_${suffix}"
  local log_file="$LOG_DIR/${suffix}.log"
  local weights=""

  prepare_v6
  write_v6_hyp "$mixup"
  if [[ "$init" == "coco" ]]; then
    weights="yolov5x.pt"
  fi

  log "Starting v6.0 job: init=$init mixup=$mixup suffix=$suffix"
  screen -S "$screen_name" -dm bash -lc \
    "cd '$workdir' && '$PYTHON_BIN' train.py --imgsz 256 --batch-size 64 --epochs '$EPOCHS' --data data/ogsod_hbb_sar.yaml --cfg models/yolov5x.yaml --weights '$weights' --hyp data/hyps/hyp.cold_paper.yaml --device '$GPU_ID' --project '$PROJECT' --name 'cold_anchor_sar_yolov5x_v6p0_${init}_${suffix}' --patience '$EPOCHS' 2>&1 | tee '$log_file'"
}

run_wave() {
  local name="$1"
  shift
  log "Preparing wave: $name"
  wait_for_no_train
  "$@"
  wait_for_wave_to_start 300
  wait_for_no_train
  log "Finished wave: $name"
}

main() {
  log "Queue started. EPOCHS=$EPOCHS GPU_ID=$GPU_ID PROJECT=$PROJECT"
  log "Wave 0 guard: wait for currently running jobs."
  wait_for_no_train

  log "Wave 1: v5.0 mixup=0.0 scratch + coco"
  start_v5_job scratch 0.0 5090d_full_scratch_mixup000
  start_v5_job coco 0.0 5090d_full_coco_mixup000
  wait_for_wave_to_start 300
  wait_for_no_train
  log "Wave 1 completed."

  log "Wave 2: v6.0 mixup=0.1 scratch + coco"
  start_v6_job scratch 0.1 5090d_full_scratch_mixup010
  start_v6_job coco 0.1 5090d_full_coco_mixup010
  wait_for_wave_to_start 300
  wait_for_no_train
  log "Wave 2 completed."

  log "All queued baseline waves completed."
}

main "$@"
