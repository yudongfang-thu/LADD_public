#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/LADD_public}"
cd "${REPO_ROOT}"

QUEUE_TS="${QUEUE_TS:-$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-0}"
EPOCHS="${EPOCHS:-300}"
BATCH="${BATCH:-64}"
IMGSZ="${IMGSZ:-640}"
SEED="${SEED:-0}"
WORKERS="${WORKERS:-8}"
POLL_SECONDS="${POLL_SECONDS:-300}"
GPU_FREE_MIN_MB="${GPU_FREE_MIN_MB:-15000}"
WATCH_MIN_BEST_MAP50="${WATCH_MIN_BEST_MAP50:-0.62}"

WAIT_LABEL="${WAIT_LABEL:-syncgeo_featrel}"
WAIT_SCREEN="${WAIT_SCREEN:-cmdi_align_syncgeo_featrel_20260618_203710}"
WAIT_RESULTS="${WAIT_RESULTS:-/root/autodl-tmp/LADD_public/comparison/cmdistill/native_reproduction/runs/vedai_yolov5_cmdistill_alignment_probe/paper80_seed0/vedai512_rgb_syncgeo_featrel_yolov5s_e300_b64_img640_s0_20260618_203710/results.csv}"
PROJECT="${PROJECT:-${REPO_ROOT}/comparison/cmdistill/native_reproduction/runs/vedai_yolov5_cmdistill_alignment_probe/paper80_seed0}"

QUEUE_DIR="${REPO_ROOT}/comparison/cmdistill/native_reproduction/logs/vedai_yolov5_cmdistill_queue/${QUEUE_TS}_logitfix"
QUEUE_LOG="${QUEUE_DIR}/queue.log"
mkdir -p "${QUEUE_DIR}"

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "${QUEUE_LOG}"
}

screen_exists() {
  screen -ls | grep -Fq "$1"
}

summarize_results() {
  local results="$1"
  /root/miniconda3/bin/python - "$results" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("missing 0 0 -1 -1")
    raise SystemExit

rows = []
with path.open(newline="") as f:
    for row in csv.DictReader(f):
        try:
            rows.append({
                "epoch": int(float(row["epoch"])),
                "map50": float(row["metrics/mAP_0.5"]),
                "map5095": float(row["metrics/mAP_0.5:0.95"]),
            })
        except Exception:
            pass

if not rows:
    print("empty 0 0 -1 -1")
    raise SystemExit

best = max(rows, key=lambda r: r["map50"])
best95 = max(rows, key=lambda r: r["map5095"])
latest = rows[-1]
print(
    f"ok {best['map50']:.6f} {latest['map50']:.6f} "
    f"{best['epoch']} {latest['epoch']} {best95['map5095']:.6f} {best95['epoch']}"
)
PY
}

wait_for_screen_done() {
  local screen_name="$1"
  local results="$2"
  local label="$3"
  local waited=0
  while screen_exists "${screen_name}"; do
    local summary
    summary="$(summarize_results "${results}" || true)"
    log "WAIT ${label}: screen=${screen_name} waited=${waited}s summary=${summary}"
    sleep "${POLL_SECONDS}"
    waited=$((waited + POLL_SECONDS))
  done
  sleep 20
  local summary
  summary="$(summarize_results "${results}" || true)"
  log "DONE ${label}: screen=${screen_name} summary=${summary}"
}

launch_probe() {
  local probe="$1"
  local kd_warmup="$2"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local screen_name="cmdi_align_${probe}_${ts}"
  local watch_screen="cmdi_align_watch_${probe}_${ts}"
  local name="vedai512_rgb_${probe}_yolov5s_e${EPOCHS}_b${BATCH}_img${IMGSZ}_s${SEED}_${ts}"
  local results="${PROJECT}/${name}/results.csv"

  log "LAUNCH ${probe}: screen=${screen_name} warmup=${kd_warmup}"
  RUN_TS="${ts}" \
  SCREEN_NAME="${screen_name}" \
  WATCH_SCREEN="${watch_screen}" \
  PROBE_NAME="${probe}" \
  DEVICE="${DEVICE}" \
  EPOCHS="${EPOCHS}" \
  BATCH="${BATCH}" \
  IMGSZ="${IMGSZ}" \
  SEED="${SEED}" \
  WORKERS="${WORKERS}" \
  GPU_FREE_MIN_MB="${GPU_FREE_MIN_MB}" \
  PAIRED_SYNC_GEO=1 \
  ALIGNED_NO_GEO=0 \
  KEEP_COLOR_AUG=0 \
  FEATURE_WEIGHT=1.0 \
  RELATION_WEIGHT=1.0 \
  LOGIT_WEIGHT=1.0 \
  FEATURE_LAYERS=shallow_deep \
  RELATION_LAYER=deepest \
  FEATURE_ADAPT=1 \
  RAW_OUTPUT_KD=0 \
  KD_WARMUP_EPOCHS="${kd_warmup}" \
  KD_GAIN=1.0 \
  WATCH_POLL_SECONDS="${POLL_SECONDS}" \
  WATCH_MIN_EPOCH=120 \
  WATCH_WINDOW=35 \
  WATCH_MIN_BEST_MAP50="${WATCH_MIN_BEST_MAP50}" \
  WATCH_MIN_IMPROVEMENT=0.008 \
  bash comparison/cmdistill/native_reproduction/scripts/launch_vedai_cmdistill_alignment_probe_autodl.sh \
    2>&1 | tee -a "${QUEUE_LOG}"

  wait_for_screen_done "${screen_name}" "${results}" "${probe}"
}

main() {
  log "logitfix queue start wait_screen=${WAIT_SCREEN} wait_results=${WAIT_RESULTS}"
  wait_for_screen_done "${WAIT_SCREEN}" "${WAIT_RESULTS}" "${WAIT_LABEL}"
  launch_probe syncgeo_allkd_fix 0.0
  launch_probe syncgeo_allkd_warm10_fix 10.0
  log "logitfix queue complete"
}

main "$@"
