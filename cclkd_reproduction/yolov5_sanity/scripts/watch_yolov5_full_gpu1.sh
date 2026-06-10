#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/root/shared-nvme/LADD_public}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv_yolov5/bin/python}"
GPU_ID="${GPU_ID:-1}"
MIN_FREE_MB="${MIN_FREE_MB:-22000}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
STALL_SECONDS="${STALL_SECONDS:-3600}"
RUN_NAME="${RUN_NAME:-yolov5x_online_cclkd_full_b64_s0_cclkd_full_yolov5x_b64_s0_gpu1}"
RUN_DIR="$REPO_ROOT/cclkd_reproduction/yolov5_sanity/results/runs/$RUN_NAME"
LAUNCHER="$REPO_ROOT/cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_cclkd_full.sh"
WATCHDOG_LOG="$RUN_DIR/watchdog.log"

mkdir -p "$RUN_DIR"

log() {
  echo "$(date '+%F %T') $*" | tee -a "$WATCHDOG_LOG"
}

gpu_free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_ID" | awk '{print $1}'
}

pid_alive() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] && ps -p "$pid" >/dev/null 2>&1
}

launcher_alive() {
  [[ -f "$RUN_DIR/pid.txt" ]] || return 1
  local pid
  pid="$(cat "$RUN_DIR/pid.txt" 2>/dev/null || true)"
  pid_alive "$pid"
}

training_pids() {
  ps -eo pid=,comm=,args= \
    | awk -v run="$RUN_NAME" '$2 ~ /python/ && $0 ~ /train_yolov5_cclkd_full.py/ && $0 ~ run {print $1}' \
    || true
}

latest_epoch() {
  local csv="$RUN_DIR/results.csv"
  [[ -f "$csv" ]] || { echo "none"; return; }
  "$PYTHON" - "$csv" <<'PY'
import csv, sys
rows=list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("none")
else:
    row={k.strip():v for k,v in rows[-1].items()}
    print((row.get("epoch") or row.get("Epoch") or "").strip() or "unknown")
PY
}

results_age_seconds() {
  local csv="$RUN_DIR/results.csv"
  [[ -f "$csv" ]] || { echo 999999999; return; }
  "$PYTHON" - "$csv" <<'PY'
import os, sys, time
print(int(time.time() - os.path.getmtime(sys.argv[1])))
PY
}

restart_launcher() {
  log "ACTION restart_or_start_launcher gpu=$GPU_ID min_free_mb=$MIN_FREE_MB"
  cd "$REPO_ROOT"
  LAUNCH=1 DRY_RUN=0 WAIT_FOR_GPU=1 MIN_FREE_MB="$MIN_FREE_MB" POLL_SECONDS=120 PYTHON="$PYTHON" \
    bash "$LAUNCHER" 64 0 "$GPU_ID" cclkd_full_yolov5x_b64_s0_gpu1 >>"$WATCHDOG_LOG" 2>&1 || {
      log "ERROR launcher restart command failed"
      return 1
    }
}

log "watchdog_start repo=$REPO_ROOT gpu=$GPU_ID interval=${INTERVAL_SECONDS}s min_free_mb=$MIN_FREE_MB"

while true; do
  free_mb="$(gpu_free_mb || echo unknown)"
  train_pids="$(training_pids | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
  epoch="$(latest_epoch)"
  age="$(results_age_seconds)"
  log "status gpu${GPU_ID}_free_mb=$free_mb launcher_alive=$(launcher_alive && echo yes || echo no) train_pids=${train_pids:-none} epoch=$epoch results_age_s=$age"

  if [[ "$epoch" =~ ^[0-9]+$ ]] && (( epoch >= 399 )); then
    log "done epoch=$epoch; watchdog_exit"
    exit 0
  fi

  if [[ -n "$train_pids" ]]; then
    if (( age > STALL_SECONDS )); then
      if grep -Eiq "out of memory|traceback|runtimeerror|exception|killed" "$RUN_DIR/nohup.log" 2>/dev/null; then
        log "WARN training appears stale and log contains error markers; leaving process untouched for manual inspection train_pids=$train_pids"
      else
        log "WARN training results stale but process alive; not killing active process"
      fi
    fi
  else
    if ! launcher_alive; then
      restart_launcher || true
    elif [[ "$free_mb" =~ ^[0-9]+$ ]] && (( free_mb >= MIN_FREE_MB )); then
      log "WARN gpu is free enough but launcher is still waiting; launcher log should break out within poll interval"
    fi
  fi

  sleep "$INTERVAL_SECONDS"
done
