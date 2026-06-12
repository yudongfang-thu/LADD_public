#!/usr/bin/env bash
set -euo pipefail
cd /mnt/dataY/ydf/projects/LADD_public
LAUNCHER="cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_cclkd_full.sh"
RUNS="cclkd_reproduction/yolov5_sanity/results/runs"

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

run_name() {
  local mode="$1" batch="$2" seed="$3" tag="$4"
  printf 'yolov5x_%s_b%s_s%s_%s' "$mode" "$batch" "$seed" "$tag"
}

launch_one() {
  local mode="$1" gpu="$2" tag="$3"
  log "launch mode=${mode} gpu=${gpu} tag=${tag}"
  LAUNCH=1 DRY_RUN=0 SMOKE_EPOCHS=80 CCLKD_YOLOV5_MODE="$mode" \
    bash "$LAUNCHER" 32 0 "$gpu" "$tag"
  local name
  name="$(run_name "$mode" 32 0 "$tag")"
  local pid_file="$RUNS/$name/pid.txt"
  local pid=""
  for _ in $(seq 1 30); do
    if [[ -s "$pid_file" ]]; then
      pid="$(cat "$pid_file")"
      break
    fi
    sleep 1
  done
  if [[ -z "$pid" ]]; then
    log "ERROR no pid file for ${name}"
    return 1
  fi
  log "started ${name} pid=${pid}"
  printf '%s\n' "$pid"
}

wait_pid_done() {
  local label="$1" pid="$2"
  log "waiting ${label} pid=${pid}"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 300
  done
  log "finished ${label} pid=${pid}"
}

W1A_TAG="detonly_same_trainer_b32_e80_wave1_gpu1"
W1B_TAG="twobranch_nokd_b32_e80_wave1_gpu3"
W2A_TAG="paper_atkd_only_b32_e80_wave2_gpu1"
W2B_TAG="paper_full_b32_e80_wave2_gpu3"

log "queue start: wave1 det_only_same_trainer+two_branch_no_kd, wave2 paper_atkd_only+paper_full"
P1="$(launch_one det_only_same_trainer 1 "$W1A_TAG" | tail -n 1)"
P2="$(launch_one two_branch_no_kd 3 "$W1B_TAG" | tail -n 1)"
wait_pid_done "wave1 det_only_same_trainer" "$P1" &
WP1=$!
wait_pid_done "wave1 two_branch_no_kd" "$P2" &
WP2=$!
wait "$WP1" "$WP2"
log "wave1 complete; launching wave2"
launch_one paper_atkd_only 1 "$W2A_TAG" >/dev/null
launch_one paper_full 3 "$W2B_TAG" >/dev/null
log "wave2 launched; queue complete"
