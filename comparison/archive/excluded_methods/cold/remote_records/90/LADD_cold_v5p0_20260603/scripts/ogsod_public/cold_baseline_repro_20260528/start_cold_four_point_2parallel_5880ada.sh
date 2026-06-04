#!/usr/bin/env bash
set -euo pipefail

cd /home/xmu/djd/ladd

LOG_DIR="/home/xmu/djd/ladd/cold_anchor/logs"
mkdir -p "$LOG_DIR"

ts="$(date +%Y%m%d_%H%M%S)"
log="${LOG_DIR}/cold_four_point_2parallel_detached_${ts}.log"
pidfile="${LOG_DIR}/cold_four_point_2parallel_5880ada.pid"

setsid bash -c \
  'cd /home/xmu/djd/ladd && exec bash scripts/ogsod_public/cold_baseline_repro_20260528/queue_cold_four_point_2parallel_5880ada.sh' \
  > "$log" 2>&1 < /dev/null &
pid="$!"

printf '%s\n' "$pid" > "$pidfile"
printf 'PID=%s\nLOG=%s\nPIDFILE=%s\n' "$pid" "$log" "$pidfile"
