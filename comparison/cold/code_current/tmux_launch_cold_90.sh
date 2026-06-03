#!/usr/bin/env bash
set -euo pipefail

# Launch the 90-server CoLD queue in tmux.
# This only starts when invoked explicitly on the 90 server.

WORKDIR="${WORKDIR:-/mnt/dataY/ydf/projects/LADD_cold_v5p0_20260603}"
RUN_ROOT="${RUN_ROOT:-$WORKDIR/cold_anchor}"
RUN_TAG="${RUN_TAG:-90_$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-cold90_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-$RUN_ROOT/logs}"
QUEUE_SCRIPT="${QUEUE_SCRIPT:-$WORKDIR/scripts/ogsod_public/cold_baseline_repro_20260528/queue_cold_offline_terms_serial_90.sh}"
mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

if [[ ! -f "$QUEUE_SCRIPT" ]]; then
  echo "Missing queue script: $QUEUE_SCRIPT" >&2
  exit 1
fi

LAUNCH_LOG="$LOG_DIR/${SESSION}.launch.log"

tmux new-session -d -s "$SESSION" \
  "cd '$WORKDIR' && RUN_TAG='$RUN_TAG' WORKDIR='$WORKDIR' RUN_ROOT='$RUN_ROOT' bash '$QUEUE_SCRIPT' 2>&1 | tee '$LAUNCH_LOG'"

echo "launched tmux session: $SESSION"
echo "launch_log=$LAUNCH_LOG"
echo "attach: tmux attach -t $SESSION"
