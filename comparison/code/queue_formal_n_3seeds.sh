#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  comparison/code/queue_formal_n_3seeds.sh <fgd|ld|cclkd|hallucidet> <gpu_id>

Runs YOLO11n seeds 0, 42, and 123 serially for one frozen comparison method.
Launch one queue per method; multiple method queues may share a GPU when memory
and throughput have been checked.

Optional:
  RUN_TAG_SUFFIX=_public4090_final_v1
  QUEUE_POLL_SECONDS=60
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

METHOD="${1:-}"
GPU_ID="${2:-}"
case "$METHOD" in
  fgd|ld|cclkd|hallucidet) ;;
  *) usage >&2; exit 1 ;;
esac
if [[ -z "$GPU_ID" ]]; then
  usage >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SUFFIX="${RUN_TAG_SUFFIX:-_public4090_final_v1}"
POLL_SECONDS="${QUEUE_POLL_SECONDS:-60}"
QUEUE_LOG_DIR="logs/formal_nomosaic_20260528/comparisons/queues"
QUEUE_LOG="${QUEUE_LOG_DIR}/queue_${METHOD}_n_3seeds_gpu${GPU_ID}${SUFFIX}.log"
mkdir -p "$QUEUE_LOG_DIR"

exec > >(tee -a "$QUEUE_LOG") 2>&1
echo "[$(date '+%F %T')] queue start method=${METHOD} gpu=${GPU_ID} suffix=${SUFFIX}"

for seed in 0 42 123; do
  run_tag="formal_nomosaic_yolo11n_${METHOD}_from_yolo_s${seed}${SUFFIX}"
  pid_path="logs/formal_nomosaic_20260528/comparisons/from_yolo_pretrain/${run_tag}_gpu${GPU_ID}.pid"
  outer_log="logs/formal_nomosaic_20260528/comparisons/from_yolo_pretrain/${run_tag}_gpu${GPU_ID}.outer.log"

  echo "[$(date '+%F %T')] launching method=${METHOD} seed=${seed} gpu=${GPU_ID}"
  RUN_TAG_SUFFIX="$SUFFIX" \
    bash comparison/code/launch_formal_from_yolo_kd_job.sh "$METHOD" n "$seed" "$GPU_ID"

  pid="$(cat "$pid_path")"
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_SECONDS"
  done

  if ! grep -q "Phase b finished" "$outer_log"; then
    echo "[$(date '+%F %T')] FAILED method=${METHOD} seed=${seed}; inspect ${outer_log}" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] finished method=${METHOD} seed=${seed}"
done

echo "[$(date '+%F %T')] queue complete method=${METHOD} gpu=${GPU_ID}"
