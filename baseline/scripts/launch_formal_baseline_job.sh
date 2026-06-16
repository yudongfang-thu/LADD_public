#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash baseline/scripts/launch_formal_baseline_job.sh <job_id> <gpu_id>

Example:
  bash baseline/scripts/launch_formal_baseline_job.sh sar_yolo11n_s0 2

List jobs:
  cat baseline/scripts/formal_baseline_jobs.tsv
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

JOB_ID="${1:-}"
GPU_ID="${2:-}"
if [[ -z "$JOB_ID" || -z "$GPU_ID" ]]; then
  usage >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOBS="${ROOT_DIR}/baseline/scripts/formal_baseline_jobs.tsv"
RUNNER="${ROOT_DIR}/baseline/scripts/run_formal_baseline.sh"

if [[ ! -f "$JOBS" ]]; then
  echo "Missing jobs file: $JOBS" >&2
  exit 1
fi

row="$(awk -F '\t' -v id="$JOB_ID" 'NR > 1 && $1 == id {print $2 "\t" $3 "\t" $4; found=1} END {if (!found) exit 1}' "$JOBS" || true)"
if [[ -z "$row" ]]; then
  echo "Unknown job_id: $JOB_ID" >&2
  echo "Run print_formal_baseline_jobs.sh to list valid jobs." >&2
  exit 1
fi

IFS=$'\t' read -r modality size seed <<< "$row"
exec "$RUNNER" "$modality" "$size" "$seed" "$GPU_ID"
