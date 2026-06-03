#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
JOBS="${ROOT_DIR}/scripts/ogsod_public/formal_nomosaic_20260528/formal_baseline_jobs.tsv"

if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$JOBS"
else
  cat "$JOBS"
fi

