#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN=1 bash scripts/paper/run_paper_baseline.sh sar n 0 0
DRY_RUN=1 bash scripts/paper/run_paper_baseline.sh rgb n 0 0
DRY_RUN=1 bash scripts/paper/run_paper_ladd_probea.sh n 0 0
DRY_RUN=1 bash scripts/paper/run_paper_comparison_kd.sh ld n 0 0
DRY_RUN=1 bash scripts/paper/run_paper_comparison_kd.sh cmdistill n 0 0
DRY_RUN=1 bash scripts/paper/run_paper_comparison_kd.sh fgd n 0 0
DRY_RUN=1 bash scripts/paper/run_paper_hallucidet.sh n 0 0
DRY_RUN=1 bash scripts/paper/run_paper_cclkd_online.sh n 0 0
