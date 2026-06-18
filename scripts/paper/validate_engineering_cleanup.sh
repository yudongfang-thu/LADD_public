#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python3 -m py_compile \
  baseline/code/train_ogsod_baseline.py \
  ladd/code/train_ladd_hbb.py \
  ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py \
  ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py \
  ladd/code/src/teacher_student_decomposition_kd_hbb/model.py \
  ladd/code_versions/current_hbb/tools/train_ladd_hbb.py \
  tools/paper_collect_results.py \
  tools/paper_validate_main_table.py \
  tools/check_current_hbb_sync.py

bash -n scripts/paper/paper_common.sh
bash -n scripts/paper/run_paper_baseline.sh
bash -n scripts/paper/run_paper_ladd_probea.sh
bash -n scripts/paper/run_paper_comparison_kd.sh
bash -n scripts/paper/run_paper_hallucidet.sh
bash -n scripts/paper/run_paper_cclkd_online.sh
bash -n scripts/paper/dry_run_all_paper_entrypoints.sh
bash -n scripts/paper/prepare_paper_dataset_yamls.sh

python3 tools/check_current_hbb_sync.py

dry_log="$(mktemp)"
trap 'rm -f "$dry_log"' EXIT
DRY_RUN=1 bash scripts/paper/dry_run_all_paper_entrypoints.sh | tee "$dry_log"

grep -q "mosaic=1.0" "$dry_log"
grep -q "close_mosaic=700" "$dry_log"
grep -q "LADD_A1B_MODE=dynamic_probe" "$dry_log"
grep -q "KD_CALIBRATION_MODE=affine" "$dry_log"

if grep -E "formal_nomosaic|no-mosaic|nomosaic" "$dry_log"; then
  echo "Dry-run output references formal/no-mosaic paths." >&2
  exit 1
fi

if grep -E " a2 |phase a2|<a2|_a2_" "$dry_log"; then
  echo "Dry-run output appears to reference A2." >&2
  exit 1
fi

echo "OK: paper engineering cleanup validation passed."
