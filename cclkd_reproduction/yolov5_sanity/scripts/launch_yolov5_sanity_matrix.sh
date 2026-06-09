#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  DRY_RUN=1 LAUNCH=0 bash cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_sanity_matrix.sh smoke

The smoke tier prints five commands:
  E1: cold_b64_pretrained
  E2: cclkd_b32_pretrained
  E3: cold_b64_scratch
  E4: rgb_teacher_b64_pretrained
  E5: x6_b32_pretrained_diag

This matrix launcher is intentionally dry-run oriented. Start jobs manually with
launch_yolov5_ogsod_baseline.sh so only one run is launched at a time.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TIER="${1:-smoke}"
DRY_RUN="${DRY_RUN:-1}"
LAUNCH="${LAUNCH:-0}"

if [[ "$TIER" != "smoke" ]]; then
  echo "Only the smoke tier is defined now. Do not default-launch a full tier." >&2
  exit 2
fi

if [[ "$LAUNCH" == "1" ]]; then
  if [[ "${ALLOW_MATRIX_LAUNCH:-0}" != "1" ]]; then
    echo "Refusing matrix launch: set LAUNCH=0 and manually start one selected command." >&2
    exit 2
  fi
  echo "ALLOW_MATRIX_LAUNCH=1 was set, but this script still will not start all smoke jobs." >&2
  echo "Use one printed launch_yolov5_ogsod_baseline.sh command for a single run." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LAUNCHER_REL="cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_ogsod_baseline.sh"

EXPERIMENTS=(
  "E1 sar x pretrained 64 0 0 cold_b64_pretrained # CoLD Table I primary baseline sanity"
  "E2 sar x pretrained 32 0 0 cclkd_b32_pretrained # Check whether CCLKD batch=32 changes baseline"
  "E3 sar x scratch 64 0 0 cold_b64_scratch # Check whether paper likely used COCO pretrained weights"
  "E4 rgb x pretrained 64 0 0 rgb_teacher_b64_pretrained # Estimate optical teacher upper bound"
  "E5 sar x6 pretrained 32 0 0 x6_b32_pretrained_diag # Diagnostic only for parameter inconsistency"
)

for item in "${EXPERIMENTS[@]}"; do
  read -r eid modality model init batch seed gpu tag _ <<<"$item"
  printf '# %s %s\n' "$eid" "${item#*# }"
  printf 'DRY_RUN=%q LAUNCH=0 bash %q %q %q %q %q %q %q %q\n' \
    "${DRY_RUN:-1}" "$LAUNCHER_REL" "$modality" "$model" "$init" "$batch" "$seed" "$gpu" "$tag"
  echo
done
