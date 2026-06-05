#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare_server_runtime.sh <OGSOD-1.0-root> <asset-root>

The asset root is an existing private runtime/archive containing:
  yolo11*.pt
  runs_public/ogsod/hbb/formal_nomosaic_20260528/baselines/

This script keeps private checkpoints and generated results out of git while
making the paper repository directly runnable.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DATASET_ROOT="${1:-}"
ASSET_ROOT="${2:-}"
if [[ -z "$DATASET_ROOT" || -z "$ASSET_ROOT" ]]; then
  usage >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="$(cd "$DATASET_ROOT" && pwd)"
ASSET_ROOT="$(cd "$ASSET_ROOT" && pwd)"
cd "$ROOT_DIR"

mkdir -p configs/datasets tools scripts/ogsod_public \
  runs_public/ogsod/hbb/formal_nomosaic_20260528 logs

sed "s|/path/to/OGSOD-1.0|${DATASET_ROOT}|g" \
  shared/configs/datasets_public/ogsod1_sar_detect.yaml \
  > configs/datasets/ogsod_hbb_sar.yaml
sed "s|/path/to/OGSOD-1.0|${DATASET_ROOT}|g" \
  shared/configs/datasets_public/ogsod1_rgb_detect.yaml \
  > configs/datasets/ogsod_hbb_rgb.yaml

ln -sfn ../ladd/code/train_ladd_hbb.py tools/train_ladd_hbb.py
ln -sfn ../baseline/code/train_ogsod_baseline.py tools/train_ogsod_baseline.py
ln -sfn ../cclkd_reproduction/code/train_cclkd_online_hbb.py tools/train_cclkd_online_hbb.py
ln -sfn ../../ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh \
  scripts/ogsod_public/run_ladd_phase.sh
ln -sfn ../../ladd/code_versions/current_hbb/scripts/ogsod_public/run_hbb_ladd_converged_chain.sh \
  scripts/ogsod_public/run_hbb_ladd_converged_chain.sh

BASELINE_DST="runs_public/ogsod/hbb/formal_nomosaic_20260528/baselines"
BASELINE_SRC="${ASSET_ROOT}/runs_public/ogsod/hbb/formal_nomosaic_20260528/baselines"
if [[ ! -e "$BASELINE_SRC" ]]; then
  echo "Missing baseline asset directory: $BASELINE_SRC" >&2
  exit 1
fi
ln -sfn "$BASELINE_SRC" "$BASELINE_DST"

for checkpoint in "$ASSET_ROOT"/yolo11*.pt "$ASSET_ROOT"/yolo26n.pt; do
  [[ -e "$checkpoint" ]] || continue
  ln -sfn "$checkpoint" "$(basename "$checkpoint")"
done

echo "Runtime prepared at $ROOT_DIR"
echo "dataset_root=$DATASET_ROOT"
echo "asset_root=$ASSET_ROOT"
