#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  DRY_RUN=1 LAUNCH=0 bash cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_sanity_matrix.sh gate

Compact CCLKD reproduction gate:
  G1: cclkd_gate_sar_x_b32_pretrained
  G2: cclkd_gate_sar_x_b64_pretrained
  G3: cclkd_gate_rgb_x_b32_pretrained

Environment:
  DRY_RUN           Print commands only (default: 1)
  LAUNCH            Set to 1 to request launching (default: 0)
  PARALLEL_LAUNCH   Required for matrix launch (default: 0)
  GPU_LIST          Comma-separated GPU ids for parallel launch, e.g. 0,1,2
  MAX_PARALLEL      Maximum jobs to start from G1/G2/G3 (default: 2)

Example:
  LAUNCH=1 DRY_RUN=0 PARALLEL_LAUNCH=1 GPU_LIST=0,1 MAX_PARALLEL=2 \
    bash cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_sanity_matrix.sh gate
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TIER="${1:-gate}"
DRY_RUN="${DRY_RUN:-1}"
LAUNCH="${LAUNCH:-0}"
PARALLEL_LAUNCH="${PARALLEL_LAUNCH:-0}"
GPU_LIST="${GPU_LIST:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

if [[ "$TIER" != "gate" && "$TIER" != "smoke" ]]; then
  echo "Only the compact gate tier is defined now." >&2
  exit 2
fi

if ! [[ "$MAX_PARALLEL" =~ ^[0-9]+$ ]]; then
  echo "MAX_PARALLEL must be a non-negative integer, got: $MAX_PARALLEL" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
LAUNCHER_REL="cclkd_reproduction/yolov5_sanity/scripts/launch_yolov5_ogsod_baseline.sh"
LAUNCHER="$REPO_ROOT/$LAUNCHER_REL"

EXPERIMENTS=(
  "G1 sar x pretrained 32 0 cclkd_gate_sar_x_b32_pretrained # Primary CCLKD Table-2-style YOLOv5-X SAR baseline gate"
  "G2 sar x pretrained 64 0 cclkd_gate_sar_x_b64_pretrained # Batch-size sanity only, because batch64 may explain the shared YOLOv5 baseline"
  "G3 rgb x pretrained 32 0 cclkd_gate_rgb_x_b32_pretrained # Optional teacher-strength gate for CCLKD, not a separate CoLD/CMDistill reproduction"
)

IFS=',' read -r -a GPUS <<<"$GPU_LIST"
if [[ "${#GPUS[@]}" -eq 0 || "${GPUS[0]}" == "" ]]; then
  echo "GPU_LIST must contain at least one GPU id, got: $GPU_LIST" >&2
  exit 2
fi

print_one() {
  local dry_run="$1"
  local launch="$2"
  local launcher_path="$3"
  local modality="$4"
  local model="$5"
  local init="$6"
  local batch="$7"
  local seed="$8"
  local gpu="$9"
  local tag="${10}"
  printf 'DRY_RUN=%q LAUNCH=%q bash %q %q %q %q %q %q %q %q\n' \
    "$dry_run" "$launch" "$launcher_path" "$modality" "$model" "$init" "$batch" "$seed" "$gpu" "$tag"
}

print_all() {
  local idx=0
  for item in "${EXPERIMENTS[@]}"; do
    read -r gid modality model init batch seed tag _ <<<"$item"
    local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    printf '# %s %s\n' "$gid" "${item#*# }"
    print_one 1 0 "$LAUNCHER_REL" "$modality" "$model" "$init" "$batch" "$seed" "$gpu" "$tag"
    echo
    idx=$((idx + 1))
  done
}

if [[ "$DRY_RUN" == "1" || "$LAUNCH" != "1" ]]; then
  print_all
  exit 0
fi

if [[ "$PARALLEL_LAUNCH" != "1" ]]; then
  echo "Use PARALLEL_LAUNCH=1 GPU_LIST=0,1 MAX_PARALLEL=2 to launch the compact gate in parallel." >&2
  exit 2
fi

launch_count=0
idx=0
for item in "${EXPERIMENTS[@]}"; do
  read -r gid modality model init batch seed tag _ <<<"$item"
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  if (( launch_count < MAX_PARALLEL )); then
    echo "Launching $gid on GPU $gpu: $tag"
    DRY_RUN=0 LAUNCH=1 bash "$LAUNCHER" "$modality" "$model" "$init" "$batch" "$seed" "$gpu" "$tag"
    launch_count=$((launch_count + 1))
  else
    if (( launch_count == MAX_PARALLEL )); then
      echo
      echo "MAX_PARALLEL=$MAX_PARALLEL reached. Remaining gate commands:"
      launch_count=$((launch_count + 1))
    fi
    print_one 0 1 "$LAUNCHER_REL" "$modality" "$model" "$init" "$batch" "$seed" "$gpu" "$tag"
  fi
  idx=$((idx + 1))
done

if (( MAX_PARALLEL < ${#EXPERIMENTS[@]} )); then
  echo "Run the remaining printed command(s) later when GPUs are available."
fi
