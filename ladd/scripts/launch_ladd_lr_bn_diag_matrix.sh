#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ladd/scripts/launch_ladd_lr_bn_diag_matrix.sh <n|s> <seed> <gpu_id> [smoke|full]

Print or launch a LADD LR/BN/schedule diagnostic matrix.

Defaults:
  DRY_RUN=1
  LAUNCH=0

Set LAUNCH=1 to call ladd/scripts/launch_formal_ladd_job.sh for each row.
Each row still uses the formal cap2 launcher and a unique RUN_TAG_SUFFIX.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SIZE="${1:-}"
SEED="${2:-}"
GPU_ID="${3:-}"
TIER="${4:-smoke}"

if [[ ! "$SIZE" =~ ^(n|s)$ || ! "$SEED" =~ ^[0-9]+$ || -z "$GPU_ID" ]]; then
  usage >&2
  exit 1
fi
if [[ "$TIER" != "smoke" && "$TIER" != "full" ]]; then
  usage >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LAUNCH="${LAUNCH:-0}"
MATRIX_DRY_RUN="${DRY_RUN:-1}"
if [[ "$LAUNCH" == "1" && -z "${DRY_RUN+x}" ]]; then
  MATRIX_DRY_RUN=0
fi

run_row() {
  local idx="$1"
  local tag="$2"
  local purpose="$3"
  shift 3
  local envs=(
    "DRY_RUN=${MATRIX_DRY_RUN}"
    "RUN_TAG_SUFFIX=_diag_${tag}"
    "LADD_DIAG_LOG_BN=1"
    "LADD_DIAG_LOG_GRAD=0"
    "$@"
  )
  local cmd=(env "${envs[@]}" bash ladd/scripts/launch_formal_ladd_job.sh cap2 "$SIZE" "$SEED" "$GPU_ID")

  echo
  echo "[$idx] tag=${tag}"
  echo "purpose=${purpose}"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [[ "$LAUNCH" == "1" ]]; then
    "${cmd[@]}"
  fi
}

idx=1
run_row "$idx" "current_stable" "current stable candidate reference" \
  B_OPTIMIZER=MuSGD B_LR0=0.001 B_LRF=0.01 \
  B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.001 \
  B_FREEZE_BN_STATS=1
idx=$((idx + 1))

run_row "$idx" "lr2e3_freeze" "test whether B lr0=1e-3 is too low" \
  B_OPTIMIZER=MuSGD B_LR0=0.002 B_LRF=0.01 \
  B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.002 \
  B_FREEZE_BN_STATS=1
idx=$((idx + 1))

run_row "$idx" "lr3e3_freeze" "test a higher but still sub-default B learning rate" \
  B_OPTIMIZER=MuSGD B_LR0=0.003 B_LRF=0.01 \
  B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.003 \
  B_FREEZE_BN_STATS=1
idx=$((idx + 1))

run_row "$idx" "tail_lr1e3_lrf0p1" "test whether the cosine tail lr is too low" \
  B_OPTIMIZER=MuSGD B_LR0=0.001 B_LRF=0.1 \
  B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.001 \
  B_FREEZE_BN_STATS=1
idx=$((idx + 1))

run_row "$idx" "nofreeze_lr1e3" "isolate peak-performance cost of BN-freeze" \
  B_OPTIMIZER=MuSGD B_LR0=0.001 B_LRF=0.01 \
  B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.001 \
  B_FREEZE_BN_STATS=0 B_FREEZE_BN_AFTER_EPOCH=-1
idx=$((idx + 1))

run_row "$idx" "delayed_bn200_lr1e3" "allow early BN adaptation then freeze stats" \
  B_OPTIMIZER=MuSGD B_LR0=0.001 B_LRF=0.01 \
  B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.001 \
  B_FREEZE_BN_STATS=0 B_FREEZE_BN_AFTER_EPOCH=200
idx=$((idx + 1))

run_row "$idx" "delayed_bn200_lr2e3" "test slightly higher LR with delayed BN-freeze" \
  B_OPTIMIZER=MuSGD B_LR0=0.002 B_LRF=0.01 \
  B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.002 \
  B_FREEZE_BN_STATS=0 B_FREEZE_BN_AFTER_EPOCH=200
idx=$((idx + 1))

run_row "$idx" "b400_lr1e3_freeze" "test whether B=800 is longer than needed" \
  EPOCHS_B=400 PATIENCE_B=400 \
  B_OPTIMIZER=MuSGD B_LR0=0.001 B_LRF=0.01 \
  B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.001 \
  B_FREEZE_BN_STATS=1
idx=$((idx + 1))

if [[ "$TIER" == "full" ]]; then
  run_row "$idx" "lr5e3_freeze" "probe upper stable LR bound under BN-freeze" \
    B_OPTIMIZER=MuSGD B_LR0=0.005 B_LRF=0.01 \
    B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.005 \
    B_FREEZE_BN_STATS=1
  idx=$((idx + 1))

  run_row "$idx" "tail_lr2e3_lrf0p1" "combine higher main LR with higher cosine tail" \
    B_OPTIMIZER=MuSGD B_LR0=0.002 B_LRF=0.1 \
    B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.002 \
    B_FREEZE_BN_STATS=1
  idx=$((idx + 1))

  run_row "$idx" "delayed_bn100_lr2e3" "test earlier delayed BN-freeze timing" \
    B_OPTIMIZER=MuSGD B_LR0=0.002 B_LRF=0.01 \
    B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.002 \
    B_FREEZE_BN_STATS=0 B_FREEZE_BN_AFTER_EPOCH=100
  idx=$((idx + 1))

  run_row "$idx" "delayed_bn400_lr2e3" "test later delayed BN-freeze timing" \
    B_OPTIMIZER=MuSGD B_LR0=0.002 B_LRF=0.01 \
    B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.002 \
    B_FREEZE_BN_STATS=0 B_FREEZE_BN_AFTER_EPOCH=400
fi
