#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
RUN_SET="${RUN_SET:-p1}"
SERVER_TAG="${SERVER_TAG:-unknown_server}"
GPU_ID="${GPU_ID:-0}"
VARIANT="${VARIANT:-cap2}"
FORMAL_LAUNCHER="${FORMAL_LAUNCHER:-ladd/scripts/launch_formal_ladd_job.sh}"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

print_header() {
  local name="$1"
  shift

  echo "================================================================"
  echo "[LADD-DIAG] name=${name}"
  echo "[LADD-DIAG] run_set=${RUN_SET}"
  echo "[LADD-DIAG] git_commit=${GIT_COMMIT}"
  echo "[LADD-DIAG] server=${SERVER_TAG}"
  echo "[LADD-DIAG] dry_run=${DRY_RUN}"
  echo "[LADD-DIAG] command:"
  printf ' %q' "$@"
  echo
  echo "================================================================"
}

run_formal() {
  local name="$1"
  local model_size="$2"
  local seed="$3"
  shift 3

  local cmd=(
    env
    "SERVER_TAG=${SERVER_TAG}"
    "GIT_COMMIT=${GIT_COMMIT}"
    "$@"
    bash "$FORMAL_LAUNCHER" "$VARIANT" "$model_size" "$seed" "$GPU_ID"
  )
  print_header "$name" "${cmd[@]}"
  if [[ "$DRY_RUN" == "0" ]]; then
    "${cmd[@]}"
  fi
}

case "$RUN_SET" in
  p1)
    run_formal "diag_h1_n_seed0_b100_smoke" n 0 \
      EPOCHS=100 EXP_SUFFIX=diag_h1_n_s0_b100_smoke \
      FREEZE_BN_STATS=1 FREEZE_BN_AFTER_EPOCH=0 \
      LADD_ASSERT_PHASE_FREEZE=1 LADD_DIAG_LOG_GRAD=1 LADD_GRAD_CLIP_NORM=0.0

    run_formal "diag_h1_s_seed0_b400" s 0 \
      EPOCHS=400 EXP_SUFFIX=diag_h1_s_s0_b400 \
      FREEZE_BN_STATS=1 FREEZE_BN_AFTER_EPOCH=0 \
      LADD_ASSERT_PHASE_FREEZE=1 LADD_DIAG_LOG_GRAD=1 LADD_GRAD_CLIP_NORM=0.0

    run_formal "diag_h1_m_seed0_a2only" m 0 \
      EXP_SUFFIX=diag_h1_m_s0_a2only LADD_CHAIN_PHASES=a1,a2 \
      FREEZE_BN_STATS=0 \
      LADD_ASSERT_PHASE_FREEZE=1 LADD_DIAG_LOG_GRAD=1 LADD_GRAD_CLIP_NORM=0.0
    ;;
  p2_s)
    run_formal "diag_h1_s_seed0_alpha_kd_0p5_b400" s 0 \
      EPOCHS=400 EXP_SUFFIX=diag_h1_s_s0_alphaKD0p5_b400 \
      ALPHA_KD=0.5 \
      FREEZE_BN_STATS=1 FREEZE_BN_AFTER_EPOCH=0 \
      LADD_ASSERT_PHASE_FREEZE=1 LADD_DIAG_LOG_GRAD=1 LADD_GRAD_CLIP_NORM=0.0

    run_formal "diag_h1_s_seed0_alpha_kd_0p25_b400" s 0 \
      EPOCHS=400 EXP_SUFFIX=diag_h1_s_s0_alphaKD0p25_b400 \
      ALPHA_KD=0.25 \
      FREEZE_BN_STATS=1 FREEZE_BN_AFTER_EPOCH=0 \
      LADD_ASSERT_PHASE_FREEZE=1 LADD_DIAG_LOG_GRAD=1 LADD_GRAD_CLIP_NORM=0.0

    run_formal "diag_h1_s_seed0_detonly_b400" s 0 \
      EPOCHS=400 EXP_SUFFIX=diag_h1_s_s0_detonly_b400 \
      ALPHA_KD=0.0 \
      FREEZE_BN_STATS=1 FREEZE_BN_AFTER_EPOCH=0 \
      LADD_ASSERT_PHASE_FREEZE=1 LADD_DIAG_LOG_GRAD=1 LADD_GRAD_CLIP_NORM=0.0
    ;;
  p2_m)
    run_formal "diag_h1_m_seed0_a2_lr3e4" m 0 \
      EXP_SUFFIX=diag_h1_m_s0_a2_lr3e4 LADD_CHAIN_PHASES=a1,a2 \
      A2_LR0=0.0003 \
      LADD_ASSERT_PHASE_FREEZE=1 LADD_DIAG_LOG_GRAD=1 LADD_GRAD_CLIP_NORM=0.0

    run_formal "diag_h1_m_seed0_a2_short25" m 0 \
      EXP_SUFFIX=diag_h1_m_s0_a2_short25 LADD_CHAIN_PHASES=a1,a2 EPOCHS_A2=25 \
      LADD_ASSERT_PHASE_FREEZE=1 LADD_DIAG_LOG_GRAD=1 LADD_GRAD_CLIP_NORM=0.0

    run_formal "diag_h1_m_seed0_a2_lambda05" m 0 \
      EXP_SUFFIX=diag_h1_m_s0_a2_lambda05 LADD_CHAIN_PHASES=a1,a2 \
      LAMBDA_REACH=0.5 LAMBDA_MATCH_INNER=0.5 LAMBDA_RANK_INNER=0.5 \
      LADD_ASSERT_PHASE_FREEZE=1 LADD_DIAG_LOG_GRAD=1 LADD_GRAD_CLIP_NORM=0.0
    ;;
  *)
    echo "Unknown RUN_SET=${RUN_SET}. Valid: p1, p2_s, p2_m" >&2
    exit 2
    ;;
esac
