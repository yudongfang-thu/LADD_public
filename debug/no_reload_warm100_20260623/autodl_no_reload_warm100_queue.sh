#!/usr/bin/env bash
set -euo pipefail

# AutoDL no-reload warm100 protocol launcher.
# Pipeline:
#   1) Train SAR detector from yolo11n.pt for 100 epochs with mosaic enabled.
#   2) Train A1 decomposition cache from the existing SAR baseline for 10 epochs.
#   3) When both caches exist, launch B700 jobs from warm100 detector with
#      decomposition modules split-loaded from A1.

REPO="${REPO:-/root/autodl-tmp/LADD_public}"
cd "$REPO"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
SIZE="${SIZE:-n}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DATA_CFG="${DATA_CFG:-configs/paper/datasets/ogsod_hbb_sar.yaml}"
TEACHER_DATA_CFG="${TEACHER_DATA_CFG:-configs/paper/datasets/ogsod_hbb_rgb.yaml}"

source scripts/paper/paper_common.sh
SAR_BASELINE="${SAR_BASELINE:-$(paper_find_baseline sar "$SIZE" "$SEED")}"
RGB_TEACHER="${RGB_TEACHER:-$(paper_find_baseline rgb "$SIZE" "$SEED")}"

RUN_ROOT="runs_public/paper/ogsod_hbb_mosaic100/no_reload_warm100/yolo11${SIZE}/seed${SEED}"
LOG_ROOT="logs/paper/ogsod_hbb_mosaic100/no_reload_warm100/yolo11${SIZE}/seed${SEED}/${TS}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

WARM_PROJECT="${RUN_ROOT}/detector_warm100"
WARM_NAME="sar_yolo11${SIZE}_no_reload_warm100_mosaic_on_e100_b${BATCH_SIZE}_s${SEED}_${TS}"
WARM_DIR="${WARM_PROJECT}/${WARM_NAME}"
WARM_LOG="${LOG_ROOT}/warm100_detector_gpu0.log"

A1_PROJECT="${RUN_ROOT}/a1_decomp_cache"
A1_TAG="a1_decomp_from_sar_baseline_yolo11${SIZE}_s${SEED}_${TS}"
A1_LOG_DIR="${LOG_ROOT}/a1_decomp_gpu1"
A1_ACTUAL_FILE="${A1_LOG_DIR}/actual_run_dir.txt"
A1_LOG="${LOG_ROOT}/a1_decomp_screen.log"

ORCH_LOG="${LOG_ROOT}/orchestrator.log"

write_meta() {
  {
    printf 'ts=%q\n' "$TS"
    printf 'size=%q\n' "$SIZE"
    printf 'seed=%q\n' "$SEED"
    printf 'batch_size=%q\n' "$BATCH_SIZE"
    printf 'sar_baseline=%q\n' "$SAR_BASELINE"
    printf 'rgb_teacher=%q\n' "$RGB_TEACHER"
    printf 'data_cfg=%q\n' "$DATA_CFG"
    printf 'teacher_data_cfg=%q\n' "$TEACHER_DATA_CFG"
    printf 'warm_project=%q\n' "$WARM_PROJECT"
    printf 'warm_name=%q\n' "$WARM_NAME"
    printf 'warm_dir=%q\n' "$WARM_DIR"
    printf 'a1_project=%q\n' "$A1_PROJECT"
    printf 'a1_tag=%q\n' "$A1_TAG"
    printf 'a1_log_dir=%q\n' "$A1_LOG_DIR"
    printf 'run_root=%q\n' "$RUN_ROOT"
    printf 'log_root=%q\n' "$LOG_ROOT"
    printf 'protocol_note=%q\n' "warm100 detector from yolo11n.pt; A1 decomp from SAR baseline; B700 detector from warm100 last.pt and decomp split-loaded from A1"
  } > "${LOG_ROOT}/no_reload_warm100_meta.env"
}

launch_warm100() {
  screen -dmS "nl_warm100_${SIZE}_s${SEED}_g0_${TS}" bash -lc "
    set -euo pipefail
    cd '$REPO'
    mkdir -p '$WARM_PROJECT' '$(dirname "$WARM_LOG")'
    PYTHONUNBUFFERED=1 python3 baseline/code/train_ogsod_baseline.py \
      --task hbb \
      --model yolo11${SIZE}.pt \
      --data '$DATA_CFG' \
      --imgsz 256 \
      --epochs 100 \
      --batch '$BATCH_SIZE' \
      --workers 8 \
      --device 0 \
      --patience 100 \
      --project '$WARM_PROJECT' \
      --name '$WARM_NAME' \
      --lr0 0.01 \
      --lrf 0.01 \
      --cos-lr \
      --mosaic 1.0 \
      --close-mosaic 0 \
      --translate 0.1 \
      --scale 0.5 \
      --fliplr 0.5 \
      --flipud 0.0 \
      --degrees 0.0 \
      --perspective 0.0 \
      --hsv-h 0.0 \
      --hsv-s 0.0 \
      --hsv-v 0.0 \
      --mixup 0.0 \
      --cutmix 0.0 \
      --erasing 0.0 \
      --save-period 100 \
      --seed '$SEED' \
      --deterministic \
      > '$WARM_LOG' 2>&1
  "
}

launch_a1_cache() {
  screen -dmS "nl_a1cache_${SIZE}_s${SEED}_g1_${TS}" bash -lc "
    set -euo pipefail
    cd '$REPO'
    env \
      SAR_BASELINE='$SAR_BASELINE' \
      RGB_TEACHER='$RGB_TEACHER' \
      DATA_CFG='$DATA_CFG' \
      TEACHER_DATA_CFG='$TEACHER_DATA_CFG' \
      PROJECT_DIR='$A1_PROJECT' \
      LOG_DIR='$A1_LOG_DIR' \
      MODEL='$SAR_BASELINE' \
      GPU_ID=1 \
      SEED='$SEED' \
      BATCH_SIZE='$BATCH_SIZE' \
      EPOCHS=10 \
      PATIENCE=10 \
      PHASE_MIN_EPOCHS=10 \
      MOSAIC=1.0 \
      CLOSE_MOSAIC=0 \
      MIXUP=0.0 \
      CUTMIX=0.0 \
      DEGREES=0.0 \
      PERSPECTIVE=0.0 \
      TRANSLATE=0.1 \
      SCALE=0.5 \
      FLIPLR=0.5 \
      FLIPUD=0.0 \
      HSV_H=0.0 \
      HSV_S=0.0 \
      HSV_V=0.0 \
      ERASING=0.0 \
      LR0=0.01 \
      LRF=0.01 \
      COS_LR=1 \
      OPTIMIZER=auto \
      WARMUP_EPOCHS=3.0 \
      WARMUP_BIAS_LR=0.1 \
      SAVE_PERIOD=10 \
      USE_MASK=1 \
      USE_FG_MASK_FOR_REACH=1 \
      USE_FG_MASK_FOR_REC=0 \
      RANK_D_NEG_CAP=2.0 \
      LAMBDA_REC=0.1 \
      LAMBDA_TASKL=1.0 \
      ALPHA_KD=1.0 \
      ALPHA_S_REC=0.1 \
      LAMBDA_REACH=1.0 \
      LAMBDA_MATCH_INNER=1.0 \
      LAMBDA_RANK_INNER=1.0 \
      REACH_INPUT_MODE=adapter \
      STUDENT_BRANCH_MODE=split \
      TEACHER_FEATURE_MODE=decomposed \
      EXIST_OK=0 \
      bash ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh hbb a1 '$A1_TAG' \
      > '$A1_LOG' 2>&1
  "
}

launch_b_job() {
  local profile="$1"
  local gpu="$2"
  local extra_env="$3"
  local warm_last="${WARM_DIR}/weights/last.pt"
  local a1_dir
  a1_dir="$(cat "$A1_ACTUAL_FILE")"
  local a1_best="${a1_dir}/weights/best.pt"
  local project="${RUN_ROOT}/b700_${profile}"
  local log_dir="${LOG_ROOT}/b700_${profile}_gpu${gpu}"
  local tag="b700_${profile}_warm100_a1cache_yolo11${SIZE}_s${SEED}_${TS}_after_e100"
  local screen_name="nl_b700_${profile}_${SIZE}_s${SEED}_g${gpu}_${TS}"

  screen -dmS "$screen_name" bash -lc "
    set -euo pipefail
    cd '$REPO'
    env \
      MODEL='$warm_last' \
      SAR_BASELINE='$SAR_BASELINE' \
      RGB_TEACHER='$RGB_TEACHER' \
      DATA_CFG='$DATA_CFG' \
      TEACHER_DATA_CFG='$TEACHER_DATA_CFG' \
      PROJECT_DIR='$project' \
      LOG_DIR='$log_dir' \
      B_DETECTOR_SOURCE='$warm_last' \
      B_DECOMP_SOURCE='$a1_best' \
      B_LOAD_STUDENT_SPLIT=0 \
      B_LOAD_STUDENT_REACHABILITY=1 \
      B_SPLIT_LOAD_STRICT=0 \
      GPU_ID='$gpu' \
      SEED='$SEED' \
      BATCH_SIZE='$BATCH_SIZE' \
      EPOCHS=700 \
      PATIENCE=700 \
      PHASE_MIN_EPOCHS=700 \
      MOSAIC=0.0 \
      CLOSE_MOSAIC=0 \
      MIXUP=0.0 \
      CUTMIX=0.0 \
      DEGREES=0.0 \
      PERSPECTIVE=0.0 \
      TRANSLATE=0.1 \
      SCALE=0.5 \
      FLIPLR=0.5 \
      FLIPUD=0.0 \
      HSV_H=0.0 \
      HSV_S=0.0 \
      HSV_V=0.0 \
      ERASING=0.0 \
      LR0=0.01 \
      LRF=0.01 \
      COS_LR=1 \
      OPTIMIZER=auto \
      WARMUP_EPOCHS=3.0 \
      WARMUP_BIAS_LR=0.1 \
      SAVE_PERIOD=100 \
      USE_MASK=1 \
      USE_FG_MASK_FOR_REACH=1 \
      USE_FG_MASK_FOR_REC=0 \
      RANK_D_NEG_CAP=2.0 \
      LAMBDA_REC=0.1 \
      LAMBDA_TASKL=1.0 \
      ALPHA_KD=1.0 \
      ALPHA_S_REC=0.1 \
      LAMBDA_REACH=1.0 \
      LAMBDA_MATCH_INNER=1.0 \
      LAMBDA_RANK_INNER=1.0 \
      REACH_INPUT_MODE=adapter \
      STUDENT_BRANCH_MODE=split \
      TEACHER_FEATURE_MODE=decomposed \
      LADD_B_LOSS_WARMUP_MODE=linear \
      LADD_B_LOSS_WARMUP_START_EPOCH=0 \
      LADD_B_LOSS_WARMUP_END_EPOCH=30 \
      LADD_B_LOSS_WARMUP_FINAL_MULT=1.0 \
      LADD_B_LOSS_WARMUP_SCOPE=core \
      LADD_KD_DECAY_MODE=none \
      EXIST_OK=0 \
      $extra_env \
      bash ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh hbb b '$tag'
  "
}

launch_orchestrator() {
  screen -dmS "nl_orch_${SIZE}_s${SEED}_${TS}" bash -lc "
    set -euo pipefail
    cd '$REPO'
    echo '[orchestrator] waiting for warm100 and A1 cache' | tee -a '$ORCH_LOG'
    while true; do
      warm_last='$WARM_DIR/weights/last.pt'
      warm_results='$WARM_DIR/results.csv'
      warm_epoch=0
      if [[ -s \"\$warm_results\" ]]; then
        warm_epoch=\$(tail -n 1 \"\$warm_results\" | cut -d, -f1)
      fi
      if [[ -f \"\$warm_last\" && \"\$warm_epoch\" =~ ^[0-9]+$ && \"\$warm_epoch\" -ge 100 && -s '$A1_ACTUAL_FILE' ]]; then
        a1_dir=\$(cat '$A1_ACTUAL_FILE')
        if [[ -f \"\$a1_dir/weights/best.pt\" ]]; then
          break
        fi
      fi
      sleep 120
    done
    echo '[orchestrator] caches ready, launching B700 jobs' | tee -a '$ORCH_LOG'
    '$REPO/debug/no_reload_warm100_20260623/autodl_no_reload_warm100_queue.sh' --launch-b-only '$TS'
  "
}

launch_b_only() {
  local warm_results="${WARM_DIR}/results.csv"
  local warm_epoch=0
  if [[ -s "$warm_results" ]]; then
    warm_epoch="$(tail -n 1 "$warm_results" | cut -d, -f1)"
  fi
  if [[ ! "$warm_epoch" =~ ^[0-9]+$ || "$warm_epoch" -lt 100 ]]; then
    echo "[$(date '+%F %T')] Refusing to launch B700: warm100 is not complete yet (epoch=${warm_epoch})." | tee -a "$ORCH_LOG"
    exit 2
  fi
  echo "[$(date '+%F %T')] Launching B700 jobs for TS=${TS}" | tee -a "$ORCH_LOG"
  launch_b_job detonly 0 "LADD_B_DET_ONLY=1 LADD_B_A2_CORE=0"
  launch_b_job probeA 0 "LADD_B_A2_CORE=1 LADD_B_FROZEN_REACH_PROBE=1 LADD_B_DETACH_REACH_PROBE=1 LADD_B_KEEP_REACH_PROBE_GRAD=0"
  launch_b_job dynamic 1 "LADD_B_A2_CORE=1 LADD_B_FROZEN_REACH_PROBE=0 LADD_B_DETACH_REACH_PROBE=0 LADD_B_KEEP_REACH_PROBE_GRAD=0"
}

if [[ "${1:-}" == "--launch-b-only" ]]; then
  TS="${2:?missing TS}"
  # Recompute paths with the provided TS.
  RUN_ROOT="runs_public/paper/ogsod_hbb_mosaic100/no_reload_warm100/yolo11${SIZE}/seed${SEED}"
  LOG_ROOT="logs/paper/ogsod_hbb_mosaic100/no_reload_warm100/yolo11${SIZE}/seed${SEED}/${TS}"
  WARM_PROJECT="${RUN_ROOT}/detector_warm100"
  WARM_NAME="sar_yolo11${SIZE}_no_reload_warm100_mosaic_on_e100_b${BATCH_SIZE}_s${SEED}_${TS}"
  WARM_DIR="${WARM_PROJECT}/${WARM_NAME}"
  A1_LOG_DIR="${LOG_ROOT}/a1_decomp_gpu1"
  A1_ACTUAL_FILE="${A1_LOG_DIR}/actual_run_dir.txt"
  ORCH_LOG="${LOG_ROOT}/orchestrator.log"
  launch_b_only
  exit 0
fi

if [[ "${1:-}" == "--orchestrate-existing" ]]; then
  TS="${2:?missing TS}"
  # Recompute paths with the provided TS and only start the waiter.
  RUN_ROOT="runs_public/paper/ogsod_hbb_mosaic100/no_reload_warm100/yolo11${SIZE}/seed${SEED}"
  LOG_ROOT="logs/paper/ogsod_hbb_mosaic100/no_reload_warm100/yolo11${SIZE}/seed${SEED}/${TS}"
  WARM_PROJECT="${RUN_ROOT}/detector_warm100"
  WARM_NAME="sar_yolo11${SIZE}_no_reload_warm100_mosaic_on_e100_b${BATCH_SIZE}_s${SEED}_${TS}"
  WARM_DIR="${WARM_PROJECT}/${WARM_NAME}"
  A1_LOG_DIR="${LOG_ROOT}/a1_decomp_gpu1"
  A1_ACTUAL_FILE="${A1_LOG_DIR}/actual_run_dir.txt"
  ORCH_LOG="${LOG_ROOT}/orchestrator.log"
  launch_orchestrator
  exit 0
fi

write_meta
launch_warm100
launch_a1_cache
launch_orchestrator

echo "launched_ts=${TS}"
echo "warm_screen=nl_warm100_${SIZE}_s${SEED}_g0_${TS}"
echo "a1_screen=nl_a1cache_${SIZE}_s${SEED}_g1_${TS}"
echo "orchestrator_screen=nl_orch_${SIZE}_s${SEED}_${TS}"
echo "log_root=${LOG_ROOT}"
