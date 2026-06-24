#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/LADD_public

FREE_GPU_THRESHOLD_MB="${FREE_GPU_THRESHOLD_MB:-8000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
PRIMARY_MIN_ROWS="${PRIMARY_MIN_ROWS:-100}"

STUDENT="runs_public/cross_dataset/cclkd_yolo11n/dronevehicle_sub2k_seed0/baselines/student_rgb/dronevehicle_sub2k_student_rgb_yolo11n_cclkdproto_e200_b64_img512_mosaic0p0_close0_mixup0p1_s0_20260623_221620/weights/best.pt"
TEACHER="runs_public/cross_dataset/cclkd_yolo11n/dronevehicle_sub2k_seed0/baselines/teacher_ir/dronevehicle_sub2k_teacher_ir_yolo11n_cclkdproto_e200_b64_img512_mosaic0p0_close0_mixup0p1_s0_gpu0_20260623_221936/weights/best.pt"
DSN_CKPT="runs_public/cross_dataset/dsn_shared_private/dronevehicle_sub2k_seed0/dronevehicle_sub2k_rgb_ir_dsn_s1_e80_b32_ld256_h512_seed0_20260623_2304/best.pt"
RGB_YAML="comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0/configs/dronevehicle_rgb_hbb.yaml"
IR_YAML="comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0/configs/dronevehicle_ir_hbb.yaml"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 2
  fi
}

latest_csv() {
  local pattern="$1"
  ls -t $pattern 2>/dev/null | head -n 1 || true
}

primary_ready() {
  local det_csv raw_csv cmd_csv dsn_csv
  det_csv=$(latest_csv 'runs_public/dronevehicle_method_search/sub2k_seed0_fullval/reload_controls/lr1e-3_nowarmup/*/results.csv')
  raw_csv=$(latest_csv 'runs_public/dronevehicle_method_search/sub2k_seed0_fullval/raw_feature_kd/ir_to_rgb_lowlr_nowarmup/*/results.csv')
  cmd_csv=$(latest_csv 'runs_public/dronevehicle_method_search/sub2k_seed0_fullval/cmdistill_style/ir_to_rgb_lowlr_nowarmup/*/results.csv')
  dsn_csv=$(latest_csv 'runs_public/dronevehicle_method_search/sub2k_seed0_fullval/dsn_shared_private/s2_student_distill/*/results.csv')
  if [[ -z "$det_csv" || -z "$raw_csv" || -z "$cmd_csv" || -z "$dsn_csv" ]]; then
    echo "[$(date '+%F %T')] waiting for primary csv files det=${det_csv:-missing} raw=${raw_csv:-missing} cmd=${cmd_csv:-missing} dsn=${dsn_csv:-missing}" >&2
    return 1
  fi
  python3 - "$PRIMARY_MIN_ROWS" "$det_csv" "$raw_csv" "$cmd_csv" "$dsn_csv" <<'PY'
import csv
import sys
from pathlib import Path

min_rows = int(sys.argv[1])
paths = sys.argv[2:]
labels = ("det", "raw", "cmd", "dsn")

def rows_and_best(path):
    rows = list(csv.DictReader(Path(path).open(newline="", errors="replace")))
    key = "metrics/mAP50-95(B)"
    best = max((float(r.get(key, "nan")) for r in rows), default=float("nan"))
    last = float(rows[-1].get(key, "nan")) if rows else float("nan")
    return len(rows), best, last

ok = True
for label, path in zip(labels, paths):
    n, best, last = rows_and_best(path)
    print(f"{label}_rows={n} {label}_best={best:.5f} {label}_last={last:.5f} path={path}")
    ok = ok and n >= min_rows
sys.exit(0 if ok else 1)
PY
}

wait_for_free_gpu() {
  while true; do
    local selected
    selected=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v th="$FREE_GPU_THRESHOLD_MB" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if (($2+0) < th) {print $1; exit}}')
    if [[ -n "$selected" ]]; then
      echo "$selected"
      return 0
    fi
    echo "[$(date '+%F %T')] waiting for free gpu memory < ${FREE_GPU_THRESHOLD_MB} MB" >&2
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >&2 || true
    sleep "$POLL_SECONDS"
  done
}

launch_variant() {
  local gpu="$1"
  local suffix="$2"
  local dsn_weight="$3"
  local decay_mode="$4"
  local decay_start="$5"
  local decay_end="$6"
  local final_mult="$7"

  local stamp run_tag project_dir log_root phase_log_dir outer_log pid_path run_name
  stamp=$(date +%Y%m%d_%H%M%S)
  run_tag="dsn_s2_${suffix}_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_${stamp}"
  project_dir="runs_public/dronevehicle_method_search/sub2k_seed0_fullval/dsn_shared_private/s2_refine_variants/${suffix}"
  log_root="logs/dronevehicle_method_search/sub2k_seed0_fullval/dsn_shared_private/s2_refine_variants/${suffix}"
  phase_log_dir="${log_root}/${run_tag}_gpu${gpu}"
  outer_log="${log_root}/${run_tag}_gpu${gpu}.outer.log"
  pid_path="${log_root}/${run_tag}_gpu${gpu}.pid"
  run_name="${run_tag}_b"
  mkdir -p "$project_dir" "$phase_log_dir" "$log_root"
  echo "[$(date '+%F %T')] launching DSN refine ${suffix} gpu=${gpu} weight=${dsn_weight} decay=${decay_mode}:${decay_start}-${decay_end}:${final_mult}"
  nohup env \
    PYTHON_BIN=/root/shared-nvme/venvs/ladd312/bin/python \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MODEL="$STUDENT" SAR_BASELINE="$STUDENT" RGB_TEACHER="$TEACHER" \
    INIT_TYPE="dsn_s2_${suffix}_dronevehicle" \
    PAPER_RUN=0 PAPER_PROTOCOL_ID="dronevehicle_sub2k_lowlr_nowarmup_dsn_s2_${suffix}" \
    DISABLE_OGSOD_PROTOCOL_GUARD=1 \
    DATA_CFG="$RGB_YAML" TEACHER_DATA_CFG="$IR_YAML" \
    GPU_ID="$gpu" SEED=0 BATCH_SIZE=64 STRICT_BATCH_SIZE=1 WORKERS=8 IMGSZ=512 \
    EPOCHS=200 PATIENCE=200 PHASE_MIN_EPOCHS=200 \
    PROJECT_DIR="$project_dir" LOG_DIR="$phase_log_dir" RUN_NAME="$run_name" SERVER_TAG=ladd4090zw1 \
    COMPARISON_KD_PROFILE=none PROFILE_KD_WEIGHT=0.0 PROFILE_KD_REPLACE_BASE=1 \
    STUDENT_BRANCH_MODE=raw TEACHER_FEATURE_MODE=raw USE_MASK=0 USE_FG_MASK_FOR_REACH=0 USE_FG_MASK_FOR_REC=0 \
    ALPHA_KD=1.0 LAMBDA_REACH=0.0 LAMBDA_REC=0.0 LAMBDA_TASKL=0.0 ALPHA_S_REC=0.0 KD_CALIBRATION_MODE=none \
    DSN_PROJECTOR_WEIGHTS="$DSN_CKPT" DSN_KD_WEIGHT="$dsn_weight" DSN_STUDENT_PROJECTOR=rgb DSN_TEACHER_PROJECTOR=peer \
    LADD_KD_DECAY_MODE="$decay_mode" LADD_KD_DECAY_START_EPOCH="$decay_start" LADD_KD_DECAY_END_EPOCH="$decay_end" LADD_KD_FINAL_MULT="$final_mult" \
    COS_LR=1 LR0=0.001 LRF=0.1 MOMENTUM=0.937 WEIGHT_DECAY=0.0005 OPTIMIZER=SGD \
    WARMUP_EPOCHS=0.0 WARMUP_BIAS_LR=0.0 WARMUP_MOMENTUM=0.937 \
    MOSAIC=0.0 CLOSE_MOSAIC=0 CLOSE_AT_EPOCH= MIXUP=0.1 CUTMIX=0.0 DEGREES=0.0 PERSPECTIVE=0.0 TRANSLATE=0.1 SCALE=0.5 FLIPLR=0.5 FLIPUD=0.0 HSV_H=0.0 HSV_S=0.0 HSV_V=0.0 ERASING=0.0 \
    SAVE_PERIOD=25 EXIST_OK=0 \
    bash ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh hbb b "$run_tag" \
    > "$outer_log" 2>&1 &
  local pid="$!"
  echo "$pid" > "$pid_path"
  echo "[$(date '+%F %T')] DSN refine ${suffix} pid=${pid} outer=${outer_log}"
  wait "$pid"
  echo "[$(date '+%F %T')] DSN refine ${suffix} finished exit=$?"
}

require_file "$STUDENT" "RGB baseline checkpoint"
require_file "$TEACHER" "IR teacher checkpoint"
require_file "$DSN_CKPT" "DSN projector checkpoint"
require_file "$RGB_YAML" "RGB data yaml"
require_file "$IR_YAML" "IR data yaml"

echo "[$(date '+%F %T')] DSN refine queue ready primary_min_rows=${PRIMARY_MIN_ROWS} free_gpu_threshold=${FREE_GPU_THRESHOLD_MB}"
while ! primary_ready; do
  sleep "$POLL_SECONDS"
done

gpu=$(wait_for_free_gpu)
launch_variant "$gpu" "w0p25_nodecay" "0.25" "none" "-1" "-1" "1.0"

gpu=$(wait_for_free_gpu)
launch_variant "$gpu" "w1p0_decay60_160_final0" "1.0" "linear" "60" "160" "0.0"

echo "[$(date '+%F %T')] DSN refine queue completed"
