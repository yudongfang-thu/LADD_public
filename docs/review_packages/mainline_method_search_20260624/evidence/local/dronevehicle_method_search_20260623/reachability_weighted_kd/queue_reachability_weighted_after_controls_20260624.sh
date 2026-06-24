#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/LADD_public

THRESHOLD_MB="${THRESHOLD_MB:-15000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MIN_ROWS="${MIN_ROWS:-20}"
REQUIRE_CMDISTILL="${REQUIRE_CMDISTILL:-1}"

STUDENT="runs_public/cross_dataset/cclkd_yolo11n/dronevehicle_sub2k_seed0/baselines/student_rgb/dronevehicle_sub2k_student_rgb_yolo11n_cclkdproto_e200_b64_img512_mosaic0p0_close0_mixup0p1_s0_20260623_221620/weights/best.pt"
TEACHER="runs_public/cross_dataset/cclkd_yolo11n/dronevehicle_sub2k_seed0/baselines/teacher_ir/dronevehicle_sub2k_teacher_ir_yolo11n_cclkdproto_e200_b64_img512_mosaic0p0_close0_mixup0p1_s0_gpu0_20260623_221936/weights/best.pt"
DECOMP="runs_public/dronevehicle_method_search/sub2k_seed0_fullval/oldsplit_90_hbb/ir_to_rgb/oldsplit90_hbb_cclkdproto_ir2rgb_from_rgbbase_P1_20260623_2313_a2/weights/last.pt"
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

summarize_csv_ready() {
  local label="$1"
  local csv_path="$2"
  python3 - "$label" "$csv_path" "$MIN_ROWS" <<'PY'
import csv
import sys
from pathlib import Path

label = sys.argv[1]
path = Path(sys.argv[2])
min_rows = int(sys.argv[3])
with path.open(newline="", errors="replace") as f:
    rows = [{(k.strip() if k else k): v for k, v in r.items()} for r in csv.DictReader(f)]

def fv(row, key):
    try:
        return float(row.get(key, "nan"))
    except Exception:
        return float("nan")

key = "metrics/mAP50-95(B)"
key50 = "metrics/mAP50(B)"
best = max(rows, key=lambda r: fv(r, key)) if rows else {}
last = rows[-1] if rows else {}
print(
    f"{label}_rows={len(rows)} {label}_best50={fv(best, key50):.5f} "
    f"{label}_best5095={fv(best, key):.5f} {label}_last5095={fv(last, key):.5f} path={path}"
)
if len(rows) < min_rows:
    sys.exit(1)
sys.exit(0)
PY
}

primary_ready() {
  DET_CSV=$(latest_csv 'runs_public/dronevehicle_method_search/sub2k_seed0_fullval/reload_controls/lr1e-3_nowarmup/*/results.csv')
  RAW_CSV=$(latest_csv 'runs_public/dronevehicle_method_search/sub2k_seed0_fullval/raw_feature_kd/ir_to_rgb_lowlr_nowarmup/*/results.csv')
  CMD_CSV=$(latest_csv 'runs_public/dronevehicle_method_search/sub2k_seed0_fullval/cmdistill_style/ir_to_rgb_lowlr_nowarmup/*/results.csv')
  if [[ -z "$DET_CSV" || -z "$RAW_CSV" ]]; then
    echo "[$(date '+%F %T')] waiting for low-lr controls det=${DET_CSV:-missing} raw=${RAW_CSV:-missing}" >&2
    return 1
  fi
  summarize_csv_ready det "$DET_CSV" || return 1
  summarize_csv_ready raw "$RAW_CSV" || return 1
  if [[ "$REQUIRE_CMDISTILL" == "1" ]]; then
    if [[ -z "$CMD_CSV" ]]; then
      echo "[$(date '+%F %T')] waiting for low-lr CMDistill sanity result file" >&2
      return 1
    fi
    summarize_csv_ready cmd "$CMD_CSV" || return 1
  fi
}

wait_for_gpu() {
  while true; do
    local selected
    selected=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v th="$THRESHOLD_MB" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if (($2+0) < th) {print $1; exit}}')
    if [[ -n "$selected" ]]; then
      echo "$selected"
      return 0
    fi
    echo "[$(date '+%F %T')] waiting for gpu memory < ${THRESHOLD_MB} MB" >&2
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits >&2 || true
    sleep "$POLL_SECONDS"
  done
}

launch_variant() {
  local gpu="$1"
  local suffix="$2"
  local kd_weight_mode="$3"
  local stamp run_tag project_dir log_root phase_log_dir outer_log pid_path run_name
  stamp=$(date +%Y%m%d_%H%M%S)
  run_tag="${suffix}_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_${stamp}"
  project_dir="runs_public/dronevehicle_method_search/sub2k_seed0_fullval/reachability_weighted_kd/${suffix}"
  log_root="logs/dronevehicle_method_search/sub2k_seed0_fullval/reachability_weighted_kd/${suffix}"
  phase_log_dir="${log_root}/${run_tag}_gpu${gpu}"
  outer_log="${log_root}/${run_tag}_gpu${gpu}.outer.log"
  pid_path="${log_root}/${run_tag}_gpu${gpu}.pid"
  run_name="${run_tag}_b"
  mkdir -p "$project_dir" "$phase_log_dir" "$log_root"
  echo "[$(date '+%F %T')] launching ${suffix} kd_weight_mode=${kd_weight_mode} gpu=${gpu} run=${run_tag}"
  nohup env \
    PYTHON_BIN=/root/shared-nvme/venvs/ladd312/bin/python \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MODEL="$STUDENT" SAR_BASELINE="$STUDENT" RGB_TEACHER="$TEACHER" \
    INIT_TYPE="reachability_weighted_kd_${suffix}_dronevehicle" \
    PAPER_RUN=0 PAPER_PROTOCOL_ID="dronevehicle_sub2k_lowlr_nowarmup_reachability_weighted_kd_${suffix}" \
    DISABLE_OGSOD_PROTOCOL_GUARD=1 \
    DATA_CFG="$RGB_YAML" TEACHER_DATA_CFG="$IR_YAML" \
    GPU_ID="$gpu" SEED=0 BATCH_SIZE=64 STRICT_BATCH_SIZE=1 WORKERS=8 IMGSZ=512 \
    EPOCHS=200 PATIENCE=200 PHASE_MIN_EPOCHS=200 \
    PROJECT_DIR="$project_dir" LOG_DIR="$phase_log_dir" RUN_NAME="$run_name" SERVER_TAG=ladd4090zw1 \
    B_DETECTOR_SOURCE="$STUDENT" B_DECOMP_SOURCE="$DECOMP" B_LOAD_STUDENT_SPLIT=1 B_LOAD_STUDENT_REACHABILITY=1 \
    COMPARISON_KD_PROFILE=none PROFILE_KD_WEIGHT=0.0 PROFILE_KD_REPLACE_BASE=0 \
    STUDENT_BRANCH_MODE=split STUDENT_DETECT_MODE=raw TEACHER_FEATURE_MODE=decomposed UNLEARNABLE_HIDDEN_RATIO=1.0 \
    USE_MASK=0 USE_FG_MASK_FOR_REACH=1 USE_FG_MASK_FOR_REC=0 TASK_LOSS_FG_ONLY=1 \
    ALPHA_KD=0.25 LAMBDA_REACH=0.0 LAMBDA_REC=0.0 LAMBDA_TASKL=0.0 ALPHA_S_REC=0.0 KD_CALIBRATION_MODE=affine \
    KD_WEIGHT_MODE="$kd_weight_mode" KD_WEIGHT_POWER=1.0 KD_AGGREGATION_MODE=token KD_TOPK_RATIO=0.5 \
    COS_LR=1 LR0=0.001 LRF=0.1 MOMENTUM=0.937 WEIGHT_DECAY=0.0005 OPTIMIZER=SGD \
    WARMUP_EPOCHS=0.0 WARMUP_BIAS_LR=0.0 WARMUP_MOMENTUM=0.937 \
    MOSAIC=0.0 CLOSE_MOSAIC=0 CLOSE_AT_EPOCH= MIXUP=0.1 CUTMIX=0.0 DEGREES=0.0 PERSPECTIVE=0.0 TRANSLATE=0.1 SCALE=0.5 FLIPLR=0.5 FLIPUD=0.0 HSV_H=0.0 HSV_S=0.0 HSV_V=0.0 ERASING=0.0 \
    SAVE_PERIOD=25 EXIST_OK=0 \
    bash ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh hbb b "$run_tag" \
    > "$outer_log" 2>&1 &
  echo "$!" > "$pid_path"
  echo "[$(date '+%F %T')] ${suffix} pid=$(cat "$pid_path") outer=${outer_log}"
}

require_file "$STUDENT" "RGB baseline checkpoint"
require_file "$TEACHER" "IR teacher checkpoint"
require_file "$DECOMP" "A2 decomposition checkpoint"
require_file "$RGB_YAML" "RGB data yaml"
require_file "$IR_YAML" "IR data yaml"

echo "[$(date '+%F %T')] reachability-weighted KD queue ready threshold=${THRESHOLD_MB} min_rows=${MIN_ROWS} require_cmdistill=${REQUIRE_CMDISTILL}"
while ! primary_ready; do
  sleep "$POLL_SECONDS"
done

gpu=$(wait_for_gpu)
launch_variant "$gpu" splitkd_unweighted none
sleep 180

gpu=$(wait_for_gpu)
launch_variant "$gpu" reachgap_weighted reachability_gap

echo "[$(date '+%F %T')] reachability-weighted KD queue finished launching"
