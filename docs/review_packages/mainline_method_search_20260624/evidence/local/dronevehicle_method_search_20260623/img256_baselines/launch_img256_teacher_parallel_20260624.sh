#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/LADD_public

GPU_ID="${GPU_ID:-0}"
EXPECTED_EXTRA_MB="${EXPECTED_EXTRA_MB:-5000}"
MAX_AFTER_MB="${MAX_AFTER_MB:-22000}"
PY="${PY:-/root/shared-nvme/venvs/ladd312/bin/python}"

RGB_WRAPPER_PID="${RGB_WRAPPER_PID:-105784}"
RGB_CHILD_PID="${RGB_CHILD_PID:-105791}"

TEACHER_DATA="comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0/configs/dronevehicle_ir_hbb.yaml"
PROJECT_ROOT="runs_public/dronevehicle_method_search/sub2k_seed0_fullval/img256_baselines"
LOG_ROOT="logs/dronevehicle_method_search/sub2k_seed0_fullval/img256_baselines/dronevehicle_sub2k_img256_baseline_pair_yolo11n_e200_b64_s0_20260624_090613_gpu0"
TEACHER_NAME="dronevehicle_sub2k_teacher_ir_yolo11n_cclkdproto_e200_b64_img256_mosaic0p0_close0_mixup0p1_s0_20260624_090613"
TEACHER_RESULTS="${PROJECT_ROOT}/teacher_ir/${TEACHER_NAME}/results.csv"
RUN_LOG="${LOG_ROOT}/teacher_ir_parallel.log"
PID_PATH="${LOG_ROOT}/teacher_ir_parallel.pid"
CLEANUP_LOG="${LOG_ROOT}/stop_serial_wrapper_after_rgb.log"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 2
  fi
}

require_file "$PY" "python interpreter"
require_file "yolo11n.pt" "YOLO init checkpoint"
require_file "$TEACHER_DATA" "IR data yaml"

if [[ -f "$TEACHER_RESULTS" ]]; then
  echo "Teacher baseline already has results: ${TEACHER_RESULTS}" >&2
  exit 0
fi

used_mb=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if (( used_mb + EXPECTED_EXTRA_MB > MAX_AFTER_MB )); then
  echo "GPU${GPU_ID} memory guard: used=${used_mb} MB expected_extra=${EXPECTED_EXTRA_MB} MB max_after=${MAX_AFTER_MB} MB" >&2
  exit 3
fi

mkdir -p "${PROJECT_ROOT}/teacher_ir" "$LOG_ROOT"

if kill -0 "$RGB_WRAPPER_PID" 2>/dev/null; then
  kill -STOP "$RGB_WRAPPER_PID" 2>/dev/null || true
  {
    echo "[$(date '+%F %T')] stopped serial wrapper pid=${RGB_WRAPPER_PID} so it will not launch duplicate teacher"
    echo "[$(date '+%F %T')] rgb child pid=${RGB_CHILD_PID} continues independently"
  } | tee -a "${LOG_ROOT}/master.log"
fi

nohup bash -lc "
set -euo pipefail
while kill -0 '${RGB_CHILD_PID}' 2>/dev/null; do
  stat=\$(ps -o stat= -p '${RGB_CHILD_PID}' 2>/dev/null | tr -d ' ')
  case \"\$stat\" in
    Z*) break ;;
  esac
  sleep 20
done
if kill -0 '${RGB_WRAPPER_PID}' 2>/dev/null; then
  kill -KILL '${RGB_WRAPPER_PID}' 2>/dev/null || true
  echo \"[\$(date '+%F %T')] killed stopped serial wrapper pid=${RGB_WRAPPER_PID}\"
fi
" > "$CLEANUP_LOG" 2>&1 < /dev/null &

{
  echo "[$(date '+%F %T')] img256 IR teacher parallel baseline start"
  echo "gpu=${GPU_ID}"
  echo "run=${TEACHER_NAME}"
} | tee -a "${LOG_ROOT}/master.log"

nohup "$PY" baseline/code/train_ogsod_baseline.py \
  --task hbb \
  --model yolo11n.pt \
  --data "$TEACHER_DATA" \
  --imgsz 256 \
  --epochs 200 \
  --batch 64 \
  --strict-batch-size \
  --workers 8 \
  --device "$GPU_ID" \
  --patience 200 \
  --project "${PROJECT_ROOT}/teacher_ir" \
  --name "$TEACHER_NAME" \
  --optimizer SGD \
  --lr0 0.01 \
  --lrf 0.01 \
  --momentum 0.937 \
  --weight-decay 0.0005 \
  --cos-lr \
  --mosaic 0.0 \
  --close-mosaic 0 \
  --mixup 0.1 \
  --cutmix 0.0 \
  --degrees 0.0 \
  --perspective 0.0 \
  --translate 0.1 \
  --scale 0.5 \
  --fliplr 0.5 \
  --flipud 0.0 \
  --hsv-h 0.0 \
  --hsv-s 0.0 \
  --hsv-v 0.0 \
  --erasing 0.0 \
  --save-period 50 \
  --seed 0 \
  --deterministic \
  > "$RUN_LOG" 2>&1 < /dev/null &

pid=$!
echo "$pid" > "$PID_PATH"
echo "[$(date '+%F %T')] launched img256 IR teacher parallel pid=${pid}" | tee -a "${LOG_ROOT}/master.log"
