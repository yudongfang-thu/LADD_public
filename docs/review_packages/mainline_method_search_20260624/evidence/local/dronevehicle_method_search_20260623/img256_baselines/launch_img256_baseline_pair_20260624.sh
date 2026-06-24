#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/LADD_public

GPU_ID="${GPU_ID:-0}"
EXPECTED_EXTRA_MB="${EXPECTED_EXTRA_MB:-6000}"
MAX_AFTER_MB="${MAX_AFTER_MB:-22000}"
PY="${PY:-/root/shared-nvme/venvs/ladd312/bin/python}"

STUDENT_DATA="comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0/configs/dronevehicle_rgb_hbb.yaml"
TEACHER_DATA="comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0/configs/dronevehicle_ir_hbb.yaml"
PROJECT_ROOT="runs_public/dronevehicle_method_search/sub2k_seed0_fullval/img256_baselines"
LOG_BASE="logs/dronevehicle_method_search/sub2k_seed0_fullval/img256_baselines"

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
require_file "$STUDENT_DATA" "RGB data yaml"
require_file "$TEACHER_DATA" "IR data yaml"

used_mb=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if (( used_mb + EXPECTED_EXTRA_MB > MAX_AFTER_MB )); then
  echo "GPU${GPU_ID} memory guard: used=${used_mb} MB expected_extra=${EXPECTED_EXTRA_MB} MB max_after=${MAX_AFTER_MB} MB" >&2
  exit 3
fi

stamp=$(date +%Y%m%d_%H%M%S)
tag="dronevehicle_sub2k_img256_baseline_pair_yolo11n_e200_b64_s0_${stamp}"
log_root="${LOG_BASE}/${tag}_gpu${GPU_ID}"
cmd_path="${log_root}/${tag}.cmd.sh"
outer_log="${log_root}/${tag}.outer.log"
master_log="${log_root}/master.log"
pid_path="${log_root}/${tag}.pid"

student_name="dronevehicle_sub2k_student_rgb_yolo11n_cclkdproto_e200_b64_img256_mosaic0p0_close0_mixup0p1_s0_${stamp}"
teacher_name="dronevehicle_sub2k_teacher_ir_yolo11n_cclkdproto_e200_b64_img256_mosaic0p0_close0_mixup0p1_s0_${stamp}"

mkdir -p "$PROJECT_ROOT/student_rgb" "$PROJECT_ROOT/teacher_ir" "$log_root"

cat > "$cmd_path" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd /root/shared-nvme/LADD_public
export PYTHONUNBUFFERED=1

echo "[\$(date '+%F %T')] img256 RGB baseline start" | tee -a "$master_log"
"$PY" baseline/code/train_ogsod_baseline.py \\
  --task hbb \\
  --model yolo11n.pt \\
  --data "$STUDENT_DATA" \\
  --imgsz 256 \\
  --epochs 200 \\
  --batch 64 \\
  --strict-batch-size \\
  --workers 8 \\
  --device "$GPU_ID" \\
  --patience 200 \\
  --project "$PROJECT_ROOT/student_rgb" \\
  --name "$student_name" \\
  --optimizer SGD \\
  --lr0 0.01 \\
  --lrf 0.01 \\
  --momentum 0.937 \\
  --weight-decay 0.0005 \\
  --cos-lr \\
  --mosaic 0.0 \\
  --close-mosaic 0 \\
  --mixup 0.1 \\
  --cutmix 0.0 \\
  --degrees 0.0 \\
  --perspective 0.0 \\
  --translate 0.1 \\
  --scale 0.5 \\
  --fliplr 0.5 \\
  --flipud 0.0 \\
  --hsv-h 0.0 \\
  --hsv-s 0.0 \\
  --hsv-v 0.0 \\
  --erasing 0.0 \\
  --save-period 50 \\
  --seed 0 \\
  --deterministic \\
  > "${log_root}/student_rgb.log" 2>&1
echo "[\$(date '+%F %T')] img256 RGB baseline done" | tee -a "$master_log"

echo "[\$(date '+%F %T')] img256 IR teacher baseline start" | tee -a "$master_log"
"$PY" baseline/code/train_ogsod_baseline.py \\
  --task hbb \\
  --model yolo11n.pt \\
  --data "$TEACHER_DATA" \\
  --imgsz 256 \\
  --epochs 200 \\
  --batch 64 \\
  --strict-batch-size \\
  --workers 8 \\
  --device "$GPU_ID" \\
  --patience 200 \\
  --project "$PROJECT_ROOT/teacher_ir" \\
  --name "$teacher_name" \\
  --optimizer SGD \\
  --lr0 0.01 \\
  --lrf 0.01 \\
  --momentum 0.937 \\
  --weight-decay 0.0005 \\
  --cos-lr \\
  --mosaic 0.0 \\
  --close-mosaic 0 \\
  --mixup 0.1 \\
  --cutmix 0.0 \\
  --degrees 0.0 \\
  --perspective 0.0 \\
  --translate 0.1 \\
  --scale 0.5 \\
  --fliplr 0.5 \\
  --flipud 0.0 \\
  --hsv-h 0.0 \\
  --hsv-s 0.0 \\
  --hsv-v 0.0 \\
  --erasing 0.0 \\
  --save-period 50 \\
  --seed 0 \\
  --deterministic \\
  > "${log_root}/teacher_ir.log" 2>&1
echo "[\$(date '+%F %T')] img256 IR teacher baseline done" | tee -a "$master_log"
EOF

chmod +x "$cmd_path"
{
  echo "[$(date '+%F %T')] launch img256 baseline pair gpu=${GPU_ID}"
  echo "tag=${tag}"
  echo "student=${student_name}"
  echo "teacher=${teacher_name}"
  echo "cmd=${cmd_path}"
  echo "outer=${outer_log}"
} | tee -a "$master_log"

nohup bash "$cmd_path" > "$outer_log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$pid_path"
echo "[$(date '+%F %T')] launched pid=${pid}" | tee -a "$master_log"
