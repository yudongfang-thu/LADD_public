#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/LADD_public

GPU_ID="${GPU_ID:-0}"
EXPECTED_EXTRA_MB="${EXPECTED_EXTRA_MB:-9000}"
MAX_AFTER_MB="${MAX_AFTER_MB:-22000}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"
MAX_WAIT_LOOPS="${MAX_WAIT_LOOPS:-360}"
PY="${PY:-/root/shared-nvme/venvs/ladd312/bin/python}"

STUDENT_DATA="comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0/configs/dronevehicle_rgb_hbb.yaml"
TEACHER_DATA="comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0/configs/dronevehicle_ir_hbb.yaml"
IMG256_BASE_ROOT="runs_public/dronevehicle_method_search/sub2k_seed0_fullval/img256_baselines"
RGB_BASE_NAME="dronevehicle_sub2k_student_rgb_yolo11n_cclkdproto_e200_b64_img256_mosaic0p0_close0_mixup0p1_s0_20260624_090613"
IR_BASE_NAME="dronevehicle_sub2k_teacher_ir_yolo11n_cclkdproto_e200_b64_img256_mosaic0p0_close0_mixup0p1_s0_20260624_090613"
RGB_RESULTS="${IMG256_BASE_ROOT}/student_rgb/${RGB_BASE_NAME}/results.csv"
IR_RESULTS="${IMG256_BASE_ROOT}/teacher_ir/${IR_BASE_NAME}/results.csv"
TEACHER_CKPT="${IMG256_BASE_ROOT}/teacher_ir/${IR_BASE_NAME}/weights/best.pt"
A1_DECOMP_CKPT="runs_public/dronevehicle_method_search/sub2k_seed0_fullval/oldsplit_a2only_controlled/ir_to_rgb/oldsplit_a2only_lowlr1e3_nowarmup_ir2rgb_yolo11n_b64_s0_20260624_053127_a1_shared_init/weights/last.pt"
YOLO_INIT="yolo11n.pt"

PROJECT_ROOT="runs_public/dronevehicle_method_search/sub2k_seed0_fullval/img256_probeA/oldsplit_probeA_yoloinit_std/ir_to_rgb"
LOG_ROOT="logs/dronevehicle_method_search/sub2k_seed0_fullval/img256_probeA/oldsplit_probeA_yoloinit_std_after_baselines_20260624_gpu${GPU_ID}"
QUEUE_LOG="${LOG_ROOT}/queue.log"
QUEUE_PID="${LOG_ROOT}/queue.pid"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 2
  fi
}

row_count() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo 0
    return
  fi
  local lines
  lines=$(wc -l < "$path" | tr -d ' ')
  if (( lines <= 0 )); then
    echo 0
  else
    echo $((lines - 1))
  fi
}

already_launched() {
  find "$PROJECT_ROOT" -name results.csv -type f 2>/dev/null | grep -q .
}

mkdir -p "$PROJECT_ROOT" "$LOG_ROOT"
echo "$$" > "$QUEUE_PID"

require_file "$PY" "python interpreter"
require_file "$YOLO_INIT" "YOLO init checkpoint"
require_file "$STUDENT_DATA" "RGB data yaml"
require_file "$TEACHER_DATA" "IR data yaml"
require_file "$A1_DECOMP_CKPT" "A1 decomposition checkpoint"

if already_launched; then
  echo "[$(date '+%F %T')] img256 ProbeA already has a results.csv under ${PROJECT_ROOT}; exit." | tee -a "$QUEUE_LOG"
  exit 0
fi

for ((i=1; i<=MAX_WAIT_LOOPS; i++)); do
  rgb_rows=$(row_count "$RGB_RESULTS")
  ir_rows=$(row_count "$IR_RESULTS")
  teacher_ready=0
  [[ -f "$TEACHER_CKPT" ]] && teacher_ready=1
  echo "[$(date '+%F %T')] wait ${i}/${MAX_WAIT_LOOPS}: rgb_rows=${rgb_rows} ir_rows=${ir_rows} teacher_best=${teacher_ready}" | tee -a "$QUEUE_LOG"

  if (( rgb_rows >= 200 && ir_rows >= 200 && teacher_ready == 1 )); then
    break
  fi

  if (( i == MAX_WAIT_LOOPS )); then
    echo "[$(date '+%F %T')] timeout waiting for img256 baselines." | tee -a "$QUEUE_LOG"
    exit 4
  fi
  sleep "$WAIT_SECONDS"
done

used_mb=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if (( used_mb + EXPECTED_EXTRA_MB > MAX_AFTER_MB )); then
  echo "[$(date '+%F %T')] GPU${GPU_ID} memory guard: used=${used_mb} MB expected_extra=${EXPECTED_EXTRA_MB} MB max_after=${MAX_AFTER_MB} MB" | tee -a "$QUEUE_LOG"
  exit 3
fi

stamp=$(date +%Y%m%d_%H%M%S)
tag="oldsplit_probeA_yoloinit_std_ir2rgb_yolo11n_e200_b64_img256_s0_${stamp}"
run_name="${tag}_b"
cmd_path="${LOG_ROOT}/${tag}.cmd.sh"
outer_log="${LOG_ROOT}/${tag}.outer.log"
master_log="${LOG_ROOT}/master.log"
pid_path="${LOG_ROOT}/${tag}.pid"

cat > "$cmd_path" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd /root/shared-nvme/LADD_public
export PYTHONUNBUFFERED=1

"$PY" ladd/code/train_ladd_hbb.py \\
  --phase b \\
  --model "$YOLO_INIT" \\
  --data "$STUDENT_DATA" \\
  --teacher-data "$TEACHER_DATA" \\
  --teacher-weights "$TEACHER_CKPT" \\
  --imgsz 256 \\
  --epochs 200 \\
  --batch 64 \\
  --strict-batch-size \\
  --workers 8 \\
  --device "$GPU_ID" \\
  --patience 200 \\
  --fraction 1.0 \\
  --project "$PROJECT_ROOT" \\
  --name "$run_name" \\
  --phase-detect-mode raw \\
  --det-loss-scale 1.0 \\
  --phase-stop-metric default \\
  --phase-min-epochs 200 \\
  --freeze-bn-after-epoch -1 \\
  --ladd-diag-log-bn 1 \\
  --ladd-diag-log-grad 0 \\
  --ladd-grad-clip-norm 0.0 \\
  --ladd-diag-log-every 1 \\
  --ladd-kd-decay-mode none \\
  --ladd-kd-decay-start-epoch -1 \\
  --ladd-kd-decay-end-epoch -1 \\
  --ladd-kd-final-mult 1.0 \\
  --ladd-kd-stop-after-epoch -1 \\
  --ladd-b-loss-warmup-mode linear \\
  --ladd-b-loss-warmup-start-epoch 0 \\
  --ladd-b-loss-warmup-end-epoch 30 \\
  --ladd-b-loss-warmup-final-mult 1.0 \\
  --ladd-b-loss-warmup-scope core \\
  --reach-target-mode detach \\
  --kd-target-mode detach \\
  --kd-calibration-mode none \\
  --lambda-rec 0.1 \\
  --lambda-taskL 1.0 \\
  --alpha-kd 1.0 \\
  --alpha-s-rec 0.1 \\
  --lambda-reach 1.0 \\
  --lambda-match-inner 1.0 \\
  --lambda-rank-inner 1.0 \\
  --delta 0.2 \\
  --rank-d-neg-cap 2.0 \\
  --reach-input-mode adapter \\
  --student-detect-mode raw \\
  --student-branch-mode split \\
  --teacher-feature-mode decomposed \\
  --unlearnable-hidden-ratio 1.0 \\
  --kd-weight-mode none \\
  --kd-aggregation-mode token \\
  --mosaic 0.0 \\
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
  --close-mosaic 0 \\
  --seed 0 \\
  --deterministic \\
  --comparison-kd-profile none \\
  --profile-kd-weight 1.0 \\
  --cmdistill-feature-weight 1.0 \\
  --cmdistill-relation-weight 1.0 \\
  --cmdistill-logit-weight 1.0 \\
  --cmdistill-temperature 4.0 \\
  --cmdistill-max-tokens 512 \\
  --cmdistill-min-confidence 0.05 \\
  --cclkd-base-temperature 2.0 \\
  --cclkd-contrastive-temperature 0.1 \\
  --cclkd-feat-weight 1.0 \\
  --cclkd-logit-weight 1.0 \\
  --cclkd-contrast-weight 0.5 \\
  --cclkd-bg-weight 0.1 \\
  --cclkd-min-confidence 0.1 \\
  --cclkd-max-tokens 512 \\
  --cclkd-temperature-min 0.5 \\
  --cclkd-temperature-max 5.0 \\
  --cclkd-entropy-scale 5.0 \\
  --lr0 0.01 \\
  --lrf 0.01 \\
  --cos-lr \\
  --optimizer SGD \\
  --momentum 0.937 \\
  --weight-decay 0.0005 \\
  --warmup-epochs 3.0 \\
  --warmup-bias-lr 0.1 \\
  --save-period 50 \\
  --use-mask \\
  --use-fg-mask-for-reach \\
  --ladd-b-a2-core \\
  --ladd-b-frozen-reach-probe \\
  --ladd-b-detach-reach-probe \\
  --b-detector-source "$YOLO_INIT" \\
  --b-decomp-source "$A1_DECOMP_CKPT" \\
  --b-load-student-reachability
EOF

chmod +x "$cmd_path"
{
  echo "[$(date '+%F %T')] launch img256 oldsplit_probeA_yoloinit_std gpu=${GPU_ID}"
  echo "tag=${tag}"
  echo "project=${PROJECT_ROOT}"
  echo "run=${run_name}"
  echo "cmd=${cmd_path}"
  echo "outer=${outer_log}"
  echo "teacher=${TEACHER_CKPT}"
  echo "a1_decomp=${A1_DECOMP_CKPT}"
  echo "rgb_baseline_results=${RGB_RESULTS}"
  echo "ir_baseline_results=${IR_RESULTS}"
} | tee -a "$master_log" "$QUEUE_LOG"

nohup bash "$cmd_path" > "$outer_log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$pid_path"
echo "[$(date '+%F %T')] launched img256 ProbeA pid=${pid}" | tee -a "$master_log" "$QUEUE_LOG"
