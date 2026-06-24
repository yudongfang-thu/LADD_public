#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/LADD_public

GPU_ID="${GPU_ID:-1}"
EXPECTED_EXTRA_MB="${EXPECTED_EXTRA_MB:-11000}"
MAX_AFTER_MB="${MAX_AFTER_MB:-22000}"
PY="${PY:-/root/shared-nvme/venvs/ladd312/bin/python}"

STUDENT_DATA="comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0/configs/dronevehicle_rgb_hbb.yaml"
TEACHER_DATA="comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0/configs/dronevehicle_ir_hbb.yaml"
TEACHER_CKPT="runs_public/cross_dataset/cclkd_yolo11n/dronevehicle_sub2k_seed0/baselines/teacher_ir/dronevehicle_sub2k_teacher_ir_yolo11n_cclkdproto_e200_b64_img512_mosaic0p0_close0_mixup0p1_s0_gpu0_20260623_221936/weights/best.pt"
A1_DECOMP_CKPT="runs_public/dronevehicle_method_search/sub2k_seed0_fullval/oldsplit_a2only_controlled/ir_to_rgb/oldsplit_a2only_lowlr1e3_nowarmup_ir2rgb_yolo11n_b64_s0_20260624_053127_a1_shared_init/weights/last.pt"
YOLO_INIT="yolo11n.pt"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 2
  fi
}

require_file "$PY" "python interpreter"
require_file "$YOLO_INIT" "YOLO init checkpoint"
require_file "$STUDENT_DATA" "RGB data yaml"
require_file "$TEACHER_DATA" "IR data yaml"
require_file "$TEACHER_CKPT" "IR teacher checkpoint"
require_file "$A1_DECOMP_CKPT" "A1 decomposition checkpoint"

used_mb=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
if (( used_mb + EXPECTED_EXTRA_MB > MAX_AFTER_MB )); then
  echo "GPU${GPU_ID} memory guard: used=${used_mb} MB expected_extra=${EXPECTED_EXTRA_MB} MB max_after=${MAX_AFTER_MB} MB" >&2
  exit 3
fi

stamp=$(date +%Y%m%d_%H%M%S)
tag="oldsplit_probeA_yoloinit_std_ir2rgb_yolo11n_e200_b64_img512_s0_${stamp}"
project_root="runs_public/dronevehicle_method_search/sub2k_seed0_fullval/oldsplit_probeA_yoloinit_std/ir_to_rgb"
log_root="logs/dronevehicle_method_search/sub2k_seed0_fullval/oldsplit_probeA_yoloinit_std/${tag}_gpu${GPU_ID}"
run_name="${tag}_b"
cmd_path="${log_root}/${tag}.cmd.sh"
outer_log="${log_root}/${tag}.outer.log"
master_log="${log_root}/master.log"
pid_path="${log_root}/${tag}.pid"

mkdir -p "$project_root" "$log_root"

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
  --imgsz 512 \\
  --epochs 200 \\
  --batch 64 \\
  --strict-batch-size \\
  --workers 8 \\
  --device "$GPU_ID" \\
  --patience 200 \\
  --fraction 1.0 \\
  --project "$project_root" \\
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
  --optimizer auto \\
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
  echo "[$(date '+%F %T')] launch oldsplit_probeA_yoloinit_std gpu=${GPU_ID}"
  echo "tag=${tag}"
  echo "project=${project_root}"
  echo "run=${run_name}"
  echo "cmd=${cmd_path}"
  echo "outer=${outer_log}"
  echo "a1_decomp=${A1_DECOMP_CKPT}"
} | tee -a "$master_log"

nohup bash "$cmd_path" > "$outer_log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$pid_path"
echo "[$(date '+%F %T')] launched pid=${pid}" | tee -a "$master_log"
