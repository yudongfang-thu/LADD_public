#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/LADD_public

PY="${PY:-/root/shared-nvme/venvs/ladd312/bin/python}"
GPU_OGSOD="${GPU_OGSOD:-0}"
MAX_AFTER_MB="${MAX_AFTER_MB:-22000}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 2
  fi
}

gpu_used_mb() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '
}

guard_gpu() {
  local gpu="$1"
  local expected="$2"
  local label="$3"
  local used
  used=$(gpu_used_mb "$gpu")
  if (( used + expected > MAX_AFTER_MB )); then
    echo "GPU${gpu} memory guard for ${label}: used=${used} expected_extra=${expected} max_after=${MAX_AFTER_MB}" >&2
    exit 3
  fi
}

require_file "$PY" "python interpreter"
require_file "yolo11n.pt" "YOLO init checkpoint"

OGSOD_SAR_DATA="debug/zw1_nomosaic_clean_cache_20260623/20260623_214553/yamls/ogsod_hbb_sar_nomosaic_zw1.yaml"
OGSOD_RGB_DATA="debug/zw1_nomosaic_clean_cache_20260623/20260623_214553/yamls/ogsod_hbb_rgb_nomosaic_zw1.yaml"
OGSOD_RGB_TEACHER="runs_public/ogsod/hbb/baseline_controls/mosaic_baselines_20260615/rgb_yolo11n_hbb_mosaicE800_closeAt100_s0_imported_cos_closeAt100_20260524/weights/best.pt"
OGSOD_A1="runs_public/paper/ogsod_hbb_mosaic100/no_reload_warm100/yolo11n/seed0/a1_decomp_cache/ogsod_hbb_ladd_a1_decomp_from_sar_baseline_yolo11n_s0_20260623_192641_img256_a1_e10_b64_s0_gpu0/weights/best.pt"

for f in "$OGSOD_SAR_DATA" "$OGSOD_RGB_DATA" "$OGSOD_RGB_TEACHER" "$OGSOD_A1"; do
  require_file "$f" "$f"
done

ROOT_LOG="logs/dronevehicle_method_search/sub2k_seed0_fullval/autodl_condition_followups_20260624"
project="runs_public/dronevehicle_method_search/ogsod_nomosaic_autodl_condition/yolo_probeA"
log_root="${ROOT_LOG}/ogsod_yolo_probeA_gpu${GPU_OGSOD}"
master_log="${log_root}/master.log"
mkdir -p "$project" "$log_root"

if find "$project" -path "*_e800_*" -name results.csv -type f 2>/dev/null | grep -q .; then
  echo "[$(date '+%F %T')] OGSOD e800 yolo_probeA already has results.csv under ${project}; skip." | tee -a "$master_log"
  exit 0
fi

guard_gpu "$GPU_OGSOD" 9000 "OGSOD yolo_probeA"

stamp=$(date +%Y%m%d_%H%M%S)
tag="ogsod_nomix_yolo_probeA_existingcache_yolo11n_e800_b64_img256_s0_${stamp}"
run_name="${tag}_b"
cmd_path="${log_root}/${tag}.cmd.sh"
outer_log="${log_root}/${tag}.outer.log"
pid_path="${log_root}/${tag}.pid"

cat > "$cmd_path" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd /root/shared-nvme/LADD_public
export PYTHONUNBUFFERED=1

"$PY" ladd/code/train_ladd_hbb.py \\
  --phase b \\
  --model yolo11n.pt \\
  --data "$OGSOD_SAR_DATA" \\
  --teacher-data "$OGSOD_RGB_DATA" \\
  --teacher-weights "$OGSOD_RGB_TEACHER" \\
  --imgsz 256 \\
  --epochs 800 \\
  --batch 64 \\
  --strict-batch-size \\
  --workers 8 \\
  --device "$GPU_OGSOD" \\
  --patience 800 \\
  --fraction 1.0 \\
  --project "$project" \\
  --name "$run_name" \\
  --phase-detect-mode raw \\
  --det-loss-scale 1.0 \\
  --phase-stop-metric default \\
  --phase-min-epochs 800 \\
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
  --mixup 0.0 \\
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
  --save-period 100 \\
  --use-mask \\
  --use-fg-mask-for-reach \\
  --ladd-b-a2-core \\
  --ladd-b-frozen-reach-probe \\
  --ladd-b-detach-reach-probe \\
  --b-detector-source yolo11n.pt \\
  --b-decomp-source "$OGSOD_A1" \\
  --b-load-student-reachability
EOF

chmod +x "$cmd_path"
{
  echo "[$(date '+%F %T')] launch OGSOD yolo_probeA gpu=${GPU_OGSOD}"
  echo "tag=${tag}"
  echo "project=${project}"
  echo "run=${run_name}"
  echo "cmd=${cmd_path}"
  echo "outer=${outer_log}"
  echo "teacher=${OGSOD_RGB_TEACHER}"
  echo "a1=${OGSOD_A1}"
} | tee -a "$master_log"

nohup bash "$cmd_path" > "$outer_log" 2>&1 < /dev/null &
echo "$!" > "$pid_path"
echo "[$(date '+%F %T')] launched OGSOD yolo_probeA pid=$(cat "$pid_path")" | tee -a "$master_log"
