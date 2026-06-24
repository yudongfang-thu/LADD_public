#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/LADD_public

PY="${PY:-/root/shared-nvme/venvs/ladd312/bin/python}"
GPU_OGSOD="${GPU_OGSOD:-0}"
GPU_DV="${GPU_DV:-1}"
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

wait_rows() {
  local csv="$1"
  local rows="$2"
  local label="$3"
  local sleep_s="${4:-120}"
  while true; do
    local got
    got=$(row_count "$csv")
    echo "[$(date '+%F %T')] wait ${label}: rows=${got}/${rows}"
    if (( got >= rows )); then
      return 0
    fi
    sleep "$sleep_s"
  done
}

require_file "$PY" "python interpreter"
require_file "yolo11n.pt" "YOLO init checkpoint"

OGSOD_SAR_DATA="debug/zw1_nomosaic_clean_cache_20260623/20260623_214553/yamls/ogsod_hbb_sar_nomosaic_zw1.yaml"
OGSOD_RGB_DATA="debug/zw1_nomosaic_clean_cache_20260623/20260623_214553/yamls/ogsod_hbb_rgb_nomosaic_zw1.yaml"
OGSOD_RGB_TEACHER="runs_public/ogsod/hbb/baseline_controls/mosaic_baselines_20260615/rgb_yolo11n_hbb_mosaicE800_closeAt100_s0_imported_cos_closeAt100_20260524/weights/best.pt"
OGSOD_A1="runs_public/paper/ogsod_hbb_mosaic100/no_reload_warm100/yolo11n/seed0/a1_decomp_cache/ogsod_hbb_ladd_a1_decomp_from_sar_baseline_yolo11n_s0_20260623_192641_img256_a1_e10_b64_s0_gpu0/weights/best.pt"

DV_RGB_DATA="comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0/configs/dronevehicle_rgb_hbb.yaml"
DV_IR_DATA="comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0/configs/dronevehicle_ir_hbb.yaml"
DV_RGB_BASE="runs_public/dronevehicle_method_search/sub2k_seed0_fullval/img256_baselines/student_rgb/dronevehicle_sub2k_student_rgb_yolo11n_cclkdproto_e200_b64_img256_mosaic0p0_close0_mixup0p1_s0_20260624_090613/weights/best.pt"
DV_IR_TEACHER="runs_public/dronevehicle_method_search/sub2k_seed0_fullval/img256_baselines/teacher_ir/dronevehicle_sub2k_teacher_ir_yolo11n_cclkdproto_e200_b64_img256_mosaic0p0_close0_mixup0p1_s0_20260624_090613/weights/best.pt"

for f in "$OGSOD_SAR_DATA" "$OGSOD_RGB_DATA" "$OGSOD_RGB_TEACHER" "$OGSOD_A1" "$DV_RGB_DATA" "$DV_IR_DATA" "$DV_RGB_BASE" "$DV_IR_TEACHER"; do
  require_file "$f" "$f"
done

ROOT_LOG="logs/dronevehicle_method_search/sub2k_seed0_fullval/autodl_condition_followups_20260624"
mkdir -p "$ROOT_LOG"

launch_ogsod_yolo_detonly() {
  local stamp tag project run_name log_root cmd_path outer_log master_log pid_path
  stamp=$(date +%Y%m%d_%H%M%S)
  tag="ogsod_nomix_yolo_detonly_existingcache_yolo11n_e800_b64_img256_s0_${stamp}"
  project="runs_public/dronevehicle_method_search/ogsod_nomosaic_autodl_condition/yolo_detonly"
  run_name="${tag}_b"
  log_root="${ROOT_LOG}/ogsod_yolo_detonly_gpu${GPU_OGSOD}"
  cmd_path="${log_root}/${tag}.cmd.sh"
  outer_log="${log_root}/${tag}.outer.log"
  master_log="${log_root}/master.log"
  pid_path="${log_root}/${tag}.pid"
  mkdir -p "$project" "$log_root"

  if find "$project" -path "*_e800_*" -name results.csv -type f 2>/dev/null | grep -q .; then
    echo "[$(date '+%F %T')] OGSOD e800 yolo_detonly already has results.csv under ${project}; skip." | tee -a "$master_log"
    return 0
  fi

  guard_gpu "$GPU_OGSOD" 9000 "OGSOD yolo_detonly"

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
  --ladd-b-det-only \\
  --b-detector-source yolo11n.pt \\
  --b-decomp-source "$OGSOD_A1" \\
  --b-load-student-reachability
EOF
  chmod +x "$cmd_path"
  {
    echo "[$(date '+%F %T')] launch OGSOD yolo_detonly gpu=${GPU_OGSOD}"
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
  echo "[$(date '+%F %T')] launched OGSOD yolo_detonly pid=$(cat "$pid_path")" | tee -a "$master_log"
}

write_dv_b_cmd() {
  local mode="$1"
  local gpu="$2"
  local a1_ckpt="$3"
  local project="$4"
  local run_name="$5"
  local cmd_path="$6"

  local method_flags=""
  if [[ "$mode" == "probeA" ]]; then
    method_flags="--ladd-b-a2-core --ladd-b-frozen-reach-probe --ladd-b-detach-reach-probe --b-load-student-reachability"
  else
    method_flags="--ladd-b-det-only --b-load-student-reachability"
  fi

  cat > "$cmd_path" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd /root/shared-nvme/LADD_public
export PYTHONUNBUFFERED=1

"$PY" ladd/code/train_ladd_hbb.py \\
  --phase b \\
  --model yolo11n.pt \\
  --data "$DV_RGB_DATA" \\
  --teacher-data "$DV_IR_DATA" \\
  --teacher-weights "$DV_IR_TEACHER" \\
  --imgsz 256 \\
  --epochs 200 \\
  --batch 64 \\
  --strict-batch-size \\
  --workers 8 \\
  --device "$gpu" \\
  --patience 200 \\
  --fraction 1.0 \\
  --project "$project" \\
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
  --save-period 50 \\
  --use-mask \\
  --use-fg-mask-for-reach \\
  $method_flags \\
  --b-detector-source yolo11n.pt \\
  --b-decomp-source "$a1_ckpt"
EOF
  chmod +x "$cmd_path"
}

launch_drone_nomix_chain() {
  local stamp root project_a1 log_root master_log a1_name a1_ckpt a1_csv
  stamp=$(date +%Y%m%d_%H%M%S)
  root="runs_public/dronevehicle_method_search/sub2k_seed0_fullval/img256_nomix_autodl_condition"
  project_a1="${root}/a1_cache"
  log_root="${ROOT_LOG}/drone_img256_nomix_gpu${GPU_DV}_${stamp}"
  master_log="${log_root}/master.log"
  a1_name="dronevehicle_img256_nomix_a1_from_img256rgb_yolo11n_b64_s0_${stamp}_a1"
  mkdir -p "$project_a1" "$log_root"

  if find "${root}/probeA" "${root}/detonly" -name results.csv -type f 2>/dev/null | grep -q .; then
    echo "[$(date '+%F %T')] Drone no-mix ProbeA/control already has results.csv under ${root}; skip." | tee -a "$master_log"
    return 0
  fi

  guard_gpu "$GPU_DV" 5500 "Drone img256 no-mix A1"

  echo "[$(date '+%F %T')] launch DroneVehicle img256 no-mix A1 gpu=${GPU_DV}" | tee -a "$master_log"
  "$PY" ladd/code/train_ladd_hbb.py \
    --phase a1 \
    --model "$DV_RGB_BASE" \
    --data "$DV_RGB_DATA" \
    --teacher-data "$DV_IR_DATA" \
    --teacher-weights "$DV_IR_TEACHER" \
    --imgsz 256 \
    --epochs 50 \
    --batch 64 \
    --strict-batch-size \
    --workers 8 \
    --device "$GPU_DV" \
    --patience 200 \
    --project "$project_a1" \
    --name "$a1_name" \
    --phase-detect-mode raw \
    --det-loss-scale 0.0 \
    --phase-stop-metric default \
    --lambda-rec 0.10 \
    --lambda-taskL 0.0 \
    --alpha-kd 1.0 \
    --alpha-s-rec 0.1 \
    --lambda-reach 1.0 \
    --lambda-match-inner 1.0 \
    --lambda-rank-inner 1.0 \
    --delta 0.2 \
    --rank-d-neg-cap 2.0 \
    --reach-target-mode coupled \
    --kd-target-mode detach \
    --reach-input-mode adapter \
    --student-detect-mode raw \
    --student-branch-mode split \
    --teacher-feature-mode decomposed \
    --kd-mechanism mse \
    --use-mask \
    --use-fg-mask-for-reach \
    --mosaic 0.0 \
    --mixup 0.0 \
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
    --close-mosaic 0 \
    --seed 0 \
    --deterministic \
    --optimizer auto \
    --lr0 0.01 \
    --lrf 0.01 \
    --cos-lr \
    --warmup-epochs 3.0 \
    --warmup-bias-lr 0.1 \
    --save-period 25 \
    > "${log_root}/${a1_name}.outer.log" 2>&1

  a1_ckpt="${project_a1}/${a1_name}/weights/best.pt"
  a1_csv="${project_a1}/${a1_name}/results.csv"
  require_file "$a1_ckpt" "Drone no-mix A1 best checkpoint"
  echo "[$(date '+%F %T')] Drone A1 done rows=$(row_count "$a1_csv") ckpt=${a1_ckpt}" | tee -a "$master_log"

  local probe_project det_project probe_name det_name probe_cmd det_cmd probe_log det_log
  probe_project="${root}/probeA/ir_to_rgb"
  det_project="${root}/detonly/ir_to_rgb"
  probe_name="probeA_yoloinit_nomix_auto_img256_yolo11n_e200_b64_s0_${stamp}_b"
  det_name="detonly_yoloinit_nomix_auto_img256_yolo11n_e200_b64_s0_${stamp}_b"
  mkdir -p "$probe_project" "$det_project"

  guard_gpu "$GPU_DV" 11000 "Drone img256 no-mix ProbeA+detonly"

  probe_cmd="${log_root}/${probe_name}.cmd.sh"
  det_cmd="${log_root}/${det_name}.cmd.sh"
  probe_log="${log_root}/${probe_name}.outer.log"
  det_log="${log_root}/${det_name}.outer.log"
  write_dv_b_cmd "probeA" "$GPU_DV" "$a1_ckpt" "$probe_project" "$probe_name" "$probe_cmd"
  write_dv_b_cmd "detonly" "$GPU_DV" "$a1_ckpt" "$det_project" "$det_name" "$det_cmd"

  echo "[$(date '+%F %T')] launch Drone no-mix ProbeA and det-only on gpu=${GPU_DV}" | tee -a "$master_log"
  nohup bash "$probe_cmd" > "$probe_log" 2>&1 < /dev/null &
  echo "$!" > "${log_root}/${probe_name}.pid"
  sleep 20
  nohup bash "$det_cmd" > "$det_log" 2>&1 < /dev/null &
  echo "$!" > "${log_root}/${det_name}.pid"
  echo "[$(date '+%F %T')] launched probe_pid=$(cat "${log_root}/${probe_name}.pid") det_pid=$(cat "${log_root}/${det_name}.pid")" | tee -a "$master_log"
}

launch_ogsod_yolo_detonly
nohup bash -lc "$(declare -f require_file gpu_used_mb guard_gpu row_count wait_rows write_dv_b_cmd launch_drone_nomix_chain); cd /root/shared-nvme/LADD_public; PY='$PY' GPU_DV='$GPU_DV' MAX_AFTER_MB='$MAX_AFTER_MB' ROOT_LOG='$ROOT_LOG' DV_RGB_DATA='$DV_RGB_DATA' DV_IR_DATA='$DV_IR_DATA' DV_RGB_BASE='$DV_RGB_BASE' DV_IR_TEACHER='$DV_IR_TEACHER' launch_drone_nomix_chain" > "${ROOT_LOG}/drone_nomix_chain_gpu${GPU_DV}.outer.log" 2>&1 < /dev/null &
echo "$!" > "${ROOT_LOG}/drone_nomix_chain_gpu${GPU_DV}.pid"
echo "[$(date '+%F %T')] launched Drone no-mix chain wrapper pid=$(cat "${ROOT_LOG}/drone_nomix_chain_gpu${GPU_DV}.pid")"
