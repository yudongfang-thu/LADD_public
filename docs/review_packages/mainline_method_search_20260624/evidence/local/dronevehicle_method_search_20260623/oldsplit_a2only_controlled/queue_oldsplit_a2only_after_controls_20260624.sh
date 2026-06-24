#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/LADD_public

THRESHOLD_MB="${THRESHOLD_MB:-15000}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MIN_ROWS="${MIN_ROWS:-20}"
REQUIRE_CMDISTILL="${REQUIRE_CMDISTILL:-1}"

PY="${PY:-/root/shared-nvme/venvs/ladd312/bin/python}"
STUDENT_CKPT="runs_public/cross_dataset/cclkd_yolo11n/dronevehicle_sub2k_seed0/baselines/student_rgb/dronevehicle_sub2k_student_rgb_yolo11n_cclkdproto_e200_b64_img512_mosaic0p0_close0_mixup0p1_s0_20260623_221620/weights/best.pt"
TEACHER_CKPT="runs_public/cross_dataset/cclkd_yolo11n/dronevehicle_sub2k_seed0/baselines/teacher_ir/dronevehicle_sub2k_teacher_ir_yolo11n_cclkdproto_e200_b64_img512_mosaic0p0_close0_mixup0p1_s0_gpu0_20260623_221936/weights/best.pt"
STUDENT_DATA="comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0/configs/dronevehicle_rgb_hbb.yaml"
TEACHER_DATA="comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0/configs/dronevehicle_ir_hbb.yaml"

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
  "$PY" - "$label" "$csv_path" "$MIN_ROWS" <<'PY'
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

summarize_result() {
  local csv_path="$1"
  "$PY" - "$csv_path" <<'PY'
import csv
import sys
from pathlib import Path
p = Path(sys.argv[1])
rows = list(csv.DictReader(p.open(newline="", errors="replace")))
key = next((k for k in rows[0].keys() if "metrics/mAP50-95" in k), "metrics/mAP50-95(B)")
key50 = next((k for k in rows[0].keys() if "metrics/mAP50(B)" in k), "metrics/mAP50(B)")
def fv(r, k):
    try:
        return float(r.get(k, "nan"))
    except Exception:
        return float("nan")
best = max(rows, key=lambda r: fv(r, key))
last = rows[-1]
print(f"best={fv(best,key50):.5f}/{fv(best,key):.5f} last={fv(last,key50):.5f}/{fv(last,key):.5f} rows={len(rows)}")
PY
}

launch_oldsplit_a2only() {
  local gpu="$1"
  local stamp tag project_root log_root master_log a1_name det_name reach_name a1_ckpt
  stamp=$(date +%Y%m%d_%H%M%S)
  tag="oldsplit_a2only_lowlr1e3_nowarmup_ir2rgb_yolo11n_b64_s0_${stamp}"
  project_root="runs_public/dronevehicle_method_search/sub2k_seed0_fullval/oldsplit_a2only_controlled/ir_to_rgb"
  log_root="logs/dronevehicle_method_search/sub2k_seed0_fullval/oldsplit_a2only_controlled/${tag}_gpu${gpu}"
  master_log="${log_root}/master.log"
  a1_name="${tag}_a1_shared_init"
  det_name="${tag}_a2_detonly_split_control"
  reach_name="${tag}_a2_reach_kd_lowlr"
  mkdir -p "$project_root" "$log_root"
  echo "[$(date '+%F %T')] oldsplit A2-only launch gpu=${gpu} tag=${tag}" | tee -a "$master_log"
  cat > "${log_root}/${tag}.cmd.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd /root/shared-nvme/LADD_public
export PYTHONUNBUFFERED=1
EOF
  chmod +x "${log_root}/${tag}.cmd.sh"

  common=(
    --teacher-weights "$TEACHER_CKPT"
    --data "$STUDENT_DATA"
    --teacher-data "$TEACHER_DATA"
    --imgsz 512 --batch 64 --strict-batch-size --workers 8 --device "$gpu" --deterministic --seed 0
    --lambda-reach 1.0 --lambda-match-inner 1.0 --lambda-rank-inner 1.0 --delta 0.3 --reach-rank-mode softplus
    --lambda-rec 0.10 --lambda-taskL 0.0 --alpha-kd 1.0 --alpha-s-rec 0.1
    --student-detect-mode raw --student-branch-mode split --teacher-feature-mode decomposed
    --kd-mechanism mse --use-fg-mask-for-reach --reach-input-mode adapter
    --optimizer SGD --lr0 0.001 --lrf 0.1 --momentum 0.937 --weight-decay 0.0005 --cos-lr
    --warmup-epochs 0.0 --warmup-bias-lr 0.0 --warmup-momentum 0.937
    --mosaic 0.0 --mixup 0.1 --cutmix 0.0 --degrees 0.0 --perspective 0.0 --translate 0.1 --scale 0.5
    --fliplr 0.5 --flipud 0.0 --hsv-h 0.0 --hsv-s 0.0 --hsv-v 0.0 --erasing 0.0 --close-mosaic 0
    --save-period 25
  )

  "$PY" ladd/code/train_ladd_hbb.py \
    --phase a1 --model "$STUDENT_CKPT" --epochs 50 --patience 200 \
    --project "$project_root" --name "$a1_name" \
    --phase-detect-mode raw --det-loss-scale 0.0 \
    --reach-target-mode coupled --kd-target-mode detach \
    "${common[@]}" >> "${log_root}/${tag}_a1.log" 2>&1
  a1_ckpt="${project_root}/${a1_name}/weights/last.pt"
  echo "[$(date '+%F %T')] A1 done ${a1_ckpt}" | tee -a "$master_log"

  "$PY" ladd/code/train_ladd_hbb.py \
    --phase a2 --model "$a1_ckpt" --epochs 100 --patience 200 \
    --project "$project_root" --name "$det_name" \
    --phase-detect-mode raw --det-loss-scale 1.0 \
    --reach-target-mode coupled --kd-target-mode detach \
    --ladd-a2-det-only \
    "${common[@]}" >> "${log_root}/${tag}_a2_detonly.log" 2>&1
  echo "[$(date '+%F %T')] A2 detonly done $(summarize_result "${project_root}/${det_name}/results.csv")" | tee -a "$master_log"

  "$PY" ladd/code/train_ladd_hbb.py \
    --phase a2 --model "$a1_ckpt" --epochs 100 --patience 200 \
    --project "$project_root" --name "$reach_name" \
    --phase-detect-mode raw --det-loss-scale 1.0 \
    --reach-target-mode coupled --kd-target-mode detach \
    "${common[@]}" >> "${log_root}/${tag}_a2_reach.log" 2>&1
  echo "[$(date '+%F %T')] A2 reach/KD done $(summarize_result "${project_root}/${reach_name}/results.csv")" | tee -a "$master_log"
}

require_file "$PY" "python interpreter"
require_file "$STUDENT_CKPT" "RGB baseline checkpoint"
require_file "$TEACHER_CKPT" "IR teacher checkpoint"
require_file "$STUDENT_DATA" "RGB data yaml"
require_file "$TEACHER_DATA" "IR data yaml"

echo "[$(date '+%F %T')] oldsplit A2-only controlled queue ready threshold=${THRESHOLD_MB} min_rows=${MIN_ROWS} require_cmdistill=${REQUIRE_CMDISTILL}"
while ! primary_ready; do
  sleep "$POLL_SECONDS"
done

gpu=$(wait_for_gpu)
launch_oldsplit_a2only "$gpu"

echo "[$(date '+%F %T')] oldsplit A2-only controlled queue finished"
