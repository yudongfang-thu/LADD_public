#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/root/shared-nvme/LADD_public}
PY=${PY:-/root/shared-nvme/venvs/ladd312/bin/python}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}

TAG=${TAG:-dynamic_plain_yoloinit}
GPU=${GPU:-0}
DETACH_REACH=${DETACH_REACH:-0}
STUDENT_BRANCH=${STUDENT_BRANCH:-split}
REACH_INPUT=${REACH_INPUT:-adapter}
ALPHA_S_REC=${ALPHA_S_REC:-0.1}
RANK_D_NEG_CAP=${RANK_D_NEG_CAP:-2.0}

cd "$REPO_ROOT"

LOG_ROOT=${LOG_ROOT:-logs/ogsod_yoloinit_probe_plain_anchor_20260625}
PROJ_ROOT=${PROJ_ROOT:-runs_public/ogsod/hbb/yoloinit_probe_plain_anchor_20260625}
SMOKE_ROOT=${SMOKE_ROOT:-runs_public/ogsod/hbb/yoloinit_probe_plain_anchor_20260625_smoke}
DATA_YAML=${DATA_YAML:-configs/paper/datasets/ogsod_hbb_sar.yaml}
TEACHER_DATA_YAML=${TEACHER_DATA_YAML:-configs/paper/datasets/ogsod_hbb_rgb.yaml}
TEACHER_WEIGHTS=${TEACHER_WEIGHTS:-runs_public/ogsod/hbb/formal_nomosaic_20260528/baselines/rgb/rgb_yolo11n_hbb_800ep_cos_nomosaic_albu_b64_s0/weights/best.pt}
B_DECOMP_SOURCE=${B_DECOMP_SOURCE:-runs_public/paper/ogsod_hbb_nomosaic/diagnostics/ladd_dynamic/yolo11n/seed0/ladd_clean_a1b_dyn_ogsod11n_diagnostic_nomosaic_dynamic_yolo11n_s0_a1_e10_b64_s0_gpu0/weights/best.pt}
mkdir -p "$LOG_ROOT"

EXTRA_ARGS=()
if [[ "$DETACH_REACH" == "1" ]]; then
  EXTRA_ARGS+=(--ladd-b-detach-reach-probe)
fi

COMMON_ARGS=(
  --phase b
  --model yolo11n.pt
  --data "$DATA_YAML"
  --teacher-data "$TEACHER_DATA_YAML"
  --teacher-weights "$TEACHER_WEIGHTS"
  --imgsz 256
  --batch 64
  --strict-batch-size
  --workers 8
  --patience 800
  --fraction 1.0
  --phase-detect-mode raw
  --det-loss-scale 1.0
  --phase-stop-metric default
  --phase-min-epochs 800
  --freeze-bn-after-epoch -1
  --ladd-diag-log-bn 1
  --ladd-diag-log-grad 0
  --ladd-grad-clip-norm 0.0
  --ladd-diag-log-every 1
  --ladd-kd-decay-mode none
  --ladd-kd-decay-start-epoch -1
  --ladd-kd-decay-end-epoch -1
  --ladd-kd-final-mult 1.0
  --ladd-kd-stop-after-epoch -1
  --ladd-b-loss-warmup-mode linear
  --ladd-b-loss-warmup-start-epoch 0
  --ladd-b-loss-warmup-end-epoch 30
  --ladd-b-loss-warmup-final-mult 1.0
  --ladd-b-loss-warmup-scope core
  --reach-target-mode detach
  --kd-target-mode detach
  --kd-calibration-mode none
  --lambda-rec 0.1
  --lambda-taskL 1.0
  --alpha-kd 1.0
  --lambda-reach 1.0
  --lambda-match-inner 1.0
  --lambda-rank-inner 1.0
  --delta 0.2
  --rank-d-neg-cap "$RANK_D_NEG_CAP"
  --reach-input-mode "$REACH_INPUT"
  --student-detect-mode raw
  --student-branch-mode "$STUDENT_BRANCH"
  --teacher-feature-mode decomposed
  --unlearnable-hidden-ratio 1.0
  --kd-weight-mode none
  --kd-aggregation-mode token
  --mosaic 0.0
  --mixup 0.0
  --cutmix 0.0
  --degrees 0.0
  --perspective 0.0
  --translate 0.1
  --scale 0.5
  --fliplr 0.5
  --flipud 0.0
  --hsv-h 0.0
  --hsv-s 0.0
  --hsv-v 0.0
  --erasing 0.0
  --close-mosaic 0
  --seed 0
  --deterministic
  --comparison-kd-profile none
  --profile-kd-weight 1.0
  --cmdistill-feature-weight 1.0
  --cmdistill-relation-weight 1.0
  --cmdistill-logit-weight 1.0
  --cmdistill-temperature 4.0
  --cmdistill-max-tokens 512
  --cmdistill-min-confidence 0.05
  --cclkd-base-temperature 2.0
  --cclkd-contrastive-temperature 0.1
  --cclkd-feat-weight 1.0
  --cclkd-logit-weight 1.0
  --cclkd-contrast-weight 0.5
  --cclkd-bg-weight 0.1
  --cclkd-min-confidence 0.1
  --cclkd-max-tokens 512
  --cclkd-temperature-min 0.5
  --cclkd-temperature-max 5.0
  --cclkd-entropy-scale 5.0
  --lr0 0.01
  --lrf 0.01
  --cos-lr
  --optimizer auto
  --warmup-epochs 3.0
  --warmup-bias-lr 0.1
  --save-period 100
  --use-mask
  --use-fg-mask-for-reach
  --ladd-b-a2-core
  --b-detector-source yolo11n.pt
  --b-decomp-source "$B_DECOMP_SOURCE"
  --b-load-student-reachability
  --alpha-s-rec "$ALPHA_S_REC"
)

SMOKE_NAME="smoke_${TAG}_${STAMP}_gpu${GPU}"
SMOKE_LOG="${LOG_ROOT}/${SMOKE_NAME}.log"
echo "[smoke] tag=${TAG} gpu=${GPU} detach_reach=${DETACH_REACH} reach_input=${REACH_INPUT} -> ${SMOKE_LOG}"
"$PY" ladd/code/train_ladd_hbb.py \
  "${COMMON_ARGS[@]}" \
  "${EXTRA_ARGS[@]}" \
  --epochs 1 \
  --fraction 0.02 \
  --phase-min-epochs 1 \
  --device "$GPU" \
  --project "${SMOKE_ROOT}/${TAG}/yolo11n/seed0" \
  --name "$SMOKE_NAME" \
  >"$SMOKE_LOG" 2>&1

RUN_NAME="ogsod_yoloinit_${TAG}_yolo11n_e800_b64_img256_s0_${STAMP}_gpu${GPU}"
CMD="${LOG_ROOT}/${RUN_NAME}.cmd.sh"
LOG="${LOG_ROOT}/${RUN_NAME}.outer.log"
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  printf 'REPO_ROOT=%q\n' "$REPO_ROOT"
  printf 'PY=%q\n' "$PY"
  declare -p COMMON_ARGS
  declare -p EXTRA_ARGS
  cat <<EOF
cd "\$REPO_ROOT"
"\$PY" ladd/code/train_ladd_hbb.py \\
  "\${COMMON_ARGS[@]}" \\
  "\${EXTRA_ARGS[@]}" \\
  --epochs 800 \\
  --device "$GPU" \\
  --project "${PROJ_ROOT}/${TAG}/yolo11n/seed0" \\
  --name "$RUN_NAME"
EOF
} >"$CMD"
chmod +x "$CMD"
nohup bash "$CMD" >"$LOG" 2>&1 &
PID=$!
echo "$PID" >"${LOG_ROOT}/${RUN_NAME}.pid"
echo "[launch] tag=${TAG} gpu=${GPU} pid=${PID} log=${LOG}"
