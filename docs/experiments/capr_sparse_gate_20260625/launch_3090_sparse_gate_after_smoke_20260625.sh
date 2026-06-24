#!/usr/bin/env bash
set -euo pipefail

cd /root/shared-nvme/LADD_public

PY=/root/shared-nvme/venvs/ladd312/bin/python
TRAIN=ladd/code/train_ladd_hbb.py
LOG_DIR=logs/capr_gatedkd_sparse_20260625
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --phase b
  --model yolo11n.pt
  --data configs/paper/datasets/ogsod_hbb_sar.yaml
  --teacher-data configs/paper/datasets/ogsod_hbb_rgb.yaml
  --teacher-weights runs_public/ogsod/hbb/formal_nomosaic_20260528/baselines/rgb/rgb_yolo11n_hbb_800ep_cos_nomosaic_albu_b64_s0/weights/best.pt
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
  --rank-d-neg-cap 2.0
  --reach-input-mode adapter
  --student-detect-mode raw
  --teacher-feature-mode decomposed
  --unlearnable-hidden-ratio 1.0
  --kd-weight-mode cap_reachability_gap
  --kd-reach-use-capped-gap
  --kd-reach-margin 1.79
  --kd-reach-tau 0.02
  --kd-reach-detach-weight
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
  --b-decomp-source runs_public/paper/ogsod_hbb_nomosaic/diagnostics/ladd_dynamic/yolo11n/seed0/ladd_clean_a1b_dyn_ogsod11n_diagnostic_nomosaic_dynamic_yolo11n_s0_a1_e10_b64_s0_gpu0/weights/best.pt
  --b-load-student-reachability
)

echo "[$(date)] Smoke sparse capR-gatedKD margin=1.79 tau=0.02"
"$PY" "$TRAIN" \
  "${COMMON_ARGS[@]}" \
  --epochs 1 \
  --phase-min-epochs 1 \
  --fraction 0.02 \
  --workers 2 \
  --device 0 \
  --project runs_public/ogsod/hbb/capr_gatedkd_smoke_20260625 \
  --name smoke_sparse_gate_m1p79_tau0p02_frac002_20260625_gpu0 \
  --alpha-s-rec 0.1 \
  > "$LOG_DIR/smoke_sparse_gate_m1p79_tau0p02_frac002_20260625_gpu0.log" 2>&1

SMOKE_DIR=runs_public/ogsod/hbb/capr_gatedkd_smoke_20260625/smoke_sparse_gate_m1p79_tau0p02_frac002_20260625_gpu0
test -s "$SMOKE_DIR/results.csv"
test -s "$SMOKE_DIR/ladd_diagnostics.csv"
grep -q "kd_reach_active_ratio" "$SMOKE_DIR/ladd_diagnostics.csv"
grep -E "Traceback|CUDA out of memory|RuntimeError|No space left|batch fallback|fallback" "$LOG_DIR/smoke_sparse_gate_m1p79_tau0p02_frac002_20260625_gpu0.log" && exit 3 || true

launch_run() {
  local gpu="$1"
  local variant="$2"
  local alpha_s_rec="$3"
  shift 3
  local project="runs_public/ogsod/hbb/capr_gatedkd_sparse_20260625/${variant}/yolo11n/seed0"
  local name="ogsod_yoloinit_${variant}_yolo11n_e800_b64_img256_s0_20260625_gpu${gpu}"
  local log="$LOG_DIR/${name}.log"
  local pidfile="$LOG_DIR/${name}.pid"
  mkdir -p "$project"
  if [[ -s "$project/$name/results.csv" ]]; then
    echo "[$(date)] Skip existing $variant at $project/$name"
    return 0
  fi
  echo "[$(date)] Launch $variant on GPU $gpu"
  nohup "$PY" "$TRAIN" \
    "${COMMON_ARGS[@]}" \
    --epochs 800 \
    --device "$gpu" \
    --project "$project" \
    --name "$name" \
    --alpha-s-rec "$alpha_s_rec" \
    "$@" \
    > "$log" 2>&1 < /dev/null &
  echo $! > "$pidfile"
  sleep 5
  if ! kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "Launch failed for $variant; tail log:" >&2
    tail -80 "$log" >&2 || true
    exit 4
  fi
}

launch_run 0 dynamic_capR2_gatedKD_sparseM1p79_tau0p02_yoloinit 0.1
launch_run 1 dynamic_capR2_gatedKD_sparseM1p79_tau0p02_wo_srec_yoloinit 0.0
launch_run 1 dynamic_capR2_gatedKD_sparseM1p79_tau0p02_shuffledT_yoloinit 0.1 --shuffle-teacher-pairs

echo "[$(date)] Sparse gate launches complete"
