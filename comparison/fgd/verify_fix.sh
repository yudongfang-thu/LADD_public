#!/usr/bin/env bash
# FGD修复验证脚本
# 用途：验证FGD loss normalization修复是否生效

set -euo pipefail

echo "========================================="
echo "FGD Loss Normalization Fix - Smoke Test"
echo "========================================="
echo ""

# 配置
MODEL_SIZE="${1:-n}"
SEED="${2:-0}"
GPU_ID="${3:-0}"
EPOCHS="${4:-20}"

echo "Configuration:"
echo "  Model: YOLO11${MODEL_SIZE}"
echo "  Seed: ${SEED}"
echo "  GPU: ${GPU_ID}"
echo "  Epochs: ${EPOCHS}"
echo ""

# 检查是否有旧的运行
RUN_NAME="fgd_fix_smoke_yolo11${MODEL_SIZE}_s${SEED}_e${EPOCHS}"
RUN_DIR="runs_public/fgd_fix_smoke/${RUN_NAME}"

if [[ -d "$RUN_DIR" ]]; then
    echo "⚠️  Found existing run directory: $RUN_DIR"
    echo "    Remove it to start fresh? (y/N)"
    read -r response
    if [[ "$response" == "y" || "$response" == "Y" ]]; then
        rm -rf "$RUN_DIR"
        echo "    Removed."
    else
        echo "    Keeping existing run."
    fi
    echo ""
fi

# 运行修复后的FGD
echo "Starting FGD smoke test..."
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID python ladd/code/train_ladd_hbb.py \
    --comparison-kd-profile fgd \
    --model yolo11${MODEL_SIZE}.pt \
    --data shared/configs/datasets_public/ogsod1_sar_detect.yaml \
    --teacher-data shared/configs/datasets_public/ogsod1_rgb_detect.yaml \
    --teacher-weights runs_public/ogsod/hbb/formal_nomosaic_20260528/baselines/yolo11${MODEL_SIZE}/seed${SEED}/weights/best.pt \
    --imgsz 256 \
    --epochs $EPOCHS \
    --batch 64 \
    --cos-lr \
    --mosaic 0.0 \
    --close-mosaic 0 \
    --seed $SEED \
    --deterministic \
    --project runs_public/fgd_fix_smoke \
    --name $RUN_NAME

echo ""
echo "========================================="
echo "Smoke test completed!"
echo "========================================="
echo ""

# 分析结果
RESULTS_CSV="${RUN_DIR}/results.csv"

if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "❌ Results file not found: $RESULTS_CSV"
    exit 1
fi

echo "Analyzing results..."
echo ""

# 提取关键指标
BEST_EPOCH=$(tail -n +2 "$RESULTS_CSV" | awk -F',' '{print NR-1, $8}' | sort -k2 -rn | head -1 | awk '{print $1}')
BEST_MAP=$(tail -n +2 "$RESULTS_CSV" | awk -F',' '{print $8}' | sort -rn | head -1)
FINAL_MAP=$(tail -n 1 "$RESULTS_CSV" | awk -F',' '{print $8}')
EPOCH_1_MAP=$(sed -n '3p' "$RESULTS_CSV" | awk -F',' '{print $8}')

echo "📊 Training Summary:"
echo "  ├─ Best mAP:    $BEST_MAP @ epoch $BEST_EPOCH"
echo "  ├─ Final mAP:   $FINAL_MAP @ epoch $EPOCHS"
echo "  └─ Epoch 1 mAP: $EPOCH_1_MAP"
echo ""

# 验证修复
PASS=true

# 检查1: Best不应该在epoch 1
if [[ "$BEST_EPOCH" == "1" ]]; then
    echo "❌ FAIL: Best mAP still at epoch 1 (training collapse not fixed)"
    PASS=false
else
    echo "✅ PASS: Best mAP not at epoch 1"
fi

# 检查2: 应该有上升趋势
if (( $(echo "$FINAL_MAP > $EPOCH_1_MAP" | bc -l) )); then
    echo "✅ PASS: Training curve shows improvement (final > epoch 1)"
else
    echo "⚠️  WARNING: Final mAP not higher than epoch 1"
fi

# 检查3: Best应该在后期
if (( BEST_EPOCH > EPOCHS / 2 )); then
    echo "✅ PASS: Best mAP in later epochs ($BEST_EPOCH > $((EPOCHS / 2)))"
else
    echo "⚠️  WARNING: Best mAP relatively early (epoch $BEST_EPOCH)"
fi

echo ""

if $PASS; then
    echo "========================================="
    echo "✅ FGD FIX VERIFIED - SMOKE TEST PASSED"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "1. Launch formal 800 epoch experiments:"
    echo "   bash comparison/code/launch_formal_transfer_kd_job.sh fgd n 0 <gpu>"
    echo "   bash comparison/code/launch_formal_transfer_kd_job.sh fgd s 0 <gpu>"
    echo ""
    echo "2. Monitor loss components in early epochs to confirm normalization"
    echo ""
    exit 0
else
    echo "========================================="
    echo "❌ FGD FIX NOT VERIFIED - ISSUES FOUND"
    echo "========================================="
    echo ""
    echo "Debug steps:"
    echo "1. Check loss scale in training logs:"
    echo "   grep 'train/kd_loss' $RUN_DIR/train_output.log | head -20"
    echo ""
    echo "2. Compare with detection loss:"
    echo "   grep 'train/box_loss\\|train/cls_loss\\|train/dfl_loss' $RUN_DIR/train_output.log | head -20"
    echo ""
    echo "3. If KD loss still >> det loss, may need further adjustment"
    echo ""
    exit 1
fi
