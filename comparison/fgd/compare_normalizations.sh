#!/usr/bin/env bash
# FGD Loss Scale 对比实验
# 目的：找到合理的loss normalization方案

set -euo pipefail

GPU_ID="${1:-0}"
SEED="${2:-0}"
EPOCHS=20

echo "========================================="
echo "FGD Loss Scale Comparison - 3 Variants"
echo "========================================="
echo ""

# 基础配置
BASE_ARGS=(
    --model yolo11n.pt
    --data shared/configs/datasets_public/ogsod1_sar_detect.yaml
    --teacher-data shared/configs/datasets_public/ogsod1_rgb_detect.yaml
    --teacher-weights runs_public/ogsod/hbb/formal_nomosaic_20260528/baselines/yolo11n/seed0/weights/best.pt
    --imgsz 256
    --epochs $EPOCHS
    --batch 64
    --cos-lr
    --mosaic 0.0
    --close-mosaic 0
    --seed $SEED
    --deterministic
)

# 方案A: 原公式 + 低权重
echo "============================================"
echo "方案A: 原公式 / bsz，α/β降10倍"
echo "============================================"
RUN_A="fgd_compare_A_origFormula_lowWeight_s${SEED}_e${EPOCHS}"

CUDA_VISIBLE_DEVICES=$GPU_ID python ladd/code/train_ladd_hbb.py \
    "${BASE_ARGS[@]}" \
    --comparison-kd-profile fgd \
    --fgd-alpha 0.0001 \
    --fgd-beta 0.00005 \
    --fgd-gamma 0.001 \
    --project runs_public/fgd_compare \
    --name "$RUN_A" \
    --fgd-normalization-mode original

echo "方案A完成: runs_public/fgd_compare/$RUN_A"
echo ""

# 方案B: 除以channels（当前实现）
echo "============================================"
echo "方案B: 除以 (num_pixels × channels)"
echo "============================================"
RUN_B="fgd_compare_B_divideChannels_s${SEED}_e${EPOCHS}"

CUDA_VISIBLE_DEVICES=$GPU_ID python ladd/code/train_ladd_hbb.py \
    "${BASE_ARGS[@]}" \
    --comparison-kd-profile fgd \
    --fgd-alpha 0.001 \
    --fgd-beta 0.0005 \
    --fgd-gamma 0.001 \
    --project runs_public/fgd_compare \
    --name "$RUN_B" \
    --fgd-normalization-mode per_channel

echo "方案B完成: runs_public/fgd_compare/$RUN_B"
echo ""

# 方案C: 只除像素数 + 降权重
echo "============================================"
echo "方案C: 除以 num_pixels，α/β降10倍"
echo "============================================"
RUN_C="fgd_compare_C_dividePixels_lowWeight_s${SEED}_e${EPOCHS}"

CUDA_VISIBLE_DEVICES=$GPU_ID python ladd/code/train_ladd_hbb.py \
    "${BASE_ARGS[@]}" \
    --comparison-kd-profile fgd \
    --fgd-alpha 0.0001 \
    --fgd-beta 0.00005 \
    --fgd-gamma 0.001 \
    --project runs_public/fgd_compare \
    --name "$RUN_C" \
    --fgd-normalization-mode per_pixel

echo "方案C完成: runs_public/fgd_compare/$RUN_C"
echo ""

# 分析对比
echo "========================================="
echo "对比分析"
echo "========================================="
echo ""

for variant in A B C; do
    case $variant in
        A) RUN_NAME="$RUN_A" DESC="原公式+低权重" ;;
        B) RUN_NAME="$RUN_B" DESC="除channels" ;;
        C) RUN_NAME="$RUN_C" DESC="除像素+低权重" ;;
    esac

    CSV="runs_public/fgd_compare/${RUN_NAME}/results.csv"

    if [[ ! -f "$CSV" ]]; then
        echo "⚠️  方案${variant} (${DESC}): 结果文件未找到"
        continue
    fi

    BEST_EPOCH=$(tail -n +2 "$CSV" | awk -F',' '{print NR-1, $8}' | sort -k2 -rn | head -1 | awk '{print $1}')
    BEST_MAP=$(tail -n +2 "$CSV" | awk -F',' '{print $8}' | sort -rn | head -1)
    FINAL_MAP=$(tail -n 1 "$CSV" | awk -F',' '{print $8}')
    EPOCH1_MAP=$(sed -n '3p' "$CSV" | awk -F',' '{print $8}')

    echo "方案${variant} (${DESC}):"
    echo "  Best:  $BEST_MAP @ epoch $BEST_EPOCH"
    echo "  Final: $FINAL_MAP"
    echo "  Ep1:   $EPOCH1_MAP"

    if [[ "$BEST_EPOCH" == "1" ]]; then
        echo "  状态: ❌ 训练崩溃（best在epoch 1）"
    elif (( $(echo "$BEST_MAP < 0.50" | bc -l) )); then
        echo "  状态: ⚠️  性能过低（可能KD过弱）"
    else
        echo "  状态: ✅ 训练正常"
    fi
    echo ""
done

echo "========================================="
echo "建议"
echo "========================================="
echo ""
echo "选择标准："
echo "1. Best不在epoch 1（训练没崩溃）"
echo "2. Best mAP尽量高（KD没削太多）"
echo "3. 训练曲线稳定上升"
echo ""
echo "如果三组都正常，选择Best mAP最高的方案"
echo "如果只有一组正常，就用那个"
echo ""
