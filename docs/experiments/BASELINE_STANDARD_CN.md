# OGSOD Baseline 训练规范与当前状态

> ⚠️ ARCHIVED DIAGNOSTIC NOTE
> This document records the older formal no-mosaic baseline protocol and historical status.
> It is not the source of paper main-table results.
> Use `paper_results/` and `docs/paper/PAPER_PROTOCOL_CN.md` for paper-facing results.

最后更新：2026-06-02

本文档是 OGSOD 正式 baseline 的统一规范、当前状态和跨机器对账。后续所有 LADD 主线、消融和对比方法实验都按本规范启动。

## 1. 正式协议

```text
OGSOD-1.0 HBB
+ imgsz=256
+ 800 epoch
+ cos_lr (lr0=0.01, lrf=0.01)
+ 全程关闭 mosaic (mosaic=0.0, close_mosaic=0)
+ 保留 Ultralytics 默认 Albumentations
+ 关闭 HSV/MixUp/CutMix/Erasing
+ 几何增强: translate=0.1, scale=0.5, fliplr=0.5
```

背离以下任何一项都视为非标准协议，必须标注：

```text
mosaic != 0.0, close_mosaic != 0, --disable-albumentations
mixup > 0, cutmix > 0, erasing > 0, cos_lr != True
epochs != 800, imgsz != 256
```

## 2. 模型容量与固定 Batch

| 模型 | 参数量 | Batch | 用途 |
|---|---:|---:|---|
| YOLO11n | 2.62M | 64 | 主机制实验，gap 最大，多 seed 优先 |
| YOLO11s | 9.46M | 64 | 小/中容量趋势验证 |
| YOLO11m | 20.11M | 32 | 中容量趋势验证 |
| YOLO11l | 25.37M | 32 | 大容量趋势补充 |
| YOLO11x | 56.97M | 16 | 最大容量实验 |

## 3. Seed 与配对

正式 seed：`0, 42, 123`

配对规则：`SAR(size, seed k) + RGB(size, seed k) -> LADD(size, seed k)`

## 4. 标准命令模板

SAR baseline（RGB 只替换 `--data` 和 `--project/--name`）：

```bash
python3 tools/train_ogsod_baseline.py \
  --task hbb --model yolo11n.pt \
  --data configs/datasets/ogsod_hbb_sar.yaml \
  --imgsz 256 --epochs 800 --batch 64 --workers 8 \
  --device $GPU_ID --patience 800 \
  --project runs_public/ogsod/hbb/formal_nomosaic_20260528/baselines/sar \
  --name sar_yolo11n_hbb_800ep_cos_nomosaic_albu_b64_s0 \
  --lr0 0.01 --lrf 0.01 --cos-lr \
  --mosaic 0.0 --close-mosaic 0 \
  --translate 0.1 --scale 0.5 --fliplr 0.5 \
  --flipud 0.0 --degrees 0.0 --perspective 0.0 \
  --hsv-h 0.0 --hsv-s 0.0 --hsv-v 0.0 \
  --mixup 0.0 --cutmix 0.0 --erasing 0.0 \
  --save-period 100 --seed 0 --deterministic
```

## 5. 当前 Baseline 结果 (90 服务器 / 3090)

### 5.1 Seed0 容量趋势

| Model | SAR best AP | RGB best AP | Gap |
|---|---:|---:|---:|
| YOLO11n | 0.55654 | 0.63018 | 0.07364 |
| YOLO11s | 0.62897 | 0.65768 | 0.02871 |
| YOLO11m | 0.65580 | 0.67909 | 0.02329 |
| YOLO11l | 0.65427 | 0.68356 | 0.02929 |
| YOLO11x | 0.65867 | 0.68284 | 0.02417 |

- `n` gap 最大，适合主机制实验
- `s` 之后 gap 约 0.02-0.03，用于容量趋势验证

### 5.2 n/s 多 Seed

| Model | Seed | SAR AP | RGB AP |
|---|---:|---:|---:|
| YOLO11n | 0 | 0.55654 | 0.63018 |
| YOLO11n | 42 | 0.55794 | 0.62664 |
| YOLO11n | 123 | 0.56128 | 0.62933 |
| YOLO11s | 0 | 0.62897 | 0.65768 |
| YOLO11s | 42 | 0.62879 | 0.66218 |
| YOLO11s | 123 | 0.62357 | 0.65987 |

### 5.3 跨机器一致性

3090（90 服务器）是 primary reference。其他机器上同协议 baseline 应与 3090 结果对比，delta ≤ 0.005 AP 视为一致。论文主表以 3090 为准。

## 6. 与 LADD 的关系

LADD 主线必须建立在同容量、同 seed 的 SAR/RGB baseline 上。具体主线设置见 [LADD_MAINLINE_STANDARD_CN.md](LADD_MAINLINE_STANDARD_CN.md)。

```text
同容量 SAR baseline + 同容量 RGB baseline + 同 seed
+ no-mosaic baseline protocol
+ A2 检测稳定修正 + cap2
```
