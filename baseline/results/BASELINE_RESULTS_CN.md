# OGSOD Baseline 结果

协议：`imgsz=256, 800ep, cos_lr, full no-mosaic, default Albumentations`
数据：OGSOD-1.0 HBB, 3 类 (bridge, harbor, storage_tank)
服务器：90 (8x RTX 3090)

## SAR Baseline

| Model | seed0 | seed42 | seed123 | mean ± std |
|---|---:|---:|---:|---:|
| YOLO11n | 0.55654 | 0.55794 | 0.56128 | 0.55859 ± 0.00244 |
| YOLO11s | 0.62897 | 0.62879 | 0.62357 | 0.62711 ± 0.00307 |
| YOLO11m | 0.65580 | — | — | — |
| YOLO11l | 0.65427 | — | — | — |
| YOLO11x | 0.65867 | — | — | — |

## RGB Baseline (Teacher)

| Model | seed0 | seed42 | seed123 |
|---|---:|---:|---:|
| YOLO11n | 0.63018 | 0.62664 | 0.62933 |
| YOLO11s | 0.65768 | 0.66218 | 0.65987 |
| YOLO11m | 0.67909 | — | — |
| YOLO11l | 0.68356 | — | — |
| YOLO11x | 0.68284 | — | — |

## 关键观察

- n 的 RGB-SAR gap 最大 (~0.074)，最适合机制验证
- s 以上 gap 缩小到 ~0.02-0.03
- 完整结果 CSV 已复制到 `baseline/results/90_formal_nomosaic_20260528/`
