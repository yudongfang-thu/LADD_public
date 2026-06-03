# Baseline

单模态 YOLO11 SAR/RGB 检测 baseline，作为 LADD 和所有对比方法的参照起点。

## 方法

标准 YOLO11 HBB 检测训练，SAR-only 或 RGB-only。使用 OGSOD-1.0 数据集，formal no-mosaic 协议。

协议：`imgsz=256, 800ep, cos_lr, mosaic=0.0, default Albumentations`

## 代码

- `code/train_ogsod_baseline.py` — 训练入口
- `scripts/launch_formal_baseline_job.sh` — 标准启动脚本
- `scripts/run_formal_baseline.sh` — 单次运行脚本

## 结果

详见 `results/BASELINE_RESULTS_CN.md`

关键数据（90 服务器, YOLO11, seed0）：

| Model | SAR AP | RGB AP |
|---|---:|---:|
| YOLO11n | 0.55654 | 0.63018 |
| YOLO11s | 0.62897 | 0.65768 |
| YOLO11m | 0.65580 | 0.67909 |

YOLO11n/s 三 seed 完成。
