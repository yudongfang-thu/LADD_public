# FGD

Focal and Global Knowledge Distillation for Detectors, CVPR 2022.

## 方法

当前修正版同时使用教师特征导出的 spatial/channel attention、GT foreground/background
分离和 batch-level relation matrix MSE。它是便携的 `FGD-style` YOLO port，不是
MMDetection 官方实现的逐行复现。

## 代码位置

FGD 在本 public 包中不是独立训练器，使用统一入口：

```text
../ladd/code/train_ladd_hbb.py --comparison-kd-profile fgd
../ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py::_fgd_style_loss
```

## 历史结果

| Model/seed | 服务器 | epoch | best/current AP50-95 | vs SAR baseline | 状态 |
|---|---|---:|---:|---:|---|
| YOLO11n seed0 | 4090D | 800 | 0.55867 best | -0.00049 | 完成，基本打平 |
| YOLO11n seed42 | 4090D | 343 | 0.46993 current | 未完成 | 运行中 |
| YOLO11n seed123 | 4090 | 134 | 0.37024 current | 未完成 | validation OOM，不计完成 |
| YOLO11s seed0 | 4090 | 192 | 0.50358 current | 未完成 | 运行中 |

这些结果来自 2026-06-04 修复前的旧实现；旧实现只有 GT 二值前景/背景权重，
没有 teacher feature attention。它们保留用于审计，但不能代表当前修正版，也不能
进入修正版主表。当前 FGD 必须重跑。

完整训练日志已归档到
[`../archive/excluded_methods/legacy_results/fgd_pre_20260604/`](../archive/excluded_methods/legacy_results/fgd_pre_20260604/)。
