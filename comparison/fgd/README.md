# FGD

Focal and Global Knowledge Distillation for Detectors, CVPR 2022.

## 方法

当前锁定版 `locked_fgd_yolo_gtbox_attention_20260618` 使用教师/学生特征导出的 spatial/channel attention、GT-box
foreground/background mask、fg/bg feature loss 和 attention mask loss。它是
便携的 `FGD-style` / FGD-YOLO adaptation，不是 MMDetection 官方实现的逐行复现。

空间与通道 attention 按 FGD 官方代码均使用 softmax，并分别乘 `H*W` 和 `C`；
temperature 固定为 0.5。当前没有实现官方的可训练 Global KD context 模块；
旧 batch-level relation、assigner-mask fallback 和 normalization sweep 已从 active
代码面移除，因此仍必须标记 `FGD-style`。

固定内部常量：

```text
alpha = 0.0001
beta = 0.00005
gamma = 0.001
temperature = 0.5
mask_mode = gt_box
relation = removed
```

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
| YOLO11n seed42 | 4090D | 343 | 0.46993 current | 未完成 | 历史 partial，当前不视为运行中 |
| YOLO11n seed123 | 4090 | 134 | 0.37024 current | 未完成 | validation OOM，不计完成 |
| YOLO11s seed0 | 4090 | 192 | 0.50358 current | 未完成 | 历史 partial，当前不视为运行中 |

这些结果来自锁定前的旧实现或修复期实验。它们仅作历史说明，不能代表当前
`locked_fgd_yolo_gtbox_attention_20260618`，也不能进入当前主表。

完整训练日志和旧 run 目录已从精简 public 分支移除，仅在历史 Git commit 中可追溯。
