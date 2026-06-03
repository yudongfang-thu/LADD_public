# FGD

Focal and Global Knowledge Distillation for Detectors, CVPR 2022.

## 方法

Foreground/background weighted feature KD + batch-level global relation matrix MSE。feature level distillation 经典方法。

## 结果

| Model | best AP | vs SAR baseline |
|---|---|---|
| YOLO11n 800ep | 0.55867 | -0.00049 |

等于 baseline，无正向提升。

完整训练日志见 `results/4090d_formal_kd_20260602/`
