# CrossKD-style

CrossKD: Cross-Head Knowledge Distillation for Object Detection, CVPR 2024.

## 方法

Prediction-level KD：student head 读取 teacher intermediate features，在 prediction 空间做蒸馏。当前为 style port 而非官方逐行复现。

## 结果

| Model | best AP | vs SAR baseline |
|---|---|---|
| YOLO11n 800ep | 0.55764 | -0.00152 |

等于 baseline，无正向提升。
