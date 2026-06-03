# HalluciDet-style

Privileged modality hallucination for SAR object detection.

## 方法

训练期使用 paired RGB/SAR，RGB 作为 privileged information 通过 hallucination 模块辅助 SAR 特征学习。推理期只输入 SAR，RGB 分支移除。

## 结果

进行中。YOLO11n seed0 800ep。
