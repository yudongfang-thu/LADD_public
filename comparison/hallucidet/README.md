# HalluciDet-style

Privileged modality hallucination for SAR object detection.

## 方法

训练期使用 paired RGB/SAR，RGB 作为 privileged information 指导 SAR student；
推理期只输入 SAR。当前 portable profile 使用置信度/前景加权特征对齐、response
map 对齐和 foreground-energy margin，但没有原文式显式 hallucination module。

因此论文中必须标注：

```text
HalluciDet-style (detection-utility guided feature alignment,
no explicit hallucination module)
```

## 结果

进行中。YOLO11n seed0 800ep。
