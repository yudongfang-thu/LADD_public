# HalluciDet-style

Privileged modality hallucination for SAR object detection.

## 方法

训练期使用 paired RGB/SAR，RGB 作为 privileged information 指导 SAR student；
推理期只输入 SAR。当前 portable profile 使用置信度/前景加权特征对齐、response
map 对齐和 foreground-energy margin，但没有原文式显式 hallucination module。

当前 profile 名称为：

```text
--comparison-kd-profile hallucidet_style
```

旧 `hallucidet` 名称不再被 CLI/launcher 接受，避免被误写成 strict HalluciDet
复现。

因此论文中必须标注：

```text
HalluciDet-style (detection-utility guided feature alignment,
no explicit hallucination module)
```

当前实现没有 image-space hallucination path，也没有 frozen RGB detector
detection-loss-through-hallucinated-image 路径。若后续要做 strict HalluciDet，
应新建独立入口，不复用当前 B-only frozen-teacher comparison launcher。

## 结果

旧 `hallucidet` 结果只能作为 `hallucidet_style_old` 参考，不能作为 HalluciDet
official reproduction。当前 profile 名称变更后需要重跑。
