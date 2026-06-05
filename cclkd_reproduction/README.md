# CCLKD 原文协议复现

最后更新：2026-06-05

本目录专门用于复现 CCLKD 原文，不属于 `comparison/` 下的受控对比方法目录。

`comparison/cclkd/` 只记录我们论文主表中的 CCLKD 对比实现边界；本目录则必须尽量
对齐 CCLKD 论文自身实验协议。二者不能混用结果。

## 目标

复现论文：

```text
Cross-modal contrastive learning-based object detection under incomplete modalities
GIS 2026
DOI: 10.1080/10095020.2026.2633014
```

论文 PDF 位于 [`paper/`](paper/)。

## 必须对齐的协议

| 项 | 要求 |
|---|---|
| 数据集 | OGSOD-1.0 HBB，类别数必须为 `nc=3` |
| 模态 | 训练期 RGB + SAR，推理期 SAR-only |
| 模型 | YOLO11s 和 YOLO11n |
| 训练方式 | CCLKD 原文定义的 online teacher-student joint training |
| epoch | 与原文一致：400 epoch |
| 数据增强 | 必须逐项按原文/作者设置对齐，不能沿用 LADD formal no-mosaic 协议替代 |
| 指标 | AP50-95、AP50，并记录 best epoch |

## 当前状态

当前 public 中已有 CCLKD loss 级组件，但 frozen RGB teacher 训练不符合 CCLKD
原文定义，不能作为原文复现结果。下一步应先补 online teacher-student trainer，
再做 YOLO11s / YOLO11n 的 400 epoch 复现实验。

## 与受控对比的关系

- `cclkd_reproduction/`：回答“我们是否能按 CCLKD 原文协议复现其方法”。
- `comparison/cclkd/`：回答“在我们论文统一协议下，CCLKD 与 LADD/FGD/LD/HalluciDet 如何比较”。

只有当本目录的原文复现跑通并确认实现可信后，CCLKD 才能进入 `comparison/`
中的正式受控对比。
