# 相关工作与推荐阅读

最后更新：2026-06-05

本文档只用于论文定位和阅读顺序，不作为实验计划。当前正式受控对比方法已经收敛为：

```text
FGD / LD / CCLKD / HalluciDet-style
```

其中 CCLKD 需要先在 [`../../cclkd_reproduction/`](../../cclkd_reproduction/) 中完成
原文协议复现，再进入统一受控对比。

## 1. 正式对比方法

| 方法 | 文献 | 角色 | 当前使用方式 |
|---|---|---|---|
| FGD | Focal and Global Knowledge Distillation for Detectors, CVPR 2022 | 通用特征蒸馏代表 | `comparison/fgd/` |
| LD | Localization Distillation for Dense Object Detection, CVPR 2022 / TPAMI 2023 | 定位输出分布蒸馏代表 | `comparison/ld/` |
| CCLKD | Cross-modal contrastive learning-based object detection under incomplete modalities, GIS 2026 | 同任务跨模态代表 | `cclkd_reproduction/` + `comparison/cclkd/` |
| HalluciDet-style | HalluciDet, WACV 2024 | privileged modality / hallucination 思路代表 | `comparison/hallucidet/` |

## 2. 任务定位文献

| 文献 | 为什么读 |
|---|---|
| M4-SAR / E2E-OSDet | 光学-SAR 检测 benchmark 与融合检测上界参考；其推理期使用双模态，部署约束不同 |
| DisCoM-KD | shared/private 或 disentanglement 类机制背景；作为概念邻近文献，不作为当前实验方法 |
| SimKD / UniKD / ReviewKD | LADD 投影、跨注意力、feature review 等机制的通用 KD 背景 |

## 3. 阅读顺序

1. CCLKD：优先核对 OGSOD、YOLO11s、400 epoch、数据增强和 online 训练协议。
2. FGD 与 LD：核对我们当前 YOLO11 HBB 适配是否有可解释边界。
3. HalluciDet：核对 privileged modality 训练期使用 RGB、推理期 SAR-only 的叙事是否准确。
4. M4-SAR 与 DisCoM-KD：用于论文定位和 related work，不进入当前方法表。

## 4. 主表口径

主表只放与当前训练/测试条件可控对齐的方法：

```text
SAR-only baseline
FGD
LD
CCLKD
HalluciDet-style
LADD
```

融合检测、不同 backbone、不同训练协议或只做背景引用的方法不进入 controlled main table。
