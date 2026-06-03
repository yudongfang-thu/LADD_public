# 对比方法记录

最后更新：2026-06-03 23:21 CST

用途：给导师快速说明本项目主表为什么选这些对比方法、它们来自哪里、核心思想是什么，以及当前实验进度。本文档只记录方法来源和实验口径；baseline 与 LADD 主方法进度见 [`BASELINE_LADD_STATUS_CN.md`](docs/experiments/BASELINE_LADD_STATUS_CN.md)。

## 1. 当前主表方法

主表保留 5 个对比方法：3 个通用检测 KD，2 个跨模态 / privileged-modality 方法。所有非 CoLD 方法按统一 `from-yolo-pretrain` 正式协议训练：student 从 `yolo11*.pt` 启动，RGB teacher 使用同容量同 seed 的 RGB baseline，测试阶段为 SAR-only。

| 类别 | 方法 | 来源 | DOI | 本项目定位 | 当前状态 |
|---|---|---|---|---|---|
| 通用 KD | FGD | CVPR 2022, Focal and Global Knowledge Distillation for Detectors | [`10.1109/CVPR52688.2022.00460`](https://doi.org/10.1109/CVPR52688.2022.00460) | 前景/背景分离的 feature KD，加全局关系蒸馏 | YOLO11n seed0 已完成；seed42 在 4090D 跑 |
| 通用 KD | CrossKD-style | CVPR 2024, CrossKD: Cross-Head Knowledge Distillation for Object Detection | [`10.1109/CVPR52733.2024.01563`](https://doi.org/10.1109/CVPR52733.2024.01563) | student feature 送入 teacher head，做跨检测头蒸馏 | YOLO11n seed0 已完成 |
| 通用 KD | LD | CVPR 2022, Localization Distillation for Dense Object Detection | [`10.1109/CVPR52688.2022.00919`](https://doi.org/10.1109/CVPR52688.2022.00919) | 对检测框定位分布做 logit / localization KD，计算量低 | YOLO11n/s seed0 正在 90 跑 |
| 跨模态 KD | CoLD | IEEE TGRS 2023, Category-Oriented Localization Distillation for SAR Object Detection and a Unified Benchmark | [`10.1109/TGRS.2023.3291356`](https://doi.org/10.1109/TGRS.2023.3291356) | OGSOD 原始 optical-to-SAR KD anchor，必须保留 | 单独作为 CoLD 复现线，不占非 CoLD 主队列 |
| 跨模态 / privileged modality | HalluciDet-style | WACV 2024, HalluciDet: Hallucinating RGB Modality for Person Detection Through Privileged Information | [`10.1109/WACV57701.2024.00147`](https://doi.org/10.1109/WACV57701.2024.00147) | 训练期使用 RGB privileged information，推理期只保留 SAR detector | YOLO11n/s seed0 正在 90 跑 |

## 2. 方法简述

**FGD** 是普通检测 KD 中非常经典的一类 feature distillation。它不把整张特征图同等看待，而是用检测监督把前景、背景和全局关系分开：前景区域强调目标相关特征，背景区域保留上下文，全局项约束不同位置之间的关系。它适合作为“强但通用”的检测 KD baseline，因为它不依赖 RGB-SAR 配对结构，能回答普通 feature KD 是否已经足够。

**CrossKD-style** 来自 CrossKD 的核心思想：不只在 student 自己的 head 上蒸馏，而是让 student 的特征经过 teacher head，再对齐 teacher 的预测行为。这样可以减少 teacher/student head 结构差异导致的监督错位。本项目实现为 YOLO11 HBB 适配版，因此记录为 `CrossKD-style`，不声称逐行复现官方实现。

**LD** 是定位蒸馏方法，重点不是蒸馏分类概率，而是蒸馏 dense detector 中边框定位的不确定性/分布信息。对我们有两个价值：第一，它计算量轻，适合多 seed；第二，它代表“只蒸馏检测输出空间”的保守 baseline，可以和 LADD 的特征分解式蒸馏形成对照。

**CoLD** 是最直接的 OGSOD 对比方法。它以 optical teacher 指导 SAR student，使用类别导向的候选区域划分和定位蒸馏，是原始 OGSOD benchmark 的核心 optical-to-SAR KD anchor。问题是公开仓库更像数据/benchmark 入口，完整训练代码复现性弱，所以本项目把 CoLD 独立为“尽力对齐原文”的复现线，不和非 CoLD 统一队列混跑。

**HalluciDet-style** 选择为第二个跨模态方法。原文任务是训练期使用 RGB privileged information 来增强 IR-only person detector，测试时不需要 RGB。这个假设和我们“训练期 RGB/SAR 配对，测试期 SAR-only”非常接近。本项目只迁移其核心思想：训练时用 RGB teacher/privileged branch 监督 SAR student 的中间表征，推理时移除 RGB 分支，因此命名为 `HalluciDet-style privileged modality hallucination`。

## 3. 当前实验进度

| 方法 | 容量/seed | 服务器 | epoch | 当前 AP50-95 | best AP50-95 | 备注 |
|---|---|---:|---:|---:|---:|---|
| FGD | YOLO11n seed0 | 4090D 归档 | 800 | 0.55514 | 0.55867@749 | 已完成，低于同 seed SAR baseline 0.55654 的幅度很小 |
| CrossKD-style | YOLO11n seed0 | 4090D 归档 | 800 | 0.55670 | 0.55764@737 | 已完成，基本与 baseline 持平 |
| FGD | YOLO11n seed42 | 4090D | 35 | 0.24907 | 0.24907@35 | 正在跑，早期爬升中 |
| LD | YOLO11n seed0 | 90 | 227 | 0.42276 | 0.42276@227 | 正在跑 |
| LD | YOLO11s seed0 | 90 | 257 | 0.54078 | 0.54078@257 | 正在跑 |
| HalluciDet-style | YOLO11n seed0 | 90 | 52 | 0.31281 | 0.31281@52 | 正在跑 |
| HalluciDet-style | YOLO11s seed0 | 90 | 52 | 0.38261 | 0.38261@52 | 正在跑 |

## 4. 选择逻辑

当前组合的优点是覆盖了三种不同对照：

1. FGD/CrossKD-style：强通用检测 KD，检验“普通检测 KD 是否足够”。
2. LD：轻量输出空间蒸馏，检验“只做 localization/logit KD 是否足够”。
3. CoLD/HalluciDet-style：跨模态或 privileged-modality 方法，检验“已有 RGB-to-SAR / train-RGB test-SAR 方案是否足够”。

目前已经完成的 FGD 和 CrossKD-style 在 YOLO11n seed0 上没有明显超过 SAR baseline，这反而有利于说明：在 formal no-mosaic、训练到收敛的设置下，普通 KD 不容易直接吃到 RGB teacher 的收益。后续关键是补齐 LD/HalluciDet-style 的 seed0 完整结果，并把 YOLO11n 的非 CoLD 方法逐步扩到 3 seed。
