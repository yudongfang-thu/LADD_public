# DSN Shared-Private Cross-Modal Distillation 调研记录

日期：2026-06-23

## 问题

拟探索的方法：

1. 加载两个已收敛单模态 detector，例如 RGB/IR 或 RGB/SAR。
2. 冻结 detector backbone / neck，训练轻量 shared-private projector：
   - `F_a -> z_a(shared) + p_a(private)`
   - `F_b -> z_b(shared) + p_b(private)`
3. 用 contrastive / VICReg / Barlow / Gram alignment 等约束拉近 `z_a, z_b`，同时约束 shared-private 解耦。
4. 后续把 frozen shared projector 作为单模态 student 的蒸馏 target，配合原 LADD student branch / residual branch 结构。

核心问题：这个方向是否已有强相似方法。

## 推荐使用的 Skills

| skill | 用途 | 本轮建议 |
|---|---|---|
| `research-lit` | 系统检索 related work，按方法簇总结 | 主用 |
| `novelty-check` | 对具体方法 claim 查新，找最近 6 个月冲突工作 | 主用，但在代码前再跑一次更完整版本 |
| `idea-discovery` | 完整方向发现流水线：lit -> idea -> novelty -> review -> refine | 暂不全量跑；当前问题已足够具体，全量流程会扩大范围 |
| `research-refine` / `experiment-plan` | 如果决定转主线，用于把方法和实验计划固化 | 下一步使用 |

## 最相关方法簇

### 1. Domain Separation Networks / shared-private 表征

- Domain Separation Networks, NeurIPS 2016.
- 典型结构就是 shared/private subspace + reconstruction + domain confusion / separation。
- 与本想法高度相关，但它主要是 domain adaptation，不是 detector KD，也不是 SAR/RGB paired detection。

差异化空间：我们不是从头训练 domain adaptation，而是从已收敛双模态 detector 中抽取 object-aware shared feature，再作为蒸馏 target。

### 2. Missing modality / shared-specific feature learning

- ShaSpec, CVPR 2023: missing modality 场景下学习 shared/specific features。
- 目标是多模态任务在缺模态情况下仍可工作，机制上与 shared-private 很接近。

风险：reviewer 可能认为“shared-specific decomposition for missing modality”是已有范式。

差异化空间：检测任务，paired remote sensing modality，frozen converged detectors，student-only inference，且以 det-only reload control 证明不是继续训练收益。

### 3. Cross-modal KD + disentanglement

- CroDiNo-KD, ECML-PKDD 2025: RGBD semantic segmentation 中用 disentanglement + contrastive learning，强调重新思考传统 teacher/student CMKD。
- DisCoM / frequency-decoupled CMKD, 2026: 明确指出 cross-modal features 同时包含 modality-generic 与 modality-specific 信息，低频更 shared，高频更 private，并用差异化 distillation 处理。

风险：这是最接近我们“先分 shared/private 再蒸馏”的方法簇。

差异化空间：它们主要是 RGBD/分类/分割/通用 CMKD；我们可以把贡献压到 object-level detector feature、SAR/RGB 或 IR/RGB paired detection、以及 frozen detector feature-bank/projector 预训练。

### 4. RGB-T / visible-infrared object detection KD

- CMDistill, IEEE J-STARS 2024: 两阶段 cross-modal detector distillation，frozen teacher，feature / relation / output 三层 distillation。
- CMKD-net, IEEE TCSVT 2026: modality-missing visible-infrared oriented object detection，用 multimodal teacher 指导 incomplete-modality student，含 multi-dimensional feature distillation 与 inter-instance relation distillation。
- CCLKD, Geo-spatial Information Science 2026: easy/hard modality 双分支，contrastive learning + KD，用于 incomplete modality object detection。

风险：DroneVehicle/VEDAI 上已有 visible-infrared missing-modality KD 直接竞争。

差异化空间：这些方法大多直接蒸馏 teacher/multimodal feature 或 relation；我们需要强调“先从两个收敛单模态 detector 中提炼 validated shared latent，再只蒸馏 shared latent，显式保留 student private/residual”。

### 5. Optical-SAR guided SAR object detection

- CoLD, IEEE TGRS 2023: optical teacher 指导 SAR detector，重点在 localization / category-oriented partition。
- GaLD, ICASSP 2025: optical-guided SAR OBB，用 Gaussian localization distillation 对齐角度信息。
- FED-CHDistill, IJRS 2025: frequency enhancement + dynamic mask + cross-head distillation，用于 optical-guided SAR object detection。

风险：在 OGSOD / optical-SAR detection 语境里，KD 已经很拥挤。

差异化空间：已有 SAR 方向多集中在 localization / prediction / mask / frequency / head-level distillation，不是 DSN-style shared-private latent extraction；但我们必须承认它们是直接相关 baseline。

## 初步新颖性判断

不是“完全没人做过”。更准确的定位应是：

> Existing CMKD either distills raw cross-modal teacher features or relation/logit cues, while shared-private methods often target domain adaptation, missing-modality classification/segmentation, or multimodal fusion. We study whether frozen converged modality-specific detectors can be used to extract a detector-aware cross-modal shared latent, and whether distilling only this validated shared latent improves single-modality detection beyond detector-only continued training.

这条线有新颖性空间，但必须避开两个坑：

1. 不能只说 DSN / shared-private，这是已有范式。
2. 不能只说 cross-modal KD，这是已有范式。

真正可讲的点应是三件事的交集：

1. **frozen converged dual-detector latent extraction**；
2. **object-aware shared latent validation**，例如 retrieval / modality classifier / foreground linear probe；
3. **reload-control-safe student distillation**，必须超过 det-only continued training 和 shuffled-pair distill。

## 最小实验建议

先在 DroneVehicle sub2k 做小闭环：

1. Stage S0: 训练 / 使用同子集 RGB student baseline 与 IR teacher baseline。
2. Stage S1: freeze 两个 detector，训练 shared-private projector。
3. Stage S1 sanity:
   - paired retrieval: `z_rgb -> z_ir`
   - modality classifier: shared `z` 是否难分模态
   - foreground/object linear probe: shared `z` 是否保留类别/目标性
   - shuffled pair control: 打乱配对后 shared 对齐应失效
4. Stage S2: 单模态 student distill shared latent。
5. 必跑对照：
   - det-only continued training
   - raw feature KD
   - CMDistill-style
   - shared-only without private/decorrelation
   - shuffled-pair shared distill

## 当前实现状态

已实现并同步到 `ladd4090-zw1`：

```text
tools/train_dsn_shared_private_projector.py
```

当前脚本做的是 Stage S1：冻结两个已收敛 YOLO11n detector，从 YOLO neck/head feature 层抽全局池化 embedding，训练 RGB 与 IR 两侧 shared/private projector。损失包含 symmetric contrastive、shared cosine alignment、feature reconstruction、shared/private separation。

当前启动的 DroneVehicle sub2k run：

```text
runs_public/cross_dataset/dsn_shared_private/dronevehicle_sub2k_seed0/dronevehicle_sub2k_rgb_ir_dsn_s1_e80_b32_ld256_h512_seed0_20260623_2304
```

2026-06-23 23:08 CST 快照：

| epoch | train loss | batch top1 | val top1 | val top5 | val top10 |
|---:|---:|---:|---:|---:|---:|
| 6 | `1.58760` | `0.46875` | `0.02995` | `0.12117` | `0.18993` |
| 8 | `1.34805` | `0.55544` | `0.04221` | `0.17631` | `0.27025` |
| 10 | `1.17173` | `0.63659` | `0.05922` | `0.24166` | `0.36828` |

初步判断：S1 已经不是纯噪声，shared latent 至少包含可检索的跨模态配对信息。是否能转化成 detector mAP 仍需 S2 验证。

## Reload-safe student distillation 的含义

这里的 `reload-safe` 不是新模块名，而是一条实验约束：S2 学生蒸馏必须和 detector-only continued-training 放在同初始化、同数据、同 schedule、同 epoch 下比较。

可接受的最小判定是：

```text
student + shared-latent distill
>
same student checkpoint + detector-only continued training
>
original baseline checkpoint
```

并且必须补 shuffled-pair shared distill。如果 shuffled-pair 也涨点，说明提升仍可能来自 reload / continued training / 正则化，而不是来自跨模态 shared latent。

## 主要参考

- Domain Separation Networks, NeurIPS 2016: https://arxiv.org/abs/1608.06019
- ShaSpec, CVPR 2023: https://arxiv.org/abs/2307.14126
- CroDiNo-KD, ECML-PKDD 2025: https://arxiv.org/abs/2505.24361
- Distilling Cross-Modal Knowledge via Feature Disentanglement, 2026: https://arxiv.org/html/2511.19887v1
- CMDistill, IEEE J-STARS 2024: https://ieeexplore.ieee.org/iel8/4609443/4609444/10715640.pdf
- CoLD, IEEE TGRS 2023: https://ieeexplore.ieee.org/iel7/36/10006360/10168956.pdf
- GaLD, ICASSP 2025: https://ieeexplore.ieee.org/iel8/10887540/10887541/10889285.pdf
- CMKD-net, IEEE TCSVT 2026: DOI 10.1109/TCSVT.2026.3670458
- CCLKD, Geo-spatial Information Science 2026: https://www.tandfonline.com/doi/abs/10.1080/10095020.2026.2633014
