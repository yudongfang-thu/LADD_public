# CCLKD 复现实验阶段汇报

更新时间：2026-06-12
对象：OGSOD-1.0 HBB，SAR student / RGB teacher，CCLKD 对比方法复现与排查。

## 1. 当前目标

我们围绕 CCLKD 做了两条线：

1. **YOLO11n 适配版 CCLKD**：在 LADD 统一训练协议下，验证 CCLKD 各组件在 YOLO11 检测头上的有效性。
2. **YOLOv5x 复现实验**：回到更接近 CCLKD 原文的 YOLOv5 系列，检查 online teacher-student、ATKD、CCL 等机制是否能在 OGSOD 上正常工作，并和标准 YOLOv5 `train.py` baseline 对齐。

当前最重要的结论是：

- YOLO11n 版本已经形成完整 400 epoch 消融证据，所有 CCLKD 组件组合相对 SAR baseline 都有正增益，但增益较小，且 full CCLKD 不是最强。
- YOLOv5x 版本已经完成标准 YOLOv5 `train.py` baseline 和四组自定义 trainer 80 epoch 控制实验。标准 baseline 明显强于自定义 trainer baseline，因此 YOLOv5x 复现的核心问题已经从“KD 是否有效”推进到“自定义 online trainer 是否与标准 YOLOv5 训练行为对齐”。

## 2. CCLKD 原文参照结果

原文 OGSOD-1.0 协议：`imgsz=256`，SGD，400 epoch，batch 32，lr 0.01，momentum 0.937，并使用 standard image augmentation 和 MixUp-based image mixing。

### 2.1 原文主结果

| 来源 | 方法 | AP50 (%) | AP (%) | 说明 |
|---|---|---:|---:|---|
| Table 5 | YOLOv5 baseline | 80.9 | 46.3 | YOLOv5 复现 gate 的 baseline 目标 |
| Table 5 | CCLKD | 88.7 | 57.3 | YOLOv5 口径 full CCLKD 目标 |
| Table 8 | CCLKD + YOLO11n | 86.8 | 53.7 | 原文给出的 YOLO11n extension 结果 |
| Table 8 | CCLKD + YOLO11s | 87.5 | 55.1 | 原文给出的 YOLO11s extension 结果 |

注意：原文 YOLO11 extension 表没有给出 YOLO11n/s 的 SAR-only baseline，因此 YOLO11n 结果只能和原文 extension 的绝对值作参照，不能严格计算同表增益。

### 2.2 原文 Table 12 组件消融

| 配置 | AP50 (%) | AP (%) |
|---|---:|---:|
| Baseline | 80.9 | 46.3 |
| LLD | 83.4 | 48.5 |
| LLD+FLD | 84.2 | 49.3 |
| LLD+FLD+RLD | 84.9 | 50.1 |
| LLD+FLD+RLD+PATM / ATKD | 87.0 | 55.1 |
| CCL only | 85.9 | 54.4 |
| Full CCLKD | 88.7 | 57.3 |

## 3. YOLO11n CCLKD 适配实验

### 3.1 协议

- 模型：YOLO11n
- 协议：LADD baseline protocol
- 输入：`imgsz=256`
- 训练：400 epoch，batch 64，seed 0
- Baseline：SAR YOLO11n 400ep reference
- Baseline 指标：AP50 = 0.77497，AP50-95 = 0.51546
- CCLKD 配置：`formulation=paper`，`ccl_mode=paper_pair`，`ccl_source=box_distribution`，`rld_mode=paper_instance`

来源：

- `cclkd_reproduction/experiment_versions/v3_paper_pair_boxdist_20260609/FINAL_ANALYSIS_20260610_CN.md`
- `cclkd_reproduction/experiment_versions/v3_paper_pair_boxdist_20260609/metrics_summary.csv`

### 3.2 完整 400 epoch 结果

| rank | variant | epoch | AP50 | AP50-95 | vs SAR baseline AP50-95 |
|---:|---|---:|---:|---:|---:|
| 1 | `lld_fld` | 400 | 0.79701 | 0.53073 | +0.01527 |
| 2 | `lld_fld_rld` | 400 | 0.79541 | 0.52793 | +0.01247 |
| 3 | `lld` | 400 | 0.79417 | 0.52762 | +0.01216 |
| 4 | `full` | 400 | 0.78764 | 0.52531 | +0.00985 |
| 5 | `atkd` | 400 | 0.78892 | 0.52487 | +0.00941 |
| 6 | `ccl_only` | 400 | 0.78560 | 0.52055 | +0.00509 |
| - | SAR baseline | 400 | 0.77497 | 0.51546 | - |

### 3.3 与原文 Table 12 的对齐程度

下表把我们的 YOLO11n v3 结果转成百分制，便于和原文 Table 12 直接看差距。需要注意：这不是严格同架构复现，原文主消融是 YOLOv5 口径；这里用于说明 YOLO11n 适配版与原文目标的距离。

| 配置 | 原文 AP50/AP (%) | 我们 YOLO11n AP50/AP (%) | AP50 差距 | AP 差距 |
|---|---:|---:|---:|---:|
| Baseline | 80.9 / 46.3 | 77.50 / 51.55 | -3.40 | +5.25 |
| LLD | 83.4 / 48.5 | 79.42 / 52.76 | -3.98 | +4.26 |
| LLD+FLD | 84.2 / 49.3 | 79.70 / 53.07 | -4.50 | +3.77 |
| LLD+FLD+RLD | 84.9 / 50.1 | 79.54 / 52.79 | -5.36 | +2.69 |
| ATKD | 87.0 / 55.1 | 78.89 / 52.49 | -8.11 | -2.61 |
| CCL only | 85.9 / 54.4 | 78.56 / 52.06 | -7.34 | -2.34 |
| Full CCLKD | 88.7 / 57.3 | 78.76 / 52.53 | -9.94 | -4.77 |

和原文 Table 8 的 YOLO11n extension 相比：

| 对比项 | 原文 CCLKD+YOLO11n AP50/AP (%) | 我们 AP50/AP (%) | AP50 差距 | AP 差距 |
|---|---:|---:|---:|---:|
| YOLO11n full | 86.8 / 53.7 | 78.76 / 52.53 | -8.04 | -1.17 |
| YOLO11n best observed (`lld_fld`) | 86.8 / 53.7 | 79.70 / 53.07 | -7.10 | -0.63 |

这说明一个比较微妙但重要的现象：我们的 YOLO11n AP50-95/AP 已经接近原文 YOLO11n extension 的 AP，但 AP50 明显低约 7-8 个点；因此问题不只是“整体 AP 低”，还可能涉及评估口径、类别分布、置信度/召回、增强或检测头适配后的 AP50 行为差异。

### 3.4 阶段判断

1. **YOLO11n CCLKD 组件不是完全无效**
   六组消融全部相对 SAR baseline 正增益，说明 CCLKD 的定位/特征蒸馏方向在当前数据集上有可观测作用。

2. **最佳组合不是 full CCLKD**
   `lld_fld` 最强，AP50-95 = 0.53073；`full` 只有 0.52531，低于 `lld_fld` 0.00542。这说明在 YOLO11n 适配版里，RLD/CCL 的额外组合没有形成干净叠加收益。

3. **CCL 有小正贡献，但互补性弱**
   `ccl_only` 相比 baseline 提升 +0.00509 AP50-95，方向不是负的；但 `full` 相比 `atkd` 只高 +0.00045，说明 CCL 与 ATKD 组合后的边际收益很小。

4. **这条线适合作为“YOLO11 适配诊断”，不应直接声称复现原文 YOLOv5x 结果**
   原文 CCLKD 基于 YOLOv5 风格检测头。YOLO11n 结果能证明我们对组件做了可运行适配，但不能替代 YOLOv5x 复现实验。

### 3.5 YOLO11n 实现版本演进说明

这里有一个容易混淆的中间版本：在最早实现之后，我们确实修过一版 RLD/PATM 对齐实现，并且当时只补了三组关键消融。这个版本保存在：

- `cclkd_reproduction/experiment_versions/v2_rld_patmfix/`

当时的三组运行快照如下：

| version | run | status | epoch | AP50 | AP50-95 | 说明 |
|---|---|---|---:|---:|---:|---|
| `v2_rld_patmfix` | `lld_fld_rld_rldpatmfix` | running snapshot | 146 | 0.64675 | 0.39554 | 三组中最高 |
| `v2_rld_patmfix` | `atkd_rldpatmfix` | running snapshot | 146 | 0.64655 | 0.39096 | ATKD-only |
| `v2_rld_patmfix` | `full_rldpatmfix` | running snapshot | 150 | 0.64389 | 0.39237 | 比 ATKD-only 高，但低于 LLD+FLD+RLD |

因此更准确的表述是：

- **是的**，后面确实有一版实现修正，只补了三组关键消融。
- 这版里 **full 比 ATKD-only 略好**，AP50-95 为 `0.39237` vs `0.39096`。
- 但这版里 full **不是三组最高**，最高是 `lld_fld_rld_rldpatmfix`，AP50-95 为 `0.39554`。
- 这版只是中间运行快照，后来被 `v3_paper_pair_boxdist_20260609` 替代；最终用于汇报主表的版本仍应是 v3 六组完整 400 epoch 结果。

## 4. YOLOv5x CCLKD 复现实验

### 4.1 已完成与正在运行的实验

当前 YOLOv5x 使用 batch 32、80 epoch、seed 0。标准 YOLOv5 `train.py` baseline 和四组自定义 trainer 控制实验均已完成。

来源：

- 最终归档：`cclkd_reproduction/yolov5_sanity/results/diagnostics_20260612_yolov5x_b32_e80_final/`
- 已上传归档：`cclkd_reproduction/yolov5_sanity/results/diagnostics_20260612_yolov5x_b32_e80_queue/`
- 当前 90 服务器运行目录：`/mnt/dataY/ydf/projects/LADD_public/cclkd_reproduction/yolov5_sanity/results/runs/`

| run | trainer | status | epoch | AP50 | AP50-95 | 说明 |
|---|---|---|---:|---:|---:|---|
| `standard_train_py` | 标准 YOLOv5 `train.py` | completed | 79/79 | 0.57056 | 0.30964 | 标准 SAR baseline |
| `det_only_same_trainer` | CCLKD 自定义 trainer，单分支检测 | completed | 79/79 | 0.33064 | 0.13490 | 自定义 trainer baseline |
| `two_branch_no_kd` | 自定义 trainer，student+teacher 双分支，无 KD | completed | 79/79 | 0.32165 | 0.13479 | 检查双分支本身是否导致崩溃 |
| `paper_atkd_only` | 自定义 trainer，ATKD/LLD+FLD+RLD，无 CCL | completed | 79/79 | 0.35592 | 0.15149 | 自定义 trainer 中最高 |
| `paper_full` | 自定义 trainer，ATKD + CCL | completed | 79/79 | 0.34732 | 0.14520 | 高于 det-only，但低于 ATKD-only |

注：以上均为 epoch 79 final / best 指标。

### 4.2 与原文 YOLOv5 目标的距离

YOLOv5x 这条线的真正对齐目标是原文 Table 5 / Table 12 的 YOLOv5 口径：

| 项 | 原文目标 AP50/AP (%) | 我们当前 AP50/AP (%) | 当前差距 | 备注 |
|---|---:|---:|---:|---|
| YOLOv5 baseline gate | 80.9 / 46.3 | 标准 `train.py` epoch79：57.06 / 30.96 | -23.84 / -15.34 | 80ep 结果；原文是 400ep |
| 自定义 trainer baseline | 80.9 / 46.3 | `det_only_same_trainer` epoch79：33.06 / 13.49 | -47.84 / -32.81 | 说明自定义 trainer 尚未对齐标准 YOLOv5 |
| Full CCLKD | 88.7 / 57.3 | `paper_full` epoch79：34.73 / 14.52 | -53.97 / -42.78 | 当前只能作机制检查，不能作复现结论 |
| ATKD | 87.0 / 55.1 | `paper_atkd_only` epoch79：35.59 / 15.15 | -51.41 / -39.95 | 自定义 trainer 中最好 |

这张表说明：YOLOv5x 当前离原文绝对数值还有明显距离。标准 `train.py` 80ep baseline 已经明显强于自定义 trainer，但仍未到原文 400ep YOLOv5 baseline 的量级；自定义 trainer 与标准 `train.py` 的差距更大。因此当前最应该汇报为“baseline gate / trainer 对齐阶段”，而不是“已经复现或失败复现 CCLKD full”。

### 4.3 YOLOv5x 当前最关键信号

1. **标准 YOLOv5 train.py baseline 明显强于自定义 trainer baseline**
   标准 `train.py` 到 epoch 79 达到 AP50-95 = 0.30964，而自定义 `det_only_same_trainer` 80 epoch 最终只有 0.13490。
   这说明现在不能直接把自定义 CCLKD 结果低归因于 CCLKD loss 本身；首先要解释为什么同样 YOLOv5x、同样 SAR 数据、同样 batch32/80epoch 下，自定义 trainer 的 detection-only baseline 低很多。

2. **双分支结构本身没有造成额外严重崩溃**
   `two_branch_no_kd` 最终 AP50-95 = 0.13479，几乎等于 `det_only_same_trainer` 的 0.13490。这说明“多一个 online teacher 分支”本身不是主要问题。

3. **ATKD/full 目前能跑通，诊断信号正常**
   在已归档的运行中快照里，`paper_atkd_only` 和 `paper_full` 都有：
   - `feature_capture_ok = 1.0`
   - `nan_or_inf_detected = 0.0`
   - COP positive ratio 约 0.96
   - full CCLKD 的 CCL loss 约 0.69，说明 CCL 分支确实被激活

4. **当前 full CCLKD 不优于 ATKD-only**
   到 epoch 79，`paper_atkd_only` AP50-95 = 0.15149，`paper_full` AP50-95 = 0.14520。
   这与 YOLO11n 中“full 不如 lld_fld/atkd 简洁组合”的趋势一致：CCL 目前没有表现出稳定正向叠加。

5. **标准 baseline 的补充非常必要**
   如果标准 `train.py` 最终显著高于自定义 trainer，那么后续 YOLOv5x CCLKD 的结论应改为：
   - 目前 CCLKD loss 机制能运行；
   - 但自定义 trainer 与标准 YOLOv5 训练存在明显性能差距；
   - 在 trainer 对齐前，不宜把 full CCLKD 的绝对 AP 作为方法有效性结论。

## 5. 当前阶段结论

### 5.1 已经可以向老师汇报的内容

1. **我们已经完成 YOLO11n CCLKD 适配版完整消融**
   6 个组件组合全部跑完 400 epoch，并且相对 SAR baseline 均为正增益。最佳是 `lld_fld`，AP50-95 从 0.51546 提升到 0.53073。

2. **YOLO11n full CCLKD 不是最佳组合**
   full CCLKD AP50-95 = 0.52531，低于 `lld_fld`。这提示 CCLKD 的 RLD/CCL 在当前 YOLO11 适配和强 baseline 协议下没有稳定叠加。

3. **YOLOv5x 复现实验已经进入关键对齐阶段**
   我们已经跑通 YOLOv5x online CCLKD trainer，并完成 det-only、two-branch-no-KD、ATKD-only、full CCLKD 和标准 YOLOv5 `train.py` baseline 五组 80 epoch 实验。

4. **YOLOv5x 当前最大问题不是 CCLKD loss 是否能运行，而是 trainer baseline 对齐**
   标准 train.py baseline 在 epoch 79 达到 AP50-95 = 0.30964，而自定义 det-only 80 epoch 只有 0.13490。这个差距需要优先定位。

### 5.2 还不能下的结论

1. 不能说 YOLOv5x CCLKD 已经完整复现原文效果。
   当前 YOLOv5x 的标准 baseline 和 CCLKD 自定义 trainer baseline 尚未对齐。

2. 不能用当前 YOLOv5x full CCLKD 的绝对 AP 评价 CCLKD 方法失败。
   因为 detection-only 自定义 trainer 本身已经明显低于标准 train.py。

3. 不能把 YOLO11n 适配结果等同于原文 YOLOv5x 复现。
   YOLO11n 是现代检测头适配验证；YOLOv5x 才更接近原文结构。

## 6. 下一步计划

### P0：归档 YOLOv5x 80 epoch 结果

五组实验均已完成，最终归档位于：

```text
cclkd_reproduction/yolov5_sanity/results/diagnostics_20260612_yolov5x_b32_e80_final/
```

归档包含 `results.csv`、CCLKD diagnostics、命令/配置、日志 head/tail/key events 和最终 summary。权重、TensorBoard event 和完整 nohup 大日志不进入 Git。

### P1：定位标准 train.py 与自定义 trainer 的差异

优先检查：

1. loss 统计口径是否一致，尤其 YOLOv5 `train.py` 的 box/obj/cls loss 与自定义 trainer 的 loss 缩放。
2. optimizer 参数组、warmup、scheduler、EMA 是否一致。
3. dataloader / augmentation / mosaic / mixup / copy-paste 是否一致。
4. label assignment、anchor、image size、batch accumulation 是否一致。
5. validation 调用是否一致，包括 confidence、IoU、NMS 和 class mapping。

### P2：trainer 对齐后再判断 CCLKD 方法本身

只有当自定义 `det_only_same_trainer` 能接近标准 YOLOv5 `train.py` baseline 后，`paper_atkd_only` / `paper_full` 的差异才适合解释为 CCLKD loss 设计问题。

## 7. 建议汇报话术

可以简洁表述为：

> 我们已经完成了 YOLO11n 版本的 CCLKD 适配消融。结果显示 CCLKD 的定位/特征蒸馏组件在 OGSOD 上有稳定正增益，最佳组合是 LLD+FLD，AP50-95 从 0.51546 提升到 0.53073。但 full CCLKD 不是最优，说明 CCL/RLD 在 YOLO11 适配下叠加收益不稳定。
>
> 为了更接近原文，我们又回到 YOLOv5x 做复现。目前 YOLOv5x online trainer 已经跑通，det-only、two-branch-no-KD、ATKD-only、full CCLKD 和标准 YOLOv5 train.py baseline 均已完成 80 epoch。标准 train.py baseline 的 AP50-95 是 0.30964，而自定义 det-only trainer 只有 0.13490；ATKD-only 是 0.15149，full 是 0.14520。因此当前重点已经转为 trainer 对齐：在 baseline 对齐之前，不能用 YOLOv5x full CCLKD 的绝对 AP 判断方法本身是否有效。

## 8. 证据索引

- CCLKD 原文参照：
  - `cclkd_reproduction/CCLKD_PAPER_AUDIT_CN.md`
  - `cclkd_reproduction/diagnostics/20260607_protocol_gap/README_CN.md`
- YOLO11n final ablation：
  - `cclkd_reproduction/experiment_versions/v3_paper_pair_boxdist_20260609/FINAL_ANALYSIS_20260610_CN.md`
  - `cclkd_reproduction/experiment_versions/v3_paper_pair_boxdist_20260609/metrics_summary.csv`
- YOLOv5x batch32/80epoch queue archive：
  - `cclkd_reproduction/yolov5_sanity/results/diagnostics_20260612_yolov5x_b32_e80_final/README.md`
  - `cclkd_reproduction/yolov5_sanity/results/diagnostics_20260612_yolov5x_b32_e80_final/summary.csv`
  - `cclkd_reproduction/yolov5_sanity/results/diagnostics_20260612_yolov5x_b32_e80_queue/README.md`
  - `cclkd_reproduction/yolov5_sanity/results/diagnostics_20260612_yolov5x_b32_e80_queue/summary.csv`
- YOLOv5x current remote runs：
  - `/mnt/dataY/ydf/projects/LADD_public/cclkd_reproduction/yolov5_sanity/results/runs/yolov5x_standard_train_b32_s0_standard_train_b32_e80_gpu0`
  - `/mnt/dataY/ydf/projects/LADD_public/cclkd_reproduction/yolov5_sanity/results/runs/yolov5x_paper_atkd_only_b32_s0_paper_atkd_only_b32_e80_wave2_gpu1`
  - `/mnt/dataY/ydf/projects/LADD_public/cclkd_reproduction/yolov5_sanity/results/runs/yolov5x_paper_full_b32_s0_paper_full_b32_e80_wave2_gpu3`
