# OGSOD LADD 正式主线训练规范

最后更新：2026-06-03

本文档记录当前 OGSOD 上 LADD 的正式主线设置。后续 LADD 主实验、多 seed、多容量扩展应优先按本规范启动；不符合本规范的实验必须标注为消融或诊断。

## 1. 主线定义

当前正式 LADD 主线是：

```text
最新收敛 SAR/RGB baseline
+ no-mosaic baseline protocol
+ A1 -> A2 -> B 完整阶段
+ A2 检测稳定修正
+ B 检测稳定修正
+ cap2 reach-rank 反坍缩
```

也就是说，主线不只是“加 cap2”，还包括 A2/B 阶段的检测稳定设置。旧 A2 默认学习率在 formal no-mosaic baseline 上已经观察到检测 loss NaN；随后 YOLO11n cap2 seed123 的旧 B 配置也出现检测 loss NaN 和 NaN/Inf 权重，因此 A2/B 都不能再使用默认高学习率配置。

## 2. Baseline 依赖

LADD 必须从同协议 baseline 出发：

- baseline 规范：[BASELINE_TRAINING_STANDARD_CN.md](BASELINE_TRAINING_STANDARD_CN.md)
- SAR/RGB 必须同容量、同 seed；
- batch 必须沿用 baseline 容量表：`n/s=64`, `m/l=32`, `x=16`；
- 旧 `400ep`、旧 `close_mosaic=100` baseline 只能作为历史诊断，不作为正式 LADD 起点。

配对规则：

```text
SAR(size, seed k) + RGB(size, seed k) -> LADD(size, seed k)
```

## 3. 阶段设置

| 阶段 | 作用 | 主线要求 |
|---|---|---|
| A1 | 训练教师解耦、reach adapter 和辅助头 | 检测损失关闭；学习教师侧分解与可达关系 |
| A2 | 让学生 backbone 在检测监督下适应 reach/KD 前置目标 | 检测损失必须开启；使用 A2 稳定学习率设置 |
| B | 正式蒸馏学生 | 教师解耦网络固定；学生侧训练检测与蒸馏 |

A1 不能替代 A2。A2 的作用不是单纯延长训练，而是避免 B 入口时 KD/检测冲击直接打坏检测器。

正式主线的阶段超参固定为：

| 阶段 | epoch | 检测损失 | 主要训练对象 | 关键开关 |
|---|---:|---:|---|---|
| A1 | `10` | `0.0` | 教师解耦、reach adapter、辅助头 | `USE_FG_MASK_FOR_REACH=1`, `USE_FG_MASK_FOR_REC=0` |
| A2 | `50` | `1.0` | 学生 backbone/split 在检测监督下适应 A 阶段目标 | `A2_OPTIMIZER=MuSGD`, `A2_LR0=0.001`, `A2_WARMUP_EPOCHS=0` |
| B | `800` | `1.0` | 学生检测与蒸馏 | 教师解耦网络冻结，`RANK_D_NEG_CAP=2.0`, `B_LR0=0.001` |

所有阶段都继承 formal no-mosaic 增强协议：`mosaic=0.0`, `close_mosaic=0`, 保留默认 Albumentations，关闭 HSV/MixUp/CutMix/Erasing。

## 4. A2 检测稳定修正

formal no-mosaic `YOLO11n_s0` 曾在 A2 约第 8 个 epoch 出现检测 loss NaN。诊断现象：

- reach loss 没有先爆；
- `kd_loss` 记录为 0；
- 首先异常的是 `train/box_loss`、`train/cls_loss`、`train/dfl_loss` 和对应 val loss；
- 因此主要问题是 A2 对已经收敛的 detector 更新过猛，而不是 cap2 或 reach loss 单独造成。

正式主线 A2 必须使用：

```text
A2_DET_LOSS_SCALE=1.0
A2_OPTIMIZER=MuSGD
A2_LR0=0.001
A2_LRF=0.01
A2_WARMUP_EPOCHS=0
A2_WARMUP_BIAS_LR=0.001
```

含义：

- `A2_DET_LOSS_SCALE=1.0`：A2 必须保留检测监督，避免学生 backbone 被 reach 目标单独拉偏；
- `A2_LR0=0.001`：比 baseline 默认 `0.01` 更温和，降低对已收敛 detector 的冲击；
- `A2_WARMUP_EPOCHS=0`：避免 warmup 阶段对 bias / detector 造成额外不稳定；
- `A2_WARMUP_BIAS_LR=0.001`：与 A2 主学习率保持一致。

若 A2 使用 `lr0=0.01` 或默认 warmup，必须标注为旧配置或诊断实验。

图示证据如下。旧 A2 默认配置在 epoch 8 出现 NaN：训练检测损失从 `2.31568` 升到最大有限值 `4.66065` 后崩溃，mAP50-95 从 `0.51288` 掉到 `0.04909`；同期 `reach_match_loss` 仍只有 `0.00222 -> 0.00626`，`kd_loss=0`，说明先坏的是检测分支。修正版 A2 完整跑完 50 epoch，训练检测损失约 `2.33951 -> 2.17645`，mAP50-95 最高 `0.56273@49`。

![A2 检测稳定修正](ladd/diagnostics/a2_stability/ladd_a2_stability_fix_20260601.png)

图文件：[ladd_a2_stability_fix_20260601.png](ladd/diagnostics/a2_stability/ladd_a2_stability_fix_20260601.png)

这张图也解释了为什么 A2 稳定修正必须计入主线，而不是一个无关实现细节：进入 B 之前，学生检测器必须保持在可用状态，否则后续蒸馏恢复会被检测崩溃吞掉。

## 5. B 检测稳定修正

formal no-mosaic `YOLO11n cap2 seed123` 的旧 B 配置使用：

```text
B_OPTIMIZER=auto
B_LR0=0.01
默认 warmup / bias lr 行为
```

该 run 在 B 阶段 epoch 429 开始出现检测损失 NaN，后续 AP 掉到 0，最终 `last.pt` 被判定为 NaN/Inf 权重。关键诊断：

- B 阶段记录中 `reach_match_loss`、`reach_rank_loss`、`d_pos/d_neg` 均为 0，因为 B 主要记录 KD/检测，不是 reach 先爆；
- `kd_loss` 在 NaN 前后仍为有限值，例如 epoch 428 附近约 0.13-0.16；
- 首先异常的是 `train/box_loss`、`train/cls_loss`、`train/dfl_loss` 和对应 val loss；
- 旧 B 初期 `lr/pg0` 约 `0.00996 -> 0.01996 -> 0.02996`，对已经过 A1/A2 的检测器冲击过强。

因此 B 的正式主线也采用温和设置：

```text
B_DET_LOSS_SCALE=1.0
B_OPTIMIZER=MuSGD
B_LR0=0.001
B_LRF=0.01
B_WARMUP_EPOCHS=0
B_WARMUP_BIAS_LR=0.001
```

seed123 的 `bstable1e3` 重跑使用上述设置，当前 144 epoch 内无 NaN，best AP50-95 已回到约 `0.56089`。后续在 117 上重跑 LADD 主线时，A2/B 都必须使用温和配置；旧 `optimizer=auto, lr0=0.01` 的 B run 只能作为诊断记录。

## 6. Cap2 反坍缩设置

正式主线使用 cap2：

```text
RANK_D_NEG_CAP=2.0
```

对应 reach rank：

```text
d_neg_eff = min(d_neg, rank_d_neg_cap)
L_rank = softplus(delta + d_pos - d_neg_eff)
```

`rank_d_neg_cap=2.0` 表示当负样本距离超过正交距离后，不再继续奖励反平行。未加 cap2 的原始 LADD 保留为消融实验，不作为当前主线。

A 阶段 reach/rec/d_neg 的历史诊断图如下。旧 rank loss 会把 `d_neg` 推向 4，也就是 L2 normalize 后的反平行极限；cap2 把 rank loss 中有效负距离截断在 2，超过正交距离后不再继续奖励反平行。

![Reach/rec/d_neg 诊断](docs/experiments/figures/reach_rec_dneg_trends.png)

图文件：[reach_rec_dneg_trends.png](docs/experiments/figures/reach_rec_dneg_trends.png)

## 7. B 入口冲击与 A2 的必要性

去掉 A2 后，B 入口的 KD loss 显著更大，且后续下降缓慢；带 A1+A2 的 short-A 链路进入 B 后 KD loss 更低，检测 mAP 也恢复更快。这个现象说明 A2 不是为了让 A 阶段多训练一些 epoch，而是作为 B 前的学生适配桥。

![B 入口 KD 冲击](ladd/diagnostics/b_collapse/ladd_b_entry_kd_shock.png)

图文件：[ladd_b_entry_kd_shock.png](ladd/diagnostics/b_collapse/ladd_b_entry_kd_shock.png)

## 8. 当前完成状态

截至 2026-06-03：

- 旧 `close_mosaic=100` 800ep LADD 已完整跑通，但不符合当前 no-mosaic 主线；
- formal no-mosaic `YOLO11n seed0` original/cap2 已完整跑完 B；
- cap2 版本是当前主线，original 版本是消融；
- `YOLO11n cap2 seed42` 已完整跑完 B，best AP50-95 为 `0.57420`；
- `YOLO11n cap2 seed123` 旧 B 出现 NaN/Inf，`bstable1e3` 正在重跑；
- `YOLO11s cap2 seed0` 正在 B 阶段；
- `YOLO11m cap2 seed0` 正在 A2 阶段，但迁移到 117 前仍需补 RGB m teacher 权重。

当前 seed0 结果：

| 实验 | best AP50-95 | 相对 SAR baseline | gap 覆盖 |
|---|---:|---:|---:|
| YOLO11n original seed0 | 0.57821 | +0.02167 | 29.4% |
| YOLO11n cap2 seed0 | 0.57662 | +0.02008 | 27.3% |

完整状态以 [EXPERIMENT_PLAN_CN.md](EXPERIMENT_PLAN_CN.md) 为准。

## 9. 启动前检查清单

启动正式 LADD 前必须确认：

1. SAR/RGB baseline 是否来自最新 no-mosaic baseline 规范；
2. SAR/RGB 是否同容量、同 seed；
3. 是否使用完整 `A1 -> A2 -> B`；
4. A2 是否 `A2_DET_LOSS_SCALE=1.0`；
5. A2 是否 `A2_LR0=0.001`；
6. A2 是否 `A2_WARMUP_EPOCHS=0`；
7. B 是否 `B_LR0=0.001`；
8. B 是否 `B_OPTIMIZER=MuSGD`；
9. B 是否 `B_WARMUP_EPOCHS=0`；
10. 是否使用 `RANK_D_NEG_CAP=2.0`；
11. 不带 cap2 的实验是否明确标注为消融；
12. 旧 close@100 / 400ep / B 默认高学习率结果是否没有混入正式主表。
