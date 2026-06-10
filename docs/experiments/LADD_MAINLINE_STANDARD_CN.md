# OGSOD LADD 正式主线训练规范

最后更新：2026-06-09

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

旧 A2 默认配置在 epoch 8 出现 NaN：训练检测损失从 `2.31568` 升到最大有限值 `4.66065` 后崩溃，mAP50-95 从 `0.51288` 掉到 `0.04909`；同期 `reach_match_loss` 仍只有 `0.00222 -> 0.00626`，`kd_loss=0`，说明先坏的是检测分支。修正版 A2 完整跑完 50 epoch，训练检测损失约 `2.33951 -> 2.17645`，mAP50-95 最高 `0.56273@49`。原始诊断图已从精简 public 分支移除。

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

seed123 的 `bstable1e3` 重跑使用上述设置后完整跑满 800 epoch，但 best 仅为
`0.56161@165`，最后退化到 `0.52875`。这说明 B 的温和学习率可以避免 NaN，
但不能单独解决后期检测器退化。后续主线必须继续使用温和 B 设置，但还需要
BN running stats 稳定修正。

## 6. B 阶段 BN-freeze 修正

YOLO11n seed0/123 的坏 run 权重没有明显 NaN/Inf，但 `last.pt` 的 BN
`running_mean/running_var` 被污染：seed0 坏 run 的 BN max running_var 到约
1726，seed123 坏 run 到约 1333，而健康 seed42 约 47.7。因此当前最新 B 稳定
候选额外加入：

```text
FREEZE_BN_STATS=1
```

含义：训练时冻结 BN running mean/var，保留 BN affine 参数梯度。当前已完成
三个 YOLO11n 关键验证：

| Run | best AP50-95 | last AP50-95 | 相对 SAR baseline |
|---|---:|---:|---:|
| YOLO11n seed0 cap2 BN-freeze | 0.57276@793 | 0.57254 | +0.01622 |
| YOLO11n seed42 cap2 BN-freeze | 0.57615@400 | 0.57295 | +0.01821 |
| YOLO11n seed123 cap2 BN-freeze | 0.57269@779 | 0.57219 | +0.01141 |

这说明 BN-freeze 不是早期猜测，而是当前最可信的 B 阶段稳定修正。YOLO11n
已经形成 seed0/42/123 三 seed 正向证据，可以作为当前主线冻结候选。需要注意的是，
YOLO11s seed0 的 BN-freeze 跑满后 best 为 `0.63388@263`，但 last 退到
`0.61759`，低于 SAR baseline `0.62897`；因此 BN-freeze 对 s 容量没有完全解决
后期退化，s 应作为单独容量诊断，而不是反向阻塞 n 主线。

## 7. Cap2 反坍缩设置

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

## 8. B 入口冲击与 A2 的必要性

去掉 A2 后，B 入口的 KD loss 显著更大，且后续下降缓慢；带 A1+A2 的 short-A 链路进入 B 后 KD loss 更低，检测 mAP 也恢复更快。这个现象说明 A2 不是为了让 A 阶段多训练一些 epoch，而是作为 B 前的学生适配桥。原始诊断图已从精简 public 分支移除。

## 9. 当前完成状态

截至 2026-06-09：

- 旧 `close_mosaic=100` 800ep LADD 已完整跑通，但不符合当前 no-mosaic 主线；
- formal no-mosaic `YOLO11n seed0` original/cap2 已完整跑完 B；
- cap2 版本是当前主线，original 版本是消融；
- `YOLO11n cap2 seed42` 已完整跑完 B，best AP50-95 为 `0.57420`；
- `YOLO11n cap2 seed123` 旧 B 出现 NaN/Inf，`bstable1e3` 跑满但后期退化；
- `YOLO11n cap2 seed0/42/123 BN-freeze` 已完整跑完 B，分别为 `0.57276@793`、`0.57615@400` 与 `0.57269@779`；
- `YOLO11s cap2 seed0` 在 90 上跑到 epoch 608，best `0.63551@605`，未满 800；双卡 4090 BN-freeze 版本跑满，best `0.63388@263`，last `0.61759`，仍有后期退化；
- `YOLO11m cap2 seed0` B 阶段异常，暂不纳入主线。

旧 `mosaic=1.0, close_mosaic=700` 收敛主线的 90 服务器原始轻量证据已补充归档：[`LADD_MOSAIC90_MAINLINE_EVIDENCE_20260528_CN.md`](LADD_MOSAIC90_MAINLINE_EVIDENCE_20260528_CN.md)。该证据说明 LADD 在开 mosaic 后关闭的旧收敛协议下可以稳定完成 B 阶段，不应把 no-mosaic/H1 退化直接解释成方法机制必然崩溃。

当前 seed0 结果：

| 实验 | best AP50-95 | 相对 SAR baseline | gap 覆盖 |
|---|---:|---:|---:|
| YOLO11n original seed0 | 0.57821 | +0.02167 | 29.4% |
| YOLO11n cap2 seed0 | 0.57662 | +0.02008 | 27.3% |
| YOLO11n cap2 seed0 BN-freeze | 0.57276 | +0.01622 | 22.0% |
| YOLO11n cap2 seed42 BN-freeze | 0.57615 | +0.01821 | 26.5% |
| YOLO11n cap2 seed123 BN-freeze | 0.57269 | +0.01141 | 16.8% |

完整状态以 [EXPERIMENT_PLAN_CN.md](EXPERIMENT_PLAN_CN.md) 为准。

## 10. 启动前检查清单

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
10. B 是否按当前候选启用 `FREEZE_BN_STATS=1`；
11. 是否使用 `RANK_D_NEG_CAP=2.0`；
12. 不带 cap2 的实验是否明确标注为消融；
13. 旧 close@100 / 400ep / B 默认高学习率结果是否没有混入正式主表。
