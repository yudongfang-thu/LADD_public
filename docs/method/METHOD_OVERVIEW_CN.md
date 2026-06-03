# LADD 方法概览与当前理解

最后更新：2026-06-01

## 1. 研究问题

LADD 解决的是 RGB 到 SAR 的跨模态目标检测蒸馏。训练时有配对 RGB/SAR 数据，推理时只能使用 SAR。直接让 SAR 学生模仿 RGB 教师会混入 RGB 私有信息，因此需要先把教师特征分成“SAR 可学习部分”和“RGB 私有部分”。

## 2. 核心结构

每个特征层上：

- 教师 RGB 特征 `f_t` 经过教师解耦网络得到 `z_t` 和 `u_t`。
- `z_t` 表示希望学生学习的可达信息。
- `u_t` 表示 RGB 私有或学生不应强行拟合的信息。
- 学生 SAR 特征 `f_s` 经过学生侧 split 得到 `z_s` 和 `r_s`。
- 蒸馏主要让 `z_s` 对齐 `z_t`。
- 检测头仍使用原始 SAR 检测特征，推理时不增加额外模块。

## 3. 当前正式主线

当前 OGSOD HBB 正式主线定义为：

```text
最新收敛 baseline
+ no-mosaic 800ep 正式训练协议
+ A1 -> A2 -> B 完整阶段
+ A2 检测稳定修正
+ cap2 反坍缩 reach rank
```

也就是说，cap2 不再只是一个附加诊断项，而是当前更合理的主线默认设置；A2 检测稳定修正也属于正式主线配置。未加 cap2 的版本保留为消融实验，用于量化旧 rank loss 的几何退化是否影响检测性能。

截至 2026-05-31，已经完整跑完的旧实验都不完全符合这个定义，因此只能作为历史可行性或诊断证据。正式结论应等待新 baseline 上的 cap2 LADD 完成。

## 4. 阶段划分

当前 OGSOD HBB 主线仍是三阶段：

| 阶段 | 训练内容 | 作用 |
|---|---|---|
| A1 | 教师解耦、可达 adapter、辅助头；检测损失关闭 | 先学习教师侧分解和可达关系 |
| A2 | 加入学生/检测损失；检测损失开启 | 让学生 backbone 适应可达/蒸馏目标，避免直接进 B 冲击过大 |
| B | 固定教师侧分解，训练学生蒸馏与检测 | 正式蒸馏阶段 |

重要经验：

- A1 不能替代 A2。去掉 A2 后，进入后续阶段时蒸馏/检测冲击过大。
- A2 必须有检测损失，否则学生 backbone 容易被可达目标拉偏。
- formal no-mosaic 协议下，A2 对已收敛 detector 的更新更敏感，需要较小学习率。

正式主线阶段设置：

| 阶段 | epoch | 检测损失 | 主线作用 |
|---|---:|---:|---|
| A1 | `10` | `0.0` | 学教师侧解耦和 reach adapter |
| A2 | `50` | `1.0` | 用检测监督稳定学生适配，防止 B 入口冲击过大 |
| B | `800` | `1.0` | 冻结教师解耦网络，训练学生检测与蒸馏 |

去掉 A2 时，B 入口 KD loss 更高且恢复更慢；带 A1+A2 的链路更容易恢复检测性能。

![B 入口 KD 冲击](ladd/diagnostics/b_collapse/ladd_b_entry_kd_shock.png)

图文件：[ladd_b_entry_kd_shock.png](ladd/diagnostics/b_collapse/ladd_b_entry_kd_shock.png)

## 5. Reach Loss 与坍缩

原始 reach rank loss 在 L2 normalize 后大致为：

```text
d_pos = ||q_s - z_t||^2
d_neg = ||q_s - u_t||^2
L_rank = softplus(delta + d_pos - d_neg)
```

因为两个单位向量的平方距离最大为 4，旧目标会持续奖励 `d_neg -> 4`。这对应 `cos(q_s, u_t) -> -1`，也就是完全反平行。实际表现是 `z_t/u_t` 容易坍缩到近似一维的反向表示。

这不是代码 bug，而是旧 rank loss 的几何最优点。

## 6. Cap2 反坍缩修正

最小修正是只在 rank 项中截断负距离：

```text
d_neg_eff = min(d_neg, rank_d_neg_cap)
L_rank = softplus(delta + d_pos - d_neg_eff)
```

其中：

- `rank_d_neg_cap=4.0`：旧行为；
- `rank_d_neg_cap=2.0`：超过正交距离后不再奖励继续反平行；
- `d_neg_eff` 只改变 rank loss 的奖励范围，不直接改变真实 `d_neg` 的记录值。

当前理解：

- cap2 能显著改变几何状态，避免继续奖励完全反平行。
- cap2 是当前正式主线默认设置，因为它是对旧 rank loss 几何目标的最小修正。
- cap2 本身不是保证涨点的工具，过强的正交约束可能伤害性能；因此 original / stronger anti-collapse 仍应作为消融对照。
- OGSOD formal no-mosaic 上，cap2 无法单独修复 A2 NaN，因为 A2 NaN 首先来自检测 loss 数值失稳。

## 7. Formal No-Mosaic A2 修正

formal no-mosaic `YOLO11n_s0` LADD 原始 A2 设置：

```text
optimizer=auto -> MuSGD
lr0=0.01
warmup_epochs=3
```

在 A2 第 8 个 epoch 左右出现检测 loss NaN。loss 诊断显示：

- reach loss 没有先爆；
- `kd_loss` 记录为 0；
- 首先变坏的是 `train/box_loss`、`train/cls_loss`、`train/dfl_loss` 以及对应 val loss。

当前修正：

```text
A2_OPTIMIZER=MuSGD
A2_LR0=0.001
A2_LRF=0.01
A2_WARMUP_EPOCHS=0
A2_WARMUP_BIAS_LR=0.001
```

该修正已让 formal no-mosaic `11n_s0` original/cap2 两个版本稳定跑完 A2，并进入 B。因此它已经计入当前正式 LADD 主线细节，而不是临时 workaround。

对应曲线如下：旧 A2 默认配置在第 8 个 epoch 检测 loss 变为 NaN，mAP50-95 掉到 `0.04909`；修正版 A2 完整跑完 50 epoch，mAP50-95 最高 `0.56273@49`。同时 `reach_match_loss` 没有先爆，说明这次失稳主要来自检测分支更新过猛。

![A2 检测稳定修正](ladd/diagnostics/a2_stability/ladd_a2_stability_fix_20260601.png)

图文件：[ladd_a2_stability_fix_20260601.png](ladd/diagnostics/a2_stability/ladd_a2_stability_fix_20260601.png)

## 8. 当前方法叙事边界

目前可以较稳地说：

- LADD 的主要价值是把 RGB 教师信息按 SAR 可达性筛选后再蒸馏。
- cap2 反坍缩修正提供了更合理的 reach rank 几何目标，因此作为当前正式主线默认设置。
- 正式 OGSOD 实验必须在收敛 baseline、同等训练预算、同等增强协议下比较。

不能直接说：

- cap2 已经完成最终性能验证；
- 任何 A2/B 冲击都来自 reach loss；
- 旧 close@100 LADD 结果可以直接替代 formal no-mosaic 结果。
