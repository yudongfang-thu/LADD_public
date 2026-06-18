# LADD 方法概览与当前理解

最后更新：2026-06-18

> 2026-06-18 更新：当前主线方法口径已固定为 `LADD Probe-A / LADD-clean A1B`，即 `A1 -> B`，不再把 A2 作为主线阶段。Probe-A 在 B 阶段动态更新 teacher decomposition/reach/taskL，但冻结 A1 学到的 student reachability probe，并在 reach loss 中 detach `q_s`。A2 与 Static/Dynamic 只保留为历史诊断或消融；准确定义见 `docs/ladd_clean_a1b_method_definition.md`。

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

当前 OGSOD HBB 主线定义已固定为：

```text
最新同协议 SAR/RGB baseline
+ nomosaic 800ep 主协议
+ A1 teacher decomposition warmup
+ Probe-A B: SAR detector training + z_s -> z_t KD
+ B 中动态 teacher decomposition/reach/taskL
+ B 中 frozen student reachability probe + detached q_s reach path
+ cap2 capped reach rank
+ sep/aux/debug losses removed
```

也就是说，cap2 仍是当前更合理的 reach rank 默认设置；A2 不再是主线必要阶段。Static `clean_a1b` 和完全 Dynamic `clean_a1b_dyn` 只作为消融；旧 A1-A2-B、formal no-mosaic 和未标记 Probe-A 的 B_A2_CORE 结果只能作为历史诊断、附录或消融，不应写作 LADD Probe-A 主表结果。

clean A1B 的完整 loss/冻结/launcher 定义见 `docs/ladd_clean_a1b_method_definition.md`。

## 4. 阶段划分

当前 OGSOD HBB 主线是两阶段：

| 阶段 | 训练内容 | 作用 |
|---|---|---|
| A1 | 教师解耦、可达 adapter、辅助头；检测损失关闭 | 先学习教师侧分解和可达关系 |
| B / Probe-A | 检测 + KD + student reconstruction；继续 teacher reconstruction/reach/taskL；冻结 student reachability probe | 正式蒸馏与检测训练，并让 `z_t/u_t` 随 B 阶段适配 |

重要经验：

- A1 后直接进 B 是当前 clean 口径；A2 在多个历史 run 中表现为不稳定或等效跳过，因此不进入主线。
- Static B 变量最少，但收益较弱；完全 Dynamic B 变量最多，s 模型曲线出现过不稳定。
- Probe-A 保留动态 teacher core 的适配能力，同时冻结 reach probe，当前 n/s 曲线更稳定，作为固定主线。

正式主线阶段设置：

| 阶段 | epoch | 检测损失 | 主线作用 |
|---|---:|---:|---|
| A1 | `10` | `0.0` | 学教师侧解耦和 reach adapter |
| B / Probe-A | `800` | `1.0` | 动态 teacher core + frozen reach probe，训练学生检测与蒸馏 |

旧 A2、Static、Dynamic 的曲线和结果可以用于解释为什么选择 Probe-A，但不再作为主方法定义。

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
- cap2 本身不是保证涨点的工具；clean A1B 当前只保留 cap2 rank 几何修正，不再保留额外 anti-collapse auxiliary loss。
- OGSOD formal no-mosaic 上，cap2 无法单独修复 A2 NaN，因为 A2 NaN 首先来自检测 loss 数值失稳。

## 7. 历史 Formal No-Mosaic A2 诊断

下面内容保留为历史诊断，用于解释为什么当前主线不再使用 A2。它不是当前
LADD Probe-A 主线的一部分。

旧 formal no-mosaic `YOLO11n_s0` LADD 原始 A2 设置：

```text
optimizer=auto -> MuSGD
lr0=0.01
warmup_epochs=3
```

在 A2 第 8 个 epoch 左右出现检测 loss NaN。loss 诊断显示：

- reach loss 没有先爆；
- `kd_loss` 记录为 0；
- 首先变坏的是 `train/box_loss`、`train/cls_loss`、`train/dfl_loss` 以及对应 val loss。

当时的修正：

```text
A2_OPTIMIZER=MuSGD
A2_LR0=0.001
A2_LRF=0.01
A2_WARMUP_EPOCHS=0
A2_WARMUP_BIAS_LR=0.001
```

该修正让旧 formal no-mosaic `11n_s0` original/cap2 两个版本稳定跑完 A2，并进入 B。但在 2026-06-18 的当前口径下，A2 已从主线移除；这段只作为历史诊断和消融背景，不再计入正式 LADD Probe-A 主线。

旧 A2 默认配置在第 8 个 epoch 检测 loss 变为 NaN，mAP50-95 掉到 `0.04909`；修正版 A2 完整跑完 50 epoch，mAP50-95 最高 `0.56273@49`。同时 `reach_match_loss` 没有先爆，说明这次失稳主要来自检测分支更新过猛。原始诊断图已从精简 public 分支移除。

## 8. 当前方法叙事边界

目前可以较稳地说：

- LADD 的主要价值是把 RGB 教师信息按 SAR 可达性筛选后再蒸馏。
- cap2 反坍缩修正提供了更合理的 reach rank 几何目标，因此作为当前正式主线默认设置。
- 正式 OGSOD 实验必须在收敛 baseline、同等训练预算、同等增强协议下比较。

不能直接说：

- cap2 已经完成最终性能验证；
- 任何 A2/B 冲击都来自 reach loss；
- 旧 close@100 LADD 结果可以直接替代 formal no-mosaic 结果。
