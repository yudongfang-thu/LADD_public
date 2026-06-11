# LADD Capacity-aware KD 诊断结果

日期：2026-06-11

快照时间：`2026-06-11 20:20:25 CST`

本文记录双卡 4090 上 capacity-aware KD 诊断的当前结果。`s alpha_kd=0.5 B400` 与 `s alpha_kd=0.25 B400` 已完成；`s B det-only r2` 已进入 B 阶段但尚未跑满，因此只作为运行中快照解释。

## 1. 证据位置

轻量证据包：

```text
ladd/results/capacity_kd_20260611/
```

包含内容：

- 原始 `results.csv` 快照。
- 每个关键阶段的 `args.yaml` 与 `manifest.txt`。
- 从大 outer log 抽取的 resume、phase diagnostic、grad clip、完成状态关键行。
- 不包含任何 `weights/`、`.pt` 或 `.pth` checkpoint。

快照 CSV：

```text
docs/experiments/ladd_capacity_kd_results_20260611_snapshot.csv
```

## 2. 代码版本

三个 LADD 诊断 run 的 `manifest.txt` 均记录：

```text
git_commit=69166620f6bc8b5885cbf285ec3ef3a7242c7e6e
```

B 阶段日志确认：

```text
effective_grad_clip_norm=10.0
freeze_bn_stats=True
```

本轮 run 的 `ladd_diag_log_grad=False`，但 phase diagnostic 明确显示 `LADD_GRAD_CLIP_NORM=0.0` 时仍使用 `effective_grad_clip_norm=10.0`，即保留 Ultralytics 默认 clip 语义。代码侧也按该语义处理 grad logging 场景，避免把日志开关变成优化器行为改动。

本次 GitHub 更新还补充了 resume/launch 代码，便于后续复现实验恢复流程；这些代码变更不改变已完成 run 的实际训练 commit，应以后续新 run 的 manifest 为准。

## 3. 结果表

YOLO11s 判据：

```text
SAR baseline = 0.62897
PASS threshold = 0.62697
```

YOLO11m 判据：

```text
SAR baseline = 0.65580
A2 safe threshold = 0.65380
```

| 优先级 | 实验 | 阶段 | 进度 | best AP50-95 | last AP50-95 | baseline | last-baseline | 状态 |
|---|---|---|---:|---:|---:|---:|---:|---|
| P1 | `s alpha_kd=0.5 B400` | B | 400/400 | 0.63074 @218 | 0.61802 | 0.62897 | -0.01095 | 完成 |
| P2 | `s alpha_kd=0.25 B400` | B | 400/400 | 0.63027 @199 | 0.61719 | 0.62897 | -0.01178 | 完成 |
| P2 | `s B det-only r2` | A1 | 10/10 | 0.62878 @1 | 0.62878 | 0.62897 | -0.00019 | 完成 |
| P2 | `s B det-only r2` | A2 | 50/50 | 0.62400 @50 | 0.62400 | 0.62897 | -0.00497 | 完成 |
| P2 | `s B det-only r2` | B | 74/400 | 0.62244 @68 | 0.62191 | 0.62897 | -0.00706 | 运行中快照 |
| P1 | `m A2 probe` | A2 | 50/50 | 0.65026 @10 | 0.63725 | 0.65580 | -0.01855 | 完成 |
| P1 | `m A2 probe` | B=1 | 1/1 | 0.62528 @1 | 0.62528 | 0.65580 | -0.03052 | 诊断完成 |

注意：两条 alpha resume 的 `results.csv` 保留了中断恢复时的重复 epoch 行。汇总时按每个 epoch 的最后一次记录计算；原始 CSV 未做删除或重写。

## 4. 关键曲线点

| 实验 | epoch | AP50-95 | train box | train cls | train dfl | train kd |
|---|---:|---:|---:|---:|---:|---:|
| `s alpha_kd=0.5` | 218 | 0.63074 | 0.69910 | 0.34083 | 0.79739 | 0.06152 |
| `s alpha_kd=0.5` | 296 | 0.62635 | 0.63682 | 0.31643 | 0.79318 | 0.05627 |
| `s alpha_kd=0.5` | 400 | 0.61802 | 0.60507 | 0.30509 | 0.79085 | 0.05420 |
| `s alpha_kd=0.25` | 199 | 0.63027 | 0.71014 | 0.34660 | 0.79755 | 0.03435 |
| `s alpha_kd=0.25` | 278 | 0.62736 | 0.64891 | 0.32148 | 0.79384 | 0.03141 |
| `s alpha_kd=0.25` | 400 | 0.61719 | 0.60076 | 0.30540 | 0.78908 | 0.03008 |
| `s B det-only r2` | 1 | 0.61806 | 0.71678 | 0.35507 | 0.79875 | 0.00000 |
| `s B det-only r2` | 72 | 0.62208 | 0.78493 | 0.37966 | 0.80424 | 0.00000 |
| `m A2 probe` | 10 | 0.65026 | 0.69250 | 0.33185 | 0.79701 | 0.00000 |
| `m A2 probe` | 50 | 0.63725 | 0.58905 | 0.29173 | 0.79015 | 0.00000 |

## 5. 诊断结论

1. 降低 `alpha_kd` 没有解决 YOLO11s B400 后期坍塌。`0.5` 与 `0.25` 都在中期达到略高于 SAR baseline 的 best，但跑满 400 后分别降到 `0.61802` 和 `0.61719`，均低于 pass threshold。

2. `alpha_kd=0.25` 不是稳定优于 `0.5`。它的 best 略低，last 也略低；同时该 run 的 A2 发生过一次 OOM 后自动降 batch，严格可比性仍需标注。

3. `s B det-only r2` 不能被解释成“纯 B 阶段问题已经排除”。该 run 设置为 `LADD_B_DET_ONLY=1`，但 `LADD_A2_DET_ONLY=0`，因此 B 阶段继承了正常 A2 的下滑 checkpoint。A2 已经从 A1 的 `0.62878` 降到 `0.62400`，说明 A2 配置本身已经对 s 有负面影响。

4. `s B det-only r2` 的 B 快照显示 `kd_loss=0`，但 AP 仍在 `0.622` 附近，尚未恢复到 baseline。这更支持“当前链路在 A2 或 A2+B 组合上已经损伤学生”的解释，而不是单纯 KD 强度过大。

5. m 模型方向的问题更早发生在 A2。`m A2 probe` 的 best `0.65026` 已低于安全阈值 `0.65380`，last `0.63725` 明显退化；不应在当前配置下启动 m full B。

## 6. 后续建议

短期继续观察：

1. 等 `s B det-only r2` 跑满 B400，确认 B det-only 是否继续下滑、停在低位或恢复。
2. 不再启动新的 s B400 变体，直到 det-only 完整结果出来。

下一组更有信息量的实验：

1. `s A2 det-only`：判断 s 的 A2 下滑是否来自 reach/aux/KD 类辅助目标。
2. `s A2 lr3e-4`：判断 A2 学习率是否过强。
3. `m A2 det-only`：优先排查 m 模型 A2 阶段损伤。
4. `m A2 lr3e-4`：若 det-only 不能恢复，再检查优化强度。

当前不建议：

- 直接开 m full B。
- 继续只扫 `alpha_kd`，因为 `0.5` 与 `0.25` 都已经跑满失败。
- 把 `s B det-only r2` 的部分 B 快照当作最终结论。
