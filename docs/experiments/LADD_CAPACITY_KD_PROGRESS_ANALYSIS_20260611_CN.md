# LADD Capacity-aware KD 阶段性分析

日期：2026-06-11

快照时间：`2026-06-11 11:56:30 CST`

更新：`2026-06-11 20:20:25 CST` 已生成完成后的诊断结果文档，见 [LADD_CAPACITY_KD_DIAG_RESULTS_20260611_CN.md](LADD_CAPACITY_KD_DIAG_RESULTS_20260611_CN.md)。本文件保留为 11:56 阶段性快照，不再作为最新结论入口。

本文记录双卡 4090 上 capacity-aware KD 诊断的阶段性状态。两个 YOLO11s B400 实验尚未完成，因此本文结论只用于调度和诊断判断，不作为最终实验结果。

## 1. 代码与环境

远端实验目录：

```text
/root/shared-nvme/LADD_public_p1
```

远端实验代码：

```text
69166620f6bc8b5885cbf285ec3ef3a7242c7e6e
fix(ladd): refresh effective KD weights at train epoch start
```

本地 GitHub 主线提交可能比该 commit 更新；后续解释结果时，以每个 run 的 `manifest.txt` 中 `git_commit` 为准。

GPU 快照：

| GPU | 显存使用 | 剩余显存 | 利用率 |
|---|---:|---:|---:|
| 0 | 23120 MiB | 962 MiB | 99% |
| 1 | 20951 MiB | 3131 MiB | 99% |

## 2. 当前数据表

| 优先级 | 实验 | 阶段 | 进度 | best AP50-95 | last AP50-95 | baseline | last-baseline | 状态 |
|---|---|---|---:|---:|---:|---:|---:|---|
| P1 | `s alpha_kd=0.5 B400` | B | 283/400 | 0.63074 @218 | 0.62803 | 0.62897 | -0.00094 | 运行中 |
| P2 | `s alpha_kd=0.25 B400` | B | 265/400 | 0.63027 @199 | 0.62901 | 0.62897 | +0.00004 | 运行中 |
| P2 | `s B det-only B400` | A1 | 3/10 | 0.62878 @3 | 0.62878 | 0.62897 | -0.00019 | 刚启动 |
| P1 | `m A2 probe` | A2 | 50/50 | 0.65026 @10 | 0.63725 | 0.65580 | -0.01855 | 已完成 |
| P1 | `m A2 probe` | B=1 | 1/1 | 0.62528 @1 | 0.62528 | 0.65580 | -0.03052 | 已完成 |

YOLO11s 判据：

```text
SAR baseline = 0.62897
PASS threshold = 0.62697
```

YOLO11m A2 判据：

```text
SAR baseline = 0.65580
A2 safe threshold = 0.65380
```

## 3. 阶段性观察

1. `s alpha_kd=0.5` 当前仍高于 pass threshold，但已经从 best `0.63074 @218` 回落到 `0.62803 @283`，last-baseline 为 `-0.00094`。

2. `s alpha_kd=0.25` 当前 last 为 `0.62901 @265`，略高于 baseline `+0.00004`，从 best `0.63027 @199` 到 last 的回落为 `-0.00126`，小于 `alpha_kd=0.5` 当前回落 `-0.00271`。

3. 上述 `0.25` 与 `0.5` 的比较不能下最终结论。`alpha_kd=0.25` 的 A2 曾触发一次 CUDA OOM，Ultralytics 自动把 batch 从 64 降到 32 后继续跑完；同时两个 run 当前 epoch 不完全一致。因此只能说 `0.25` 暂时显示更小 late drop，但可比性有缺口。

4. 两个 s B 阶段中，训练 box/cls/dfl loss 仍在下降，而 AP 在 best 之后轻微回落。这更像泛化/目标错配或 late-stage KD 负迁移，而不是训练数值发散。

5. `m A2 probe` 已经给出明确负信号：A2 best `0.65026` 低于安全阈值 `0.65380`，A2 last `0.63725` 进一步下降。B=1 的 `0.62528` 只是延续受损 A2 后的诊断结果。当前不应启动 m full B。

## 4. Loss 快照

| 实验 | epoch | AP50-95 | train box | train cls | train dfl | train kd |
|---|---:|---:|---:|---:|---:|---:|
| `s alpha_kd=0.5` | 218 | 0.63074 | 0.69910 | 0.34083 | 0.79739 | 0.06152 |
| `s alpha_kd=0.5` | 283 | 0.62803 | 0.65316 | 0.32228 | 0.79357 | 0.05773 |
| `s alpha_kd=0.25` | 199 | 0.63027 | 0.71014 | 0.34660 | 0.79755 | 0.03435 |
| `s alpha_kd=0.25` | 265 | 0.62901 | 0.65563 | 0.32484 | 0.79376 | 0.03282 |
| `m A2 probe` | 10 | 0.65026 | 0.69250 | 0.33185 | 0.79701 | 0.00000 |
| `m A2 probe` | 50 | 0.63725 | 0.58905 | 0.29173 | 0.79015 | 0.00000 |

## 5. 当前解释

对 YOLO11s：

- `alpha_kd=0.5` 和 `0.25` 都已经把当前 last 保持在 pass threshold 以上，说明降低 KD 强度是合理方向。
- `0.25` 当前 late drop 更小，但由于 A2 OOM 降 batch，不能把它直接判定为优于 `0.5`。
- `s B det-only B400` 已启动，它是判断“B 长训练自身是否会退化”的关键对照。如果 det-only 稳，而两个 KD run 掉，负迁移证据更强；如果 det-only 也掉，则需要检查 B 长训练协议本身。

对 YOLO11m：

- 当前 m 的主要问题不是 B 后期，而是 A2 已经把 detector 拉低。
- 下一轮 m 方向应优先做 `m A2 det-only` 和 `m A2 lr3e-4`，而不是 m full B。

## 6. 后续动作

短期：

1. 等 `s alpha_kd=0.5 B400` 和 `s alpha_kd=0.25 B400` 跑满 400。
2. 继续观察 `s B det-only B400` 是否发生 OOM 或自动降 batch。
3. 两个 s B400 完成后运行 summary 工具，生成最终 `LADD_CAPACITY_KD_DIAG_RESULTS_20260610_CN.md`。

下一轮候选：

1. `m A2 det-only`
2. `m A2 lr3e-4`
3. `s KD decay 200->300 B400`
4. `s KD stop@220 B400`

不建议：

- 在当前 m A2 probe 失败后直接启动 m full B。
- 在 `s B det-only` 尚未跑出信号前继续加更多 s B400 变体。

## 7. 关联文件

- 阶段性 CSV：[ladd_capacity_kd_progress_20260611_snapshot.csv](ladd_capacity_kd_progress_20260611_snapshot.csv)
- 诊断计划：[LADD_CAPACITY_AWARE_KD_DIAG_PLAN_20260610_CN.md](LADD_CAPACITY_AWARE_KD_DIAG_PLAN_20260610_CN.md)
- 已启动实验记录：[LADD_CAPACITY_KD_ACTIVE_RUNS_20260611_CN.md](LADD_CAPACITY_KD_ACTIVE_RUNS_20260611_CN.md)
