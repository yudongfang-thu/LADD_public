# LADD 主线稳定性诊断归档 2026-06-09

本目录用于保存当前 LADD 主线在 A2/B 阶段遇到的稳定性问题、修正路径和可复核训练日志。归档目标是让后续分析能区分三件事：

1. A2/B 旧学习率导致的检测分支 NaN；
2. B 阶段不 NaN 但后期 AP 退化；
3. BN running stats freeze 对 YOLO11n 与 YOLO11s 的不同效果。

## 文件结构

| 文件/目录 | 内容 |
|---|---|
| `remote_4090dual/` | 从双卡 4090 `/root/shared-nvme/LADD_public` 拉取的完整 `results.csv` 和 `args.yaml` |
| `summaries/mainline_runs_summary.csv` | A1/A2/B 各 run 的 last/best AP 汇总 |
| `summaries/b_stage_key_epoch_points.csv` | B 阶段关键 epoch 的 AP、loss、lr 摘要 |
| `summaries/mainline_issue_timeline.csv` | A2 NaN、B NaN、B 退化、BN-freeze 修复、YOLO11s 退化的事件链 |

本目录不包含 `.pt` / `.pth` checkpoint。

## 已拉取的 4090 证据

| Run | 阶段 | 结论 |
|---|---|---|
| `yolo11n cap2 seed42 bnfreeze1e3 final_v2` | A1/A2/B | B 跑满 800，best AP50-95 `0.57615@400`，last `0.57295` |
| `yolo11s cap2 seed0 bnfreeze1e3 final_v2` | A1/A2/B | B 跑满 800，best AP50-95 `0.63388@263`，last `0.61759` |

## 当前读法

### 1. A2 崩溃与 B 崩溃

A2 旧配置在 YOLO11n seed0 上约 epoch 8 出现检测 loss NaN；B 旧配置在 YOLO11n seed123 上约 epoch 429 开始出现检测 loss NaN。二者都首先表现为 `box/cls/dfl` 检测损失失稳，而不是 reach/KD 先爆。因此它们应归为同一类：阶段切换后 detector 被默认高学习率、warmup 或 bias LR 冲击。

主线稳定修正：

```text
A2/B_OPTIMIZER=MuSGD
A2/B_LR0=0.001
A2/B_LRF=0.01
A2/B_WARMUP_EPOCHS=0
A2/B_WARMUP_BIAS_LR=0.001
```

### 2. B 后期退化

YOLO11n seed123 的 `bstable1e3` 能跑满 800 epoch，但 best AP50-95 只有 `0.56161@165`，last 降到 `0.52875`。这说明降低学习率可以防 NaN，但不能单独解决长训练退化。

BN stats 诊断显示坏 run 的 BN running variance 远高于健康 run，因此加入：

```text
FREEZE_BN_STATS=1
```

训练时冻结 BN running mean/var，保留 BN affine 参数梯度。

### 3. YOLO11n 与 YOLO11s 的差异

YOLO11n 上，BN-freeze 已形成较强正向证据：

| Model | Seed | best AP50-95 | last AP50-95 | 读法 |
|---|---:|---:|---:|---|
| YOLO11n | 0 | `0.57276@793` | `0.57254` | 90 服务器完成，正向 |
| YOLO11n | 42 | `0.57615@400` | `0.57295` | 4090 完成，正向 |
| YOLO11n | 123 | `0.57269@779` | `0.57219` | 90 服务器完成，修复 seed123 退化 |

YOLO11s 上，BN-freeze 没有完全解决后期退化：

| Model | Seed | best AP50-95 | last AP50-95 | SAR baseline | 读法 |
|---|---:|---:|---:|---:|---|
| YOLO11s | 0 | `0.63388@263` | `0.61759` | `0.62897` | best 正向，但 last 低于 baseline |

因此当前最稳的主线证据应放在 YOLO11n；YOLO11s 需要单独排查或重新跑最终稳定协议，不能直接作为完整容量闭环。

## 与主文档的关系

- 主线标准：[docs/experiments/LADD_MAINLINE_STANDARD_CN.md](../../../docs/experiments/LADD_MAINLINE_STANDARD_CN.md)
- 主线结果汇总：[ladd/results/LADD_RESULTS_CN.md](../LADD_RESULTS_CN.md)
- Baseline 规范与状态：[docs/experiments/BASELINE_STANDARD_CN.md](../../../docs/experiments/BASELINE_STANDARD_CN.md)
- 方法定义与实现入口：[docs/method/METHOD_DEFINITIONS_AND_IMPLEMENTATION_CN.md](../../../docs/method/METHOD_DEFINITIONS_AND_IMPLEMENTATION_CN.md)
