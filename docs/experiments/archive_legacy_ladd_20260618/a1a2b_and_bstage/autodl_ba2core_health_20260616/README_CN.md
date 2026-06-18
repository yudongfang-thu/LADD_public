# AutoDL no-mosaic YOLO-init skipA2/A2-core 对比

日期：2026-06-16

## 结论

AutoDL 上当前运行的 `formal_nomosaic_yolo11n_cap2_s0_yoloinit_a1_bA2core_B800_autodl_20260616` 可以和 no-mosaic 同协议 LADD 曲线比较。

它的协议是：

- 模型：YOLO11n
- seed：0
- 增强：`mosaic=0.0, close_mosaic=0, mixup=0.0, cutmix=0.0`
- B 阶段 schedule：800 epoch, `lr0=0.001`, `lrf=0.01`, `optimizer=MuSGD`
- 初始化链路：`yolo11n.pt -> A1 best -> B`
- B 阶段启用：`--ladd-b-a2-core`

所以它更准确的命名应是：

```text
YOLO-init A1 -> B(A2-core)
```

它不是纯粹的：

```text
yolo11n.pt -> B
```

也不是完整的：

```text
A1 -> A2 -> B
```

## 最新快照

最新同步自 AutoDL：

- B epoch：270
- AP50：0.75853
- AP：0.49436

同协议主要对照：

| run | latest epoch | latest AP | best AP |
|---|---:|---:|---:|
| YOLO-init A1 -> B(A2-core), AutoDL | 270 | 0.49436 | 0.49436 |
| N0 YOLO-init det-only B800sched | 332 | 0.45155 | 0.45155 |
| N1 SAR baseline-best continue | 332 | 0.57494 | 0.57521 |
| N2 A2-best continue | 229 | 0.54271 | 0.55681 |
| N2 A2-last continue | 319 | 0.46290 | 0.56073 |
| N3 YOLO-init + A2 decomp | 360 | 0.46670 | 0.46670 |

## 判断

这条 AutoDL 线说明 `YOLO-init A1 -> B(A2-core)` 明显优于 YOLO-init det-only 和旧的 YOLO-init + A2 decomp 线，说明 A1 加 B 内 A2-core 是有效的。

但它目前还没有追上以 SAR baseline 或 A2 checkpoint 为入口的 no-mosaic LADD：

- 距离 `N1 SAR baseline-best continue` 还有约 0.081 AP；
- 距离 `N2 A2-best continue` 的 best 还有约 0.062 AP；
- 相比健康的 historical no-mosaic LADD best 约 0.576，仍低约 0.082 AP。

因此它可以作为同协议对比线，但现在不能证明“YOLO-init skipA2/A2-core 优于完整 LADD”。更合理的定位是：它验证了更短链路有恢复能力，但需要继续跑到 800 epoch，并至少补一个同协议 `A1 best -> B(A2-core)` 与 `A1 -> A2 -> B` 的成对对照。

## 文件

- 曲线图：`figures/fig1_autodl_ba2core_ap_early_compare.png`
- loss 健康图：`figures/fig2_autodl_ba2core_loss_health.png`
- 汇总表：`autodl_ba2core_health_curve_summary_20260616.csv`
- 最新 AutoDL B 结果：`raw/current/b/results.csv`
- 最新 AutoDL B diagnostics：`raw/current/b/ladd_diagnostics.csv`

