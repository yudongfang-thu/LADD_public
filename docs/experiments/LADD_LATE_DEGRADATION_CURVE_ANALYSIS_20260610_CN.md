# LADD 后期退化曲线诊断记录

日期：2026-06-10

用途：把当前 no-mosaic/H1 退化 run 与旧 90 服务器 mosaic/close@100 稳定 run 的后期 loss 曲线做轻量对比，判断退化更像优化崩溃、BN 污染，还是 late overfit / 泛化退化。

本记录只使用已归档的轻量证据，不包含 checkpoint 权重。

## 1. 原始输入与分析日志

分析 artifact：

```text
docs/experiments/ladd_late_degradation_curve_20260610_artifacts/
```

包含：

- `curve_run_summary.csv`：每条 run 的 best/last AP、掉点和最后 100 epoch AP 斜率。
- `curve_best_late_window_delta.csv`：best epoch 附近 41 epoch 窗口与最后 100 epoch 的 loss 均值差。
- `curve_key_epoch_points.csv`：关键 epoch 的 AP、train/val loss、KD loss、LR。
- `h1_s_b400_diagnostic_keypoints.csv`：H1 s b400 的 BN / grad / KD 诊断关键点。
- `curve_analysis_command_log.txt`：本次分析过程的轻量日志。

主要原始 CSV：

| Run | 原始文件 |
|---|---|
| H1 `s b400` | `docs/experiments/ladd_h1_diag_20260610_artifacts/h1_s_b400/results.csv` |
| H1 `s b400` 诊断 | `docs/experiments/ladd_h1_diag_20260610_artifacts/h1_s_b400/ladd_diagnostics.csv` |
| H1 `n b100` | `docs/experiments/ladd_h1_diag_20260610_artifacts/h1_n_b100/results.csv` |
| no-mosaic BN-freeze `s b800` | `ladd/results/mainline_stability_20260609/remote_4090dual/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11s/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11s_cap2_s0_bnfreeze1e3_public4090dual_final_v2_b_e800_b64_s0_gpu1/results.csv` |
| no-mosaic BN-freeze `n b800` | `ladd/results/mainline_stability_20260609/remote_4090dual/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11n/cap2/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s42_bnfreeze1e3_public4090dual_final_v2_b_e800_b64_s42_gpu0/results.csv` |
| 旧 90 mosaic/close@100 cap2 | `docs/experiments/ladd_mosaic90_20260528_artifacts/ladd_b_runs/*/results.csv` |

H1 `s b400` 的 master/outer/chain logs 已在同一 artifact 目录中归档。4090 dual mainline stability 包目前只有拉取后的 `results.csv`/`args.yaml`，没有额外完整 stdout log。

## 2. Run 级概览

| Run | best AP50-95 | last AP50-95 | last-best | 读法 |
|---|---:|---:|---:|---|
| H1 `s b400` | 0.63127@216 | 0.62036@400 | -0.01091 | 400 epoch 内已出现明显后期退化 |
| H1 `n b100` | 0.55872@100 | 0.55872@100 | +0.00000 | smoke 通过，不能观察长尾退化 |
| BN-freeze `s b800` | 0.63388@263 | 0.61759@800 | -0.01629 | s 容量跑满后明显退化 |
| BN-freeze `n b800` | 0.57615@400 | 0.57295@800 | -0.00320 | n 也有轻微回落，但仍稳定高于 baseline |
| 90 mosaic cap2 seed0 | 0.56841@798 | 0.56792@800 | -0.00049 | 后期基本稳定 |
| 90 mosaic cap2 seed42 | 0.56799@750 | 0.56044@800 | -0.00755 | 有回落，但仍高于同协议 SAR baseline |
| 90 mosaic cap2 seed123 | 0.56163@800 | 0.56163@800 | +0.00000 | 后期持续爬升到最后 |

## 3. 后期 loss 形态

关键对比不是 AP 掉了多少，而是 AP 掉的时候 loss 如何变化：

| Run | train box | train cls | train KD | val box | val cls |
|---|---:|---:|---:|---:|---:|
| H1 `s b400` late100 vs best window | -11.1% | -9.0% | -11.2% | +1.5% | +8.2% |
| BN-freeze `s b800` late100 vs best window | -21.3% | -17.8% | -27.2% | +2.5% | +17.4% |
| BN-freeze `n b800` late100 vs best window | -12.7% | -11.0% | -16.7% | +2.3% | +5.2% |
| 90 mosaic cap2 seed0 late100 vs best window | +2.1% | +2.0% | +2.0% | -0.5% | -0.0% |
| 90 mosaic cap2 seed123 late100 vs best window | +1.9% | +1.8% | +2.1% | -0.2% | -0.1% |

退化 run 的共同形态是：

- train `box/cls` loss 继续下降；
- `train/kd_loss` 继续下降；
- 但 validation loss，尤其 `val/cls_loss`，在 best epoch 后上升；
- AP 随着 validation loss 漂移而下降。

因此当前 no-mosaic/H1 的 s 退化更像 late overfit / 泛化退化，而不是检测 loss 爆炸或 KD loss 爆炸。

## 4. H1 s b400 诊断项

`h1_s_b400/ladd_diagnostics.csv` 显示：

- `nan_or_inf_detected` 全程为 0；
- BN running stats 在 H1 诊断中保持常量，例如 `bn_running_var_max=155.75`；
- `kd_loss` 从 epoch 1 的 `1.13078` 降到 epoch 400 的 `0.09803`；
- best epoch 后，`grad_norm_total` 有波动但没有 NaN/Inf 标记。

这说明 H1 修复后，s 的后期退化不能再简单解释为 BN running stats 继续被污染，也不是数值崩溃。BN-freeze 仍然是 n 的稳定修复，但它不是 s 容量的充分修复。

## 5. 与旧 90 mosaic/close@100 的差异

旧 90 mosaic/close@100 cap2 run 的形态明显不同：

- AP 从较低点持续爬升，best 多数出现在 750-800 epoch；
- 后期 `val/cls_loss` 基本持平，seed0/123 甚至略降；
- 没有出现 no-mosaic s run 里 train/KD 继续变好但 val cls 明显变差的形态。

这支持一个重要判断：LADD 并非机制上必然 B 阶段崩溃；当前问题更可能来自 no-mosaic 协议、长尾学习率/训练时长、KD 强度、容量差异和泛化之间的交互。

## 6. 对后续实验的含义

当前最有信息量的下一组实验仍然是 `P2_s`：

| 实验 | 验证问题 |
|---|---|
| `diag_h1_s_seed0_alpha_kd_0p5_b400` | 降低 KD 强度能否缓解 `val/cls_loss` 后期上升 |
| `diag_h1_s_seed0_alpha_kd_0p25_b400` | 如果 0.5 仍退化，进一步判断 KD 牵引强弱 |
| `diag_h1_s_seed0_detonly_b400` | 判断退化是否即使无 KD 也存在，即 no-mosaic+s 自身训练长尾问题 |

如果 det-only 不退化，而 KD 降权后退化减轻，则问题主要是 KD 对 s 容量后期泛化的牵引过强。如果 det-only 也退化，则应优先检查 no-mosaic 下 s 容量的训练时长、cos LR tail、regularization 和 checkpoint selection。
