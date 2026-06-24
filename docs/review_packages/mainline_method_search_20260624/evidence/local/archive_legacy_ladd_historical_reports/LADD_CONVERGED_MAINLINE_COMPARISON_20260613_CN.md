# LADD 收敛主线历史对比与曲线补充

日期：2026-06-13

本文件补充两类之前汇报中容易漏掉、但对解释 LADD 主线非常关键的证据：

1. 早期 formal no-mosaic 主线里，`B_FREEZE_BN_STATS=0` 的 LADD 确实有很强的健康 run：`YOLO11n original/no-cap2 seed0` best `0.57821@730`，是目前 n no-mosaic LADD 里最高的单点；`cap2 seed0/42 no-BN-freeze` 也分别达到 `0.57662/0.57420`。但同一设置在 seed123 old-B 上会检测 loss NaN 并 collapse 到 final `0.00000`。
2. 更早的 90 服务器 mosaic100/close@100 主线中，LADD 在六条 n seed/method run 上都没有 collapse，best 相对同协议 SAR baseline 大约提升 `+0.02072` 到 `+0.02750`，这是历史上 LADD 提升最大的主线证据。

因此汇报口径应该是：LADD 不是“从来学不到”；相反，早期 no-BN-freeze 和 mosaic100 都能冲到很高。但 formal no-mosaic 收敛协议下，稳定性从 seed123 开始暴露，BN-freeze 修复了 n 的三 seed 稳定性，同时也降低了一部分峰值；s/m 则仍有容量相关退化。

## 1. 数据与图件

汇总 CSV：

```text
ladd/results/converged_mainline_ladd_20260613/converged_mainline_ladd_summary_20260613.csv
docs/experiments/ladd_converged_mainline_ladd_summary_20260613.csv
ladd/results/converged_mainline_ladd_20260613/converged_mainline_ladd_phase_summary_20260613.csv
docs/experiments/ladd_converged_mainline_ladd_phase_summary_20260613.csv
```

绘图脚本：

```text
ladd/results/converged_mainline_ladd_20260613/gen_converged_mainline_ladd_overview.py
```

图件：

![formal no-mosaic n curves](https://cdn.jsdelivr.net/gh/yudongfang-thu/LADD_public@main/docs/experiments/figures/ladd_converged_mainline_ladd_20260613/fig1_nomosaic_n_no_bnfreeze_bnfreeze_curves.png)

![mosaic100 n curves](https://cdn.jsdelivr.net/gh/yudongfang-thu/LADD_public@main/docs/experiments/figures/ladd_converged_mainline_ladd_20260613/fig2_mosaic100_n_ladd_curves.png)

![all mainline best gain](https://cdn.jsdelivr.net/gh/yudongfang-thu/LADD_public@main/docs/experiments/figures/ladd_converged_mainline_ladd_20260613/fig3_all_mainline_best_gain.png)

![all mainline final gain](https://cdn.jsdelivr.net/gh/yudongfang-thu/LADD_public@main/docs/experiments/figures/ladd_converged_mainline_ladd_20260613/fig4_all_mainline_final_gain.png)

![formal no-mosaic A1/A2/B chain](https://cdn.jsdelivr.net/gh/yudongfang-thu/LADD_public@main/docs/experiments/figures/ladd_converged_mainline_ladd_20260613/fig5_nomosaic_n_a1a2b_stage_chain_ap.png)

![formal no-mosaic A1/A2 entrance diagnostics](https://cdn.jsdelivr.net/gh/yudongfang-thu/LADD_public@main/docs/experiments/figures/ladd_converged_mainline_ladd_20260613/fig6_nomosaic_n_a1a2_entrance_diagnostics.png)

![mosaic100 A1/A2/B chain](https://cdn.jsdelivr.net/gh/yudongfang-thu/LADD_public@main/docs/experiments/figures/ladd_converged_mainline_ladd_20260613/fig7_mosaic100_n_a1a2b_stage_chain_ap.png)

本地图片路径：

- [fig1_nomosaic_n_no_bnfreeze_bnfreeze_curves.png](./figures/ladd_converged_mainline_ladd_20260613/fig1_nomosaic_n_no_bnfreeze_bnfreeze_curves.png)
- [fig2_mosaic100_n_ladd_curves.png](./figures/ladd_converged_mainline_ladd_20260613/fig2_mosaic100_n_ladd_curves.png)
- [fig3_all_mainline_best_gain.png](./figures/ladd_converged_mainline_ladd_20260613/fig3_all_mainline_best_gain.png)
- [fig4_all_mainline_final_gain.png](./figures/ladd_converged_mainline_ladd_20260613/fig4_all_mainline_final_gain.png)
- [fig5_nomosaic_n_a1a2b_stage_chain_ap.png](./figures/ladd_converged_mainline_ladd_20260613/fig5_nomosaic_n_a1a2b_stage_chain_ap.png)
- [fig6_nomosaic_n_a1a2_entrance_diagnostics.png](./figures/ladd_converged_mainline_ladd_20260613/fig6_nomosaic_n_a1a2_entrance_diagnostics.png)
- [fig7_mosaic100_n_a1a2b_stage_chain_ap.png](./figures/ladd_converged_mainline_ladd_20260613/fig7_mosaic100_n_a1a2b_stage_chain_ap.png)

## 2. 早期 no-BN-freeze 主线：峰值强，但 seed123 暴露崩溃

`B_FREEZE_BN_STATS=0` 不是简单的坏设置。它在健康 seed 上峰值很强：

- `YOLO11n original/no-cap2 seed0`: best `0.57821@730`, last `0.57517`, 相对 n SAR seed0 best `+0.02167`。
- `YOLO11n cap2 seed0 no-BN-freeze`: best `0.57662@725`, last `0.57504`, 相对 n SAR seed0 best `+0.02008`。
- `YOLO11n cap2 seed42 no-BN-freeze`: best `0.57420@735`, last `0.57293`, 相对 n SAR seed42 best `+0.01626`。

问题出在稳定性边界：同样主线在 `seed123 old-B` 上记录到 epoch 483，best 只有 `0.52182@1`，final 为 `0.00000`，对应历史诊断中的 detection loss NaN / last.pt NaN-Inf 事件。随后 `bstable1e3` 能跑满 800，但 best/final 为 `0.56161/0.52875`，说明只降 B LR 能防 NaN，却不能防 late regression。

BN-freeze 的作用因此更像稳定性修复，而不是单纯涨点技巧：它让 n seed0/42/123 都回到正收益闭环，但 seed0 峰值低于 no-BN-freeze 的健康 run。

### A1/A2/B 连续链读法

补充的 A1->A2->B 连续曲线说明：`seed123 old-B` 的 A1/A2 入口并不坏。`seed123` 在 A1 last 为 `0.56128`，A2 best/last 为 `0.56574/0.56574`，已经高于同 seed SAR baseline best `0.56128`；崩溃发生在进入旧 B 后。`seed123 bstable1e3` 说明降低 B 学习率可以避免直接 NaN，但它的 B best/final 只有 `0.56161/0.52875`，仍然明显 late-regress。因此更准确的表述是：

- 降低 LR / 去 warmup：解决“旧 B 高 LR 路径直接 NaN”的一部分数值稳定问题。
- 只降低 LR：不能解决 seed123 的长期退化。
- BN-freeze：在 n seed123 上把同一个 A1/A2 入口稳定到 B best/final `0.57269/0.57219`，是目前 n 三 seed 闭环的关键稳定修复。
- 代价：BN-freeze seed0 峰值 `0.57276` 低于健康 no-BN-freeze seed0 `0.57662` 和 original/no-cap2 seed0 `0.57821`，所以它更像保守稳定化，而不是最高性能设置。

## 3. mosaic100/close@100：历史提升最大且没有 collapse

mosaic100/close@100 主线的同协议 SAR baseline best/final 为 `0.54091/0.53836`。六条 LADD B run 都稳定在 `0.56+`：

- legacy mean best `0.56631`，平均 gain `+0.02540`。
- cap2 mean best `0.56601`，平均 gain `+0.02510`。
- 单条最高为 `cap2 seed0` best `0.56841@798`，相对同协议 SAR best `+0.02750`。

这批结果很重要，因为它说明 LADD 在“带 mosaic 前 100 epoch、后 700 epoch 收敛”的旧协议下既能涨点，也没有当前 formal no-mosaic 中 seed123/s/m 暴露的崩溃或后期退化模式。后续汇报中应把它作为反证：问题不应被描述成 LADD 机制天然不可训练，而是训练协议、BN running stats、容量和阶段设置的交互。

## 4. 全部主线表

### mosaic100 / close@100

| protocol   |   server | model   |   seed | method   | bn_stats   | status   |   epochs_recorded |   best_epoch |   best_map |   last_map |   sar_baseline_best |   best_minus_sar_best |   last_minus_sar_final |   best_final_drop | notes                                                                 |
|:-----------|---------:|:--------|-------:|:---------|:-----------|:---------|------------------:|-------------:|-----------:|-----------:|--------------------:|----------------------:|-----------------------:|------------------:|:----------------------------------------------------------------------|
| mosaic100  |       90 | n       |      0 | cap2     | normal     | complete |               800 |          798 |    0.56841 |    0.56792 |             0.54091 |               0.0275  |                0.02956 |           0.00049 | largest historical LADD gain under old close@100 protocol             |
| mosaic100  |       90 | n       |     42 | cap2     | normal     | complete |               800 |          750 |    0.56799 |    0.56044 |             0.54091 |               0.02708 |                0.02208 |           0.00755 | old close@100 protocol; no collapse                                   |
| mosaic100  |       90 | n       |    123 | cap2     | normal     | complete |               800 |          800 |    0.56163 |    0.56163 |             0.54091 |               0.02072 |                0.02327 |           0       | old close@100 protocol; no collapse                                   |
| mosaic100  |       90 | n       |      0 | legacy   | normal     | complete |               755 |          746 |    0.56678 |    0.56638 |             0.54091 |               0.02587 |                0.02802 |           0.0004  | old close@100 protocol; mosaic open for first 100 epochs; no collapse |
| mosaic100  |       90 | n       |     42 | legacy   | normal     | complete |               800 |          763 |    0.56688 |    0.5592  |             0.54091 |               0.02597 |                0.02084 |           0.00768 | old close@100 protocol; no collapse                                   |
| mosaic100  |       90 | n       |    123 | legacy   | normal     | complete |               800 |          770 |    0.56526 |    0.5622  |             0.54091 |               0.02435 |                0.02384 |           0.00306 | old close@100 protocol; no collapse                                   |

### formal no-mosaic: YOLO11n

| protocol        | server   | model   |   seed | method           | bn_stats   | status          |   epochs_recorded |   best_epoch |   best_map |   last_map |   sar_baseline_best |   best_minus_sar_best |   last_minus_sar_final |   best_final_drop | notes                                                                                       |
|:----------------|:---------|:--------|-------:|:-----------------|:-----------|:----------------|------------------:|-------------:|-----------:|-----------:|--------------------:|----------------------:|-----------------------:|------------------:|:--------------------------------------------------------------------------------------------|
| formal_nomosaic | 90       | n       |      0 | cap2_a2mu1e3     | normal     | complete        |               800 |          725 |    0.57662 |    0.57504 |             0.55654 |               0.02008 |                0.02377 |           0.00158 | healthy no-BN-freeze seed0; strong best/final                                               |
| formal_nomosaic | 90       | n       |     42 | cap2_a2mu1e3     | normal     | complete        |               800 |          735 |    0.5742  |    0.57293 |             0.55794 |               0.01626 |                0.01849 |           0.00127 | healthy no-BN-freeze seed42; positive but below seed0                                       |
| formal_nomosaic | 90       | n       |      0 | cap2_bnfreeze1e3 | freeze     | complete        |               800 |          793 |    0.57276 |    0.57254 |             0.55654 |               0.01622 |                0.02127 |           0.00022 | stable BN-freeze candidate; slightly lower peak than healthy no-freeze seed0                |
| formal_nomosaic | dual4090 | n       |     42 | cap2_bnfreeze1e3 | freeze     | complete        |               800 |          400 |    0.57615 |    0.57295 |             0.55794 |               0.01821 |                0.01851 |           0.0032  | stable BN-freeze seed42; cross-machine evidence                                             |
| formal_nomosaic | 90       | n       |    123 | cap2_bnfreeze1e3 | freeze     | complete        |               800 |          779 |    0.57269 |    0.57219 |             0.56128 |               0.01141 |                0.01143 |           0.0005  | BN-freeze fixes seed123 collapse/late regression into positive stable run                   |
| formal_nomosaic | 90       | n       |    123 | cap2_bstable1e3  | normal     | late_regression |               800 |          165 |    0.56161 |    0.52875 |             0.56128 |               0.00033 |               -0.03201 |           0.03286 | MuSGD lr1e-3 prevents NaN but final severely regresses                                      |
| formal_nomosaic | 90       | n       |    123 | cap2_old_b       | normal     | nan_crash       |               483 |            1 |    0.52182 |    0       |             0.56128 |              -0.03946 |               -0.56076 |           0.52182 | old B default/high-LR path; detection loss NaN around epoch 429 and final AP collapses to 0 |
| formal_nomosaic | 90       | n       |      0 | original         | normal     | complete        |               800 |          730 |    0.57821 |    0.57517 |             0.55654 |               0.02167 |                0.0239  |           0.00304 | highest n no-mosaic LADD best AP; kept as ablation/diagnostic, not final mainline           |

### formal no-mosaic: YOLO11s / YOLO11m

| protocol        | server   | model   |   seed | method           | bn_stats   | status           |   epochs_recorded |   best_epoch |   best_map |   last_map |   sar_baseline_best |   best_minus_sar_best |   last_minus_sar_final |   best_final_drop | notes                                                                     |
|:----------------|:---------|:--------|-------:|:-----------------|:-----------|:-----------------|------------------:|-------------:|-----------:|-----------:|--------------------:|----------------------:|-----------------------:|------------------:|:--------------------------------------------------------------------------|
| formal_nomosaic | 90       | m       |      0 | cap2_a2mu1e3     | normal     | abnormal_partial |               121 |            1 |    0.59796 |    0.52361 |             0.6558  |              -0.05784 |               -0.12542 |           0.07435 | B entrance is already far below m SAR baseline; not valid mainline result |
| formal_nomosaic | 90       | s       |      0 | cap2_a2mu1e3     | normal     | partial_positive |               608 |          605 |    0.63551 |    0.63527 |             0.62897 |               0.00654 |                0.01294 |           0.00024 | stopped at epoch 608; positive mid/late evidence, but not full closure    |
| formal_nomosaic | dual4090 | s       |      0 | cap2_bnfreeze1e3 | freeze     | late_regression  |               800 |          263 |    0.63388 |    0.61759 |             0.62897 |               0.00491 |               -0.00474 |           0.01629 | full 800; best positive but final below SAR final                         |

## 5. 汇报时建议放法

1. 先放 A1/A2/B 连续链：seed123 的 A1/A2 入口是健康的，分叉发生在 B；这比只展示 B 阶段更清楚。
2. 再放 no-mosaic n B-stage zoom 图：健康 no-BN-freeze run 很强，但 seed123 old-B 崩溃；这解释为什么“最开始性能最好”与“后来必须稳定修复”并不矛盾。
3. 再放 mosaic100 图：旧协议下 LADD 六条都正向，且提升最大；这说明方法潜力存在，当前问题是 formal no-mosaic 主线的新稳定性问题。
4. 最后放总表：BN-freeze 是 n 三 seed 闭环最稳的主线候选；s 的 BN-freeze 虽有 positive best，但 final 低于 SAR final；m 从 B 入口就异常，不能进入 full mainline。
