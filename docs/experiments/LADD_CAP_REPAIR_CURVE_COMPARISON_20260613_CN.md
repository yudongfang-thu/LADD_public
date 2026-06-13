# LADD cap 主线与后续修改曲线对比

日期：2026-06-13

这份文档把主线 cap 版本 LADD 与后续几轮修改实验放在同一组曲线里比较。重点不是只看最高点，而是同时看 peak、final、best-final gap，以及 detector loss / reach / rec 这些辅助信号是否解释性能漂移。

## 1. 数据与图件

数据汇总：

```text
ladd/results/ladd_cap_repair_curve_comparison_20260613/ladd_cap_repair_curve_summary_20260613.csv
```

绘图脚本：

```text
ladd/results/ladd_cap_repair_curve_comparison_20260613/gen_curve_comparison.py
```

图件：

![s B-stage AP curves](https://cdn.jsdelivr.net/gh/yudongfang-thu/LADD_public@main/docs/experiments/figures/ladd_cap_repair_curve_comparison_20260613/fig1_s_b_stage_ap_curves.png)

![s B-stage detector losses](https://cdn.jsdelivr.net/gh/yudongfang-thu/LADD_public@main/docs/experiments/figures/ladd_cap_repair_curve_comparison_20260613/fig2_s_b_stage_detector_losses.png)

![s A2 AP and auxiliary losses](https://cdn.jsdelivr.net/gh/yudongfang-thu/LADD_public@main/docs/experiments/figures/ladd_cap_repair_curve_comparison_20260613/fig3_s_a2_ap_and_aux_losses.png)

![n B-stage AP curves](https://cdn.jsdelivr.net/gh/yudongfang-thu/LADD_public@main/docs/experiments/figures/ladd_cap_repair_curve_comparison_20260613/fig4_n_b_stage_ap_curves.png)

![m A2 AP curves](https://cdn.jsdelivr.net/gh/yudongfang-thu/LADD_public@main/docs/experiments/figures/ladd_cap_repair_curve_comparison_20260613/fig5_m_a2_ap_curves.png)

![s B final delta](https://cdn.jsdelivr.net/gh/yudongfang-thu/LADD_public@main/docs/experiments/figures/ladd_cap_repair_curve_comparison_20260613/fig6_s_b_final_delta_bar.png)

如果当前网络环境阻止 CDN 图片加载，可直接打开仓库内本地图片：

- [fig1_s_b_stage_ap_curves.png](./figures/ladd_cap_repair_curve_comparison_20260613/fig1_s_b_stage_ap_curves.png)
- [fig2_s_b_stage_detector_losses.png](./figures/ladd_cap_repair_curve_comparison_20260613/fig2_s_b_stage_detector_losses.png)
- [fig3_s_a2_ap_and_aux_losses.png](./figures/ladd_cap_repair_curve_comparison_20260613/fig3_s_a2_ap_and_aux_losses.png)
- [fig4_n_b_stage_ap_curves.png](./figures/ladd_cap_repair_curve_comparison_20260613/fig4_n_b_stage_ap_curves.png)
- [fig5_m_a2_ap_curves.png](./figures/ladd_cap_repair_curve_comparison_20260613/fig5_m_a2_ap_curves.png)
- [fig6_s_b_final_delta_bar.png](./figures/ladd_cap_repair_curve_comparison_20260613/fig6_s_b_final_delta_bar.png)

## 2. YOLO11s：B 阶段修改没有解决 late regression

主线 cap BN-freeze B800 的 best 为 `0.63388@263`，final 为 `0.61759`，best-final gap 达到 `0.01629`。这说明 s 的问题不是“学不到”，而是中后期把已经学到的性能退掉了。

后续几次修改给出了更清楚的拆分：

- `alphaKD0.5/0.25 B400` 的 best 仍在 `0.630` 左右，但 final 分别落到 `0.61802/0.61719`，降低 KD 强度没有解决 final 退化。
- `B det-only r2 B400` best 为 `0.63025@226`，final 为 `0.61923`，说明即使 B 阶段关掉 LADD/KD 辅助项，也仍存在 late regression。
- `repair det-only lr1e-4 B120` best 为 `0.63125@13`，final 为 `0.62556`，短期比 SAR final 高，但仍低于 safe threshold `0.62697`。
- `repair weakKD0.1 B120` best 为 `0.62827@10`，final 为 `0.62267`，没有保住 A2 的高点。

Detector loss 曲线进一步说明：不少 run 的 train detector loss 持续下降，但 AP 在后期下降。这更像 generalization / validation drift，而不是训练 loss 爆炸。

## 3. YOLO11s：A2 的问题可以被低 LR/短 A2 缓解，但 B 仍会再损伤

A2 full50 主线 best/final 为 `0.62664/0.62349`；`s A2 lr3e-4 short13` 能把 A2 final 锁到 `0.63057`。这说明 A2 阶段本身可以通过“低 LR + 短训练”获得更干净的起点。

但后续 B 阶段没有稳定继承这个起点：`short13 + B det-only200` 从 A2 `0.63057` 进入 B 后只到 `0.62436` best、`0.61880` final；repair 的 `det-only lr1e-4 B120` 有短期恢复，但 final 仍回落。结论是：A2 selection 是必要修复，但不是完整修复。

reach/rec 曲线没有出现明显爆炸。`reach_match` 快速下降，`reach_rank` 和 `t_rec+s_rec` 更像平稳收敛或缓慢变化；它们不能单独解释 s 的 B-stage AP 退化。

## 4. YOLO11n：主线 cap 是正向且稳定，repair 不是更强替代

n cap mainline BN-freeze B800 best/final 为 `0.57615/0.57295`，都高于 n SAR baseline `0.55654/0.55127`。`n repair weakKD0.25 B200` 为 `0.56476/0.56419`，方向是正的，但弱于已知 mainline。

这支持一个清晰口径：n 上主线 cap 版本有稳定正证据；repair 变体只是辅助诊断，不应替代 n 主线。

## 5. YOLO11m：短 A2、低 LR、A2 freeze BN 都没有救回来

m 的 A2 曲线没有任何一条达到 safe threshold `0.65380`。少数 run 的 early best 略高于 m SAR final `0.64903`，例如 `m A2 probe` 的 `0.65026` 和 `m A2 lr3e-4 short10` 的 `0.64929`，但 final 都回落到 SAR final 以下。目前最高的 repair 类结果是 `m repair lr1e-4 short5`，best/final 为 `0.64735/0.64416`，仍没过 m SAR final。

这说明 m 的问题不是简单的“训练太长”或“A2 LR 太大”。短 A2、低 LR、det-only、A2 freeze BN 都不能把 m 拉回安全区间，因此 m 不适合直接进入 full B。

## 6. 数值摘要

### s B-stage

| label                         | phase   |   epochs |   best_epoch |   best_map |   last_map |   best_final_drop |   best_minus_baseline_best |   last_minus_baseline_final |
|:------------------------------|:--------|---------:|-------------:|-----------:|-----------:|------------------:|---------------------------:|----------------------------:|
| s cap mainline BN-freeze B800 | B       |      800 |          263 |    0.63388 |    0.61759 |           0.01629 |                    0.00491 |                    -0.00474 |
| s alphaKD0.5 B400             | B       |      400 |          218 |    0.63074 |    0.61802 |           0.01272 |                    0.00177 |                    -0.00431 |
| s alphaKD0.25 B400            | B       |      400 |          199 |    0.63027 |    0.61719 |           0.01308 |                    0.0013  |                    -0.00514 |
| s B det-only r2 B400          | B       |      400 |          226 |    0.63025 |    0.61923 |           0.01102 |                    0.00128 |                    -0.0031  |
| s short13 + B det-only200     | B       |      200 |           84 |    0.62436 |    0.6188  |           0.00556 |                   -0.00461 |                    -0.00353 |
| s repair weakKD0.1 B120       | B       |      120 |           10 |    0.62827 |    0.62267 |           0.0056  |                   -0.0007  |                     0.00034 |
| s repair det-only lr1e-4 B120 | B       |      120 |           13 |    0.63125 |    0.62556 |           0.00569 |                    0.00228 |                     0.00323 |

### s A2

| label                    | phase   |   epochs |   best_epoch |   best_map |   last_map |   best_final_drop |   best_minus_baseline_best |   last_minus_baseline_final |
|:-------------------------|:--------|---------:|-------------:|-----------:|-----------:|------------------:|---------------------------:|----------------------------:|
| s cap mainline A2 full50 | A2      |       50 |           12 |    0.62664 |    0.62349 |           0.00315 |                   -0.00233 |                     0.00116 |
| s A2 det-only full50     | A2      |       50 |           13 |    0.62795 |    0.62222 |           0.00573 |                   -0.00102 |                    -0.00011 |
| s A2 lr3e-4 full50       | A2      |       50 |           13 |    0.63309 |    0.62443 |           0.00866 |                    0.00412 |                     0.0021  |
| s A2 lr3e-4 short13      | A2      |       13 |           13 |    0.63057 |    0.63057 |           0       |                    0.0016  |                     0.00824 |
| s A2 lr1e-4 short15      | A2      |       15 |           13 |    0.63051 |    0.62647 |           0.00404 |                    0.00154 |                     0.00414 |

### n B-stage

| label                         | phase   |   epochs |   best_epoch |   best_map |   last_map |   best_final_drop |   best_minus_baseline_best |   last_minus_baseline_final |
|:------------------------------|:--------|---------:|-------------:|-----------:|-----------:|------------------:|---------------------------:|----------------------------:|
| n cap mainline BN-freeze B800 | B       |      800 |          400 |    0.57615 |    0.57295 |           0.0032  |                    0.01961 |                     0.02168 |
| n repair weakKD0.25 B200      | B       |      200 |          197 |    0.56476 |    0.56419 |           0.00057 |                    0.00822 |                     0.01292 |

### m A2

| label                           | phase   |   epochs |   best_epoch |   best_map |   last_map |   best_final_drop |   best_minus_baseline_best |   last_minus_baseline_final |
|:--------------------------------|:--------|---------:|-------------:|-----------:|-----------:|------------------:|---------------------------:|----------------------------:|
| m A2 probe full50               | A2      |       50 |           10 |    0.65026 |    0.63725 |           0.01301 |                   -0.00554 |                    -0.01178 |
| m A2 det-only full50            | A2      |       50 |            4 |    0.64521 |    0.63892 |           0.00629 |                   -0.01059 |                    -0.01011 |
| m A2 lr3e-4 retry2 interrupted  | A2      |       40 |            4 |    0.64707 |    0.64123 |           0.00584 |                   -0.00873 |                    -0.0078  |
| m A2 lr3e-4 full50 retry3       | A2      |       50 |            8 |    0.64611 |    0.63911 |           0.007   |                   -0.00969 |                    -0.00992 |
| m A2 short10                    | A2      |       10 |            4 |    0.64411 |    0.64276 |           0.00135 |                   -0.01169 |                    -0.00627 |
| m A2 lr3e-4 short10             | A2      |       10 |            4 |    0.64929 |    0.6399  |           0.00939 |                   -0.00651 |                    -0.00913 |
| m repair lr3e-4 short4          | A2      |        4 |            4 |    0.64325 |    0.64325 |           0       |                   -0.01255 |                    -0.00578 |
| m repair lr1e-4 short5          | A2      |        5 |            2 |    0.64735 |    0.64416 |           0.00319 |                   -0.00845 |                    -0.00487 |
| m repair freezeBN short4 AutoDL | A2      |        4 |            4 |    0.64189 |    0.64189 |           0       |                   -0.01391 |                    -0.00714 |

## 7. 汇报建议

1. 先讲现象：s 主线 B800 在中期达到高点，但 final 明显退化；这张图比单表格更能说明 late regression。
2. 再讲第一轮修复：降低 alphaKD、B det-only、B 低 LR 都能改变局部形态，但没有根治 final drift。
3. 然后讲 A2 selection：低 LR + 短 A2 可以得到更好的 A2 起点，说明 A2 不是完全坏掉，但 B 的继承机制仍不稳定。
4. 最后讲容量差异：n 主线稳定，s 有后期退化，m 在 A2 阶段就已经低于安全线。这个容量差异是下一轮方法设计最关键的约束。
