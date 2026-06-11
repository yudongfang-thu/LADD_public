# Formal Protocol Late-Regression 分析（2026-06-12）

## 数据源与口径

输入数据：

- `ladd/results/ladd90_formal_baselines_20260612/summary/ladd90_formal_baseline_summary_20260612.csv`
- `docs/experiments/LADD_CAPACITY_KD_DIAG_RESULTS_20260611_CN.md`
- `docs/experiments/ladd_capacity_kd_results_20260611_snapshot.csv`
- `ladd/results/capacity_kd_20260611/*/results.csv`
- `comparison/FORMAL_TRANSFER_STATUS_20260611_CN.md`
- 本地已有 HalluciDet archived partial `results.csv`

计算口径：

- `best-final drop = best AP50-95 - final/latest AP50-95`。
- `method final - SAR baseline best` 与 `method final - SAR baseline final` 使用同模型同 seed SAR baseline；若 comparison snapshot 只有 latest，则 latest 作为当前 final/latest。
- `excess drop = method drop - SAR baseline drop`。
- `last20/50/100 slope` 使用窗口首末 AP 差除以 epoch 差；同时在 CSV 中保留窗口 delta。
- loss delta 使用 5-epoch 窗口：best epoch 前后各 2 个 epoch 的均值，与最后 5 个 epoch 均值比较，记录 `final-window - best-window`。
- LADD `s B det-only r2` 的 B 结果来自 2026-06-11 20:20 snapshot，不代表完整 400 epoch 终值。
- LD/FGD/HalluciDet-style 的 s/n comparison 行来自 2026-06-11 状态文档；其中 running 行是 latest snapshot，不是完成结果。

完整机器可读表：`docs/experiments/formal_protocol_late_regression_summary_20260612.csv`。

## 1. 90 Formal Baseline 本身存在 Best-Final Gap

90 服务器 formal no-mosaic 800ep baseline 的 SAR seed0 结果：

| model | best | best_ep | final | drop | last100_delta | val_loss_delta |
| --- | --- | --- | --- | --- | --- | --- |
| yolo11l | 0.65427 | 735 | 0.64892 | 0.00535 | -0.00373 | 0.02075 |
| yolo11m | 0.65580 | 704 | 0.64903 | 0.00677 | -0.00604 | 0.03258 |
| yolo11n | 0.55654 | 734 | 0.55127 | 0.00527 | -0.00186 | 0.03233 |
| yolo11s | 0.62897 | 729 | 0.62233 | 0.00664 | -0.00494 | 0.03418 |
| yolo11x | 0.65867 | 685 | 0.64801 | 0.01066 | -0.00867 | 0.03980 |

RGB seed0 结果：

| model | best | best_ep | final | drop | last100_delta | val_loss_delta |
| --- | --- | --- | --- | --- | --- | --- |
| yolo11l | 0.68356 | 618 | 0.66892 | 0.01464 | -0.01352 | 0.06678 |
| yolo11m | 0.67909 | 663 | 0.67159 | 0.00750 | -0.00691 | 0.02867 |
| yolo11n | 0.63018 | 723 | 0.62737 | 0.00281 | -0.00136 | 0.02372 |
| yolo11s | 0.65768 | 647 | 0.64958 | 0.00810 | -0.00637 | 0.03626 |
| yolo11x | 0.68284 | 539 | 0.65820 | 0.02464 | -0.01516 | 0.09711 |

s/m/l/x 的 SAR 与 RGB baseline drop 对比：

| modality | model | best | best_ep | final | drop | last50_delta | last100_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sar | yolo11s | 0.62897 | 729 | 0.62233 | 0.00664 | -0.00533 | -0.00494 |
| rgb | yolo11s | 0.65768 | 647 | 0.64958 | 0.00810 | -0.00446 | -0.00637 |
| sar | yolo11m | 0.65580 | 704 | 0.64903 | 0.00677 | -0.00188 | -0.00604 |
| rgb | yolo11m | 0.67909 | 663 | 0.67159 | 0.00750 | -0.00428 | -0.00691 |
| sar | yolo11l | 0.65427 | 735 | 0.64892 | 0.00535 | -0.00486 | -0.00373 |
| rgb | yolo11l | 0.68356 | 618 | 0.66892 | 0.01464 | -0.00837 | -0.01352 |
| sar | yolo11x | 0.65867 | 685 | 0.64801 | 0.01066 | -0.00540 | -0.00867 |
| rgb | yolo11x | 0.68284 | 539 | 0.65820 | 0.02464 | -0.00737 | -0.01516 |

结论：formal no-mosaic 800ep 协议本身有 late-regression/best-final gap。SAR seed0 中，YOLO11s baseline drop 为 `0.00664`，YOLO11m baseline drop 为 `0.00677`。这意味着主表使用 best AP 是合理的；只看最后一轮会系统性低估已收敛 detector。

## 2. LADD Capacity KD 的额外退化

| method | phase | best | best_ep | final/latest | drop | final-SARbest | final-SARfinal | excess_drop | last50_delta | val_loss_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s alpha_kd=0.5 B400 | B | 0.63074 | 218 | 0.61802 | 0.01272 | -0.01095 | -0.00431 | 0.00608 | -0.00229 | 0.04225 |
| s alpha_kd=0.25 B400 | B | 0.63027 | 199 | 0.61719 | 0.01308 | -0.01178 | -0.00514 | 0.00644 | -0.00152 | 0.04419 |
| s B det-only r2 | A1 | 0.62878 | 1 | 0.62878 | 0.00000 | -0.00019 | 0.00645 | -0.00664 | 0.00000 | 0.00000 |
| s B det-only r2 | A2 | 0.62400 | 50 | 0.62400 | 0.00000 | -0.00497 | 0.00167 | -0.00664 | 0.00723 | 0.00241 |
| s B det-only r2 | B | 0.62244 | 68 | 0.62191 | 0.00053 | -0.00706 | -0.00042 | -0.00611 | 0.01582 | -0.00122 |
| m A2 probe | A2 | 0.65026 | 10 | 0.63725 | 0.01301 | -0.01855 | -0.01178 | 0.00624 | 0.00442 | -0.04985 |
| m A2 probe | B1 | 0.62528 | 1 | 0.62528 | 0.00000 | -0.03052 | -0.02375 | -0.00677 |  | 0.00000 |

关键判断：

1. `s alpha_kd=0.5 B400` 与 `s alpha_kd=0.25 B400` 的 best 都略高于 s SAR baseline best，但 final 分别低于 s SAR baseline final。这不是普通 protocol late-regression 可以完全解释的现象，因为它们的 excess drop 明显大于 s baseline drop。
2. `s B det-only r2` 的 A2 已经低于 s baseline best 与 baseline final；因此 B 阶段即使 det-only，也继承了一个已损伤的 A2 checkpoint。这个 run 不能证明 B det-only 本身有害，只能说明链条状态已经被 A2 改变。
3. `m A2 probe` 的 A2 best 与 final 都低于 m baseline best/final；m 的损伤在 A2 阶段已经出现。

## 3. Comparison 方法的 Late-Regression 状态

以下 comparison 行来自已有状态 summary；running 行使用 latest 作为当前 final/latest。

| method | model | best | best_ep | latest/final | drop | final-SARbest | final-SARfinal | excess_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fgd | yolo11n | 0.52982 | 1 | 0.38095 | 0.14887 | -0.17559 | -0.17032 | 0.14360 |
| ld | yolo11n | 0.57035 | 640 | 0.56989 | 0.00046 | 0.01335 | 0.01862 | -0.00481 |
| hallucidet_style | yolo11n | 0.57365 | 785 | 0.57239 | 0.00126 | 0.01585 | 0.02112 | -0.00401 |
| fgd | yolo11s | 0.61673 | 1 | 0.58786 | 0.02887 | -0.04111 | -0.03447 | 0.02223 |
| ld | yolo11s | 0.64390 | 612 | 0.64386 | 0.00004 | 0.01489 | 0.02153 | -0.00660 |
| hallucidet_style | yolo11s | 0.64310 | 639 | 0.63124 | 0.01186 | 0.00227 | 0.00891 | 0.00522 |

现有 comparison 证据显示：

- LD 的 n/s latest 与 best 接近，late-regression 较小。
- HalluciDet-style 的 n final 与 best 接近，s 有明显 best-final gap，因此正式主表也应使用 best AP。
- FGD 的 n/s 当前 snapshot best 出现在 epoch 1，latest 明显低于 best；该曲线需要单独解释，不应只用 latest 代表方法能力。

## 4. 为什么主表应该用 Best AP

90 formal baseline 已经证明，协议本身会在 late stage 出现 best-final gap，尤其大容量和 RGB teacher 也会出现最后 AP 低于历史 best 的情况。若主表使用 last AP，会把 protocol late-regression 与方法质量混在一起。

因此：

- 主表、跨方法排名、claim 支撑应使用 best AP50-95。
- final/last AP 应保留在诊断表中，用于观察稳定性、late-regression 与是否存在额外退化。
- 对 LADD A2/B 诊断，必须同时比较 SAR baseline best 与 SAR baseline final。

## 5. 对 A2 阈值解释的修正

当前 A2 诊断不应只用一个 safe threshold。更稳妥的解释方式是：

- 与 SAR baseline best 比较：判断方法是否达到同协议 detector 的最佳能力上界附近。
- 与 SAR baseline final 比较：判断 final/last 是否只是落在协议自然 late-regression 范围内。
- 与 SAR baseline drop 比较：判断是否存在超过 protocol late-regression 的额外损伤。

例如 s baseline best/final/drop 是 `0.62897 / 0.62233 / 0.00664`。如果某个 LADD A2/B run 的 best 可以超过 0.62697 但 final 跌到 baseline final 以下，应该被标记为“可冲高但后期不稳定”，而不是简单判定为完全失败或完全通过。

## 6. 下一步建议

1. A2 损伤定位应增加 shorter-A2 或 A2 early-stop probe；当前 capacity KD 结果已经说明只看 best 不够，final 与 excess drop 会揭示额外退化。
2. 对 m 模型应优先检查 A2 阶段优化状态，因为 `m A2 probe` 的 A2 best 与 final 均低于 m baseline best/final。
3. 后续所有 method summary 同时报告 best、final、drop、excess drop，避免把 protocol late-regression 误读为方法崩溃，或反过来掩盖 LADD 的额外退化。
