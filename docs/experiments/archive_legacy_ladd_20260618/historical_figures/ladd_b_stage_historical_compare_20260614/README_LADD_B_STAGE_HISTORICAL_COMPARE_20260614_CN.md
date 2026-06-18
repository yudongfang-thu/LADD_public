# LADD B 阶段历史曲线对比（2026-06-14）

本地报告对比当前 B-entrance 诊断实验和之前 LADD 主线 B 阶段记录。

关键读法：
- 当前 B-entrance 实验从收敛 baseline 或 selected checkpoint 出发，所以 epoch 1 AP 已经接近 SAR baseline 区间。
- 历史 mosaic100 LADD B 从更低的 A2 末端开始，在 B 阶段长期恢复；它的上升曲线不能直接等价于从收敛 detector 做 B-only continuation。
- 历史 no-mosaic n 主线说明完整 A1/A2/B 链条下 B 仍有长程增长，尤其和当前 split-load 设置不是同一个实验。
- 历史 s BN-freeze 的 best 为正，但 best-final drop 很大；这和当前 s 入口实验的担心一致：不崩溃也可能 late-regress。

生成图：
- `fig1_n_current_vs_historical_b_ap.png/pdf`
- `fig2_s_current_vs_historical_b_ap.png/pdf`
- `fig3_selected_b_loss_zoom_120.png/pdf`

汇总表：

| label                                      | model   | family              |   epochs |   first_epoch_ap |    best |   best_epoch |    last |   best_final_drop |   best_first120 |   best_first120_epoch |
|:-------------------------------------------|:--------|:--------------------|---------:|-----------------:|--------:|-------------:|--------:|------------------:|----------------:|----------------------:|
| N1 current: SAR baseline cont. B100        | n       | current             |      100 |          0.54123 | 0.56615 |           99 | 0.56594 |           0.00021 |         0.56615 |                    99 |
| N2 current: A2-best cont. B100             | n       | current             |      100 |          0.51911 | 0.55872 |          100 | 0.55872 |           0.00000 |         0.55872 |                   100 |
| N3 current: SAR-base + A2-last decomp B100 | n       | current             |      100 |          0.51435 | 0.55722 |          100 | 0.55722 |           0.00000 |         0.55722 |                   100 |
| N4 current: N3 + KD ramp B120              | n       | current             |      120 |          0.53806 | 0.56379 |          113 | 0.56311 |           0.00068 |         0.56379 |                   113 |
| old n: mosaic100 cap2 s0 B800              | n       | old_mosaic          |      800 |          0.39969 | 0.56841 |          798 | 0.56792 |           0.00049 |         0.41054 |                   120 |
| old n: mosaic100 legacy s0 B755            | n       | old_mosaic          |      755 |          0.39806 | 0.56678 |          746 | 0.56638 |           0.00040 |         0.41128 |                   120 |
| old n: no-mosaic cap2 s0 no-BN-freeze B800 | n       | old_nomosaic        |      800 |          0.53675 | 0.57662 |          725 | 0.57504 |           0.00158 |         0.53675 |                     1 |
| old n: no-mosaic cap2 s0 BN-freeze B800    | n       | old_nomosaic        |      800 |          0.52679 | 0.57276 |          793 | 0.57254 |           0.00022 |         0.55213 |                   117 |
| old n: no-mosaic s123 old-B crash          | n       | old_crash           |      483 |          0.52182 | 0.52182 |            1 | 0.00000 |           0.52182 |         0.52182 |                     1 |
| old n: no-mosaic s123 B-lr1e-3 late-reg.   | n       | old_late_regression |      800 |          0.53575 | 0.56161 |          165 | 0.52875 |           0.03286 |         0.56058 |                   113 |
| S1 current: SAR baseline cont. B100        | s       | current             |      100 |          0.61471 | 0.62493 |           62 | 0.62238 |           0.00255 |         0.62493 |                    62 |
| S2 current: A2-best cont. B100             | s       | current             |      100 |          0.60866 | 0.62599 |           54 | 0.62174 |           0.00425 |         0.62599 |                    54 |
| S3 current: SAR-base + A2-last decomp B100 | s       | current             |      100 |          0.60625 | 0.62553 |           65 | 0.62262 |           0.00291 |         0.62553 |                    65 |
| S4 current: S3 + KD ramp B120              | s       | current             |      120 |          0.61462 | 0.62521 |           62 | 0.62111 |           0.00410 |         0.62521 |                    62 |
| old s: no-mosaic cap2 s0 no-BN-freeze B608 | s       | old_nomosaic        |      608 |          0.60675 | 0.63551 |          605 | 0.63527 |           0.00024 |         0.60675 |                     1 |
| old s: no-mosaic cap2 s0 BN-freeze B800    | s       | old_nomosaic        |      800 |          0.60570 | 0.63388 |          263 | 0.61759 |           0.01629 |         0.62654 |                   118 |
