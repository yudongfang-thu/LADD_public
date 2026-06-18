# Formal no-mosaic baseline 与 LADD B 阶段对比（2026-06-14）

本报告只看 formal no-mosaic 记录，不纳入 mosaic100 历史实验。

关键读法：
- formal no-mosaic SAR baseline 本身在 800ep 后段存在 best-final gap，因此 final 低于 best 是协议现象的一部分。
- 当前 B-entrance 从收敛/selected checkpoint 出发，epoch1 AP 已经高；它不是从 YOLO 初始权重开始的 baseline 曲线。
- no-mosaic 历史 LADD full-chain B 证明 B 阶段可以在完整链条中继续获得长程增益；当前 split-load B-only 目前没有复现这种增益。
- s 模型历史 BN-freeze B800 与当前 S 曲线共同提示：早期平台期后仍可能出现 late-regression，B100/B120 只看入口不够完整。

生成图：
- `fig0_formal_nomosaic_sar_baseline_curves.png/pdf`
- `fig1_nomosaic_n_baseline_ladd_b_compare.png/pdf`
- `fig2_nomosaic_s_baseline_ladd_b_compare.png/pdf`

汇总表：

| label                                     | model   | kind               |   epochs |   first_ap |    best |   best_epoch |    last |   best_final_drop |   best_first120 |   best_first120_epoch |
|:------------------------------------------|:--------|:-------------------|---------:|-----------:|--------:|-------------:|--------:|------------------:|----------------:|----------------------:|
| SAR n baseline train 800ep                | n       | baseline           |      800 |    0.05211 | 0.55654 |          734 | 0.55127 |           0.00527 |         0.35655 |                   120 |
| old no-mosaic n cap2 s0 no-BN-freeze B800 | n       | old_ladd_b         |      800 |    0.53675 | 0.57662 |          725 | 0.57504 |           0.00158 |         0.53675 |                     1 |
| old no-mosaic n cap2 s0 BN-freeze B800    | n       | old_ladd_b         |      800 |    0.52679 | 0.57276 |          793 | 0.57254 |           0.00022 |         0.55213 |                   117 |
| old no-mosaic n s123 old-B crash          | n       | old_ladd_b         |      483 |    0.52182 | 0.52182 |            1 | 0.00000 |           0.52182 |         0.52182 |                     1 |
| current N1 SAR baseline cont. B100        | n       | current_b_entrance |      100 |    0.54123 | 0.56615 |           99 | 0.56594 |           0.00021 |         0.56615 |                    99 |
| current N3 SAR-base + A2-last decomp B100 | n       | current_b_entrance |      100 |    0.51435 | 0.55722 |          100 | 0.55722 |           0.00000 |         0.55722 |                   100 |
| current N4 N3 + KD ramp B120              | n       | current_b_entrance |      120 |    0.53806 | 0.56379 |          113 | 0.56311 |           0.00068 |         0.56379 |                   113 |
| SAR s baseline train 800ep                | s       | baseline           |      800 |    0.05413 | 0.62897 |          729 | 0.62233 |           0.00664 |         0.43690 |                   120 |
| old no-mosaic s cap2 s0 no-BN-freeze B608 | s       | old_ladd_b         |      608 |    0.60675 | 0.63551 |          605 | 0.63527 |           0.00024 |         0.60675 |                     1 |
| old no-mosaic s cap2 s0 BN-freeze B800    | s       | old_ladd_b         |      800 |    0.60570 | 0.63388 |          263 | 0.61759 |           0.01629 |         0.62654 |                   118 |
| current S1 SAR baseline cont. B100        | s       | current_b_entrance |      100 |    0.61471 | 0.62493 |           62 | 0.62238 |           0.00255 |         0.62493 |                    62 |
| current S3 SAR-base + A2-last decomp B100 | s       | current_b_entrance |      100 |    0.60625 | 0.62553 |           65 | 0.62262 |           0.00291 |         0.62553 |                    65 |
| current S4 S3 + KD ramp B120              | s       | current_b_entrance |      120 |    0.61462 | 0.62521 |           62 | 0.62111 |           0.00410 |         0.62521 |                    62 |
