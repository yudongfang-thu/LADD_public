# LADD Mosaic vs No-Mosaic 协议重合实验对比

日期：2026-06-14

本文件专门回答一个问题：旧的 `mosaic100/close@100` 收敛协议和当前 `formal no-mosaic 800ep` 收敛协议下，是否存在相同或相近的 LADD/baseline 重合实验，能否说明训练协议本身在影响 LADD 表现。

结论先行：**最对称的重合证据主要在 YOLO11n 上**。旧 `mosaic100/close@100` 有 n 的 SAR baseline、RGB baseline 和 LADD legacy/cap2 三 seed；当前 `formal no-mosaic` 也有 n 的 SAR/RGB baseline 和 LADD cap2/original/BN-freeze 多 seed。s/m 暂时没有同等对称的 mosaic 对照，因此不要把 s/m 的 no-mosaic 现象直接外推为 protocol 结论。

## 1. 可比协议

| protocol | mosaic | close_mosaic | epochs | baseline 口径 | 主要重合容量 |
|---|---:|---:|---:|---|---|
| `mosaic100/close@100` | `1.0` | `700` | 800 target | SAR YOLO11n best/final `0.54091/0.53836` | n |
| `formal no-mosaic` | `0.0` | `0` | 800 | SAR YOLO11n seed0/42/123 best `0.55654/0.55794/0.56128` | n/s/m，但 n 最对称 |

注意：no-mosaic 的 SAR baseline 本身比 mosaic100 baseline 高约 `+0.016` 到 `+0.020` AP50-95。因此必须同时看 **raw AP** 和 **相对同协议 SAR baseline 的 gain**，不能只看绝对 AP。

## 2. 干净性能均值

性能均值只计算健康完成 run。`nan_crash`、已知 `late_regression`、abnormal/partial run 不进入平均性能；这些 run 只进入下一节稳定性事件表。这样做的目的不是隐藏失败，而是避免把“训练崩溃”混成“方法正常收敛后的平均 AP”。

| comparison_set            | protocol            | variant                     | seeds_included   |   n_included_in_performance_mean |   best_mean |   last_mean |   best_gain_mean |   final_gain_mean | excluded_from_performance_mean                          | notes                                                                                |
|:--------------------------|:--------------------|:----------------------------|:-----------------|---------------------------------:|------------:|------------:|-----------------:|------------------:|:--------------------------------------------------------|:-------------------------------------------------------------------------------------|
| n_cap2_main_overlap       | mosaic100_close100  | cap2 normal BN              | 0,42,123         |                                3 |     0.56601 |     0.56333 |          0.0251  |           0.02497 |                                                         | 三 seed 都健康完成；可作为 mosaic cap2 性能均值。                                    |
| n_legacy_context          | mosaic100_close100  | legacy normal BN            | 0,42,123         |                                3 |     0.56631 |     0.56259 |          0.0254  |           0.02423 |                                                         | 三 seed 都健康完成；作为旧 legacy 方法背景。                                         |
| n_cap2_main_overlap       | formal_nomosaic_800 | cap2 normal BN healthy only | 0,42             |                                2 |     0.57541 |     0.57398 |          0.01817 |           0.02113 | nomosaic_cap2_s123_old_crash;nomosaic_cap2_s123_bstable | 只计算 seed0/42 两条健康完成 run；seed123 的 collapse/late-regression 不进性能均值。 |
| n_cap2_stabilized_overlap | formal_nomosaic_800 | cap2 BN-freeze              | 0,42,123         |                                3 |     0.57387 |     0.57256 |          0.01528 |           0.01707 |                                                         | 三 seed 都健康完成；这是 no-mosaic 稳定化后的三 seed 性能均值。                      |
| n_original_context        | formal_nomosaic_800 | original normal BN          | 0                |                                1 |     0.57821 |     0.57517 |          0.02167 |           0.0239  |                                                         | 单 seed 健康高点，只作背景，不和三 seed 均值混算。                                   |

## 3. 失败与稳定性事件

| group_id                         |   attempted_runs |   attempted_unique_seeds |   healthy_complete_runs |   healthy_complete_unique_seeds |   nan_crash_runs |   late_regression_runs | statuses                                                                                                                                         |
|:---------------------------------|-----------------:|-------------------------:|------------------------:|--------------------------------:|-----------------:|-----------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------|
| mosaic100_cap2_normal_bn         |                3 |                        3 |                       3 |                               3 |                0 |                      0 | mosaic_cap2_s0:complete;mosaic_cap2_s42:complete;mosaic_cap2_s123:complete                                                                       |
| mosaic100_legacy_normal_bn       |                3 |                        3 |                       3 |                               3 |                0 |                      0 | mosaic_legacy_s0:complete;mosaic_legacy_s42:complete;mosaic_legacy_s123:complete                                                                 |
| nomosaic_cap2_normal_bn_attempts |                4 |                        3 |                       2 |                               2 |                1 |                      1 | nomosaic_cap2_s0_a2mu:complete;nomosaic_cap2_s42_a2mu:complete;nomosaic_cap2_s123_old_crash:nan_crash;nomosaic_cap2_s123_bstable:late_regression |
| nomosaic_cap2_bnfreeze           |                3 |                        3 |                       3 |                               3 |                0 |                      0 | nomosaic_cap2_s0_bnfreeze:complete;nomosaic_cap2_s42_bnfreeze:complete;nomosaic_cap2_s123_bnfreeze:complete                                      |
| nomosaic_original_seed0          |                1 |                        1 |                       1 |                               1 |                0 |                      0 | nomosaic_original_s0:complete                                                                                                                    |

核心读法：

- `mosaic100 cap2` 三 seed 平均 best gain 是 `+0.02510`，final gain 是 `+0.02497`，且没有 collapse。
- `formal no-mosaic cap2 no-BN-freeze` 的健康 seed0/42 平均 best gain 是 `+0.01817`，绝对 AP 更高，但相对同协议 baseline 的 gain 小了一截；seed123 的崩溃/退化不参与该性能均值，只作为稳定性失败证据。
- `formal no-mosaic cap2 BN-freeze` 三 seed 稳定闭环，平均 best gain `+0.01528`，final gain `+0.01707`，比 mosaic100 的相对增益更小。
- no-mosaic seed123 normal-BN 暴露了 collapse / late-regression；mosaic100 seed123 没有类似问题。

## 4. seed 级重合对比

| pair_id                  |   seed |   mosaic_best |   mosaic_best_gain |   mosaic_last |   mosaic_final_gain | nomosaic_key                 | nomosaic_status   |   nomosaic_best |   nomosaic_best_gain |   nomosaic_last |   nomosaic_final_gain |   raw_best_delta_nomosaic_minus_mosaic |   best_gain_delta_nomosaic_minus_mosaic |
|:-------------------------|-------:|--------------:|-------------------:|--------------:|--------------------:|:-----------------------------|:------------------|----------------:|---------------------:|----------------:|----------------------:|---------------------------------------:|----------------------------------------:|
| cap2_seed0_no_bnfreeze   |      0 |       0.56841 |            0.0275  |       0.56792 |             0.02956 | nomosaic_cap2_s0_a2mu        | complete          |         0.57662 |              0.02008 |         0.57504 |               0.02377 |                                0.00821 |                                -0.00742 |
| cap2_seed42_no_bnfreeze  |     42 |       0.56799 |            0.02708 |       0.56044 |             0.02208 | nomosaic_cap2_s42_a2mu       | complete          |         0.5742  |              0.01626 |         0.57293 |               0.01849 |                                0.00621 |                                -0.01082 |
| cap2_seed123_old_b_crash |    123 |       0.56163 |            0.02072 |       0.56163 |             0.02327 | nomosaic_cap2_s123_old_crash | nan_crash         |         0.52182 |             -0.03946 |         0       |              -0.56076 |                               -0.03981 |                                -0.06018 |
| cap2_seed123_bstable     |    123 |       0.56163 |            0.02072 |       0.56163 |             0.02327 | nomosaic_cap2_s123_bstable   | late_regression   |         0.56161 |              0.00033 |         0.52875 |              -0.03201 |                               -2e-05   |                                -0.02039 |
| cap2_seed0_bnfreeze      |      0 |       0.56841 |            0.0275  |       0.56792 |             0.02956 | nomosaic_cap2_s0_bnfreeze    | complete          |         0.57276 |              0.01622 |         0.57254 |               0.02127 |                                0.00435 |                                -0.01128 |
| cap2_seed42_bnfreeze     |     42 |       0.56799 |            0.02708 |       0.56044 |             0.02208 | nomosaic_cap2_s42_bnfreeze   | complete          |         0.57615 |              0.01821 |         0.57295 |               0.01851 |                                0.00816 |                                -0.00887 |
| cap2_seed123_bnfreeze    |    123 |       0.56163 |            0.02072 |       0.56163 |             0.02327 | nomosaic_cap2_s123_bnfreeze  | complete          |         0.57269 |              0.01141 |         0.57219 |               0.01143 |                                0.01106 |                                -0.00931 |
| seed0_original_nomosaic  |      0 |       0.56841 |            0.0275  |       0.56792 |             0.02956 | nomosaic_original_s0         | complete          |         0.57821 |              0.02167 |         0.57517 |               0.0239  |                                0.0098  |                                -0.00583 |

最关键的三条：

1. seed0/42 上，no-mosaic LADD 的 **绝对 best AP** 比 mosaic100 高，这是因为 no-mosaic baseline 更强；但它的 **best gain** 比 mosaic100 小约 `0.007` 到 `0.011`。
2. seed123 上，mosaic100 cap2 是健康完整 run，best/final `0.56163/0.56163`；no-mosaic normal-BN 先后出现 old-B collapse 和 bstable late-regression。
3. no-mosaic BN-freeze 能把 seed123 稳住到 best/final `0.57269/0.57219`，但相对 no-mosaic SAR baseline 的 best gain 只有 `+0.01141`，仍明显低于 mosaic100 seed123 的 `+0.02072`。

## 5. 图件

本轮协议对比新增图：

![n protocol group gain compare](../figures/ladd_protocol_overlap_20260614/fig1_n_protocol_group_gain_compare.png)

![n protocol seed pair delta](../figures/ladd_protocol_overlap_20260614/fig2_n_protocol_seed_pair_delta.png)

已有图件可以直接用于汇报：

![formal no-mosaic n curves](../figures/ladd_converged_mainline_ladd_20260613/fig1_nomosaic_n_no_bnfreeze_bnfreeze_curves.png)

![mosaic100 n curves](../figures/ladd_converged_mainline_ladd_20260613/fig2_mosaic100_n_ladd_curves.png)

![all mainline best gain](../figures/ladd_converged_mainline_ladd_20260613/fig3_all_mainline_best_gain.png)

![all mainline final gain](../figures/ladd_converged_mainline_ladd_20260613/fig4_all_mainline_final_gain.png)

## 6. 目前能支持的判断

可以支持：

- LADD 不是天然不可训练；旧 mosaic100/close@100 下 n 三 seed、legacy/cap2 都有稳定正收益。
- no-mosaic 协议本身把 SAR baseline 推高，因此 LADD 的相对增益自然更难做大。
- no-mosaic 下 LADD 的稳定性余量更窄，seed123 normal-BN 暴露出 collapse / late regression，而 mosaic100 同 seed 没有。
- BN-freeze 更像 no-mosaic 下的稳定性修复：它能补 seed123 稳定性，但不恢复 mosaic100 下那种 `+0.025` 左右的平均相对增益。

暂时不能支持：

- 不能说 no-mosaic 一定让 LADD 失效，因为 n seed0/42 和 BN-freeze 三 seed 仍然是正收益。
- 不能把 s/m 的 no-mosaic 退化直接归因于 mosaic 与否，因为没有同容量、同 seed、同实现版本的 mosaic 对称组。
- 不能把 mosaic100 与 no-mosaic 的 absolute AP 直接排在同一个主表里做优劣结论；主表要按协议分层。

## 7. 建议下一步最小桥接实验

如果要验证“训练协议是主因”，建议只做小而对称的 bridge，不再铺大矩阵：

| 优先级 | 实验 | 目的 |
|---:|---|---|
| P0 | 用当前代码重跑 n seed0 `mosaic100/close@100` SAR baseline + LADD cap2 | 排除旧代码/旧服务器差异，复现旧协议大增益是否还在 |
| P1 | 用当前代码重跑 n seed123 `mosaic100/close@100` LADD cap2 | 检查 no-mosaic seed123 的 collapse 是否由协议触发 |
| P2 | n seed0/123 no-mosaic 与 mosaic100 共用同一 A1/A2/B 记录，统一提取 A2 入口与 B 曲线 | 判断差异来自 A2 起点、B 入口冲击，还是后期调度/BN |

如果 P0/P1 仍稳定并保持约 `+0.02` 以上相对 gain，那么训练协议假设很强；如果复现不了，则需要回头查旧代码、旧 checkpoint、旧服务器/环境差异。

## 8. 关联数据

CSV：

```text
docs/experiments/ladd_mainline_diagnosis/ladd_protocol_mosaic_vs_nomosaic_overlap_20260614.csv
docs/experiments/ladd_mainline_diagnosis/ladd_protocol_mosaic_vs_nomosaic_group_summary_20260614.csv
docs/experiments/ladd_mainline_diagnosis/ladd_protocol_mosaic_vs_nomosaic_clean_performance_20260614.csv
docs/experiments/ladd_mainline_diagnosis/ladd_protocol_mosaic_vs_nomosaic_stability_events_20260614.csv
```

来源：

```text
ladd/results/converged_mainline_ladd_20260613/converged_mainline_ladd_summary_20260613.csv
ladd/results/ladd90_formal_baselines_20260612/summary/ladd90_formal_baseline_summary_20260612.csv
docs/experiments/LADD_CONVERGED_MAINLINE_COMPARISON_20260613_CN.md
docs/experiments/LADD_MOSAIC90_MAINLINE_EVIDENCE_20260528_CN.md
docs/experiments/figures/ladd_protocol_overlap_20260614/
```
