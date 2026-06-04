# NCLD 偏低的数据诊断

数据来源：

- `experiment_records/90_current_online_noiwm_20260604/extracted/runs/ogsod_cold_online_terms/*/results.txt`
- `experiment_records/90_current_online_noiwm_20260604/extracted/runs/ogsod_cold_online_terms/*/cold_stats.csv`
- 汇总表：`analysis/same_epoch_diagnostic_table.csv`

## 配置一致性

90 上三条 no-IWM 实验的关键配置一致，只有 `cold_terms` 不同：

| 配置项 | 值 |
| --- | --- |
| `cold_loss_mode` | `candidate` |
| `cold_iwm_mode` | `none` |
| `batch_size` | 32 |
| `effective_batch_size` | 64 |
| `teacher_det_weight` | 1.0 |
| `lambda_loc_cold` | 1.0 |
| `lambda_cls_cold` | 0.0 |
| `alpha_non_target` | 2.0 |
| `temperature` | 20.0 |
| `candidate_topk` | 1000 |
| `candidate_min_conf` | 0.001 |

初始化权重 MD5 相同：`4f7eee7ab596ed6f9496520cb304c7cb`。

## 同 epoch 59 对比

| 实验 | P | R | mAP50 | mAP50-95 | loc_cold | candidate_count | terms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| NCLD no-IWM | 0.7177 | 0.2396 | 0.2451 | 0.1082 | 0.001892 | 22336.5 | `n_terms=265.2` |
| TCLD no-IWM | 0.5652 | 0.2888 | 0.2987 | 0.1351 | 0.000787 | 17697.5 | `t_terms=188.7` |
| BOTH no-IWM | 0.5103 | 0.2863 | 0.2910 | 0.1281 | 0.001432 | 22234.8 | `t=201.7,n=269.1` |

## 80-86 epoch 窗口均值

| 实验 | P avg | R avg | mAP50 avg | mAP50-95 avg |
| --- | ---: | ---: | ---: | ---: |
| NCLD no-IWM | 0.7399 | 0.2610 | 0.2728 | 0.1258 |
| TCLD no-IWM | 0.5425 | 0.3049 | 0.3206 | 0.1515 |
| BOTH no-IWM | 0.5240 | 0.2905 | 0.2972 | 0.1352 |

## 数据支持的结论

1. NCLD 不是因为候选数少而低。epoch 59 时 NCLD candidate_count 为 22336.5，高于 TCLD 的 17697.5，接近 BOTH 的 22234.8。

2. NCLD 不是因为 CoLD loss 太小而低。epoch 59 时 NCLD loc_cold 为 0.001892，约为 TCLD 0.000787 的 2.4 倍。

3. NCLD 的直接指标问题是 recall 低。epoch 59 时 NCLD precision 为 0.7177，但 recall 只有 0.2396；TCLD precision 较低但 recall 为 0.2888。80-86 窗口也保持同样模式。

4. 验证 loss 也支持 NCLD 较差。epoch 59 时 NCLD 的 `val_box=0.06298`, `val_cls=0.00532`，TCLD 为 `val_box=0.05917`, `val_cls=0.00387`。

## 当前解释

在当前实现里，NCLD-only 不是没有训练信号，而是训练信号偏强且把模型推向高 precision、低 recall 的保守检测状态。换言之，NCLD 的问题不是“没学到”，而是“学到的约束不利于提升召回”。

## 2026-06-04 最新监控补充

截至 2026-06-04 08:22 左右，90 服务器 no-IWM 三条实验继续保持同一趋势：

| 实验 | 最新进度 | P | R | mAP50 | mAP50-95 | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| NCLD no-IWM | epoch 102/399 完成 | 0.7547 | 0.2772 | 0.2979 | 0.1430 | precision 仍高，recall 仍低 |
| TCLD no-IWM | epoch 205/399 运行中 | 0.6251 | 0.4795 | 0.4841 | 0.2332 | 当前最好 |
| BOTH no-IWM | epoch 162/399 运行中 | 0.5679 | 0.4259 | 0.4236 | 0.1979 | 低于 TCLD |

117 服务器的 `BOTH + IWM(mean)` 不能作为 IWM 有效性的正证据：该 run 在 epoch 59 左右达到 `mAP50=0.3781, mAP50-95=0.1810`，但 epoch 82 降到 `mAP50=0.2490, mAP50-95=0.1175`，出现明显退化。

因此，当前数据判断更新为：no-IWM 线上实验仍显示 TCLD 主导；IWM(mean) 的 117 run 暂时标记为不稳定，不能用来解释或修正 NCLD 偏低问题。

2026-06-04 08:35 补充同步后，趋势没有改变。最新同步结果为：NCLD epoch 102 `mAP50-95=0.1433`，TCLD epoch 207 `mAP50-95=0.2350`，BOTH epoch 164 `mAP50-95=0.2005`，117 IWM epoch 83 `mAP50-95=0.1210`。详见 `analysis/LATEST_SYNC_ANALYSIS_20260604_CN.md`。

## 下一步验证

建议优先做最小诊断实验：

1. `NCLD alpha=1.0`：检验当前 `alpha_non_target=2.0` 是否对 NCLD-only 过强。
2. `NCLD + IWM`：检验 IWM 是否能稳定非目标蒸馏并改善 recall。
3. `BOTH + no-IWM` 与 `BOTH + IWM` 同机器同 batch 对照：当前 117 IWM 和 90 no-IWM 跨机器，不适合作强归因。
