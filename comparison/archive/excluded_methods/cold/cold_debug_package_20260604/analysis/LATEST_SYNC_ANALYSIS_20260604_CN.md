# 2026-06-04 最新同步数据分析

同步时间：2026-06-04 08:35-08:37，Asia/Shanghai。

本次补充同步了 90 和 117 正在运行的 CoLD 实验原始文件，包括 `results.txt`、`cold_stats.csv`、`opt.yaml`、`hyp.yaml`、完整训练日志和 GPU/进程/log tail 状态。

## 新增原始证据位置

| 来源 | 包内路径 |
| --- | --- |
| 90 no-IWM 三实验最新同步 | `experiment_records/90_current_online_noiwm_20260604/latest_sync_20260604_083549/` |
| 117 BOTH + IWM(mean) 最新同步 | `experiment_records/117_current_iwm_and_history_20260604/latest_sync_20260604_083549/` |

## 新增派生分析表

| 文件 | 用途 |
| --- | --- |
| `analysis/latest_sync_run_summary_20260604_083549.csv` | 每条 run 的最新 epoch 指标、loss、candidate 统计 |
| `analysis/latest_sync_best_epoch_summary_20260604_083549.csv` | 每条 run 按 mAP50-95 选出的当前最佳 epoch |
| `analysis/latest_sync_window_trends_20260604_083549.csv` | 0-9、49-59、80-89、100-109、140-149、latest10 等窗口均值 |

## 最新指标表

| 实验 | 最新 epoch | P | R | mAP50 | mAP50-95 | candidate_count | loc_cold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NCLD no-IWM, 90 GPU3 | 102 | 0.7310 | 0.2891 | 0.2978 | 0.1433 | 18102.7 | 0.001869 |
| TCLD no-IWM, 90 GPU4 | 207 | 0.6796 | 0.4621 | 0.4850 | 0.2350 | 13618.2 | 0.000676 |
| BOTH no-IWM, 90 GPU5 | 164 | 0.5644 | 0.4305 | 0.4282 | 0.2005 | 16374.4 | 0.001359 |
| BOTH + IWM(mean), 117 GPU0 | 83 | 0.5215 | 0.2464 | 0.2559 | 0.1210 | 42670.6 | 0.000292 |

## 当前最佳 epoch 表

| 实验 | best epoch | best mAP50 | best mAP50-95 | 备注 |
| --- | ---: | ---: | ---: | --- |
| NCLD no-IWM, 90 GPU3 | 102 | 0.2978 | 0.1433 | 当前仍在缓慢上升，但 recall 偏低 |
| TCLD no-IWM, 90 GPU4 | 207 | 0.4850 | 0.2350 | 当前最强 |
| BOTH no-IWM, 90 GPU5 | 164 | 0.4282 | 0.2005 | 低于 TCLD |
| BOTH + IWM(mean), 117 GPU0 | 63 | 0.3852 | 0.1871 | epoch 63 后明显退化 |

## 数据结论

1. no-IWM 在线实验继续呈现 `TCLD > BOTH > NCLD`。这与 CoLD 原文中 NCLD 主导、BOTH 接近或略强于 NCLD 的消融趋势不一致。

2. NCLD 偏低仍不能解释为候选不足或 loss 过小。最新 epoch 102 中，NCLD 的 candidate_count 为 18102.7，loc_cold 为 0.001869；TCLD 最新 epoch 207 的 candidate_count 为 13618.2，loc_cold 为 0.000676。NCLD 信号并不弱，但指标仍明显低于 TCLD。

3. NCLD 的主要症状仍是高 precision、低 recall。最新 epoch 102 为 P=0.7310, R=0.2891；TCLD 最新 epoch 207 为 P=0.6796, R=0.4621。也就是说，NCLD 不是简单“不收敛”，而是把检测器推向更保守的状态。

4. 117 的 `BOTH + IWM(mean)` 当前不能支持“IWM 有效”的结论。它的最佳点是 epoch 63，mAP50-95=0.1871；最新 epoch 83 降到 0.1210。窗口均值也显示退化：49-59 窗口 mAP50-95 均值为 0.1695，80-83 窗口均值为 0.1157。

## 还应继续补的证据

如果要让老师进一步定位代码问题，下一轮最有价值的不是继续只报 mAP，而是补以下诊断：

1. 每个 epoch 的 target / non-target 候选框数量、类别分布、平均 teacher confidence。
2. NCLD 候选框在不同类别上的数量和 loss 贡献，确认是否某些类别支配了 non-target loss。
3. 每类 AP / recall 曲线，确认 NCLD 的 recall 下降是否集中在小目标或少数类别。
4. 学生和教师各自的检测 loss 曲线，确认在线教师是否与学生共同漂移到不利点。
5. 同机器、同 batch 的 `BOTH no-IWM` vs `BOTH + IWM(mean)`，避免 90/117 跨机器归因。
