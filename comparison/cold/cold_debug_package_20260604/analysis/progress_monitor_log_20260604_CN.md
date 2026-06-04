# CoLD 当前实验进度监控记录

记录时间：2026-06-04 08:22 左右；补充同步时间：2026-06-04 08:35-08:37，Asia/Shanghai。

说明：本文件记录人工监控时从训练日志、`results.txt` 和 GPU 状态读取到的最新状态。它用于补充包内已复制的原始日志快照；如果远程实验继续运行，`experiment_records/` 下的原始文件可能早于本文件。

## 当前运行状态

| 服务器 | GPU | 实验 | 进度 | 速度/资源 | 最新 P | 最新 R | 最新 mAP50 | 最新 mAP50-95 | 判断 |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 90 | 3 | NCLD no-IWM | epoch 102/399 完成 | 约 2.22 s/it；GPU3 与其他进程共享 | 0.7547 | 0.2772 | 0.2979 | 0.1430 | 仍是高 precision、低 recall |
| 90 | 4 | TCLD no-IWM | epoch 205/399 运行中 | 约 1.95 it/s | 0.6251 | 0.4795 | 0.4841 | 0.2332 | 当前三条 no-IWM 中最好 |
| 90 | 5 | BOTH no-IWM | epoch 162/399 运行中 | 约 1.47 it/s | 0.5679 | 0.4259 | 0.4236 | 0.1979 | 低于 TCLD，高于 NCLD |
| 117 | 0 | BOTH + IWM(mean) | epoch 83/399 运行中 | 约 3.5 s/it；显存约 19.4GB/49.1GB | 0.5027 | 0.2380 | 0.2490 | 0.1175 | 指标从 epoch 59 后明显退化 |

## 最新趋势判断

90 服务器三条 no-IWM 在线 CoLD 没有发现 OOM、NaN 或 Traceback。当前排序为：

`TCLD no-IWM > BOTH no-IWM > NCLD no-IWM`

这仍然不符合 CoLD 原文消融中 NCLD 主导、BOTH 接近或略强于 NCLD 的趋势。NCLD 的核心问题继续表现为高 precision、低 recall，而不是候选框不足或 loss 过小。

117 服务器的 `BOTH + IWM(mean)` 需要单独标记为不稳定：该 run 在 epoch 59 左右曾达到 `mAP50=0.3781, mAP50-95=0.1810`，但最新 epoch 82 降到 `mAP50=0.2490, mAP50-95=0.1175`。这不能作为 IWM 有效性的正证据。

## 后续同步要求

后续如果继续给老师同步，应同时更新：

1. `analysis/current_run_latest_summary.csv`
2. `analysis/progress_monitor_log_20260604_CN.md`
3. 对应远程 run 的 `results.txt`、`cold_stats.csv` 和训练 log 快照

其中第 3 项是原始证据；第 1、2 项只是面向阅读的摘要。

## 2026-06-04 08:35 补充同步

已补充同步最新原始证据：

- `experiment_records/90_current_online_noiwm_20260604/latest_sync_20260604_083549/`
- `experiment_records/117_current_iwm_and_history_20260604/latest_sync_20260604_083549/`

已新增派生分析：

- `analysis/LATEST_SYNC_ANALYSIS_20260604_CN.md`
- `analysis/latest_sync_run_summary_20260604_083549.csv`
- `analysis/latest_sync_best_epoch_summary_20260604_083549.csv`
- `analysis/latest_sync_window_trends_20260604_083549.csv`

同步后的最新指标略有更新：TCLD no-IWM 到 epoch 207，mAP50-95=0.2350；BOTH no-IWM 到 epoch 164，mAP50-95=0.2005；NCLD no-IWM 到 epoch 102，mAP50-95=0.1433；117 BOTH + IWM(mean) 到 epoch 83，mAP50-95=0.1210。
