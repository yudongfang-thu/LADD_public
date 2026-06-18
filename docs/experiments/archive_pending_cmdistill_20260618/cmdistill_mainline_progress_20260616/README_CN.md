# CMDistill 主协议进度归档 2026-06-16

本目录归档 2026-06-16 上午从 AutoDL 拉回的 CMDistill / LADD 主协议进度，以及两个已停止的 YOLO11n no-extra-aug 400 epoch baseline 负面证据。

## 归档范围

- 主协议：formal no-mosaic 800 epoch，`imgsz=256`，`mosaic=0.0`，`close_mosaic=0`，batch 64。
- 可比对象：SAR baseline、RGB teacher、历史同协议 LADD、当前 AutoDL LADD、当前 CMDistill n/s。
- 负面校准：`cclkd_table2_noextraaug_20260616` 下的 SAR/RGB YOLO11n 400 epoch no-extra-aug baseline，已停止，不作为 CMDistill 正式协议。

## 当前主协议进度

数据表：`mainline_progress_summary.csv`

| model | run | latest completed epoch | latest AP50 | latest AP | best AP |
|---|---|---:|---:|---:|---:|
| n | SAR baseline n | 801 | 0.82340 | 0.55127 | 0.55654 |
| n | RGB teacher n | 801 | 0.92787 | 0.62737 | 0.63018 |
| n | Prev LADD cap2 n | 801 | 0.85155 | 0.57295 | 0.57615 |
| n | Current LADD cap2 n | 247 | 0.74891 | 0.48544 | 0.48544 |
| n | CMDistill n | 304 | 0.73608 | 0.46990 | 0.46990 |
| s | SAR baseline s | 801 | 0.90602 | 0.62233 | 0.62897 |
| s | RGB teacher s | 801 | 0.93707 | 0.64958 | 0.65768 |
| s | Prev LADD cap2 s | 801 | 0.89676 | 0.61759 | 0.63388 |
| s | CMDistill s | 304 | 0.84889 | 0.56709 | 0.56710 |

说明：`latest completed epoch` 使用 `results.csv` 的 `epoch + 1`，与曲线横轴一致。当前 CMDistill n/s 和 Current LADD n 都还在训练中，不能作为最终结果。

## 当前判断

- CMDistill n/s 训练稳定，诊断中 `nonfinite_metrics_or_cmdistill=0`，`nan_or_inf_detected=0`。
- CMDistill s 当前明显强于 CMDistill n，但二者都还低于对应 SAR baseline 的最终/最佳水平。
- 当前 LADD cap2 n 在同阶段略高于 CMDistill n，但仍是中途结果。
- 历史同协议 LADD n/s 已经提供完整 800 epoch 参照；CMDistill 必须继续跑完或至少跑到足够长的前哨阶段后再判断。

## 曲线

- `figures/mainline_progress_ap.png`
- `figures/mainline_progress_ap50.png`
- `figures/mainline_progress_ap_snapshot_20260616_1001.png`
- `figures/mainline_progress_ap50_snapshot_20260616_1001.png`

绘图脚本：

- `plot_mainline_progress.py`

## 已停止 no-extra-aug baseline

数据表：`stopped_noextraaug_summary.csv`

| run | latest completed epoch | latest AP50 | latest AP | best AP epoch | best AP |
|---|---:|---:|---:|---:|---:|
| SAR no-extra YOLO11n 400 stopped | 399 | 0.46086 | 0.25156 | 242 | 0.27494 |
| RGB no-extra YOLO11n 400 stopped | 401 | 0.75812 | 0.45467 | 234 | 0.46811 |

结论：该 no-extra-aug baseline 协议明显不适合作为短期 CMDistill/CCLKD 对齐协议。它与当前 formal no-mosaic 800 主协议不可直接混比，因此已经停止，保留作负面校准证据。

## 原始证据

- 主协议 CSV：`source/`
- 主协议 diagnostics：`diagnostics/`
- 已停止 baseline CSV/args/command/log tail：`stopped_noextraaug_baselines/`
- CMDistill manifest 和日志尾部：`logs/`
- GPU/进程快照：`process_gpu_snapshot_20260616_1002_concise.txt`
- 远端压缩证据包：`remote_snapshot/cmdistill_mainline_archive_20260616_1000.tgz`

权重文件没有归档，也不应提交到 GitHub。
