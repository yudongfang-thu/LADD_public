# LADD 90 服务器 YOLO11 Formal Baseline 证据记录（2026-06-12）

## 结论

90 服务器 `inspur-NF5468M5` 上保留了 YOLO11 多容量 SAR/RGB baseline 的 formal 收敛协议结果。证据来自：

`/mnt/dataY/ydf/projects/LADD_og/runs_public/ogsod/hbb/formal_nomosaic_20260528/baselines/`

这些 run 均为 800 epoch 收敛训练协议：

- `epochs=800`
- `patience=800`
- `imgsz=256`
- `cos_lr=true`
- `mosaic=0.0`
- `close_mosaic=0`
- `deterministic=true`
- n/s 使用 batch 64，m/l 使用 batch 32，x 使用 batch 16

这说明 90 服务器上的单模态 YOLO11 baseline 本身不是崩溃状态；n/s/m/l/x 的 SAR/RGB baseline 都有完整 `results.csv`、`args.yaml` 和对应原始训练 log。

## 本次同步到仓库的证据

轻量证据包位于：

`ladd/results/ladd90_formal_baselines_20260612/`

包含：

- `summary/ladd90_formal_baseline_summary_20260612.csv`
- `args/*.yaml`
- `results/*.csv`
- `log_extracts/*.log`
- `manifest.txt`

完整原始 log 没有直接提交，因为单个 log 约 30-126 MB，且包含大量进度条控制字符。仓库中提交的是从原始 log 抽取的关键片段，包括 final validation、`Optimizer stripped`、`Results saved`、最终 `all ... mAP50-95` 等信息。完整原始 log 的 90 服务器路径和字节大小已记录在 summary CSV 与 manifest 中。

权重、checkpoint、完整 run 目录、TensorBoard event、wandb 均未提交。

## SAR Baseline

| model | seed | batch | best AP50-95 | best epoch | last AP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 0 | 64 | 0.55654 | 734 | 0.55127 |
| YOLO11n | 42 | 64 | 0.55794 | 739 | 0.55444 |
| YOLO11n | 123 | 64 | 0.56128 | 797 | 0.56076 |
| YOLO11s | 0 | 64 | 0.62897 | 729 | 0.62233 |
| YOLO11s | 42 | 64 | 0.62879 | 735 | 0.62486 |
| YOLO11s | 123 | 64 | 0.62357 | 750 | 0.62013 |
| YOLO11m | 0 | 32 | 0.65580 | 704 | 0.64903 |
| YOLO11l | 0 | 32 | 0.65427 | 735 | 0.64892 |
| YOLO11x | 0 | 16 | 0.65867 | 685 | 0.64801 |

## RGB Baseline / Teacher

| model | seed | batch | best AP50-95 | best epoch | last AP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLO11n | 0 | 64 | 0.63018 | 723 | 0.62737 |
| YOLO11n | 42 | 64 | 0.62664 | 739 | 0.62567 |
| YOLO11n | 123 | 64 | 0.62933 | 789 | 0.62876 |
| YOLO11s | 0 | 64 | 0.65768 | 647 | 0.64958 |
| YOLO11s | 42 | 64 | 0.66218 | 683 | 0.65091 |
| YOLO11s | 123 | 64 | 0.65987 | 710 | 0.65519 |
| YOLO11m | 0 | 32 | 0.67909 | 663 | 0.67159 |
| YOLO11l | 0 | 32 | 0.68356 | 618 | 0.66892 |
| YOLO11x | 0 | 16 | 0.68284 | 539 | 0.65820 |

## 代表性原始 Log 路径

SAR:

- `/mnt/dataY/ydf/projects/LADD_og/logs/formal_nomosaic_20260528/baselines/sar_yolo11n_hbb_800ep_cos_nomosaic_albu_b64_s0_gpu2.log`
- `/mnt/dataY/ydf/projects/LADD_og/logs/formal_nomosaic_20260528/baselines/sar_yolo11s_hbb_800ep_cos_nomosaic_albu_b64_s0_gpu2.log`
- `/mnt/dataY/ydf/projects/LADD_og/logs/formal_nomosaic_20260528/baselines/sar_yolo11m_hbb_800ep_cos_nomosaic_albu_b32_s0_gpu2.log`
- `/mnt/dataY/ydf/projects/LADD_og/logs/formal_nomosaic_20260528/baselines/sar_yolo11l_hbb_800ep_cos_nomosaic_albu_b32_s0_gpu4.log`
- `/mnt/dataY/ydf/projects/LADD_og/logs/formal_nomosaic_20260528/baselines/sar_yolo11x_hbb_800ep_cos_nomosaic_albu_b16_s0_gpu4.log`

RGB:

- `/mnt/dataY/ydf/projects/LADD_og/logs/formal_nomosaic_20260528/baselines/rgb_yolo11n_hbb_800ep_cos_nomosaic_albu_b64_s0_gpu5.log`
- `/mnt/dataY/ydf/projects/LADD_og/logs/formal_nomosaic_20260528/baselines/rgb_yolo11s_hbb_800ep_cos_nomosaic_albu_b64_s0_gpu5.log`
- `/mnt/dataY/ydf/projects/LADD_og/logs/formal_nomosaic_20260528/baselines/rgb_yolo11m_hbb_800ep_cos_nomosaic_albu_b32_s0_gpu5.log`
- `/mnt/dataY/ydf/projects/LADD_og/logs/formal_nomosaic_20260528/baselines/rgb_yolo11l_hbb_800ep_cos_nomosaic_albu_b32_s0_gpu3.log`
- `/mnt/dataY/ydf/projects/LADD_og/logs/formal_nomosaic_20260528/baselines/rgb_yolo11x_hbb_800ep_cos_nomosaic_albu_b16_s0_gpu5.log`

## 对当前 A2 损伤诊断的意义

这批 90 baseline 结果提供了一个重要对照：在同一 formal nomosaic 收敛协议下，单模态 SAR/RGB detector 可以稳定完成 800 epoch，并且 m/l/x 容量没有出现训练崩溃。因此当前 4090D 上观察到的 LADD A2/B 阶段退化，不应简单归因于 YOLO11 detector 或 OGSOD formal protocol 本身不可收敛。

特别是当前诊断使用的阈值：

- YOLO11s SAR seed0 baseline：0.62897
- YOLO11m SAR seed0 baseline：0.65580

均来自这批 90 formal baseline，并已同步对应 `results.csv`、`args.yaml` 和 log extract 作为证据。
