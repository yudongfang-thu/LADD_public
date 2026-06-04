# CoLD 复现问题分析包

整理时间：2026-06-04

用途：给老师/合作者集中查看 CoLD 复现过程中的代码版本、论文依据、实验日志、结果统计和当前问题诊断。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `paper/` | CoLD 原文 PDF，以及 `pdftotext` 提取的全文文本 |
| `method_extraction/` | 从原文提取的 CPM / TCLD / NCLD / IWM / OKD / 消融关键词上下文，以及中文方法说明 |
| `code_versions/` | 早期 OBB/YOLO11 版本、当前 YOLOv5-v5.0 HBB 版本、本地启动脚本 |
| `docs/` | 历史复现诊断、最终复现总结、迁移说明 |
| `experiment_records/` | 5090D baseline 快照、90 当前三条 no-IWM 实验、117 当前 IWM 实验和历史日志 |
| `analysis/` | 当前结果汇总 CSV、同 epoch 对比 CSV、进度监控记录、NCLD 偏低数据诊断 |
| `manifests/` | 文件清单 |

## 推荐阅读顺序

1. `analysis/NCLD_LOW_DATA_DIAGNOSIS_CN.md`
2. `analysis/LATEST_SYNC_ANALYSIS_20260604_CN.md`
3. `analysis/progress_monitor_log_20260604_CN.md`
4. `analysis/same_epoch_diagnostic_table.csv`
5. `analysis/current_run_latest_summary.csv`
6. `method_extraction/COLD_METHOD_EXTRACTION_CN.md`
7. `code_versions/CODE_VERSION_INDEX_CN.md`
8. `docs/cold_repro/COLD_REPRO_FINAL_CN.md`

## 当前最重要现象

同 epoch 对齐后，90 服务器 no-IWM 三条实验呈现：

| 实验 | epoch 59 mAP50 | epoch 59 mAP50-95 | 主要现象 |
| --- | ---: | ---: | --- |
| NCLD no-IWM | 0.2451 | 0.1082 | Precision 高、Recall 低 |
| TCLD no-IWM | 0.2987 | 0.1351 | no-IWM 内部最强 |
| BOTH no-IWM | 0.2910 | 0.1281 | 低于 TCLD |
| BOTH + IWM, 117 | 0.3781 | 0.1810 | 跨机器/跨 batch，不能直接归因，但未崩 |

目前数据支持的判断是：NCLD 偏低不是因为候选数少，也不是因为 loss 太小；它更像把 detector 推向了高 precision、低 recall 的保守状态。2026-06-04 08:35 补充同步后仍显示 `TCLD > BOTH > NCLD`。117 的 `BOTH + IWM(mean)` 在 epoch 63 后出现明显退化，暂时不能作为 IWM 有效性的正证据。详见 `analysis/LATEST_SYNC_ANALYSIS_20260604_CN.md`、`analysis/NCLD_LOW_DATA_DIAGNOSIS_CN.md` 和 `analysis/progress_monitor_log_20260604_CN.md`。

## 收集边界

- 已复制代码、脚本、日志、`opt.yaml`、`hyp.yaml`、`results.txt`、`cold_stats.csv`、论文 PDF 和提取文本。
- 未复制 `.pt` checkpoint 权重，避免包体过大；相关权重 MD5 已记录在远程状态文件中。
- 未复制 SSH key、密码文件、数据集原始图片。

## 关键远程记录位置

| 来源 | 包内路径 |
| --- | --- |
| 90 当前 no-IWM 三实验 | `experiment_records/90_current_online_noiwm_20260604/extracted/` |
| 117 当前 IWM 与历史 CoLD 日志 | `experiment_records/117_current_iwm_and_history_20260604/extracted/` |
| 5090D baseline 快照 | `experiment_records/5090d_baseline_snapshot_20260601/` |
