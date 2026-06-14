# 实验记录索引

最后更新：2026-06-15

本文档现在只作为入口页。当前本地实验数据较多，直接从旧报告或全仓 `results.csv` 得结论容易误用，因此请优先从项目地图和 registry 进入。

## 0. 新入口

| 需要 | 文档 |
|---|---|
| 项目三条线总地图 | [PROJECT_EXPERIMENT_MAP_20260614_CN.md](PROJECT_EXPERIMENT_MAP_20260614_CN.md) |
| 本地数据整理方案 | [DATA_ORGANIZATION_PLAN_20260614_CN.md](DATA_ORGANIZATION_PLAN_20260614_CN.md) |
| 全局实验 registry | [registry/README_20260614_CN.md](registry/README_20260614_CN.md) |
| 三条线 inventory CSV | [project_line_inventory_20260614.csv](project_line_inventory_20260614.csv) |

## 1. 三条工作线

| 工作线 | 状态页 | 说明 |
|---|---|---|
| CCLKD 复现消融线 | [cclkd_reproduction/CCLKD_LINE_STATUS_20260614_CN.md](cclkd_reproduction/CCLKD_LINE_STATUS_20260614_CN.md) | 区分 YOLOv5x paper reproduction gate 与 YOLO11 controlled comparison |
| 其他对比方法线 | [comparison_methods/COMPARISON_LINE_STATUS_20260614_CN.md](comparison_methods/COMPARISON_LINE_STATUS_20260614_CN.md) | 管理 LD / FGD / HalluciDet / CCLKD 受控对比 |
| LADD 主线诊断线 | [ladd_mainline_diagnosis/LADD_LINE_FORENSIC_MAP_20260614_CN.md](ladd_mainline_diagnosis/LADD_LINE_FORENSIC_MAP_20260614_CN.md) | 复盘历史健康主线与近期崩溃/平台实验 |

LADD 当前复盘主文档：[ladd_mainline_diagnosis/LADD_FORENSIC_REVIEW_20260614_CN.md](ladd_mainline_diagnosis/LADD_FORENSIC_REVIEW_20260614_CN.md)

LADD 训练协议重合对比：[ladd_mainline_diagnosis/LADD_PROTOCOL_MOSAIC_VS_NOMOSAIC_OVERLAP_20260614_CN.md](ladd_mainline_diagnosis/LADD_PROTOCOL_MOSAIC_VS_NOMOSAIC_OVERLAP_20260614_CN.md)

LADD 阶段与协议复盘分析：[ladd_mainline_diagnosis/LADD_STAGE_PROTOCOL_FORENSICS_20260615_CN.md](ladd_mainline_diagnosis/LADD_STAGE_PROTOCOL_FORENSICS_20260615_CN.md)

## 2. 当前使用规则

1. 不要直接遍历全仓 `results.csv` 得实验结论；先查 [registry/experiment_registry_20260614.csv](registry/experiment_registry_20260614.csv)。
2. 同一 run 的多个副本通过 [registry/duplicate_results_20260614.csv](registry/duplicate_results_20260614.csv) 去重。
3. `smoke / probe / diag / snapshot / partial / old` 默认只能作为诊断证据。
4. LADD 主线当前进入 forensic review 阶段，暂停继续堆新实验。
5. 旧文档如与新入口冲突，以新入口和分线状态页为准。
