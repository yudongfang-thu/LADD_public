# 实验记录索引

最后更新：2026-06-18

本文档现在只作为入口页。当前本地实验数据较多，直接从旧报告或全仓 `results.csv` 得结论容易误用，因此请优先从项目地图、registry 和各条线的当前状态页进入。

## 0. 新入口

| 需要 | 文档 |
|---|---|
| 项目三条线总地图 | [PROJECT_EXPERIMENT_MAP_20260614_CN.md](PROJECT_EXPERIMENT_MAP_20260614_CN.md) |
| 本地数据整理方案 | [DATA_ORGANIZATION_PLAN_20260614_CN.md](DATA_ORGANIZATION_PLAN_20260614_CN.md) |
| 全局实验 registry | [registry/README_20260614_CN.md](registry/README_20260614_CN.md) |
| 三条线 inventory CSV | [project_line_inventory_20260614.csv](project_line_inventory_20260614.csv) |
| LADD clean A1B 方法定义 | [../ladd_clean_a1b_method_definition.md](../ladd_clean_a1b_method_definition.md) |
| LADD Probe-A 主线训练规范 | [LADD_MAINLINE_STANDARD_CN.md](LADD_MAINLINE_STANDARD_CN.md) |
| LADD clean A1B 当前曲线 | [ladd_mosaic100_mainline_curves_20260618/](ladd_mosaic100_mainline_curves_20260618/) |
| 旧 LADD 实验归档 | [archive_legacy_ladd_20260618/README_CN.md](archive_legacy_ladd_20260618/README_CN.md) |
| CMDistill 待定实验归档 | [archive_pending_cmdistill_20260618/README_CN.md](archive_pending_cmdistill_20260618/README_CN.md) |

## 1. 三条工作线

| 工作线 | 状态页 | 说明 |
|---|---|---|
| CCLKD 复现消融线 | [cclkd_reproduction/CCLKD_LINE_STATUS_20260614_CN.md](cclkd_reproduction/CCLKD_LINE_STATUS_20260614_CN.md) | 区分 YOLOv5x paper reproduction gate 与 YOLO11 controlled comparison |
| 其他对比方法线 | [comparison_methods/COMPARISON_LINE_STATUS_20260614_CN.md](comparison_methods/COMPARISON_LINE_STATUS_20260614_CN.md) | 管理 LD / FGD / HalluciDet / CCLKD 受控对比 |
| LADD Probe-A 主线 | [LADD_MAINLINE_STANDARD_CN.md](LADD_MAINLINE_STANDARD_CN.md) | 当前主方法为 `clean_a1b_dynprobe`，即 A1 -> B / Probe-A，不经过 A2 |

LADD 当前主方法定义：[../ladd_clean_a1b_method_definition.md](../ladd_clean_a1b_method_definition.md)

LADD 当前结果入口：[ladd_mosaic100_mainline_curves_20260618/](ladd_mosaic100_mainline_curves_20260618/)

LADD 旧诊断入口：[archive_legacy_ladd_20260618/README_CN.md](archive_legacy_ladd_20260618/README_CN.md)

## 2. 当前使用规则

1. 不要直接遍历全仓 `results.csv` 得实验结论；先查 [registry/experiment_registry_20260614.csv](registry/experiment_registry_20260614.csv)。
2. 同一 run 的多个副本通过 [registry/duplicate_results_20260614.csv](registry/duplicate_results_20260614.csv) 去重。
3. `smoke / probe / diag / snapshot / partial / old` 默认只能作为诊断证据。
4. LADD 主线当前固定为 `clean_a1b_dynprobe`；`clean_a1b` static 与 `clean_a1b_dyn` dynamic 只能作为消融实验。
5. 旧 A1-A2-B、旧 mosaic90、旧 no-mosaic/formal repair、BN-freeze、short-B、loss audit 等记录已归档到 [archive_legacy_ladd_20260618/](archive_legacy_ladd_20260618/)。
6. CMDistill native / VEDAI 方向当前暂不作为主线入口，已移入 [archive_pending_cmdistill_20260618/](archive_pending_cmdistill_20260618/) 和 `comparison/cmdistill/archive_pending_20260618/`。
7. 旧文档如与新入口冲突，以 clean A1B 方法定义、主线训练规范和本索引为准。
