# LADD 文档入口

最后更新：2026-06-18

## 快速导航

| 需要 | 文档 |
|---|---|
| 论文主协议与主表 gate | [paper/PAPER_PROTOCOL_CN.md](paper/PAPER_PROTOCOL_CN.md) |
| 论文方法命名白名单 | [paper/METHOD_NAME_WHITELIST_CN.md](paper/METHOD_NAME_WHITELIST_CN.md) |
| paper results schema | [../paper_results/README.md](../paper_results/README.md) |
| paper launcher | [../scripts/paper/README.md](../scripts/paper/README.md) |
| 实验总索引 | [experiments/EXPERIMENT_INDEX_CN.md](experiments/EXPERIMENT_INDEX_CN.md) |
| LADD-clean / A1B 主方法定义 | [ladd_clean_a1b_method_definition.md](ladd_clean_a1b_method_definition.md) |
| 方法定义与实现入口 | [method/METHOD_DEFINITIONS_AND_IMPLEMENTATION_CN.md](method/METHOD_DEFINITIONS_AND_IMPLEMENTATION_CN.md) |
| 对比方法来源、简介和 DOI | [experiments/COMPARISON_METHODS_RECORD_CN.md](experiments/COMPARISON_METHODS_RECORD_CN.md) |
| 对比方法实现复核 | [../comparison/IMPLEMENTATION_REVIEW_CN.md](../comparison/IMPLEMENTATION_REVIEW_CN.md) |
| Baseline 规范与状态 | [experiments/BASELINE_STANDARD_CN.md](experiments/BASELINE_STANDARD_CN.md) |
| LADD 主线规范 | [experiments/LADD_MAINLINE_STANDARD_CN.md](experiments/LADD_MAINLINE_STANDARD_CN.md) |
| 项目实验地图 | [experiments/PROJECT_EXPERIMENT_MAP_20260614_CN.md](experiments/PROJECT_EXPERIMENT_MAP_20260614_CN.md) |
| CCLKD 原文复现 | [../cclkd_reproduction/README.md](../cclkd_reproduction/README.md) |
| LADD 方法概述 | [method/METHOD_OVERVIEW_CN.md](method/METHOD_OVERVIEW_CN.md) |
| 相关工作 | [literature/RELATED_WORK_CN.md](literature/RELATED_WORK_CN.md) |

## 目录说明

| 目录 | 内容 |
|---|---|
| `method/` | LADD 方法描述 |
| `experiments/` | 实验规范、计划、状态 |
| `literature/` | 文献调研与相关工作 |

## 当前状态摘要

1. 当前 LADD 主方法口径已固定为 `LADD Probe-A / LADD-clean A1B`：SAR baseline 初始化，A1 后直接进 B，B 阶段使用 dynamic teacher core + frozen reach probe，A2 只保留为历史诊断/消融
2. 当前 paper-facing 主协议固定为 mosaic100：`imgsz=256, 800ep, mosaic=1.0, close_mosaic=700, cos_lr, deterministic`；baseline、LADD、comparison methods 必须同协议重跑，主表结果只从 `paper_results/` 的 verified rows 取数
3. 对比实验：当前方法口径为 FGD-style、LD、CMDistill-style、HalluciDet-YOLO adaptation、CCLKD online；旧 `hallucidet_style` profile 已移除
4. CCLKD：原文复现走 `cclkd_reproduction/`，LADD 统一协议对比走 online launcher；frozen-teacher loss 组件不能单独写作 CCLKD 复现
5. 服务器记录：公开分支保留结果摘要、关键 `results.csv`/`args.yaml` 证据和代码；权重与连接信息不发布
