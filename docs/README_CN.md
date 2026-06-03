# LADD 文档入口

最后更新：2026-06-03

## 快速导航

| 需要 | 文档 |
|---|---|
| 实验总索引 | [experiments/EXPERIMENT_INDEX_CN.md](experiments/EXPERIMENT_INDEX_CN.md) |
| Baseline + LADD 最新台账 | [experiments/BASELINE_LADD_STATUS_CN.md](experiments/BASELINE_LADD_STATUS_CN.md) |
| 对比方法来源、简介和 DOI | [experiments/COMPARISON_METHODS_RECORD_CN.md](experiments/COMPARISON_METHODS_RECORD_CN.md) |
| Baseline 规范与状态 | [experiments/BASELINE_STANDARD_CN.md](experiments/BASELINE_STANDARD_CN.md) |
| LADD 主线规范 | [experiments/LADD_MAINLINE_STANDARD_CN.md](experiments/LADD_MAINLINE_STANDARD_CN.md) |
| 对比实验 | [experiments/COMPARISON_EXPERIMENTS_CN.md](experiments/COMPARISON_EXPERIMENTS_CN.md) |
| 消融计划 | [experiments/ABLATION_PLAN_CN.md](experiments/ABLATION_PLAN_CN.md) |
| LADD 方法概述 | [method/METHOD_OVERVIEW_CN.md](method/METHOD_OVERVIEW_CN.md) |
| CoLD 复现总结 | [cold_repro/COLD_REPRO_FINAL_CN.md](cold_repro/COLD_REPRO_FINAL_CN.md) |
| 对比方法调研 | [literature/COMPARISON_METHOD_SURVEY_20260528_CN.md](literature/COMPARISON_METHOD_SURVEY_20260528_CN.md) |
| 相关工作 | [literature/RELATED_WORK_CN.md](literature/RELATED_WORK_CN.md) |

## 目录说明

| 目录 | 内容 |
|---|---|
| `method/` | LADD 方法描述 |
| `experiments/` | 实验规范、计划、状态 |
| `cold_repro/` | CoLD 复现完整记录 |
| `literature/` | 文献调研与相关工作 |
| `archive/` | 历史文档（仅供溯源，不作当前结论） |

## 当前状态摘要

1. 正式 OGSOD 协议：`imgsz=256, 800ep, cos_lr, full no-mosaic, default Albumentations`
2. LADD 主线：A2 稳定修正 + cap2，YOLO11n 三 seed 在跑
3. 对比实验：FGD/CrossKD YOLO11n seed0 已完成（无正向提升），LD/HalluciDet-style seed0 正在跑
4. 服务器记录：公开包只保留匿名化的结果、配置、日志和代码；不包含 SSH、密码、密钥或完整连接命令
