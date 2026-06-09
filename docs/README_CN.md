# LADD 文档入口

最后更新：2026-06-09

## 快速导航

| 需要 | 文档 |
|---|---|
| 实验总索引 | [experiments/EXPERIMENT_INDEX_CN.md](experiments/EXPERIMENT_INDEX_CN.md) |
| Baseline + LADD 最新台账 | [experiments/BASELINE_LADD_STATUS_CN.md](experiments/BASELINE_LADD_STATUS_CN.md) |
| 对比方法来源、简介和 DOI | [experiments/COMPARISON_METHODS_RECORD_CN.md](experiments/COMPARISON_METHODS_RECORD_CN.md) |
| 对比方法实现复核 | [../comparison/IMPLEMENTATION_REVIEW_CN.md](../comparison/IMPLEMENTATION_REVIEW_CN.md) |
| Baseline 规范与状态 | [experiments/BASELINE_STANDARD_CN.md](experiments/BASELINE_STANDARD_CN.md) |
| LADD 主线规范 | [experiments/LADD_MAINLINE_STANDARD_CN.md](experiments/LADD_MAINLINE_STANDARD_CN.md) |
| LADD 主线稳定性诊断归档 | [../ladd/results/mainline_stability_20260609/README_CN.md](../ladd/results/mainline_stability_20260609/README_CN.md) |
| 对比实验 | [experiments/COMPARISON_EXPERIMENTS_CN.md](experiments/COMPARISON_EXPERIMENTS_CN.md) |
| CCLKD 原文复现 | [../cclkd_reproduction/README.md](../cclkd_reproduction/README.md) |
| 消融计划 | [experiments/ABLATION_PLAN_CN.md](experiments/ABLATION_PLAN_CN.md) |
| LADD 方法概述 | [method/METHOD_OVERVIEW_CN.md](method/METHOD_OVERVIEW_CN.md) |
| 相关工作 | [literature/RELATED_WORK_CN.md](literature/RELATED_WORK_CN.md) |

## 目录说明

| 目录 | 内容 |
|---|---|
| `method/` | LADD 方法描述 |
| `experiments/` | 实验规范、计划、状态 |
| `literature/` | 文献调研与相关工作 |

## 当前状态摘要

1. 正式 OGSOD 协议：`imgsz=256, 800ep, cos_lr, full no-mosaic, default Albumentations`
2. LADD 主线：A2/B 温和学习率 + cap2 + B BN-freeze；YOLO11n BN-freeze 三 seed 已形成正向证据
3. 对比实验：当前只保留 FGD/LD/HalluciDet-style/CCLKD 四方法；CCLKD 先走独立原文复现目录，FGD/LD 修复前结果作废
4. 服务器记录：公开分支保留结果摘要、关键 `results.csv`/`args.yaml` 证据和代码；权重与连接信息不发布
