# 实验记录索引

最后更新：2026-06-03

本文档是当前实验状态的入口。所有历史文档保留在 `archive/` 中。

## 文档导航

| 需要 | 文档 |
|---|---|
| 给导师看的 baseline + LADD 最新台账 | [BASELINE_LADD_STATUS_CN.md](BASELINE_LADD_STATUS_CN.md) |
| 给导师看的对比方法来源、简介和 DOI | [COMPARISON_METHODS_RECORD_CN.md](COMPARISON_METHODS_RECORD_CN.md) |
| Baseline 训练规范与当前状态 | [BASELINE_STANDARD_CN.md](BASELINE_STANDARD_CN.md) |
| LADD 主线规范与状态 | [LADD_MAINLINE_STANDARD_CN.md](LADD_MAINLINE_STANDARD_CN.md) |
| 对比实验计划与实现 | [COMPARISON_EXPERIMENTS_CN.md](COMPARISON_EXPERIMENTS_CN.md) |
| 消融实验计划 | [ABLATION_PLAN_CN.md](ABLATION_PLAN_CN.md) |
| CoLD 复现总结 | [../cold_repro/COLD_REPRO_FINAL_CN.md](../cold_repro/COLD_REPRO_FINAL_CN.md) |
| LADD 方法概述 | [../method/METHOD_OVERVIEW_CN.md](../method/METHOD_OVERVIEW_CN.md) |
| 对比方法调研 | [../literature/COMPARISON_METHOD_SURVEY_20260528_CN.md](../literature/COMPARISON_METHOD_SURVEY_20260528_CN.md) |

## 当前正式主线

```text
收敛 SAR/RGB baseline (no-mosaic 协议)
+ LADD A1 -> A2 -> B
+ A2 检测稳定修正 + cap2 反坍缩
```

- 正式协议：`imgsz=256, 800ep, cos_lr, full no-mosaic, default Albumentations`
- LADD 主线：`A1=10 -> A2=50 -> B=800`, cap2 reach-rank
- A2 修正：`MuSGD, lr0=0.001, warmup_epochs=0`

## 当前实验状态

| 线 | 服务器 | 状态 |
|---|---|---|
| LADD 主线 | 90 + 4090D | YOLO11n seed0/42 已有正向完成点；seed0/123 BN-freeze 修正版在 90 诊断；YOLO11s 多 run 运行中 |
| 非 CoLD 对比 | 90 + 4090D | FGD/CrossKD-style n seed0 已完成；LD/HalluciDet-style n/s seed0 运行中；FGD n seed42 运行中 |
| CoLD 复现 | 90 | 独立慢跑/诊断线，不占非 CoLD 主队列 |
| 消融实验 | — | 未开始 |

## 服务器

| 服务器 | 用途 |
|---|---|
| 90 (8x3090) | baseline 主参考，LADD 诊断，非 CoLD 对比，CoLD 独立复现 |
| 4090D | 当前接管 LADD/FGD 并行实验 |
| 117 (RTX 5880 Ada) | 因 IO/网络过慢，暂不作为非 CoLD 主力 |
