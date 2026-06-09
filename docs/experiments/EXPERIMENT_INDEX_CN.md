# 实验记录索引

最后更新：2026-06-09

本文档是当前实验状态的入口。当前 public 分支只保留主线代码、正式对比方法和关键摘要。

## 文档导航

| 需要 | 文档 |
|---|---|
| 给导师看的 baseline + LADD 最新台账 | [BASELINE_LADD_STATUS_CN.md](BASELINE_LADD_STATUS_CN.md) |
| 给导师看的对比方法来源、简介和 DOI | [COMPARISON_METHODS_RECORD_CN.md](COMPARISON_METHODS_RECORD_CN.md) |
| 对比方法代码复核 | [../../comparison/IMPLEMENTATION_REVIEW_CN.md](../../comparison/IMPLEMENTATION_REVIEW_CN.md) |
| CCLKD 原文协议复现 | [../../cclkd_reproduction/README.md](../../cclkd_reproduction/README.md) |
| Baseline 训练规范与当前状态 | [BASELINE_STANDARD_CN.md](BASELINE_STANDARD_CN.md) |
| LADD 主线规范与状态 | [LADD_MAINLINE_STANDARD_CN.md](LADD_MAINLINE_STANDARD_CN.md) |
| LADD 主线稳定性诊断归档 | [../../ladd/results/mainline_stability_20260609/README_CN.md](../../ladd/results/mainline_stability_20260609/README_CN.md) |
| 对比实验计划与实现 | [COMPARISON_EXPERIMENTS_CN.md](COMPARISON_EXPERIMENTS_CN.md) |
| 消融实验计划 | [ABLATION_PLAN_CN.md](ABLATION_PLAN_CN.md) |
| LADD 方法概述 | [../method/METHOD_OVERVIEW_CN.md](../method/METHOD_OVERVIEW_CN.md) |

## 当前正式主线

```text
收敛 SAR/RGB baseline (no-mosaic 协议)
+ LADD A1 -> A2 -> B
+ A2/B 检测稳定修正 + cap2 反坍缩 + B BN-freeze
```

- 正式协议：`imgsz=256, 800ep, cos_lr, full no-mosaic, default Albumentations`
- LADD 主线：`A1=10 -> A2=50 -> B=800`, cap2 reach-rank
- A2/B 修正：`MuSGD, lr0=0.001, warmup_epochs=0`
- B 稳定修正：`FREEZE_BN_STATS=1`

## 当前实验状态

| 线 | 服务器 | 状态 |
|---|---|---|
| LADD 主线 | 90 + 双卡 4090 | YOLO11n BN-freeze seed0/42/123 已完成且正向；YOLO11s seed0 BN-freeze best 正向但 last 退化 |
| 受控对比 | 运行中 | 四方法为 FGD/LD/HalluciDet-style/CCLKD；CCLKD 正在按 LADD baseline 协议做 paper-aligned 消融 |
| 消融实验 | 筹备中 | 先冻结 YOLO11n 主线，再排 LADD 主方法消融 |

## 服务器

| 服务器 | 用途 |
|---|---|
| 90 (8x3090) | baseline 主参考、LADD 诊断与受控对比 |
| 双卡 4090 | 当前 CCLKD 与 LADD 关键补跑节点 |
| 4090D | 历史结果恢复；当前不作为新实验状态来源 |
| 117 (RTX 5880 Ada) | 因 IO/网络过慢，暂不作为当前受控实验主力 |
