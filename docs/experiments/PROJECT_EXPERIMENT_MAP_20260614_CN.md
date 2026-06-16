# LADD 项目实验地图（2026-06-14）

本文是当前本地仓库的实验入口页。目的不是替代各实验报告，而是把已经混杂在本地的证据分成清晰的工作线，防止重复计数、误用旧结果，或者把诊断实验当成主线结论。

## 0. 当前总判断

当前项目主要有三条线：

| 线 | 目标 | 当前状态 | 新入口 |
|---|---|---|---|
| CCLKD 复现消融线 | 验证 CCLKD paper formulation 与原文协议/受控协议下的有效性 | YOLO11 受控消融与 YOLOv5x 原文复现必须分开；旧错误 formulation 已作废 | [CCLKD_LINE_STATUS_20260614_CN.md](cclkd_reproduction/CCLKD_LINE_STATUS_20260614_CN.md) |
| 其他对比方法线 | 维护 FGD / LD / CMDistill / HalluciDet-YOLO / CCLKD 的对比方法实现与训练证据 | HalluciDet-style 已废弃；frozen-teacher KD、standalone HalluciDet 和 online CCLKD 必须分口径核验 | [COMPARISON_LINE_STATUS_20260614_CN.md](comparison_methods/COMPARISON_LINE_STATUS_20260614_CN.md) |
| LADD 主线诊断线 | 复盘 LADD 主线从健康结果到当前崩溃/平台现象的因果链 | 已有大量 A2/B/BN/schedule/split-load 诊断；当前重点是 forensic review，而不是继续开新实验 | [LADD_LINE_FORENSIC_MAP_20260614_CN.md](ladd_mainline_diagnosis/LADD_LINE_FORENSIC_MAP_20260614_CN.md) |

全局 registry 入口：

- [experiment_registry_20260614.csv](registry/experiment_registry_20260614.csv)
- [duplicate_results_20260614.csv](registry/duplicate_results_20260614.csv)
- [registry README](registry/README_20260614_CN.md)

整理方案：

- [DATA_ORGANIZATION_PLAN_20260614_CN.md](DATA_ORGANIZATION_PLAN_20260614_CN.md)

## 1. 数据层级

当前不建议大规模移动原始 evidence。先采用四层结构：

| 层级 | 含义 | 例子 | 使用规则 |
|---|---|---|---|
| `raw / shutdown evidence` | 从服务器迁回的原始轻量证据 | `ladd/results/ladd4090_shutdown_sync_20260614/` | 只作 provenance，不直接作为论文数字来源 |
| `curated summary` | 人工/脚本整理后的结果表 | `docs/experiments/*summary*.csv`, `ladd/results/*/summary/` | 优先用于汇报和复盘 |
| `registry` | 全局扫描索引，负责去重和定位 | `docs/experiments/registry/` | 查 run 是否存在、是否重复、来自哪台服务器 |
| `local archive` | 过时报告、旧计划、临时分析 | `/Users/yudongfang/Desktop/光sar/LADD_public_local_archives/` | 移出仓库，保留历史但不污染 Git 工作区 |

## 2. 统一使用规则

1. 不要直接遍历全仓 `results.csv` 得出实验数量或主表数字；必须先查 registry 和分线 summary。
2. 每条结果进入当前结论前，必须至少能追到：`results.csv`、`args.yaml` 或命令、服务器来源、commit 或 manifest、是否含 diagnostics。
3. 同一 run 的多个副本只保留一个 canonical 解释；副本通过 `duplicate_results_20260614.csv` 查。
4. `smoke / probe / diag / snapshot / partial / old` 默认不是主线结果，除非文档显式说明它是某个诊断问题的证据。
5. LADD 主线诊断中，`B100/B120` 压缩 schedule 不能直接解释 `B800` 后期趋势，只能说明入口和早期行为。
6. 任何 `.pt/.pth`、TensorBoard event、wandb、大 run 目录都不进入 GitHub 证据包。

## 3. 当前最重要的整理任务

| 优先级 | 任务 | 目的 |
|---|---|---|
| P0 | LADD forensic review | 重新钉死主线 LADD 的真实 loss/phase/optimizer 定义，区分历史健康主线与近期漂移实验 |
| P0 | 三条线 canonical summary | 每条线只保留一个当前可信状态页，旧文档移出仓库到 local archive |
| P1 | registry 字段人工标注 | 自动 `validity=candidate_or_unknown` 不等于可进论文，需要补 `claim_usable` / `role` |
| P1 | 曲线图索引 | 将 LADD 关键曲线按问题组织，而不是按生成时间堆放 |
| P2 | 轻量 evidence 清理 | 移出仓库内的大 tar 包和非必要 shutdown 副本，保留 manifest |

## 4. 推荐汇报口径

当前不应说“实验一无所获”。更准确的口径是：

1. CCLKD 线已经明确区分了 paper reproduction gate 与 LADD controlled comparison，旧错误 formulation 结果不再混用。
2. 对比方法线已经形成方法实现/证据框架，但部分训练结果需要重新按 registry 核验。
3. LADD 主线诊断线已经证明：训练框架/B800 schedule 本身不是唯一根因；近期问题集中在 A2/B checkpoint、B 阶段 LADD loss 组合、数值稳定性和协议漂移。
4. 现在需要停止堆实验，先完成 forensic review，再决定是否只做一个严格 replay。
