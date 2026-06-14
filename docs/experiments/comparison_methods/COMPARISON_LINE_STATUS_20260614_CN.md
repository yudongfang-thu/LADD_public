# 其他对比方法线状态（2026-06-14）

## 1. 这条线在回答什么

对比方法线用于维护 LADD 论文主表/附表中的外部方法对照。它和 CCLKD 复现线、LADD 主线诊断线分开管理。

当前包含：

| 方法 | 当前角色 | 当前状态 |
|---|---|---|
| LD | 受控对比方法 | 已有训练证据，需要按 registry 和最新同步结果复核完成度 |
| FGD | 受控对比方法 | normalization 修复后需要 smoke/正式 run 证据闭环 |
| CCLKD | 受控对比方法 + 独立复现线 | YOLO11 controlled comparison 归入本线；paper reproduction 归入 CCLKD 线 |
| HalluciDet | 受控/复现候选 | strict HalluciDet 与旧 HalluciDet-style 必须分开 |
| HalluciDet-style | 历史诊断 | 已废弃，不进入当前主表 |

## 2. Canonical 入口

| 类型 | 路径 |
|---|---|
| 方法总览 | [../../../comparison/README.md](../../../comparison/README.md) |
| 方法代码映射 | [../COMPARISON_METHODS_RECORD_CN.md](../COMPARISON_METHODS_RECORD_CN.md) |
| 当前状态草稿 | [../../../comparison/FINAL_STATUS_20260613.md](../../../comparison/FINAL_STATUS_20260613.md) |
| 4090 关机同步证据 | [../../../comparison/results_shutdown_sync_20260614/](../../../comparison/results_shutdown_sync_20260614/) |
| 全局 registry | [../registry/experiment_registry_20260614.csv](../registry/experiment_registry_20260614.csv) |

## 3. 当前风险

| 风险 | 影响 | 处理 |
|---|---|---|
| 旧 HalluciDet-style 和 strict HalluciDet 混名 | 会把废弃方法当正式对比 | 文档中统一标记 HalluciDet-style deprecated |
| LD/FGD/CCLKD 训练长度不一致 | 不能直接比较 best/final | 主表只用 protocol-matched run |
| comparison evidence 和 LADD shutdown evidence 混放 | 容易重复计数 | 统一通过 registry 查 run |
| `FINAL_STATUS_20260613.md` 带有预期值 | 不能作为结果表 | 只作为计划/状态草稿，最终数字另建 summary |

## 4. 推荐当前结论格式

对比方法进入论文或汇报前，每个方法至少需要一行这样的记录：

| method | model | seed | protocol | epochs | best AP | final AP | source | status |
|---|---|---:|---|---:|---:|---:|---|---|
| LD | n/s | 0 | formal no-mosaic | 800 | TBD | TBD | registry canonical path | verified/running/invalid |

未完成核验前，不建议继续写“优于/低于 LADD”的结论。

## 5. 下一步

| 优先级 | 任务 | 验收 |
|---|---|---|
| P0 | 从 registry 提取 comparison family 的 canonical run 表 | 每个方法只有一个当前状态 |
| P0 | 标注 HalluciDet-style 为 archive/diagnostic | 主表不会误引用 |
| P1 | LD/FGD 结果核验 | 明确是否完成 800ep、是否 protocol matched |
| P1 | 将 CCLKD YOLO11 controlled comparison 与 CCLKD paper reproduction 分开汇报 | 两套表不混用 |

