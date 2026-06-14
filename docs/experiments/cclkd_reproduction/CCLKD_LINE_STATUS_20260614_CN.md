# CCLKD 复现消融线状态（2026-06-14）

## 1. 这条线在回答什么

CCLKD 线不是 LADD 主线的一部分。它有两个独立目的：

| 分支 | 目的 | 能回答的问题 | 不能回答的问题 |
|---|---|---|---|
| YOLO11 受控消融 | 在 LADD formal no-mosaic 协议下比较 CCLKD 组件 | CCLKD 作为受控对比方法在同协议下是否有效 | 不能证明 CCLKD 原文可复现 |
| YOLOv5x 原文协议复现 | 按 CCLKD paper protocol 检查实现是否靠谱 | 实现是否能接近原文 Table 5/12 | 不能直接进入 LADD 主表 |

这两个分支的结果不能混用。

## 2. Canonical 入口

| 类型 | 路径 |
|---|---|
| 综合状态 | [../../../cclkd_reproduction/CCLKD_REPRODUCTION_STATUS_20260614.md](../../../cclkd_reproduction/CCLKD_REPRODUCTION_STATUS_20260614.md) |
| 复现协议 | [../../../cclkd_reproduction/PROTOCOL_CN.md](../../../cclkd_reproduction/PROTOCOL_CN.md) |
| 版本记录 | [../../../cclkd_reproduction/experiment_versions/](../../../cclkd_reproduction/experiment_versions/) |
| YOLOv5 sanity | [../../../cclkd_reproduction/yolov5_sanity/results/](../../../cclkd_reproduction/yolov5_sanity/results/) |
| 全局 registry | [../registry/experiment_registry_20260614.csv](../registry/experiment_registry_20260614.csv) |

## 3. 已作废或只能诊断的结果

以下结果不能进入论文主表：

| 结果类型 | 原因 |
|---|---|
| 2026-06-08 之前 CCL 错误 formulation | 使用了不正确的 CCL 对象 |
| RLD 量级修复前结果 | 与 paper formulation 不一致 |
| frozen-teacher CCLKD | 不符合 CCLKD online teacher-student 训练定义 |
| YOLOv5x paper protocol 结果 | 可用于复现 gate，不能作为 LADD controlled comparison |
| 只有 full、没有 Table 12 结构的零散结果 | 不能支撑组件消融结论 |

## 4. 当前读法

1. CCLKD 实现线已经从“能不能跑”进入“是否对齐原文 protocol / 是否形成有效 controlled comparison”的阶段。
2. YOLOv5x 分支的 baseline gate 很关键：如果 det-only 自身达不到原文量级，就不能急着解释 KD 增益不足。
3. YOLO11 分支应作为对比方法线的一部分看待，优先统一协议、统一 baseline、统一训练长度。

## 5. 下一步

| 优先级 | 任务 | 验收 |
|---|---|---|
| P0 | 重新核对 YOLOv5x det-only baseline gate | 与原文 Table 5 的 AP50/AP 差距有明确解释 |
| P1 | 整理 YOLO11 CCLKD controlled comparison 状态 | 明确哪些 run 可以与 LD/FGD/LADD 同表 |
| P1 | 将作废结果在 registry 中标注为 diagnostic/invalid | 避免后续重复引用 |

