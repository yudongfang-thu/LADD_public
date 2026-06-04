# CoLD 复现证据地图

最后更新：2026-06-03 23:50 CST

CoLD 是当前最需要外部排查的对比方法。本文档只说明证据在哪里、每个版本解决什么问题、目前卡在哪里。

## 1. 原文目标

CoLD: Category-Oriented Localization Distillation for SAR Object Detection and a Unified Benchmark, IEEE TGRS 2023, DOI `10.1109/TGRS.2023.3291356`。

原文报告 OGSOD-1.0 HBB / YOLOv5x:

| Setting | AP50-95 |
|---|---:|
| YOLOv5 baseline | 0.463 |
| CoLD full | 0.567 |
| NCLD only | 0.563 |
| TCLD only | 0.502 |

我们复现的核心困难是：candidate CPM/OKD 能看到方向性收益，但 TCLD/NCLD 趋势和速度开销与原文不一致。

## 2. 版本与证据

| 版本 | 服务器/来源 | 核心设置 | 证据位置 | 主要结论 |
|---|---|---|---|---|
| v1 matched | 4090D / 旧 90 记录 | 只在 GT positive anchors 上做 matched distillation | `v1_matched_4090d/` | AP 约 0.470，远低于原文；覆盖不到大规模 NCLD |
| v2 candidate | 117/5880 Ada 归档 | topk candidate CPM + online OKD，50ep | `v2_candidate_5880ada/`, `COLD_REPRO_FINAL_CN.md` | NCLD 有方向性收益，但 TCLD > NCLD，与原文相反 |
| v3 frozen teacher | 117/5880 Ada 归档 | frozen RGB teacher offline KD | `v3_frozen_5880ada/` | 低于 baseline，说明 OKD 很关键 |
| 5090D baseline snapshot | 本地归档 | YOLOv5x SAR baseline variants | `remote_records/5090d_baseline_snapshot_20260601/` | 记录原文协议 baseline 坐标 |
| 90 migration | 90 服务器 | 2026-06-03 迁移到独立 CoLD 工作树，online candidate 400ep | `remote_records/90/LADD_cold_v5p0_20260603/` | 正在跑/已产生若干 `results.txt` 和日志 |

## 3. 当前 90 记录

90 最新 CoLD 目录已复制以下内容：

- `cold_anchor/logs/*.log`: online NCLD/TCLD/both candidate 运行日志。
- `cold_anchor/runs/ogsod_cold_online_terms/*/results.txt`: YOLOv5 风格结果文件。
- `scripts/ogsod_public/cold_baseline_repro_20260528/*.sh`: 队列和启动脚本。
- `scripts/ogsod_public/cold_baseline_repro_20260528/train_cold_v5p0_hbb.py`: 当前训练实现。

为了公开安全，包里不包含 SSH 连接方式、密码、密钥或完整服务器入口命令。

## 4. 外部排查问题

1. 我们的 `candidate` 近似是否真的等价于原文 CPM 的候选框集合和 distribution 计算。
2. YOLOv5 v5.0 没有 DFL bins，当前用候选框集合 softmax KL 近似定位分布，这是否偏离 CoLD 原公式。
3. online OKD 中 teacher/student 同步训练的调度是否和原文一致。
4. 为什么 TCLD 在短跑中强于 NCLD，而原文 NCLD 是主导项。
5. 当前 Python 逐图逐类循环造成约 4-5x 慢速，是否需要向量化才能接近原文 3.6% overhead。
