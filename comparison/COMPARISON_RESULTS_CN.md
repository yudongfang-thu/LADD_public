# 对比实验结果

最后更新：2026-06-05 16:45 CST

协议：`from-yolo-pretrain`, formal no-mosaic, 800 epoch, same-capacity same-seed RGB teacher, SAR-only inference。训练长度不是对比指标，只有跑到收敛或明确异常退出后才进入最终主表。

## 历史已完成（不进入修正版主表）

| 方法 | Model/seed | 服务器 | epoch | best AP50-95 | vs same-seed SAR baseline | 判断 |
|---|---|---|---:|---:|---:|---|
| FGD 旧实现 | YOLO11n seed0 | 4090D | 800 | 0.55867@749 | -0.00049 | 缺 teacher attention，需重跑 |

## 正在运行/待补

| 方法 | Model/seed | 服务器 | 最近记录 | current/best AP50-95 | 状态 |
|---|---|---|---:|---:|---|
| FGD 旧实现 | 多个 run | 4090/4090D | 中期记录 | public 内已有 CSV | 实现已修复，旧 run 不进入主表 |
| LD 旧实现 | YOLO11n/s seed0 | 90 | 中期记录 | public 内已有 CSV | 实际为 soft-logit KD，结果作废 |
| FGD-style 修正版 | YOLO11n seed0 | 双卡 4090 | 已停止 | 作废 | 目标机 yaml 为 `nc=5`，不进入主表 |
| LD 修正版 | YOLO11n seed0 | 双卡 4090 | 已停止 | 作废 | 目标机 yaml 为 `nc=5`，不进入主表 |
| CCLKD | YOLO11n seed0 | 双卡 4090 | 已停止 | 作废 | frozen-teacher 旧实现 + `nc=5`，不进入主表 |
| HalluciDet-style | YOLO11n seed0 | 双卡 4090 | 已停止 | 作废 | 目标机 yaml 为 `nc=5`，不进入主表 |

## 当前判断

2026-06-05 审计发现，双卡 4090 active dataset yaml 被错误迁移为 `nc=5`
旧类别表，而正式 OGSOD HBB 协议是 `nc=3`：

```text
0: bridge
1: harbor
2: storage_tank
```

因此 2026-06-04 晚上在双卡 4090 启动的旧四方法 smoke / formal partial runs
全部作废。结果和日志已从 active 路径移走，服务器侧归档到
`/root/shared-nvme/archive/invalid_5class_yaml_20260605_162122`。当前没有任何
双卡 4090 修正版正式结果可用于主表。

CCLKD 旧实现同时存在方法偏差。2026-06-05 loss 级代码已修正 LLD/FLD/RLD；
`cclkd_reproduction/code/` 已补原文定义的 online teacher-student trainer。CCLKD 在
online 入口完成 GPU smoke 和原文条件复现实验前，不得写入正式受控对比结果表。

旧 FGD/LD 原始训练数据已从 public 当前分支移除，仅 Git 历史可追溯；代码位置和当前 profile 映射见
[`METHOD_CODE_MAP_CN.md`](METHOD_CODE_MAP_CN.md)。
