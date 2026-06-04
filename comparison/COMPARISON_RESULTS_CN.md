# 对比实验结果

最后更新：2026-06-04 08:55 CST

协议：`from-yolo-pretrain`, formal no-mosaic, 800 epoch, same-capacity same-seed RGB teacher, SAR-only inference。训练长度不是对比指标，只有跑到收敛或明确异常退出后才进入最终主表。

## 历史已完成（不进入修正版主表）

| 方法 | Model/seed | 服务器 | epoch | best AP50-95 | vs same-seed SAR baseline | 判断 |
|---|---|---|---:|---:|---:|---|
| FGD 旧实现 | YOLO11n seed0 | 4090D | 800 | 0.55867@749 | -0.00049 | 缺 teacher attention，需重跑 |
| CrossKD-style | YOLO11n seed0 | 4090D | 800 | 0.55764@737 | -0.00152 | 方法已淘汰 |

## 正在运行/待补

| 方法 | Model/seed | 服务器 | 最近记录 | current/best AP50-95 | 状态 |
|---|---|---|---:|---:|---|
| FGD 旧实现 | 多个 run | 4090/4090D | 中期记录 | public 内已有 CSV | 实现已修复，旧 run 不进入主表 |
| CrossKD-style | seed42/123 | 4090 | 已停止 | public 内已有 CSV | 方法已淘汰 |
| LD 旧实现 | YOLO11n/s seed0 | 90 | 中期记录 | public 内已有 CSV | 实际为 soft-logit KD，结果作废 |
| CCLKD-style | 待定 | 待定 | 未启动 | - | 先 smoke |
| HalluciDet-style | YOLO11n/s seed0 | 90 | 早期记录 | public 内已有 CSV | 仍需跑满，不能提前下结论 |

## 当前判断

2026-06-04 修复后，正式四方法为 FGD/LD/CCLKD-style/HalluciDet-style。FGD 和
LD 的旧结果不能用于评价修正版；CrossKD 已停止并淘汰。当前没有足够的修正版
结果支撑方法优劣结论。

旧 FGD/LD、CrossKD 与 CoLD 数据统一见
[`archive/excluded_methods/`](archive/excluded_methods/)；代码位置和当前 profile 映射见
[`METHOD_CODE_MAP_CN.md`](METHOD_CODE_MAP_CN.md)。
