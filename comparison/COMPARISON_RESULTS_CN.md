# 对比实验结果

最后更新：2026-06-04 08:55 CST

协议：`from-yolo-pretrain`, formal no-mosaic, 800 epoch, same-capacity same-seed RGB teacher, SAR-only inference。训练长度不是对比指标，只有跑到收敛或明确异常退出后才进入最终主表。

## 已完成

| 方法 | Model/seed | 服务器 | epoch | best AP50-95 | vs same-seed SAR baseline | 判断 |
|---|---|---|---:|---:|---:|---|
| FGD | YOLO11n seed0 | 4090D | 800 | 0.55867@749 | -0.00049 | 基本打平，无正向 |
| CrossKD-style | YOLO11n seed0 | 4090D | 800 | 0.55764@737 | -0.00152 | 基本打平，无正向 |

## 正在运行/待补

| 方法 | Model/seed | 服务器 | 最近记录 | current/best AP50-95 | 状态 |
|---|---|---|---:|---:|---|
| FGD | YOLO11n seed42 | 4090D | epoch 343 | 0.46993 current | 运行中，预计 2026-06-04 晚间完成 |
| FGD | YOLO11n seed123 | 4090 | epoch 134 | 0.37024 best/current | validation OOM 掉线，不计为完成 |
| FGD | YOLO11s seed0 | 4090 | epoch 192 | 0.50358 current | 运行中 |
| CrossKD-style | YOLO11n seed42 | 4090 | epoch 195 | 0.40349 current | 运行中 |
| CrossKD-style | YOLO11n seed123 | 4090 | epoch 186 | 0.39508 current | 运行中 |
| LD | YOLO11n/s seed0 | 90 | 中期记录 | public 内已有 CSV | 需确认是否为从头训练后的有效版本 |
| HalluciDet-style | YOLO11n/s seed0 | 90 | 早期记录 | public 内已有 CSV | 仍需跑满，不能提前下结论 |

## 当前判断

FGD/CrossKD-style 的完整 seed0 结果说明普通检测 KD 在当前 formal no-mosaic 协议下没有明显吃到 RGB teacher 的收益。这个结论目前只覆盖 YOLO11n seed0，不能代替多 seed 结论。

4090 上 FGD seed123 的 OOM 更像是同卡并发过多导致 validation 阶段显存不足，不应解释为方法失败；如果后续重启，应从可恢复 checkpoint 或重新排队，并减少并发。

完整数据见各方法 `results/` 子目录。代码位置和 profile 映射见 [`METHOD_CODE_MAP_CN.md`](METHOD_CODE_MAP_CN.md)。
