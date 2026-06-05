# CCLKD 复现协议清单

最后更新：2026-06-05

本清单是启动 CCLKD 复现实验前的硬约束。任何不满足本清单的结果都不能称为
CCLKD 原文复现。

## 1. 实验矩阵

| Model | seed | epoch | 目的 |
|---|---:|---:|---|
| YOLO11s | 0 | 400 | 对齐论文中 YOLO11s 报告设置的第一优先级复现 |
| YOLO11n | 0 | 400 | 小模型容量补充，验证方法在 n 上是否仍有效 |
| YOLO11s | 42/123 | 400 | 单 seed 跑通后再补稳定性 |
| YOLO11n | 42/123 | 400 | 单 seed 跑通后再补稳定性 |

## 2. 启动前阻断项

1. dataset yaml 必须由脚本检查为 `nc=3`，类别名与 OGSOD-1.0 三类一致。
2. 数据增强必须从 CCLKD 论文/作者设置逐项抄出并写入本目录，不能使用
   LADD formal no-mosaic 设置代替。
3. trainer 必须是 online teacher-student joint training；frozen RGB teacher
   只能作为实现调试，不可进入复现表。
4. teacher/student 的优化器、学习率、warmup、batch、imgsz、epoch 需要和
   CCLKD 原文设置逐项对齐。若论文没有公开某项，必须在记录中标注“论文未公开，
   使用可复现近似值”，并单独列出影响。
5. 训练日志必须保存完整命令、git commit、dataset yaml、hyp/augment 配置和
   best epoch。

## 3. 与 LADD formal 协议的差别

LADD 受控主表使用：

```text
800 epoch
full no-mosaic
same-capacity same-seed frozen RGB teacher
```

CCLKD 原文复现使用：

```text
400 epoch
paper-matched augmentation
online teacher-student joint training
```

因此两套结果不可直接混入同一张 controlled main table。CCLKD 复现结果用于确认
方法实现是否可信；之后如需进入 LADD 主表，需要另起“统一受控协议”的 CCLKD run。

## 4. 待补代码

- `train_cclkd_online_hbb.py`：online teacher-student 训练入口。
- `launch_cclkd_paper_repro_job.sh`：只允许 YOLO11s/YOLO11n、400 epoch 和
  paper-matched augmentation。
- `check_cclkd_repro_protocol.py`：启动前校验 `nc=3`、epoch、augmentation 和
  online trainer 标志。
