# CCLKD 原文协议复现实验异常记录

最后更新：2026-06-06 14:55 CST

## 1. 现象摘要

当前 `cclkd_reproduction/` 的 paper-protocol 复现实验可以稳定运行，不再出现此前的 validation loss / EMA mutation 崩溃；但已经完成的 seed0 结果明显低于 CCLKD 论文报告量级，说明复现实验本身仍存在重大差距，需要进一步排查。

已完成结果：

| run | epoch | precision | recall | mAP50 | AP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLO11n seed0 | 400/400 | 0.696845 | 0.593599 | 0.624522 | 0.385318 |
| YOLO11s seed0 | 400/400 | 0.804146 | 0.715463 | 0.754318 | 0.484594 |

正在运行 / 等待：

| run | 状态 | 当前 epoch | 当前 AP50-95 |
|---|---|---:|---:|
| YOLO11n seed42 | running | 49 | 0.226373 |
| YOLO11s seed42 | running | 32 | 0.240044 |
| YOLO11n seed123 | 已手动并行启动 | 尚未写 results.csv | - |
| YOLO11s seed123 | 已手动并行启动 | 尚未写 results.csv | - |

队列日志显示 seed0 已完成，seed42 已按队列启动。由于显存充足，seed123 已在 2026-06-06 14:48-15:00 之间手动并行启动，避免继续串行等待。

## 2. 当前复现实验协议

启动脚本：`cclkd_reproduction/code/launch_cclkd_paper_repro_job.sh`

核心配置：

- `imgsz=256`
- `epochs=400`
- `batch=32`
- `optimizer=SGD`
- `lr0=0.01`
- `mosaic=1.0`
- `mixup=0.1`
- `close_mosaic=10`
- student 输入 SAR，teacher 输入 RGB
- teacher/student 均从对应 `yolo11n.pt` / `yolo11s.pt` COCO 预训练初始化
- teacher 为 trainable online teacher，并带 RGB detection loss
- student 以 SAR-only detector 做 validation

数据 YAML：

- `ogsod_hbb_sar.yaml`
- `ogsod_hbb_rgb.yaml`
- 均为 `nc=3`，类别为 `bridge / harbor / storage_tank`

### 2.1 数据规模复核

本次上传的诊断包里包含原始 queue log，可以确认复现实验实际使用的是
CCLKD 论文量级的大切片 OGSOD，而不是 CoLD 复现中常见的 `2870/1162`
小切片协议。

压缩包 `../cclkd_repro_diag_20260606_144738.tar.gz` 中的
`queue_s_gpu1_20260606_014745.log` 记录了 Ultralytics 实际扫描到的
训练/验证规模：

```text
train: Scanning /root/shared-nvme/OGSOD-1.0/sar/labels/train ... 1/14664
...
all 3667 9614 ...
```

双卡服务器上对应目录的实际文件计数也已复核：

| split | SAR images | SAR labels | RGB images | RGB labels |
|---|---:|---:|---:|---:|
| train | 14664 | 14664 | 14664 | 14664 |
| test | 3667 | 3667 | 3667 | 3667 |

因此，当前 CCLKD reproduction 的低 AP 不能解释为“只用了 2870 张训练图，
400 epoch 等效训练强度不足”。更可能的排查方向应放在 CCLKD 机制适配、
online teacher 训练状态、teacher-side validation、以及 paper-protocol
SAR/RGB standalone baseline。

另一个独立问题是：双卡服务器上的 `cold_anchor` / `yolov5_v5p0` 目录中仍存在
`nc=5` 的旧 CoLD/Yolov5 YAML，但这些 YAML 不被 `LADD_public` 当前
CCLKD reproduction 使用；本次 CCLKD reproduction 使用的是本目录快照中的
`nc=3` public YAML。

## 3. 需要老师重点看的证据文件

本目录小文件：

- `summary.json`：远程结果自动摘要
- `*_results.csv`：已完成/运行中的结果曲线快照
- `train_cclkd_online_hbb.py`：当前 online CCLKD trainer 实现快照
- `launch_cclkd_paper_repro_job.sh`：当前复现启动脚本快照
- `check_cclkd_repro_protocol.py`：启动前协议检查脚本快照
- `ogsod_hbb_sar.yaml` / `ogsod_hbb_rgb.yaml`：训练数据配置快照

完整日志压缩包：

- `../cclkd_repro_diag_20260606_144738.tar.gz`

压缩包内包含：

- `queue_n_gpu0_20260606_014745.log`
- `queue_s_gpu1_20260606_014745.log`
- 完整 results.csv / code / yaml 快照

## 4. 已观察到的异常点

1. seed0 跑满 400 epoch 后，YOLO11s AP50-95 只有 `0.484594`，明显低于 CCLKD 论文报告的 AP 量级，也低于我们希望复现到的同任务结果。
2. YOLO11n seed0 更低，AP50-95 为 `0.385318`。
3. 日志中没有发现 `Traceback`、`RuntimeError`、`KeyError`、`CUDA out of memory` 等硬崩溃。
4. seed0 的最终 epoch 同时也是 best epoch，说明不是 early collapse 后恢复失败，而是整条曲线最终仍停在较低水平。
5. 训练日志中早期 `cclkd_loss` 会出现一段为 0 的阶段，随后变为非零；这可能来自前景 token / class-positive pair 条件触发，也可能暴露候选 token 构造与论文 COP 机制的差异，需要核查。

## 5. 可能排查方向

这些不是最终结论，只是根据当前证据列出的优先检查项：

1. **论文机制适配差异**：原文基于 YOLOv5 风格候选框 / objectness / region feature；当前实现适配为 YOLO11 HBB + DFL + dense token / assigned foreground token，可能导致 LLD / FLD / RLD 对象并不完全等价。
2. **online teacher 训练细节**：当前 teacher 是 trainable RGB detector，并有 RGB detection loss；但还需要核对原文 teacher-student 参数更新、loss 权重、teacher/student forward 顺序和 feature 提取层是否一致。
3. **数据增强细节**：当前已对齐 `mosaic=1.0`、`mixup=0.1`、`imgsz=256`、`400ep`，但 Ultralytics 默认 `hsv/erasing/auto_augment/close_mosaic` 等细节仍可能与论文实现不同。
4. **评价协议差异**：需要确认论文报告 AP 的精确定义、HBB/OBB、test split、是否使用同一 OGSOD-1.0 HBB 标注。
5. **学生/教师 standalone paper-protocol baseline**：建议补同协议 YOLO11n/s SAR-only 与 RGB-only baseline，用于判断当前低分来自 CCLKD loss 本身，还是 paper-protocol baseline 已经偏低。

## 6. 运行状态备注

远程服务器：双卡 4090，路径 `/root/shared-nvme/LADD_public`

2026-06-06 14:48-15:00 后已将原本排队等待的 seed123 也并行启动：

- YOLO11n seed123：GPU0，与 seed42 / LADD n42 并发
- YOLO11s seed123：GPU1，与 seed42 / LADD s0 并发

完整日志压缩包是在并行 seed123 完全启动前生成的，因此主要覆盖 seed0 完成日志与 seed42 运行日志；seed123 后续日志仍在远程 `logs/cclkd_reproduction/parallel_*_s123_*.log` 中实时增长。

这会让原始串行 queue 在 seed42 完成后尝试再次启动 seed123 时遇到已存在目录或已运行任务；后续监控时应以实际 run 目录和进程为准，而不是只看旧 queue 的最终状态。
