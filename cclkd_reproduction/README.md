# CCLKD 原文协议复现

最后更新：2026-06-05

本目录专门用于复现 CCLKD 原文，不属于 `comparison/` 下的受控对比方法目录。

`comparison/cclkd/` 只记录我们论文主表中的 CCLKD 对比实现边界；本目录则必须尽量
对齐 CCLKD 论文自身实验协议。二者不能混用结果。

## 目标

复现论文：

```text
Cross-modal contrastive learning-based object detection under incomplete modalities
GIS 2026
DOI: 10.1080/10095020.2026.2633014
```

论文 PDF 位于 [`paper/`](paper/)。

## 必须对齐的协议

| 项 | 要求 |
|---|---|
| 数据集 | OGSOD-1.0 HBB，类别数必须为 `nc=3` |
| 模态 | 训练期 RGB + SAR，推理期 SAR-only |
| 模型 | YOLO11s 和 YOLO11n |
| 训练方式 | CCLKD 原文定义的 online teacher-student joint training |
| 输入尺寸 | `imgsz=256` |
| epoch | 与原文一致：400 epoch |
| 数据增强 | 必须逐项按原文/作者设置对齐，不能沿用 LADD formal no-mosaic 协议替代 |
| 指标 | AP50-95、AP50，并记录 best epoch |

## 当前代码入口

当前目录已补齐原文复现专用入口：

| 文件 | 用途 |
|---|---|
| [`code/train_cclkd_online_hbb.py`](code/train_cclkd_online_hbb.py) | Online teacher-student HBB 训练入口；SAR student 与 RGB teacher 同步训练，teacher 有独立 detection loss 并参与 optimizer |
| [`code/check_cclkd_repro_protocol.py`](code/check_cclkd_repro_protocol.py) | 启动前硬校验：`nc=3`、YOLO11n/s、`imgsz=256`、400ep、batch=32、SGD lr=0.01、mosaic=1.0、online trainer |
| [`code/launch_cclkd_paper_repro_job.sh`](code/launch_cclkd_paper_repro_job.sh) | 原文协议 launcher，只允许 YOLO11n / YOLO11s |
| [`ABLATION_PLAN_CN.md`](ABLATION_PLAN_CN.md) | YOLO11n 消融计划；逐项映射原文 Table 12 的 LLD / FLD / RLD / PATM / CCL |

注意：本地公开仓库只完成静态校验。正式训练环境需要先做 tiny smoke，确认
student detection loss、teacher detection loss 和 CCLKD loss 都非零，再启动 400 epoch。

示例：

```bash
export STUDENT_DATA=/path/to/ogsod_hbb_sar.yaml
export TEACHER_DATA=/path/to/ogsod_hbb_rgb.yaml
bash cclkd_reproduction/code/launch_cclkd_paper_repro_job.sh s 0 0
```

## 与受控对比的关系

- `cclkd_reproduction/`：回答“我们是否能按 CCLKD 原文协议复现其方法”。
- `comparison/cclkd/`：回答“在我们论文统一协议下，CCLKD 与 LADD/FGD/LD/HalluciDet 如何比较”。

只有当本目录的原文复现跑通并确认实现可信后，CCLKD 才能进入 `comparison/`
中的正式受控对比。受控对比若保留 CCLKD，也应使用同一个 online 方法定义，而不是
frozen-teacher 近似。
