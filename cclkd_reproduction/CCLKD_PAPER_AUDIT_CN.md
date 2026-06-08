# CCLKD 原文信息与当前实现审计

更新时间：2026-06-08

本文档用于固定 CCLKD 的原文定义、当前代码映射和实验协议边界。后续任何 CCLKD 实验必须先核对本文档，避免再次把“原文实现对齐”和“训练协议对齐”混在一起。

## 结论

当前目标不是完整复现 CCLKD 原文训练设置，而是在 LADD formal baseline 协议下评估一个尽量贴近 CCLKD 原文方法定义的 online teacher-student 实现。因此：

- 方法实现必须使用 `--cclkd-formulation paper`。
- 训练协议必须使用 LADD formal no-mosaic baseline 协议。
- 消融必须按原文 Table 12 的组件结构完整跑，不只跑 full。
- 2026-06-08 之前使用错误 CCL、错误 protocol 或只跑 full 的 CCLKD 结果均只能作为诊断，不能进入主表或消融表。

## 原文方法定义

论文：`Cross-modal contrastive learning-based object detection under incomplete modalities`，Geo-spatial Information Science，online published 2026-04-07。

原文 CCLKD 是双分支 online teacher-student 训练：

- teacher network 输入 easy-to-detect modality，例如 OGSOD 的 optical/RGB。
- student network 输入 hard-to-detect modality，例如 OGSOD 的 SAR。
- inference 阶段只使用 student/hard modality。
- teacher 不是离线 frozen teacher，而是在训练时用 easy modality 的 detection loss 共同优化。

原文核心模块：

| 模块 | 原文含义 |
|---|---|
| COP | 基于 teacher 预测类别和 GT 类别，把候选框划分为 target-category / non-target-category，给 ATKD 和 CCL 提供 category mask。 |
| PATM | Prediction-guided adaptive temperature mechanism。根据 teacher 对某一类正样本的预测熵映射到类别温度 `T_j`，默认范围 `[0.5, 5.0]`，熵尺度参数初始化为 `5.0`。 |
| LLD | Logit-level distillation。对候选框定位分布做 category-aware KL，包括 target 与 non-target 分布。 |
| FLD | Feature-level distillation。对候选框区域内的 feature map 做 box-aligned bilinear sampling，得到区域特征后做蒸馏。 |
| RLD | Relationship-level distillation。对同类候选框特征构造自相关矩阵，约束 student 保留 teacher 的类内结构关系。 |
| CCL | Category-constrained contrastive learning。按类别构造正/负候选框对，并做 class-balanced InfoNCE，使同类 teacher-student 特征更近、异类更可分。 |

原文 loss 结构：

- teacher detection loss：标准检测损失。
- student detection loss：标准检测损失。
- ATKD loss：`LLD + FLD + RLD`，其中 PATM 控制温度。
- CCL loss：类别约束对比损失。
- 总体上是 detection loss + KD loss + CCL loss 的 joint training。

## 原文实验设置

原文 Table 2 的 OGSOD-1.0 设置：

| Dataset | Input size | Optimizer | Epoch | Batch size | LR | Momentum |
|---|---:|---|---:|---:|---:|---:|
| OGSOD-1.0 | 256 x 256 | SGD | 400 | 32 | 0.01 | 0.937 |

原文还说明使用 standard image augmentation 和 MixUp-based image mixing。该设置属于“复现 CCLKD 原文训练协议”，不是当前 LADD 主表受控比较协议。

当前 LADD controlled comparison 要使用：

| 项 | 设置 |
|---|---|
| imgsz | 256 |
| epochs | 当前 CCLKD 消融先跑 400；LADD formal baseline 主协议为 800 |
| batch | n/s = 64 |
| lr | `lr0=0.01`, `lrf=0.01`, `cos-lr=True` |
| mosaic | `0.0` |
| mixup/cutmix | `0.0` |
| close-mosaic | `0` |
| deterministic | True |

因此当前正确实验不是 `cclkd_reproduction/code/launch_cclkd_paper_repro_job.sh`，而是 `comparison/code/launch_formal_online_cclkd_ablation_job.sh`。

## 原文 Table 12 消融

原文 Table 12 的结构和数值如下：

| LLD | FLD | RLD | PATM | CCL | mAP50 | mAP |
|---|---|---|---|---|---:|---:|
|  |  |  |  |  | 80.9 | 46.3 |
| yes |  |  |  |  | 83.4 | 48.5 |
| yes | yes |  |  |  | 84.2 | 49.3 |
| yes | yes | yes |  |  | 84.9 | 50.1 |
| yes | yes | yes | yes |  | 87.0 | 55.1 |
|  |  |  |  | yes | 85.9 | 54.4 |
| yes | yes | yes | yes | yes | 88.7 | 57.3 |

当前 LADD baseline 协议下要跑的 CCLKD 消融：

| ablation | 权重设置 | 温度设置 |
|---|---|---|
| `lld` | `LLD=1, FLD=0, RLD=0, CCL=0` | fixed T = 1 |
| `lld_fld` | `LLD=1, FLD=1, RLD=0, CCL=0` | fixed T = 1 |
| `lld_fld_rld` | `LLD=1, FLD=1, RLD=1, CCL=0` | fixed T = 1 |
| `atkd` | `LLD=1, FLD=1, RLD=1, CCL=0` | PATM `[0.5, 5.0]` |
| `ccl_only` | `LLD=0, FLD=0, RLD=0, CCL=1` | not used by KD terms |
| `full` | `LLD=1, FLD=1, RLD=1, CCL=1` | PATM `[0.5, 5.0]` |

## 当前代码映射

当前实现入口：

- Online trainer：`cclkd_reproduction/code/train_cclkd_online_hbb.py`
- LADD 协议消融 launcher：`comparison/code/launch_formal_online_cclkd_ablation_job.sh`
- CCLKD 原文协议 launcher：`cclkd_reproduction/code/launch_cclkd_paper_repro_job.sh`

关键代码状态：

| 原文要求 | 当前代码 |
|---|---|
| online teacher-student | `CCLKDOnlineHBBTrainer` 同时构建 student 和 teacher，teacher 参数 `requires_grad=True`。 |
| teacher RGB detection loss | `teacher_det_loss` 独立计算，并进入 total loss。 |
| student SAR detection loss | `student_det_loss` 独立计算，并进入 total loss。 |
| teacher-side COP | `formulation=paper` 时使用 `assign_source = teacher_main`。 |
| PATM | `formulation=paper` 使用 `_adaptive_temperature_class(...)`，按 class 正样本熵得到一个类别温度。 |
| target/non-target LLD | `formulation=paper` 中正样本和采样负样本都进入 DFL KL。 |
| box-aligned FLD/RLD/CCL | `formulation=paper` 通过 `_sample_box_features(...)` 使用 `grid_sample` 从 neck feature map 按 box 采样。 |
| class-balanced CCL | 按当前 mini-batch COP 类别频次做 `1/n_j` 归一权重。 |

## 当前实现仍是 YOLO11 适配的地方

这些不是“已完全等价于原文”的部分，后续写论文或报告必须明确：

1. 原文公式基于候选框级 `p_box`、`p_cls` 和区域特征；当前 YOLO11 HBB 实现把 anchor/token assignment、DFL logits 和 neck feature map 映射到这个定义。
2. 原文 FLD 文字描述为区域特征经 `1x1 conv` 降维后蒸馏；当前实现没有额外学习一个 `1x1 conv` 投影，而是直接对 box-sampled neck feature 做 KL。
3. 原文 CCL 的 Algorithm 2 以 target/non-target candidate-box pair 为单位；当前 `paper` formulation 使用正候选框 teacher-student 相似度和负候选框 teacher-student 相似度构造二分类 InfoNCE，是 YOLO11 检测头上的近似实现。
4. `grid_sample` 的 CUDA backward 非完全 deterministic，PyTorch 会给 warn-only 提示；这不应视为训练失败，但会影响 bit-level determinism。
5. `cclkd_reproduction/code/train_cclkd_online_hbb.py` 默认参数仍是 `--cclkd-formulation adapted`，这是为了兼容旧诊断入口；正式 CCLKD 实验必须由 launcher 显式传入 `paper`。

## 当前正确实验入口

在 4090 服务器 `/root/shared-nvme/LADD_public` 中，从 repo root 启动：

```bash
EPOCHS=400 BATCH_SIZE=64 comparison/code/launch_formal_online_cclkd_ablation_job.sh n lld 0 0
EPOCHS=400 BATCH_SIZE=64 comparison/code/launch_formal_online_cclkd_ablation_job.sh n lld_fld 0 1
EPOCHS=400 BATCH_SIZE=64 comparison/code/launch_formal_online_cclkd_ablation_job.sh n lld_fld_rld 0 0
EPOCHS=400 BATCH_SIZE=64 comparison/code/launch_formal_online_cclkd_ablation_job.sh n atkd 0 1
EPOCHS=400 BATCH_SIZE=64 comparison/code/launch_formal_online_cclkd_ablation_job.sh n ccl_only 0 0
EPOCHS=400 BATCH_SIZE=64 comparison/code/launch_formal_online_cclkd_ablation_job.sh n full 0 1
```

两张 4090 同时只能直接跑两个任务；不要用队列脚本。每轮完成后人工启动下一轮。

当前 run 输出路径：

```text
runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/online_cclkd/yolo11n/paper_ablation/
logs/formal_nomosaic_20260528/comparisons/online_cclkd_paper_ablation/
```

## 作废结果记录原则

以下结果不能进入正式表：

1. Frozen-teacher CCLKD：不符合原文 online teacher-student 定义。
2. `--cclkd-formulation adapted` 的结果：只能作为 YOLO11 token-level 诊断。
3. 2026-06-08 之前 CCL 使用 DFL 回归特征或错误负样本方向的结果。
4. 使用 CCLKD 原文 paper protocol 但被误当作 LADD baseline 协议的结果。
5. 只跑 full、没有完整 Table 12 结构消融的结果。

## 最低启动前检查

启动任何 CCLKD 消融前必须确认：

```bash
DRY_RUN=1 EPOCHS=400 BATCH_SIZE=64 \
  comparison/code/launch_formal_online_cclkd_ablation_job.sh n full 0 0
```

输出中必须同时出现：

- `--cclkd-formulation paper`
- `--mosaic 0.0`
- `--mixup 0.0`
- `--close-mosaic 0`
- `--batch 64`
- `--cos-lr`
- `--deterministic`

如果任一项不满足，不允许启动。
