# 协议与 CCLKD 实现审计记录

最后更新：2026-06-05

本文记录 2026-06-05 对 `LADD_public` 和双卡 4090 部署的复核结论。当前不启动新实验；本记录用于人工检查代码与结果有效性。

## 1. 已确认的协议错误

双卡 4090 服务器 `/root/shared-nvme/ladd` 的 active OGSOD HBB yaml 曾错误配置为 5 类：

```text
nc: 5
0: ship
1: storage-tank
2: baseball-diamond
3: tennis-court
4: basketball-court
```

正式 OGSOD HBB 协议应为 3 类：

```text
nc: 3
0: bridge
1: harbor
2: storage_tank
```

影响：

- 双卡 4090 上从 3 类 A2 / baseline 权重继续训练的 LADD B 阶段被强制改成 5 类 head，结果无效。
- 双卡 4090 上从 YOLO 预训练启动的 comparison runs 使用了错误任务定义，结果无效。
- 这些结果不能进入主表、不能作为 smoke 通过证据、不能用于比较趋势。

处置：

- 服务器训练进程已停止。
- 服务器错误结果已归档到 `/root/shared-nvme/archive/invalid_5class_yaml_20260605_162122`。
- public 仓库 `shared/configs/datasets_public/ogsod1_{sar,rgb}_detect.yaml` 已改为 3 类。

## 2. 需要人工重点检查的配置文件

| 文件 | 检查点 |
|---|---|
| `shared/configs/datasets_public/ogsod1_sar_detect.yaml` | `nc=3`，类别为 bridge/harbor/storage_tank |
| `shared/configs/datasets_public/ogsod1_rgb_detect.yaml` | 同上 |
| `comparison/COMPARISON_RESULTS_CN.md` | 双卡 4090 旧 smoke/formal partial runs 标为作废 |
| `docs/experiments/EXPERIMENT_PLAN_CN.md` | 当前阶段为审计与人工复核，不是运行中 |
| `comparison/cclkd/README.md` | CCLKD 不能声称官方严格复现 |

## 3. CCLKD 旧实现问题

旧 `_cclkd_style_loss()` 主要做：

- teacher-confidence feature MSE；
- teacher/student class-logit KL；
- foreground token 上按 label 相等构造 InfoNCE。

该实现缺少或弱化了论文关键结构：

- COP 不是基于 teacher dominant category 与 GT label 一致性的 category mask；
- adaptive temperature 不是按类别正样本 entropy 映射到 `[0.5, 5.0]`；
- LLD 没有显式利用 YOLO regression spatial distribution；
- FLD 不是 category-masked feature distribution KD；
- RLD 缺失；
- CCL 不是论文描述的 target / non-target category distribution contrast；
- teacher 分支不是论文 Table 3 所述 joint teacher-student online distillation 训练方式。

## 4. CCLKD 当前实现

当前代码位置：

```text
ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py
ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/loss.py
```

当前 `_cclkd_style_loss()` 已改为 paper-structured loss-level reimplementation：

- COP：`teacher_scores.sigmoid().argmax` 与 `target_scores.argmax` 一致，并满足 teacher confidence threshold 后进入类别正样本。
- ATKD temperature：按类别正样本 teacher probability 的 binary entropy 归一化后，映射到 `[0.5, 5.0]`。
- LLD：只对 YOLO11 DFL raw regression logits 做 localization distribution KD，不包含分类 logit KL。
- FLD：类别正样本 token feature MSE。
- RLD：同类 token 的 `R^T R / n` feature-dimension correlation matrix MSE。
- CCL：按类别频次反比加权，对 target / non-target teacher-student neck spatial features 做 contrastive loss；DFL regression distribution 只用于 LLD。

新增参数：

```text
--cclkd-temperature-min 0.5
--cclkd-temperature-max 5.0
--cclkd-entropy-scale 5.0
```

这些参数已同步到：

```text
ladd/code/train_ladd_hbb.py
ladd/code_versions/current_hbb/tools/train_ladd_hbb.py
ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py
ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/trainer.py
ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh
comparison/code/launch_formal_from_yolo_kd_job.sh
comparison/code/launch_formal_transfer_kd_job.sh
```

## 5. 仍然不能声称严格官方复现的原因

即使当前 loss 实现比旧版更接近论文，仍不能写作官方严格复现：

- 论文没有公开可运行代码。
- 论文实现基于 YOLOv5 风格 candidate boxes / objectness / regional feature extraction，本仓库是 YOLO11 HBB + DFL。
- 当前用 YOLO11 DFL raw logits 适配论文 spatial distribution。
- 当前用 per-level dense token feature 近似 candidate-box region feature。
- frozen-teacher 对比入口中的 teacher 仍来自给定 RGB teacher 权重，不是完整 joint teacher-student online training branch。这是最大架构差距；必须改用 `cclkd_reproduction/code/` 的 online trainer 完成 smoke 和复现后，CCLKD 才能进入正式对比主表讨论。

推荐写法：

```text
CCLKD online YOLO11 adaptation pending GPU smoke and 400-epoch paper-protocol reproduction
```

不推荐写法：

```text
official CCLKD reproduction
strict CCLKD reproduction
controlled CCLKD main-table result
```

## 6. 原文条件复现要求

后续 CCLKD 复现实验必须先验证 online trainer：

1. RGB teacher branch 与 SAR student branch 同步训练；
2. teacher branch 使用 RGB 图像和检测监督更新；
3. student branch 使用 SAR 图像和检测监督更新；
4. CCLKD loss 从 online teacher outputs/features 蒸馏到 student；
5. 先按原文最接近条件做 YOLO11s 和 YOLO11n / 400 epoch 复现，再决定是否进入受控对比；
6. 受控对比若保留 CCLKD，也应使用同一个 online 方法定义，而不是 frozen teacher 近似。

## 7. 已完成的非训练验证

已执行：

```text
python3 -m py_compile ...
bash -n comparison/code/launch_formal_from_yolo_kd_job.sh
bash -n comparison/code/launch_formal_transfer_kd_job.sh
bash -n ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh
```

结论：语法和 shell 脚本解析通过。当前本地 macOS Python 为
`/opt/homebrew/bin/python3`，未安装 `torch`，因此 `python3 ladd/code/train_ladd_hbb.py --help`
和 CPU 合成张量 loss 调用不能在本机完成。未做 GPU smoke，未启动训练。

## 8. 人工复核后才能继续的步骤

1. 复核 public 仓库 diff。
2. 使用 `cclkd_reproduction/code/train_cclkd_online_hbb.py` 复核 teacher/student optimizer、loss 和数据流。
3. 复核双卡服务器部署前的 yaml、代码 hash 和 `--help` 输出。
4. 只做 1 epoch / tiny fraction smoke，确认日志包含 `nc=3` 且 teacher/student detection loss 与 CCLKD loss 均非零。
5. smoke 通过后，先做原文条件 YOLO11s 和 YOLO11n / 400 epoch 复现，再讨论受控对比。
