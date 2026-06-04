# 对比方法实现复核

最后更新：2026-06-04

本文档给外部老师复核当前受控对比方法的代码语义。正式候选为
`FGD / LD / CCLKD-style / HalluciDet-style`。所有方法共享同一套
YOLO11 HBB 训练入口、formal no-mosaic 数据增强和 800 epoch 收敛口径。

## 1. 结论

| 方法 | 当前实现 | 可否直接作为严格复现 | 后续实验要求 |
|---|---|---|---|
| FGD | teacher spatial/channel attention + GT fg/bg weighting + batch relation | 否，属于 FGD-style YOLO port | 修复前结果不能代表当前实现，需重跑 |
| LD | 前景 anchor 的 YOLO DFL regression logits KL | 可以作为 LD 的 YOLO/DFL 适配 | 旧 soft-logit 结果作废，需重跑 |
| CCLKD-style | teacher-confidence adaptive feature/logit KD + category-constrained contrastive KD | 否，论文无公开可运行代码，且当前缺 relationship-level 项 | 必须先 smoke，再正式跑 |
| HalluciDet-style | detection-utility guided feature/response/margin alignment | 否，没有显式 hallucination module | 写作时必须标注 `-style` |

## 2. 本次修复

### LD

旧 `_ld_style_loss()` 接收的是分类 logits，实际等价于 soft-logit KD，与
Localization Distillation 不符。本次把 student/teacher 检测头输出中的
`boxes` 原始 DFL logits 从总 loss 穿透到每个尺度，在 GT-assigned foreground
anchor 上按四条边分别做温度 KL。

YOLO11 训练态检测头已经返回：

```text
boxes:  [B, 4 * reg_max, N]
scores: [B, num_classes, N]
feats:  multi-scale feature maps
```

因此无需修改 Ultralytics 检测头。

当前 profile 框架会先计算各尺度 LD，再按有效尺度平均；这属于 YOLO 适配选择，
不保证与原实现的样本/尺度加权完全一致，正式写作应注明。

### FGD

旧实现只有 GT 二值前景/背景权重，没有教师特征注意力。本次加入由教师特征
绝对响应生成的连续 spatial/channel attention，并与 GT fg/bg 权重相乘。
这比“完全用 teacher attention 替换 GT mask”更接近 FGD：原方法同时包含
teacher attention 和前景/背景分离。

当前 batch-wise cosine relation 项仍是便携近似，不等同于官方实现的全部
global context 模块，因此写作使用 `FGD-style (teacher-attention weighted)`。

### CCLKD-style

DOI `10.1080/10095020.2026.2633014` 对应论文
*Cross-modal contrastive learning-based object detection under incomplete modalities*。
论文未提供可运行代码，当前实现只保留可明确映射的设计意图：

1. 使用 teacher prediction confidence 决定 token 权重和自适应温度；
2. 同时对齐 feature 与分类预测分布；
3. 在 GT-assigned foreground anchor 上，以类别构造跨模态正负样本。

当前实现没有完整 relationship-level distillation，也把 candidate box 级 CCL
近似为 assigned anchor-token CCL。因此代码和论文中都必须写作
`CCLKD-style portable implementation`，不能声称官方严格复现。

### HalluciDet-style

当前实现没有独立 hallucination pathway/module，但保留了训练期 RGB
privileged information、检测效用加权对齐和 SAR-only 推理约束。保留现状，
写作时明确 `no explicit hallucination module`。

## 3. 淘汰方法处置

- `CrossKD / MGD / MMANet / C2KD` profile 仅保留用于审计历史代码。
- formal launcher 会直接拒绝这些 profile，防止误启动。
- CoLD 已降级为纯历史归档，不再作为当前实验线。
- CrossKD 与旧 FGD/LD 的历史结果可作为失败/实现修正记录，但不能进入修正后的主表。
- 统一归档入口：[`archive/excluded_methods/README.md`](archive/excluded_methods/README.md)。

## 4. 代码入口

| 功能 | 文件 |
|---|---|
| Profile loss 与 teacher/student 输出穿透 | `../ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py` |
| CLI 参数 | `../ladd/code/train_ladd_hbb.py` |
| Trainer 参数传递 | `../ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py` |
| 正式 from-YOLO launcher | `code/launch_formal_from_yolo_kd_job.sh` |
| 正式 transfer launcher | `code/launch_formal_transfer_kd_job.sh` |

两个 launcher 是部署到完整 LADD 工作区使用的模板。Public 包刻意不包含 checkpoint、
数据集和完整 runtime 配置，因此不能在本仓库内直接启动正式训练。

## 5. 启动前检查

1. 对四个 profile 分别执行 `--help` 和短 smoke。
2. LD smoke 必须确认 teacher/student `boxes` 都是 `[B, N, 4*reg_max]`，且 loss 非零。
3. CCLKD-style smoke 必须监控 contrastive 矩阵显存和 NaN；实现最多保留 512 个 foreground token。
4. FGD/LD 修复前已经运行的实验全部使用旧 loss，不得与修复后实验混合统计。
