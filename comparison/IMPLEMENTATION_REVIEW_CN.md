# 对比方法实现复核

最后更新：2026-06-05

本文档给外部老师复核当前受控对比方法的代码语义。当前正式方法为
`FGD / LD / CCLKD / HalluciDet-style`。其中 FGD、LD、HalluciDet-style 使用
frozen-teacher 受控对比入口；CCLKD 必须先在
[`../cclkd_reproduction/`](../cclkd_reproduction/) 中使用 online teacher-student
原文复现入口完成 smoke 和 400ep 复现，再回到 `comparison/` 执行统一协议对比。

## 1. 结论

| 方法 | 当前实现 | 可否直接作为严格复现 | 后续实验要求 |
|---|---|---|---|
| FGD | 官方形式的 softmax spatial/channel attention + GT fg/bg weighting + batch relation 近似 | 否，属于 FGD-style YOLO port | 修复前结果不能代表当前实现，需重跑 |
| LD | 前景 anchor 的 YOLO DFL regression logits KL，shape 异常直接失败 | 可以作为 LD 的 YOLO/DFL 适配 | 旧 soft-logit 结果作废，需重跑 |
| CCLKD paper-structured reimplementation | COP + entropy temperature + localization-only LLD / FLD-MSE / RLD feature-correlation / class-balanced CCL 的 YOLO11 loss 适配；online trainer 已补 | 否，需先完成原文协议 smoke/复现 | 暂不正式跑，先 smoke online 复现入口 |
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

Teacher 虽处于 eval 模式，但当前 Ultralytics Detect head 会返回
`(decoded_predictions, raw_predictions_dict)`，本实现从第二项提取原始 DFL
logits。当前增加了 fail-fast 检查，并使用独立 `ld_temperature=10.0`。

### FGD

旧实现只有 GT 二值前景/背景权重，没有教师特征注意力。本次加入由教师特征
绝对响应生成的连续 spatial/channel attention，并与 GT fg/bg 权重相乘。
这比“完全用 teacher attention 替换 GT mask”更接近 FGD：原方法同时包含
teacher attention 和前景/背景分离。

当前 batch-wise cosine relation 项仍是便携近似，不等同于官方实现的全部
global context 模块，因此写作使用 `FGD-style (teacher-attention weighted)`。
attention 保留官方的 softmax、`H*W`/`C` 缩放，并使用官方默认温度 0.5。

### CCLKD paper-structured reimplementation

DOI `10.1080/10095020.2026.2633014` 对应论文
*Cross-modal contrastive learning-based object detection under incomplete modalities*。
论文未提供可运行代码。2026-06-05 版本按论文结构重新实现：

1. COP：teacher dominant class 与 GT assigned label 一致时形成类别正样本 mask；
2. adaptive temperature：按类别正样本 teacher probability entropy 映射到 `[0.5, 5.0]`；
3. LLD：只对 YOLO11 DFL raw regression logits 做 localization distribution KD，不做分类 logit KL；
4. FLD：类别正样本 feature MSE；
5. RLD：同类 token 的 `R^T R / n` feature-dimension correlation matrix MSE；
6. CCL：按类别频次反比加权，对 target / non-target spatial distributions 做 contrastive loss。

仍需注明适配边界：YOLO11 没有论文 YOLOv5 candidate-box/objectness 的完全同构公开实现，
因此本实现用 DFL raw logits 作为 spatial distribution，用 dense token feature 近似
candidate region feature。更关键的是，frozen-teacher 对比入口仍不符合 CCLKD 原文；
原文方法定义包含 joint online teacher-student training branch。因此当前 frozen-teacher
loss 只能作为实现部件，不能作为 CCLKD 复现或正式对比结果入口；应使用
`cclkd_reproduction/code/train_cclkd_online_hbb.py` 先做 smoke / formal。

### HalluciDet-style

当前实现没有独立 hallucination pathway/module，但保留了训练期 RGB
privileged information、检测效用加权对齐和 SAR-only 推理约束。保留现状，
写作时明确 `no explicit hallucination module`。

## 3. 代码入口

| 功能 | 文件 |
|---|---|
| Profile loss 与 teacher/student 输出穿透 | `../ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py` |
| CLI 参数 | `../ladd/code/train_ladd_hbb.py` |
| Trainer 参数传递 | `../ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py` |
| 正式 from-YOLO frozen-teacher launcher | `code/launch_formal_from_yolo_kd_job.sh` |
| 正式 transfer frozen-teacher launcher | `code/launch_formal_transfer_kd_job.sh` |
| CCLKD 原文复现目录 | `../cclkd_reproduction/` |

两个 launcher 是部署到完整 LADD 工作区使用的模板。Public 包刻意不包含 checkpoint、
数据集和完整 runtime 配置，因此不能在本仓库内直接启动正式训练。

## 4. 启动前检查

1. 对 FGD/LD/HalluciDet-style 分别执行 `--help` 和短 smoke。
2. LD smoke 必须确认 teacher/student `boxes` 都是 `[B, N, 4*reg_max]`，且 loss 非零。
3. CCLKD 必须先 smoke online teacher-student trainer；frozen-teacher smoke 不再作为 CCLKD 通过证据。
4. FGD/LD 修复前已经运行的实验全部使用旧 loss，不得与修复后实验混合统计。
5. 第二轮复核意见响应与未采纳原因见
   [`REVIEW_FEEDBACK_RESPONSE_CN.md`](REVIEW_FEEDBACK_RESPONSE_CN.md)。
