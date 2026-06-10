# 对比方法实现复核

最后更新：2026-06-10

本文档给外部老师复核当前受控对比方法的代码语义。当前正式方法为
`FGD-style / LD / CCLKD / HalluciDet-style`。其中 FGD-style、LD、HalluciDet-style 使用
frozen-teacher 受控对比入口；CCLKD 必须先在
[`../cclkd_reproduction/`](../cclkd_reproduction/) 中使用 online teacher-student
原文复现入口完成 smoke 和 400ep 复现，再回到 `comparison/` 执行统一协议对比。

## 1. 结论

| 方法 | 当前实现 | 可否直接作为严格复现 | 后续实验要求 |
|---|---|---|---|
| FGD-style | fg/bg feature loss + teacher/student attention mask loss；GT-box mask 默认启用；legacy batch relation 默认关闭 | 否，属于 FGD-YOLO adaptation | 修复前结果不能代表当前实现，需重跑 |
| LD | raw YOLO11 DFL logits 的 foreground/main LD + teacher-quality VLR-style candidate LD，shape 异常直接失败 | 可以作为 LD 的 YOLO/DFL 适配 | 旧 foreground-only / soft-logit 结果作废，需重跑 |
| CCLKD paper-structured reimplementation | COP + entropy temperature + localization-only LLD / FLD-MSE / RLD feature-correlation / class-balanced CCL 的 YOLO11 loss 适配；online trainer 已补 | 否，需先完成原文协议 smoke/复现 | 暂不正式跑，先 smoke online 复现入口 |
| HalluciDet-style | detection-utility guided feature/response/margin alignment，profile 名称为 `hallucidet_style` | 否，没有显式 hallucination module | 旧 `hallucidet` 名称禁用；写作时必须标注 `-style` |

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

当前 LD-YOLO11 adaptation 使用 raw DFL localization logits，并包含两部分：
foreground/main LD 和 teacher-quality VLR-style candidate LD。main LD 在
TaskAlignedAssigner 的 foreground anchors 上执行，权重来自 assigned target
quality 与 teacher assigned-class confidence；VLR-style LD 在非 foreground
anchors 中选取 teacher confidence 与 GT IoU 加权的候选位置。YOLO11
TaskAlignedAssigner 不暴露官方 LD/ATSS 的 `get_vlr_region()` API，因此该
VLR 是 YOLO 适配，不是官方 region selector 的逐行复现。

Teacher 虽处于 eval 模式，但当前 Ultralytics Detect head 会返回
`(decoded_predictions, raw_predictions_dict)`，本实现从第二项提取原始 DFL
logits。当前增加了 fail-fast 检查，并使用独立 `ld_temperature=10.0`。
默认 `ld_main_weight=0.25`、`ld_vlr_weight=0.25`，对齐官方 LD loss weight
量级，同时保持 `ld_use_vlr=1`。

### FGD

旧实现只有 teacher-attention weighted token MSE 和 batch relation 近似。本次
改为更接近官方 FGD focal 分支的 YOLO11 适配：

- teacher/student 都计算 spatial attention 和 channel attention；
- feature loss 拆成 foreground loss 和 background loss；
- 默认 `fgd_mask_mode=gt_box`，用 pixel xyxy GT boxes 投影到各层 feature map，
  box 内填 `1 / area`，background mask 可归一化；
- 新增 attention `mask_loss`，对齐 student 与 teacher attention；
- 官方 trainable global relation 模块本轮不实现；旧 batch-wise relation 已改为
  legacy opt-in，`fgd_lambda=0.0` 默认关闭，不能继续冒充官方 relation。

因此正式写作应使用 `FGD-style` 或 `FGD-YOLO adaptation`，并注明
`focal + attention mask implemented; official trainable global relation disabled by default`。
默认内部权重按官方量级设置：`fgd_alpha=0.001`、`fgd_beta=0.0005`、
`fgd_gamma=0.001`、`fgd_lambda=0.0`。其中 `fgd_lambda` 保持为 0，因为当前未实现
official trainable global relation，不能默认打开 legacy batch relation。

### CCLKD paper-structured reimplementation

DOI `10.1080/10095020.2026.2633014` 对应论文
*Cross-modal contrastive learning-based object detection under incomplete modalities*。
论文未提供可运行代码。2026-06-05 版本按论文结构重新实现：

1. COP：teacher dominant class 与 GT assigned label 一致时形成类别正样本 mask；
2. adaptive temperature：按类别正样本 teacher probability entropy 映射到 `[0.5, 5.0]`；
3. LLD：只对 YOLO11 DFL raw regression logits 做 localization distribution KD，不做分类 logit KL；
4. FLD：类别正样本 feature MSE；
5. RLD：同类 token 的 `R^T R / n` feature-dimension correlation matrix MSE；
6. CCL：按类别频次反比加权，对 target / non-target neck spatial features 做 contrastive loss；DFL regression distribution 只用于 LLD。

仍需注明适配边界：YOLO11 没有论文 YOLOv5 candidate-box/objectness 的完全同构公开实现，
因此本实现用 DFL raw logits 适配 LLD localization distribution，用 dense token
feature 近似 candidate region feature 并承担 CCL。更关键的是，frozen-teacher 对比入口仍不符合 CCLKD 原文；
原文方法定义包含 joint online teacher-student training branch。因此当前 frozen-teacher
loss 只能作为实现部件，不能作为 CCLKD 复现或正式对比结果入口；应使用
`cclkd_reproduction/code/train_cclkd_online_hbb.py` 先做 smoke / formal。

### HalluciDet-style

当前实现没有独立 image-space hallucination pathway/module，也没有
“SAR/IR -> hallucinated RGB/image representation -> frozen RGB detector
detection loss”路径。它只保留训练期 RGB privileged information、特征/响应/
margin 对齐和 SAR-only YOLO11 student 推理约束。

profile 名称已改为 `hallucidet_style`；旧 `hallucidet` 名称不再被 CLI 和
launcher 接受。旧 hallucidet 运行目录或表格字段若保留，只能标为
`hallucidet_style_old`，不能写成 HalluciDet official reproduction。若以后做
strict HalluciDet，应单独新建入口，例如
`comparison/hallucidet_image_reproduction/`，实现 SAR/IR image 到 3-channel
hallucinated image，再由 frozen RGB detector detection loss 反传训练；本轮
不混入当前 B-only frozen-teacher comparison launcher。

## 3. 代码入口

| 功能 | 文件 |
|---|---|
| Profile loss 与 teacher/student 输出穿透 | `../ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py` |
| CLI 参数 | `../ladd/code/train_ladd_hbb.py` |
| Trainer 参数传递 | `../ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py` |
| 正式 from-YOLO frozen-teacher launcher | `code/launch_formal_from_yolo_kd_job.sh` (`fgd|ld|hallucidet_style`) |
| 正式 transfer frozen-teacher launcher | `code/launch_formal_transfer_kd_job.sh` (`fgd|ld|hallucidet_style`) |
| CCLKD 原文复现目录 | `../cclkd_reproduction/` |

两个 launcher 是部署到完整 LADD 工作区使用的模板。Public 包刻意不包含 checkpoint、
数据集和完整 runtime 配置，因此不能在本仓库内直接启动正式训练。

## 4. 启动前检查

1. 对 `fgd`/`ld`/`hallucidet_style` 分别执行 `--help`、dry-run 和短 smoke。
2. LD smoke 必须确认 teacher/student `boxes` 都是 `[B, N, 4*reg_max]`，且 loss 非零。
3. CCLKD 必须先 smoke online teacher-student trainer；frozen-teacher smoke 不再作为 CCLKD 通过证据。
4. FGD 修复前结果全部作废；LD 修复前 soft-logit / foreground-only 结果全部作废；
   旧 hallucidet 结果只能作为 `hallucidet_style_old` 参考，不能写作 HalluciDet。
5. 第二轮复核意见响应与未采纳原因见
   [`REVIEW_FEEDBACK_RESPONSE_CN.md`](REVIEW_FEEDBACK_RESPONSE_CN.md)。
