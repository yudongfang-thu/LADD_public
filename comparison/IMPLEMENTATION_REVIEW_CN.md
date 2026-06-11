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

## 5. 4090 服务器 smoke 记录

2026-06-10 在双卡 4090 服务器 `ladd4090` 上完成 synthetic smoke。为避免污染
服务器正式运行目录，使用本地 `git archive HEAD` 将 commit `8027757` 的干净快照
传到临时目录：

```text
ladd4090:/tmp/LADD_public_smoke_20260610
```

执行命令与结果：

```bash
python3 comparison/code/smoke_check_comparison_losses.py
# comparison loss smoke checks passed

python3 ladd/code_versions/current_hbb/tools/train_ladd_hbb.py --help | grep -E "fgd-alpha|fgd-mask-mode|ld-use-vlr|hallucidet_style"
# 输出包含 hallucidet_style、fgd-alpha、fgd-mask-mode、ld-use-vlr

bash -n comparison/code/launch_formal_from_yolo_kd_job.sh
bash -n comparison/code/launch_formal_transfer_kd_job.sh
bash -n ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh
# 均通过，exit 0
```

本次只验证 comparison loss、CLI 参数和 launcher 语法；未启动正式训练。

### 5.1 4090 runtime smoke 进度

2026-06-10 进一步在 `ladd4090:/root/shared-nvme/LADD_public` 上启动 20 epoch
runtime smoke。用户原始命令指定 GPU0/GPU1/GPU2；实际服务器只有 GPU0/GPU1，
且 GPU0 被已有任务占用、GPU2 不存在，因此本次将 `fgd`、`ld`、`hallucidet_style`
改为 GPU1 串行执行。命令统一使用：

```bash
COMPARISON_IMPL_VERSION=v2_20260610_runtime_smoke \
EPOCHS_B=20 PATIENCE_B=20 PROFILE_KD_WEIGHT=1.0 \
comparison/code/launch_formal_transfer_kd_job.sh <method> n 0 <gpu_id>
```

当前记录时间：2026-06-10 18:54 CST。队列日志显示
`serial runtime smoke complete`，三个方法均完成 20/20 epoch。`results.csv`
和对应 `b.log` / outer log 已检查，未发现 NaN、OOM、shape error、
RuntimeError 或 Traceback。

| 方法 | GPU | 状态 | train/kd_loss | train/box_loss | train/cls_loss | train/dfl_loss | mAP50-95(B) | 检查结论 |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `fgd` | 1 | 完成 20/20 epoch | 3.38704 | 1.37135 | 0.72154 | 0.89573 | 0.45006 | smoke passed；KD loss 大于单项 det loss，且与三项 det loss 之和同量级，权重可能偏强，正式重跑前需继续评估 |
| `ld` | 1 | 完成 20/20 epoch | 0.00749 | 1.12776 | 0.56678 | 0.84884 | 0.51471 | smoke passed；KD loss 有限且非零 |
| `hallucidet_style` | 1 | 完成 20/20 epoch | 0.34971 | 1.12845 | 0.56490 | 0.84887 | 0.51491 | smoke passed；KD loss 有限且非零 |

Run directories：

```text
fgd:
runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/transferred_kd/yolo11n/fgd/transfer_fgd_hbb_ogsod11n_formal_nomosaic_yolo11n_fgd_v2_20260610_runtime_smoke_transfer_s0_b_e20_b64_s0_gpu1

ld:
runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/transferred_kd/yolo11n/ld/transfer_ld_hbb_ogsod11n_formal_nomosaic_yolo11n_ld_v2_20260610_runtime_smoke_transfer_s0_b_e20_b64_s0_gpu1

hallucidet_style:
runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/transferred_kd/yolo11n/hallucidet_style/transfer_hallucidet_style_hbb_ogsod11n_formal_nomosaic_yolo11n_hallucidet_style_v2_20260610_runtime_smoke_transfer_s0_b_e20_b64_s0_gpu1
```

FGD 重点判断：最后一轮 `train/kd_loss=3.38704`，大于 `box/cls/dfl`
任一单项 loss，略高于三项 det loss 之和 `2.98862`，但不是高几个数量级。
因此当前结论是：FGD smoke passed but weight may be too strong。本轮只记录该判断，
不直接修改 FGD 权重或机制。

服务器 runtime 中还暴露了两个部署细节：

1. comparison launcher 经根目录 `scripts/ogsod_public/run_ladd_phase.sh` symlink
   调用 phase 脚本时，服务器布局下可能误判 repo root。当前 launcher 已改为显式调用
   `ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh`。
2. `run_ladd_phase.sh` 在训练结束后尝试调用
   `/root/shared-nvme/LADD_public/tools/summarize_tskd_results.py`，该 public
   runtime 中不存在该汇总脚本。该错误发生在训练完成后，`results.csv` 和 run
   directory 已正常生成；本轮只记录该环境问题，不修改 LADD phase 主线脚本。

### 5.2 formal transfer 恢复与平台期记录

2026-06-11 在 `ladd4090` 上恢复并继续 `fgd`、`ld`、`hallucidet_style`
的 YOLO11n/s seed0 formal transfer run。共享盘曾触发 `Disk quota exceeded`，
导致少量 epoch 出现 `results.csv` 已写入但 `last.pt` 未更新的孤儿记录；恢复时
按 checkpoint epoch 对齐 CSV，并将恢复日志转写到 `/tmp/ladd_resume_logs/`。

截至 2026-06-11 21:07 CST：

- `n_hallucidet_style` 与 `s_hallucidet_style` 已完成 800/800 epoch。
- 两条 HalluciDet-style 均未早停；`args.yaml` 为 `epochs=800`、`patience=800`。
- `n_hallucidet_style` 最后 50 epoch 基本平台，best 0.57365@785，final
  0.57239@800。
- `s_hallucidet_style` 在 0.64310@639 达到 best，之后轻微退化到
  final 0.63124@800；正式汇总应使用 `best.pt` 对应指标。
- `ld` 两条仍在运行但已接近平台：`n_ld` 0.57035@640，`s_ld` 0.64390@612。
- `fgd` 两条仍在运行且较慢；曲线远低于早期 best，需要在结果表中单独标注。

详见轻量记录：
[`FORMAL_TRANSFER_STATUS_20260611_CN.md`](FORMAL_TRANSFER_STATUS_20260611_CN.md)。
