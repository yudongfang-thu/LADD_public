# 非 CCLKD 对比方法实现审查

最后更新：2026-06-10

本文只审查 `FGD-style`、`LD` 和 `HalluciDet-style` 三个 frozen-teacher
受控对比 profile。CCLKD 不在本文范围内。

## 1. 总结

当前代码没有发现会让三个 profile 静默失效的明显实现错误；但只有 LD
较接近“核心方法的 YOLO/DFL 适配”。FGD 和 HalluciDet 都必须继续标注
`-style`，不能写成官方逐行复现。

| 方法 | 当前代码判断 | 写作口径 |
|---|---|---|
| FGD-style | teacher spatial/channel attention、GT fg/bg weighting、batch relation 近似均可运行 | `FGD-style (teacher-attention weighted)` |
| LD | 对 YOLO11 raw DFL regression logits 做 foreground KL，温度和 fail-fast 逻辑正确 | `LD adapted to YOLO11 DFL logits` |
| HalluciDet-style | 使用 RGB privileged teacher 做 feature/response/margin alignment，但没有显式 hallucination module | `HalluciDet-style, no explicit hallucination module` |

## 2. FGD-style

代码位置：

```text
ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py::_fgd_style_loss
```

本地实现与官方 FGD 对齐的部分：

- teacher feature 的 spatial attention 使用 `softmax(att / T) * H * W`；
- channel attention 使用 `softmax(att / T) * C`；
- 默认 `fgd_temperature=0.5`；
- 使用 foreground/background 区域区分，而不是全图无差别 MSE。

需要保留的边界：

- 官方 FGD 有 student/teacher attention mask loss，本地没有单独的 mask loss；
- 官方 foreground mask 按 GT box 区域面积归一化，本地使用 trainer 已分配的
  foreground token mask 和常量 background 权重；
- 官方 Global KD 是带 `conv_mask_*` 和 `channel_add_conv_*` 的 GCNet-style
  context 模块，本地 relation 项是 batch-level cosine relation matrix MSE；
- 因此本地实现是可移植 YOLO port，不是官方 MMDetection FGD 复现。

官方证据：

- 论文：<https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.pdf>
- 代码：<https://raw.githubusercontent.com/yzd-v/FGD/master/mmdet/distillation/losses/fgd.py>

## 3. LD

代码位置：

```text
ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py::_ld_style_loss
```

本地实现正确点：

- 使用 student/teacher 检测头 raw DFL regression logits；
- 只在 assigned foreground anchors 上蒸馏；
- 每个 token reshape 为 `[-1, 4, reg_max]` 后按四条边做 KL；
- 默认 `ld_temperature=10.0`，与官方配置一致；
- teacher eval forward 取 raw prediction dict，若拿不到匹配 DFL logits 直接
  `RuntimeError`，避免旧实现那类静默退化。

需要保留的边界：

- 官方 LD 基于 GFL/ATSS head，包含 `loss_ld`、`loss_ld_vlr`、classification
  KD 和可选 imitation loss；
- 本地当前只实现 localization distribution KD 主体，没有完整 VLR / KD / IM
  recipe；
- 因此可作为 LD 的 YOLO11 DFL 适配，但不应声称完全复现官方 MMDetection LD。

官方证据：

- 论文：<https://openaccess.thecvf.com/content/CVPR2022/papers/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.pdf>
- 配置：<https://raw.githubusercontent.com/HikariTJU/LD/master/configs/ld/ld_r18_gflv1_r101_fpn_coco_1x.py>
- Head 实现：<https://raw.githubusercontent.com/HikariTJU/LD/master/mmdet/models/dense_heads/ld_head.py>

## 4. HalluciDet-style

代码位置：

```text
ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py::_hallucidet_style_loss
```

本地实现保留的思想：

- 训练期使用 paired RGB/SAR；
- RGB teacher 作为 privileged information；
- SAR-only inference；
- 前景和 teacher response 更高的位置获得更强 alignment 权重。

与官方 HalluciDet 的关键差异：

- 官方训练 IR-to-RGB encoder-decoder / hallucination pathway；
- 官方将 hallucinated image 输入冻结 detector，用 detection loss 直接驱动
  hallucination；
- 官方还可包含 RGB/IR pixel reconstruction 和 perceptual reconstruction；
- 本地没有生成 RGB-like 图像，没有独立 hallucination module，也没有 detector
  loss 直接作用于 hallucinated image。

因此本地实现只能写作：

```text
HalluciDet-style (detection-utility guided feature alignment,
no explicit hallucination module)
```

官方证据：

- 论文：<https://openaccess.thecvf.com/content/WACV2024/html/Medeiros_HalluciDet_Hallucinating_RGB_Modality_for_Person_Detection_Through_Privileged_Information_WACV_2024_paper.html>
- 代码仓库：<https://github.com/heitorrapela/HalluciDet>
- 训练脚本：<https://raw.githubusercontent.com/heitorrapela/HalluciDet/main/train_hallucidet.py>

## 5. 启动与数据配置风险

当前两个 non-CCLKD launcher 都限制 method 为 `fgd|ld|hallucidet`，不会误启动
CCLKD：

```text
comparison/code/launch_formal_from_yolo_kd_job.sh
comparison/code/launch_formal_transfer_kd_job.sh
```

需要在完整实验工作区核验的数据配置点：

- `from_yolo` launcher 显式设置
  `configs/datasets/ogsod_hbb_sar.yaml` 和
  `configs/datasets/ogsod_hbb_rgb.yaml`；
- `transfer` launcher 默认继承
  `data/ogsod_public_prepared/yamls/ogsod_hbb_sar.yaml` 和
  `data/ogsod_public_prepared/yamls/ogsod_hbb_rgb.yaml`；
- public 包中只保留
  `shared/configs/datasets_public/ogsod1_sar_detect.yaml` 和
  `shared/configs/datasets_public/ogsod1_rgb_detect.yaml`。

补实验前应确认这些 YAML 的 `nc`、类别名、路径、split 完全一致，并把实际 YAML
路径写入 run manifest。此前双卡 4090 的 `nc=5` YAML 错误说明这个检查不能省略。

## 6. 补实验前检查清单

1. 对 `fgd`、`ld`、`hallucidet` 各跑 1-2 epoch smoke。
2. LD smoke 必须记录 student/teacher DFL logits shape，期望为 `[B, N, 4 * reg_max]`。
3. 三个方法都应记录非零 `train/kd_loss` 或等价 profile loss。
4. 每个 run 的 manifest 记录 student init、RGB teacher checkpoint、SAR/RGB YAML、
   `PROFILE_KD_REPLACE_BASE`、profile hyperparameters。
5. 修复前 FGD/LD/HalluciDet 旧结果不得与当前实现结果混合统计。
