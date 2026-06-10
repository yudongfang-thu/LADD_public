# 对比方法复核意见响应

最后更新：2026-06-10

本文记录当前 public 包中非 CCLKD comparison 方法的复核结论和实现边界。
CCLKD 不在本轮小修范围内；相关代码和协议说明保持现状。

## 1. 当前结论

| 方法 | 当前处置 | 写作口径 |
|---|---|---|
| FGD-style | 已实现 fg/bg feature loss + teacher/student attention mask loss；GT-box mask 默认启用 | `FGD-style / FGD-YOLO adaptation` |
| LD | 已实现 raw YOLO11 DFL main foreground LD + VLR-style candidate LD | `LD adapted to YOLO11 DFL logits` |
| HalluciDet-style | profile 名称统一为 `hallucidet_style`；旧 `hallucidet` 名称禁用 | `HalluciDet-style, not strict HalluciDet` |

## 2. FGD

FGD 官方 attention 形式仍按官方代码保留：

```text
S_attention = H * W * softmax(spatial_map / temp)
C_attention = C     * softmax(channel_map / temp)
```

当前实现包含：

- teacher/student spatial attention 和 channel attention；
- GT-box area-normalized foreground mask，`fgd_mask_mode=gt_box` 为默认；
- foreground feature loss 和 background feature loss；
- attention mask loss，对齐 student 与 teacher attention；
- `fgd_temperature=0.5`。

当前没有实现官方 trainable global relation 模块。官方 Global KD 需要可训练的
spatial context pooling 和 channel-add 模块，应注册在 model 中并进入 optimizer；
本轮没有扩大到该模型结构改动。

旧 batch-level cosine relation matrix MSE 已降级为 legacy opt-in：

```text
fgd_lambda = 0.0
```

默认关闭，不能继续描述为官方 relation。FGD 内部权重按官方量级设置：

```text
fgd_alpha = 0.001
fgd_beta  = 0.0005
fgd_gamma = 0.001
fgd_lambda = 0.0
```

因此，修复前 FGD 结果全部作废，当前结果应写作 `FGD-style` 或
`FGD-YOLO adaptation`。

## 3. LD

Teacher 仍保持 eval。当前 Ultralytics Detect head 在非 export eval forward 中
返回 `(decoded_predictions, raw_predictions_dict)`，实现从第二项提取 raw DFL
logits；若 teacher/student DFL shape 不匹配会 fail-fast。

当前 LD-YOLO adaptation 包含：

- foreground/main LD：在 TaskAlignedAssigner foreground anchors 上对 raw DFL
  logits 做 KL；
- main 权重：assigned target quality 乘 teacher assigned-class confidence；
- VLR-style candidate LD：在非 foreground anchors 上，用 teacher confidence 与
  teacher decoded box 到 GT 的 IoU 构造候选权重；
- `ld_temperature=10.0`；
- `ld_main_weight=0.25`、`ld_vlr_weight=0.25`。

YOLO11 TaskAlignedAssigner 不暴露官方 LD/ATSS 的 `get_vlr_region()` API，因此
当前 VLR 是 YOLO 适配，不是官方 region selector 的逐行复现。修复前 soft-logit
KD 或 foreground-only LD 结果全部作废，需要重跑。

## 4. HalluciDet-style

当前 portable baseline 的 profile 名称统一为：

```text
hallucidet_style
```

旧 `hallucidet` 名称不再被 CLI 和 launcher 接受，避免结果表误写成 strict
HalluciDet。

当前实现只包含 privileged RGB teacher 的 feature/response/margin alignment：

- no explicit image-space hallucination module；
- no SAR/IR -> hallucinated RGB/image representation path；
- no frozen RGB detector detection loss through hallucinated image；
- inference remains SAR-only YOLO11 student。

旧 `hallucidet` 运行若保留，只能标为 `hallucidet_style_old`，不能作为 HalluciDet
official reproduction。若后续实现 strict HalluciDet，应新建单独入口，不复用当前
B-only frozen-teacher comparison launcher。

## 5. 正式运行要求

1. `fgd`、`ld`、`hallucidet_style` 首次正式运行前都要先做短 smoke。
2. LD smoke 必须记录 teacher/student raw DFL logits shape，期望为 `[B, N, 4*reg_max]`。
3. RUN_TAG/RUN_NAME 必须包含 comparison implementation version，避免覆盖旧结果。
4. 修复前 FGD、LD 和旧 `hallucidet` 结果不得进入当前主表。
