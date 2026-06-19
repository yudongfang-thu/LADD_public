# Paper Method Name Whitelist

最后更新：2026-06-18

本文限定论文表格和图注中允许使用的方法名，避免把 adaptation、reimplementation 和 official reproduction 混写。

## 允许的方法名

| method key | paper display name | 说明 |
|---|---|---|
| `sar_baseline` | SAR baseline | SAR-only YOLO detector |
| `rgb_teacher` | RGB teacher | RGB-only YOLO detector，作为 teacher/checkpoint source |
| `vanilla_feature_kd` | Vanilla feature KD | 如果保留，必须说明为受控适配 |
| `fgd` | FGD-style / FGD-YOLO adaptation | 受控 YOLO feature KD adaptation，不写作 official FGD reproduction |
| `ld` | LD | 受控 localization distillation adaptation |
| `cmdistill` | CMDistill-style / paper-aligned adaptation | 受控实现，必须 `KD_CALIBRATION_MODE=affine` |
| `hallucidet_yolo` | HalluciDet-YOLO adaptation | standalone adaptation，不是 `hallucidet_style` KD profile |
| `cclkd_online` | CCLKD online comparison | optional；必须走 online trainer，不是 frozen-teacher profile |
| `ladd` | LADD, ours | `clean_a1b_dynprobe`, A -> B, no A2 |
| `ladd_static_ablation` | LADD Static ablation | `clean_a1b`，只用于消融 |
| `ladd_dynamic_ablation` | LADD Dynamic ablation | `clean_a1b_dyn`，只用于消融 |

## 禁止混用的名称

以下写法不能用于 paper main table：

```text
official CMDistill reproduction
official FGD reproduction
official HalluciDet reproduction
frozen-teacher CCLKD official result
HalluciDet-style KD profile
old LADD A1-A2-B mainline
BN-freeze mainline
formal no-mosaic as current mainline
```

如果需要讨论这些历史记录，必须标注为 diagnostic、archive、robustness appendix 或 reproduction-side evidence。
