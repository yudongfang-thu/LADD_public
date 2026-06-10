# LD

Localization Distillation，CVPR 2022 / TPAMI 2023，经典检测 KD baseline。

## 方法

当前修正版直接蒸馏 YOLO11 检测头的 DFL regression logits，并包含
foreground/main LD 与 teacher-quality VLR-style candidate LD：

```text
student/teacher boxes: [B, N, 4 * reg_max]
foreground assigned anchors -> quality-weighted temperature KL
non-foreground teacher-quality candidates -> VLR-style temperature KL
```

默认 `ld_temperature=10.0`，与 LD 官方配置一致。Teacher eval forward 的 tuple
第二项保留原始 DFL logits；当前代码会在无法取得匹配 logits 时直接失败，而不是
静默返回零 loss。

YOLO11 TaskAlignedAssigner 不暴露官方 LD/ATSS 的 `get_vlr_region()` API；
当前 VLR-style 区域由 teacher confidence 与 teacher decoded box 到 GT 的 IoU
加权构造，因此是 YOLO 适配，不是官方 region selector 的逐行复现。

2026-06-04 以前的 `_ld_style_loss()` 实际蒸馏分类 logits，属于 soft-logit KD；
2026-06-10 以前的修正版也只有 foreground KL，没有 VLR-style candidate LD。
旧实验结果全部作废，必须使用当前修正版重跑。

## 结果

修正版已通过 117 真实 GPU smoke，之后重跑 YOLO11n/s。

旧 soft-logit 结果不进入当前主表；原始归档数据已从精简 public 分支移除。
