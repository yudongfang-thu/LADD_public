# LD

Localization Distillation，CVPR 2022 / TPAMI 2023，经典检测 KD baseline。

## 方法

当前修正版直接蒸馏 YOLO11 检测头的 DFL regression logits：

```text
student/teacher boxes: [B, N, 4 * reg_max]
foreground assigned anchors -> reshape [-1, 4, reg_max] -> temperature KL
```

默认 `ld_temperature=10.0`，与 LD 官方配置一致。Teacher eval forward 的 tuple
第二项保留原始 DFL logits；当前代码会在无法取得匹配 logits 时直接失败，而不是
静默返回零 loss。

2026-06-04 以前的 `_ld_style_loss()` 实际蒸馏分类 logits，属于 soft-logit KD，
不是 LD。旧实验结果全部作废，必须使用修正版重跑。

## 结果

修正版已通过 117 真实 GPU smoke，之后重跑 YOLO11n/s。

旧 soft-logit 结果不进入当前主表；原始归档数据已从精简 public 分支移除。
