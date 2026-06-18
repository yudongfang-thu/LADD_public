# HalluciDet Legacy Archive

最后更新：2026-06-18

当前 active HalluciDet 实现固定为：

```text
HalluciDet-YOLO official-style U-Net adaptation
SAR -> replicate3 -> segmentation_models_pytorch U-Net(resnet34, ImageNet) -> frozen RGB YOLO11
```

本目录只保留早期 custom U-Net standalone 分析记录。它们不能作为当前 HalluciDet-YOLO
official-style U-Net 结果，也不能进入论文主表。

有效最终结果保留在：

- `comparison/hallucidet/results_autodl_sync_20260616/official_b64_800ep/`
- `comparison/hallucidet/analysis/official_b64_compare_20260615/`
