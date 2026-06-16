# HalluciDet-YOLO11 Adaptation

最后更新：2026-06-16

本目录是当前唯一有效的 HalluciDet 相关入口。旧
`--comparison-kd-profile hallucidet_style` feature/response/margin KD baseline
已经从 LADD comparison profile 系统移除。

## 方法

当前 standalone 协议为：

```text
SAR image
  -> hallucination network
  -> 3-channel hallucinated representation
  -> frozen RGB YOLO11 detector
  -> detection loss / validation metrics
```

训练时只更新 hallucination network；RGB YOLO11 detector 作为 frozen privileged
detector。验证也走同一条 `SAR -> hallucination -> frozen RGB YOLO` 路径。

有效入口：

```bash
python comparison/hallucidet/train_hallucidet.py \
  --data shared/configs/datasets_public/ogsod1_sar_detect.yaml \
  --teacher-data shared/configs/datasets_public/ogsod1_rgb_detect.yaml \
  --teacher-weights <rgb_teacher_best.pt> \
  --imgsz 256
```

## 边界

- 这是 detection-loss-only HalluciDet-YOLO11 adaptation。
- 不是 strict official HalluciDet reproduction；原文使用 Faster R-CNN/FCOS/RetinaNet
  等检测器，本目录适配到 YOLO11。
- 没有 RGB paired reconstruction loss、perceptual loss 或 image-level RGB target
  matching objective。
- 不是已移除的 `hallucidet_style` feature/response/margin KD baseline。

## 结果

旧 `hallucidet`/`hallucidet_style` 结果只能作为历史 diagnostic，不能作为当前
HalluciDet-YOLO adaptation 或 official HalluciDet reproduction。详细实现与 smoke
规则见 [`IMPLEMENTATION_GUIDE.md`](IMPLEMENTATION_GUIDE.md)。
