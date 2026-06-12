# HalluciDet 实现总结

日期：2026-06-12

## 当前状态

当前目录实现的是一个独立的 HalluciDet-YOLO11 adaptation 入口，已经和原先 `hallucidet_style` comparison baseline 分开。它更接近论文协议：

```text
SAR image -> hallucination network -> hallucinated 3-channel representation -> frozen RGB YOLO11 detector
```

训练时只更新 hallucination network，RGB detector 参数冻结；推理时使用 hallucination network + frozen RGB detector 这一整条链。

## 已完成

1. `hallucidet_model.py`
   - U-Net 风格 hallucination network。
   - detector forward 不再使用 `torch.no_grad()`，因此 detection loss 可以反传到 hallucination network。
   - RGB detector 参数 `requires_grad=False`，只作为冻结检测器使用。

2. `train_hallucidet.py`
   - 接入 YOLODataset / build_dataloader。
   - 接入 frozen detector raw prediction + YOLO detection loss。
   - `lambda_reg` 现在作用于非 detach 的 `cls + lambda_reg * (box + dfl)`。
   - validation 复用 YOLO DetectionValidator 的 mAP 统计逻辑，输出 `metrics/mAP50(B)` 和 `metrics/mAP50-95(B)`。
   - 写出轻量 `results.csv`。

3. `test_gradient_smoke.py`
   - 默认离线 toy-detector smoke 检查基础 autograd 链路。
   - 传入 `--teacher-weights` 后可做真实 YOLO `v8DetectionLoss` 单 batch smoke。
   - 不再隐式联网下载 `yolo11n.pt`。

## 仍需验证

- 需要在 4090 服务器用真实 RGB teacher 权重运行：

```bash
python3 comparison/hallucidet/test_gradient_smoke.py \
  --teacher-weights <rgb_teacher_best.pt> \
  --device 0 \
  --imgsz 256
```

- 真实 YOLO loss smoke 通过后，再启动 20 epoch runtime smoke。
- 当前是 YOLO11 adaptation，不是逐行复刻原论文 Faster R-CNN/FCOS/RetinaNet 实现。

## 与旧 `hallucidet_style` 的区别

- `hallucidet_style`：训练 SAR YOLO student 的 feature/response/margin alignment baseline，不是 strict HalluciDet。
- 当前实现：训练 SAR->RGB-like hallucination network，冻结 RGB detector，loss 来自 frozen RGB detector 的 detection loss，更接近 HalluciDet 原始协议。
