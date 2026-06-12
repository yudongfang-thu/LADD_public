# HalluciDet 实现总结

日期：2026-06-13

## 当前状态

当前目录实现的是唯一保留的 detection-loss-only HalluciDet-YOLO11 adaptation 入口。
原先 `hallucidet_style` feature/response/margin comparison baseline 已从
`--comparison-kd-profile`、loss dispatch 和 launcher 中移除，避免干扰。当前入口更接近论文协议：

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
   - `lambda_reg` 作用于非 detach 的 `cls + lambda_reg * (box + dfl)`。
   - validation 复用 YOLO DetectionValidator 的 mAP 统计逻辑，输出 `metrics/mAP50(B)` 和 `metrics/mAP50-95(B)`。
   - 写出轻量 `results.csv`。
   - `best.pt` 按 mAP50-95 最大保存；`last.pt` 每个 epoch 更新。
   - 支持 `--resume last.pt`，恢复 hallucination net、optimizer、scheduler、best metric，并从 `epoch + 1` 继续。

3. `test_gradient_smoke.py`
   - 默认离线 toy-detector smoke 检查基础 autograd 链路。
   - 传入 `--teacher-weights` 后可做真实 YOLO `v8DetectionLoss` 单 batch smoke。
   - `--resume-smoke` 可验证 checkpoint resume、results.csv 追加、best/last checkpoint 语义。
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
- 当前没有 RGB paired reconstruction loss，也没有 perceptual loss。

## 与已移除 `hallucidet_style` 的区别

- 已移除的 `hallucidet_style`：训练 SAR YOLO student 的 feature/response/margin alignment baseline，不是 strict HalluciDet。
- 当前实现：训练 SAR->RGB-like hallucination network，冻结 RGB detector，loss 来自 frozen RGB detector 的 detection loss，更接近 HalluciDet 原始协议。
