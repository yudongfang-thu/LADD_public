# HalluciDet-YOLO11 Adaptation Guide

**Date**: 2026-06-18

This directory is the only active HalluciDet entry. The old
`--comparison-kd-profile hallucidet_style` feature/response/margin baseline has been
removed from the LADD comparison profile system.

## Protocol

Current standalone protocol:

```text
SAR image
  -> replicate3 input adapter
  -> official-style U-Net hallucination network
  -> 3-channel hallucinated representation
  -> frozen RGB YOLO11 detector
  -> detection loss / detection result
```

Training updates only the hallucination network. The RGB detector is a pretrained RGB
detector and all of its parameters are frozen. The detector forward is still tracked by
autograd, because the detection loss must backpropagate through the detector operations
to the hallucinated representation and then to the hallucination network.

## What This Is

- Detection-loss-only HalluciDet-YOLO11 adaptation.
- Fixed active implementation: `segmentation_models_pytorch.Unet(resnet34, imagenet)`,
  `--hallucination-input-mode replicate3`.
- Uses YOLO box/cls/dfl loss components as `cls + lambda_reg * (box + dfl)`.
- Uses SAR labels and SAR images from the YOLO dataset yaml.
- Uses the frozen RGB YOLO detector as privileged-modality supervision.
- Validation runs `SAR -> hallucination -> frozen RGB YOLO -> YOLO metrics`.
- `results.csv` records `val/loss` and all mAP metrics returned by the validator.

## What This Is Not

- Not strict official HalluciDet reproduction: the paper uses detectors such as Faster
  R-CNN/FCOS/RetinaNet, while this entry adapts the protocol to YOLO11.
- No RGB paired reconstruction loss.
- No perceptual loss.
- No image-level RGB target matching objective.
- Not the removed `hallucidet_style` feature/response/margin KD baseline.
- Not the archived custom U-Net standalone implementation.

## Difference From Removed `hallucidet_style`

The removed `hallucidet_style` was a portable KD baseline inside the LADD comparison
loss path:

```text
SAR YOLO student + RGB teacher feature/response/margin alignment
```

This standalone entry is an image/representation hallucination protocol:

```text
SAR -> hallucination network -> frozen RGB YOLO detector
```

Do not launch or report the removed profile as a current method; old runs are historical
diagnostics only.

## Checkpoint Policy

- `last.pt` is updated every epoch.
- `best.pt` is selected by highest mAP50-95.
- Primary metric key is resolved in this order:
  - `metrics/mAP50-95(B)`
  - `metrics/mAP50-95`
  - `map50_95`
- If none of those metrics exist, the trainer falls back to `-val/loss` and prints a warning.
- `--save-period N` optionally writes `epoch_xxxx.pt`; it does not control `last.pt`.
- `--resume last.pt` resumes hallucination net, optimizer, scheduler, best metric, and starts from `epoch + 1`.

## Input Normalization

Train, validation, and model forward use the same policy:

- if image values look like `0..255`, divide by `255`;
- otherwise keep the tensor as already normalized;
- no per-batch min-max normalization.

The locked official-style U-Net uses a sigmoid output and directly produces `[0, 1]`
images for the frozen YOLO detector.

## Smoke Checks

Local syntax and lightweight gradient checks:

```bash
python3 -m py_compile comparison/hallucidet/*.py
python3 comparison/hallucidet/test_gradient_smoke.py --device cpu
python3 comparison/hallucidet/test_gradient_smoke.py --device cpu --resume-smoke
```

Real frozen-detector loss smoke on the GPU server:

```bash
python3 comparison/hallucidet/test_gradient_smoke.py \
  --teacher-weights <rgb_teacher_best.pt> \
  --device 0 \
  --imgsz 256
```

Resume smoke with real training should be run before formal experiments:

```bash
python3 comparison/hallucidet/train_hallucidet.py \
  --data shared/configs/datasets_public/ogsod1_sar_detect.yaml \
  --teacher-data shared/configs/datasets_public/ogsod1_rgb_detect.yaml \
  --teacher-weights <rgb_teacher_best.pt> \
  --imgsz 256 \
  --hallucination-input-mode replicate3 \
  --epochs 1 \
  --batch 8 \
  --device 0 \
  --project runs_public/hallucidet_smoke \
  --name resume_check

python3 comparison/hallucidet/train_hallucidet.py \
  --data shared/configs/datasets_public/ogsod1_sar_detect.yaml \
  --teacher-data shared/configs/datasets_public/ogsod1_rgb_detect.yaml \
  --teacher-weights <rgb_teacher_best.pt> \
  --imgsz 256 \
  --hallucination-input-mode replicate3 \
  --epochs 2 \
  --batch 8 \
  --device 0 \
  --project runs_public/hallucidet_smoke \
  --name resume_check \
  --resume runs_public/hallucidet_smoke/resume_check/last.pt
```

The expected behavior is that `results.csv` contains epochs `0,1`, `last.pt` stores
epoch `1`, and `best.pt` follows the best mAP50-95 epoch.
