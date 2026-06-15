# CMDistill Review Packet

Updated: 2026-06-15

This directory is the self-contained review packet for the CMDistill
implementation in `yudongfang-thu/LADD_public`.

## Purpose

CMDistill is now treated as a high-priority comparison method because the CCLKD
paper reports it as a strong OGSOD/YOLO benchmark competitor. This packet is for
reviewing whether our implementation is faithful enough to run as a controlled
comparison baseline.

## Source Priority

1. CMDistill paper is the source of truth for method definition.
2. CCLKD paper is used only to justify CMDistill's OGSOD/YOLO benchmark
   relevance.
3. PKD open-source code is used only as a reference for the PCC/Pearson
   feature-normalization tensor detail that CMDistill describes but does not
   provide as code.

## Files To Review

Paper and notes:

- `paper/CMDistill__2025_JSTARS__Cross_Modal_Distillation_Framework_for_AAV_Image_Object_Detection.pdf`
- `paper/README.md`
- `IMPLEMENTATION_AUDIT.md`
- `references/PKD_REFERENCE.md`
- `references/pkd_loss_mmrazor.py`

Core implementation:

- `../../ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py`
  - `_cmdistill_style_loss`
  - `_cmdistill_pcc_feature_loss`
  - `_cmdistill_relation_loss`
  - `_cmdistill_output_loss`
  - `_pkd_channel_standardize_map`
- `../../ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/loss.py`
  - synchronized copy used by formal launch scripts
- `../../comparison/code/smoke_check_comparison_losses.py`
  - synthetic checks for CMDistill gradients and PKD-normalization equivalence

Entrypoints:

- `../../comparison/code/launch_formal_from_yolo_kd_job.sh`
- `../../comparison/code/launch_formal_transfer_kd_job.sh`
- `../../ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh`

## Implemented Mapping

| CMDistill component | Current implementation |
|---|---|
| PCCFD | 1x1 student adaptive layer via `KD_CALIBRATION_MODE=affine`; shallowest and deepest feature maps; channel-wise Pearson normalization; MSE/2. |
| SLRD | Deepest feature map only; normalized token affinity matrix; L1 loss. |
| IBCLD | Decoded student/teacher box IoU loss plus BCE from student logits to teacher sigmoid probabilities. |
| Total | Student detection loss plus weighted CMDistill losses controlled by `CMDISTILL_FEATURE_WEIGHT`, `CMDISTILL_RELATION_WEIGHT`, and `CMDISTILL_LOGIT_WEIGHT`. |

## Known Adaptation Boundaries

- Original CMDistill: IR teacher to RGB student, RGB-only inference, YOLOv5s,
  `640 x 640` input.
- Our OGSOD adaptation: RGB teacher to SAR student, SAR-only inference, YOLO11
  controlled comparison protocol, `256 x 256` input.
- No official CMDistill code was found. This is a paper-aligned
  reimplementation/adaptation, not a line-by-line official reproduction.

## Local Validation

Commands already run locally:

```bash
python -m py_compile \
  ladd/code/train_ladd_hbb.py \
  ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py \
  ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py \
  ladd/code_versions/current_hbb/tools/train_ladd_hbb.py \
  ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/loss.py \
  ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/trainer.py \
  comparison/code/smoke_check_comparison_losses.py

python comparison/code/smoke_check_comparison_losses.py

bash -n \
  comparison/code/launch_formal_from_yolo_kd_job.sh \
  comparison/code/launch_formal_transfer_kd_job.sh \
  ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh

DRY_RUN=1 bash comparison/code/launch_formal_from_yolo_kd_job.sh cmdistill n 0 0
DRY_RUN=1 bash comparison/code/launch_formal_transfer_kd_job.sh cmdistill n 0 0

git diff --check
```

No GPU training run has been launched for this CMDistill implementation yet.
