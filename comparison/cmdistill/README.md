# CMDistill-style Controlled Comparison

This directory documents the CMDistill-style baseline in the controlled comparison
framework. CMDistill is currently a high-priority comparison method because the
CCLKD paper reports CMDistill as a strong OGSOD/YOLO11 benchmark competitor.

For external code review, start from [`REVIEW_PACKET.md`](REVIEW_PACKET.md).
For a copy-ready review prompt, see [`PRO_REVIEW_PROMPT.md`](PRO_REVIEW_PROMPT.md).
The first review response is tracked in
[`REVIEW_ROUND1_RESPONSE.md`](REVIEW_ROUND1_RESPONSE.md).
The second review cleanup is tracked in
[`REVIEW_ROUND2_RESPONSE.md`](REVIEW_ROUND2_RESPONSE.md).
The third review cleanup is tracked in
[`REVIEW_ROUND3.md`](REVIEW_ROUND3.md).
The fourth pre-smoke review is tracked in
[`REVIEW_ROUND4.md`](REVIEW_ROUND4.md).

The native VEDAI/other-dataset reproduction track is currently pending and has
been moved out of the active entry path. For historical context, see
[`archive_pending_20260618/native_reproduction/README_CN.md`](archive_pending_20260618/native_reproduction/README_CN.md).

## Paper Asset

Primary paper:

- [`paper/CMDistill__2025_JSTARS__Cross_Modal_Distillation_Framework_for_AAV_Image_Object_Detection.pdf`](paper/CMDistill__2025_JSTARS__Cross_Modal_Distillation_Framework_for_AAV_Image_Object_Detection.pdf)
- DOI: `10.1109/JSTARS.2024.3479717`
- Venue: IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2025, vol. 18, pp. 1395-1409.

Source priority for implementation:

1. CMDistill paper definitions and equations.
2. CCLKD paper tables only for OGSOD benchmark positioning.
3. PKD open-source implementation only for under-specified PCC tensor
   normalization details, not for defining CMDistill itself.

## Scope

`cmdistill` is a non-official CMDistill-style paper-aligned OGSOD/YOLO11
adaptation. No official CMDistill code was found. Results must not be reported
as official CMDistill reproduction.

The implemented profile follows the three components described in the CMDistill
paper and used as a strong comparison method in the CCLKD OGSOD benchmark:

- PCCFD: CMDistill Pearson-correlation feature distillation. The code uses a
  1x1 student adaptive layer through `KD_CALIBRATION_MODE=affine`, then applies
  channel-wise Pearson normalization and MSE/2 on the shallowest and deepest
  feature maps. The channel-wise reduction follows PKD only where CMDistill
  does not specify tensor-layout details.
- SLRD: semantic-level relation distillation. The code samples tokens from the
  deepest feature map per image and matches teacher/student affinity matrices
  with L1. Batch items are not mixed when building relation matrices.
- IBCLD: IoU-based binary classification logic distillation. The code aligns
  decoded teacher/student boxes with `1 - IoU` and aligns multi-class logic with
  binary cross entropy against teacher sigmoid probabilities. IBCLD is computed
  once on the full concatenated detector output, not inside the per-FPN feature
  loop.
- Total CMDistill KD is explicitly normalized as
  `feature_weight * mean(PCCFD_shallow, PCCFD_deep) +
  relation_weight * SLRD_deep + logit_weight * IBCLD_full_outputs`.

It uses a frozen RGB teacher and SAR student, matching the existing FGD/LD
controlled comparison protocol. It is not an online CCLKD trainer.

## Entry

From YOLO-pretrain protocol:

```bash
bash comparison/code/launch_formal_from_yolo_kd_job.sh cmdistill n 0 0
```

Transferred baseline protocol:

```bash
bash comparison/code/launch_formal_transfer_kd_job.sh cmdistill n 0 0
```

Both launchers pass `COMPARISON_KD_PROFILE=cmdistill` into
`ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh`.

## Tunable Weights

Environment variables:

- `CMDISTILL_FEATURE_WEIGHT`, default `1.0`
- `CMDISTILL_RELATION_WEIGHT`, default `1.0`
- `CMDISTILL_LOGIT_WEIGHT`, default `1.0`
- `KD_CALIBRATION_MODE`, default `affine` in the CMDistill launchers
- `CMDISTILL_MAX_TOKENS`, default `512`
- `CMDISTILL_MIN_CONFIDENCE`, default `0.05`

`CMDISTILL_TEMPERATURE` remains accepted for CLI compatibility with the earlier
diagnostic profile, but the strict IBCLD implementation does not use
temperature-scaled KL.

`CMDISTILL_MAX_TOKENS` caps SLRD relation tokens per image for memory control.
`CMDISTILL_MIN_CONFIDENCE` adds teacher-confident predictions to the IBCLD
candidate set in addition to assigner foreground tokens.
Valid formal CMDistill-style runs require `KD_CALIBRATION_MODE=affine`; runs
with a different calibration mode are not valid CMDistill comparison runs.

## Validation

Run:

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
```
