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
- `REVIEW_ROUND1_RESPONSE.md`
- `REVIEW_ROUND2_RESPONSE.md`
- `REVIEW_ROUND3.md`
- `REVIEW_ROUND4.md`
- `references/PKD_REFERENCE.md`
- `references/pkd_loss_mmrazor.py`

Core implementation:

- `../../ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py`
  - `_cmdistill_style_loss`
  - `_cmdistill_pcc_feature_loss`
  - `_cmdistill_relation_loss`
  - `_cmdistill_output_loss`
  - `_cmdistill_combine_components`
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
| SLRD | Deepest feature map only; per-image normalized token affinity matrix; L1 loss; spatial-token cap for memory control. |
| IBCLD | Decoded student/teacher box IoU loss plus BCE from student logits to teacher sigmoid probabilities, computed once on full concatenated detector outputs. |
| Total | Student detection loss plus `feature_weight * mean(PCCFD_shallow, PCCFD_deep) + relation_weight * SLRD_deep + logit_weight * IBCLD_full_outputs`. |

## Known Adaptation Boundaries

- Original CMDistill: IR teacher to RGB student, RGB-only inference, YOLOv5s,
  `640 x 640` input.
- Our OGSOD adaptation: RGB teacher to SAR student, SAR-only inference, YOLO11
  controlled comparison protocol, `256 x 256` input.
- No official CMDistill code was found. This is a paper-aligned
  reimplementation/adaptation, not a line-by-line official reproduction.
- `CMDISTILL_MAX_TOKENS` and `CMDISTILL_MIN_CONFIDENCE` are YOLO11/OGSOD
  adaptations and should be reported with any result.
- `CMDISTILL_TEMPERATURE` is reserved/accepted by CLI but is not used by strict
  IBCLD.
- Valid formal CMDistill-style runs require `KD_CALIBRATION_MODE=affine`. Runs
  with `KD_CALIBRATION_MODE!=affine` are not valid CMDistill comparison runs.

## Round 1 Review Response

The first external review returned `needs code fix first`. The following fixes
have been applied:

- SLRD no longer mixes tokens across images in a batch.
- IBCLD is no longer computed inside each FPN feature-profile call; it is
  computed once at detector-output level.
- CMDistill now warns if the adaptive layer is disabled by using a
  non-`affine` KD calibration mode.
- Smoke tests now cover SLRD no-batch-mixing, feature/relation level selection,
  IBCLD call separation, teacher detach, and confidence-candidate behavior.

## Round 2 Cleanup

The second external review verdict was `ready for smoke training`, with cleanup
recommended before formal long runs. The following changes have been applied:

- CMDistill no longer uses generic `profile_kd_loss / profile_levels`
  normalization. PCCFD is averaged over shallowest and deepest feature maps,
  SLRD is taken from the deepest feature map, and IBCLD is computed once on the
  full detector output.
- `ladd_diagnostics.csv` now receives lightweight CMDistill smoke stats from
  the loss object: PCC levels, SLRD token count, IBCLD candidate ratio, IBCLD
  foreground count, teacher-confidence added count, and IBCLD cls/box losses.
- Synthetic smoke checks verify that CMDistill component normalization is not
  equivalent to dividing by all FPN levels.

This status supports short GPU smoke training only. It does not justify a
formal 800-epoch run without first checking real data-flow diagnostics.

## Round 3 Cleanup

The third external review verdict was `ready for 1-epoch GPU smoke`. The
cleanup commit records the review in `REVIEW_ROUND3.md`, updates CMDistill run
tags to `v3_smoke_ready_20260615`, adds an epoch-1 stdout smoke summary, splits
nonfinite diagnostics into metric/CMDistill and BN components, and standardizes
the public naming as CMDistill-style.

## Round 4 Pre-Smoke Review

The fourth external review again returned `ready for 1-epoch GPU smoke`. It
found no P0/P1 blockers before smoke training. Remaining items are formal-run
hardening tasks for after smoke: hard-fail on missing CMDistill prerequisites,
add a deeper `_compute_decomposition_losses()` synthetic guard, and evaluate
`CMDISTILL_MIN_CONFIDENCE` from real smoke diagnostics before long runs.

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
