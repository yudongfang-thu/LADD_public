# CMDistill Review Round 1 Response

Updated: 2026-06-15

## Input Verdict

The first external review verdict was:

> needs code fix first

The review identified two blocking implementation issues before GPU training:

1. SLRD relation matrices mixed tokens from different images in the same batch.
2. IBCLD was attached to the per-FPN feature-profile loop instead of being
   computed once at detector-output level.

## Fixes Applied

### SLRD

`_cmdistill_relation_loss()` now keeps the batch dimension:

- feature maps are flattened as `[B, K, C]`;
- token sampling is done on the spatial-token axis `K`;
- relation matrices are computed per image with `torch.bmm()`;
- no cross-image token pairs are used.

### IBCLD

`_cmdistill_style_loss()` now contains only feature-side CMDistill losses:

- PCCFD on shallowest and deepest feature levels;
- SLRD on the deepest feature level.

`_cmdistill_output_loss()` is called once from `_compute_decomposition_losses()`
on the full concatenated detector outputs:

- student/teacher decoded boxes;
- student logits and teacher logits;
- assigner foreground plus teacher-confidence candidates.

### Calibration Guard

The loss initialization now warns when `COMPARISON_KD_PROFILE=cmdistill` is used
without `KD_CALIBRATION_MODE=affine`, because CMDistill expects an adaptive
student feature layer.

### Smoke Coverage

`comparison/code/smoke_check_comparison_losses.py` now checks:

- SLRD batch result equals the mean of independent per-image relation losses;
- PCCFD is active only on first and last feature levels;
- SLRD is active only on the last feature level;
- IBCLD does not backpropagate through per-level feature style loss;
- IBCLD does backpropagate through full-output logits and decoded boxes;
- teacher tensors remain detached;
- teacher-confidence filtering increases candidate count when expected.

## Remaining Adaptation Notes

This is still a CMDistill-style paper-aligned adaptation, not an official
reproduction:

- original CMDistill: IR teacher to RGB student, YOLOv5s, 640 input;
- this project: RGB teacher to SAR student, YOLO11, 256 input, OGSOD protocol;
- `CMDISTILL_MAX_TOKENS` caps relation tokens per image for memory control;
- `CMDISTILL_MIN_CONFIDENCE` filters output-distillation candidates;
- `CMDISTILL_TEMPERATURE` is accepted by CLI but currently unused.

## Current Status

After these fixes, the implementation status is:

> ready for local smoke validation and short GPU smoke training

Formal long GPU training should still wait until the local validation commands in
`REVIEW_PACKET.md` pass after this commit.
