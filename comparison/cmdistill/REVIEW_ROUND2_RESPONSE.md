# CMDistill Review Round 2 Response

Updated: 2026-06-15

## Input Verdict

The second external review verdict was:

> ready for smoke training

The review recommended a small cleanup before formal 800-epoch GPU runs:

1. make CMDistill component normalization explicit;
2. expose IBCLD candidate and component-loss diagnostics;
3. keep documentation clear that this is not official CMDistill code.

## Fixes Applied

### Explicit Component Normalization

CMDistill no longer uses the generic `profile_kd_loss / profile_levels`
normalization. The implementation now constructs:

```text
L_cmd =
    feature_weight  * mean(PCCFD_shallow, PCCFD_deep)
  + relation_weight * SLRD_deep
  + logit_weight    * IBCLD_full_outputs
```

FGD, LD, and CCLKD continue to use the existing generic comparison-profile path.

### Smoke Diagnostics

The loss object records a lightweight `_cmdistill_last_stats` cache with:

- `cmdistill_pcc_levels`
- `cmdistill_slrd_tokens`
- `cmdistill_ibcld_candidate_ratio`
- `cmdistill_ibcld_fg_count`
- `cmdistill_ibcld_teacher_conf_added_count`
- `cmdistill_ibcld_cls_loss`
- `cmdistill_ibcld_box_loss`

The HBB trainer appends these fields into `ladd_diagnostics.csv`, together with
additional summary values:

- `cmdistill_pcc_loss`
- `cmdistill_relation_loss`
- `cmdistill_ibcld_loss`
- `cmdistill_total_loss`

### Synthetic Smoke

`comparison/code/smoke_check_comparison_losses.py` now verifies:

- explicit CMDistill normalization uses PCCFD mean over two selected levels;
- SLRD is not diluted by all FPN levels;
- the result is not equivalent to `(PCCFD + SLRD) / num_fpn_levels`;
- IBCLD remains separate from `_cmdistill_style_loss`;
- IBCLD stats are populated and finite;
- SLRD remains per-image and does not mix batch items.

## Remaining Requirement Before Formal Runs

This implementation is ready for short GPU smoke training, not formal long
training. Before an 800-epoch run, a real data-flow smoke should confirm:

- no shape error;
- no NaN/Inf;
- PCCFD, SLRD, and IBCLD are finite;
- `cmdistill_ibcld_candidate_ratio` is neither zero nor effectively one;
- teacher tensors remain detached;
- validation and metrics/diagnostics files are written normally.

## Reporting Constraint

This remains a paper-aligned CMDistill-style OGSOD/YOLO11 adaptation:

- original CMDistill: IR teacher to RGB student, YOLOv5s, 640 input;
- this project: RGB teacher to SAR student, SAR-only inference, YOLO11, 256
  input, no mosaic;
- no official CMDistill code was used.

Formal reports should not call the result an official CMDistill reproduction.
