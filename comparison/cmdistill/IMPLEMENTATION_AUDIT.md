# CMDistill Implementation Audit

Updated: 2026-06-15

## Source Priority

The implementation must follow the CMDistill paper first:

1. `CMDistill: Cross-Modal Distillation Framework for AAV Image Object Detection`, IEEE JSTARS 2025, DOI `10.1109/JSTARS.2024.3479717`.
2. CCLKD paper benchmark tables, only for OGSOD positioning and expected competitor strength.
3. PKD open-source code, only for the tensor-level PCC normalization detail that CMDistill describes conceptually but does not specify as code.

## Paper-To-Code Mapping

| CMDistill component | Paper definition | Current code |
|---|---|---|
| PCCFD | Pearson-correlation feature distillation on selected FPN layers; paper selects shallowest and deepest feature layers and uses an adaptive layer before feature loss. | `_cmdistill_pcc_feature_loss()` in `ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py`; launchers set `KD_CALIBRATION_MODE=affine` to enable the 1x1 student adaptive layer. |
| SLRD | Affinity matrices from high-level semantic features, supervised with L1; only deepest semantic feature graph is used for efficiency. | `_cmdistill_relation_loss()` uses per-image normalized token affinity and L1 on the deepest feature level. The token cap samples the spatial-token axis within each image and does not mix batch items. |
| IBCLD | Logic distillation combines teacher-student predicted-box IoU loss and binary classification logic loss. | `_cmdistill_output_loss()` uses `1 - IoU(decoded student box, decoded teacher box)` plus BCE from student logits to teacher sigmoid probabilities. It is called once on the full concatenated detector output. |
| Total loss | `L_total = L_det + lambda1 L_fea + lambda2 L_rela + lambda3 L_log`. | Controlled comparison profile explicitly uses `mean(PCCFD_shallow, PCCFD_deep)`, `SLRD_deep`, and full-output `IBCLD`; weights are exposed as `CMDISTILL_FEATURE_WEIGHT`, `CMDISTILL_RELATION_WEIGHT`, and `CMDISTILL_LOGIT_WEIGHT`. |

## Known Adaptation Boundaries

- Original CMDistill uses IR teacher to RGB student for AAV inference with RGB-only input. Our OGSOD adaptation uses RGB teacher to SAR student for SAR-only inference.
- Original CMDistill experiments use YOLOv5s at `640 x 640`; our controlled LADD/OGSOD formal protocol uses YOLO11 variants at `256 x 256`, no mosaic, and SAR/RGB paired data.
- No official CMDistill code was found. This is a paper-aligned reimplementation, not a line-by-line reproduction.
- PKD is not a CMDistill replacement. It is only a reference for implementing Pearson feature normalization when mapping Eq. (1) to tensors.
- `CMDISTILL_MAX_TOKENS` is an OGSOD/YOLO11 memory-control adaptation for SLRD.
- `CMDISTILL_MIN_CONFIDENCE` is an OGSOD/YOLO11 candidate-filtering adaptation for IBCLD.
- `CMDISTILL_TEMPERATURE` is accepted for CLI compatibility but is currently unused by strict IBCLD.
- Valid formal CMDistill-style runs require `KD_CALIBRATION_MODE=affine`. Runs
  with `KD_CALIBRATION_MODE!=affine` are not valid CMDistill comparison runs.

## Review Round 1 Fixes

- Fixed SLRD batch mixing: relation matrices are now computed per image with
  `torch.bmm()` instead of over flattened `B * H * W` tokens.
- Moved IBCLD out of the per-FPN feature loop: logic distillation is computed
  once on full detector outputs.
- Added a warning when `COMPARISON_KD_PROFILE=cmdistill` is used without
  `KD_CALIBRATION_MODE=affine`.
- Extended smoke checks for SLRD no-batch-mixing, PCCFD/SLRD level selection,
  IBCLD call separation, teacher detach, and candidate-count behavior.

## Review Round 2 Cleanup

- Replaced the generic profile-level averaging for CMDistill with explicit
  component normalization:
  `feature_weight * mean(PCCFD_shallow, PCCFD_deep) +
  relation_weight * SLRD_deep + logit_weight * IBCLD_full_outputs`.
- Added `_cmdistill_last_stats` and diagnostics fields for smoke inspection:
  `cmdistill_pcc_levels`, `cmdistill_slrd_tokens`,
  `cmdistill_ibcld_candidate_ratio`, `cmdistill_ibcld_fg_count`,
  `cmdistill_ibcld_teacher_conf_added_count`, `cmdistill_ibcld_cls_loss`, and
  `cmdistill_ibcld_box_loss`.
- Extended synthetic smoke checks to verify the explicit normalization is not
  equivalent to dividing CMDistill feature/relation losses by all FPN levels.

## Review Round 3 Cleanup

- Default CMDistill run tags now use `v3_smoke_ready_20260615`.
- `ladd_diagnostics.csv` keeps the legacy `nan_or_inf_detected` flag and also
  splits it into `nonfinite_metrics_or_cmdistill` and `nonfinite_bn_stats`.
- Epoch-1 CMDistill runs print one `cmdistill_smoke_stats` summary line to the
  outer log for quick smoke inspection.
- Naming is standardized as CMDistill-style controlled comparison.

## Validation

Current local validation:

```bash
python -m py_compile \
  ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py \
  ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/loss.py \
  comparison/code/smoke_check_comparison_losses.py

python comparison/code/smoke_check_comparison_losses.py

DRY_RUN=1 bash comparison/code/launch_formal_from_yolo_kd_job.sh cmdistill n 0 0
```
