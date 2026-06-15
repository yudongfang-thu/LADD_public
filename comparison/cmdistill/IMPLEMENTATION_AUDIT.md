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
| SLRD | Affinity matrices from high-level semantic features, supervised with L1; only deepest semantic feature graph is used for efficiency. | `_cmdistill_relation_loss()` uses normalized token affinity and L1 on the deepest feature level. |
| IBCLD | Logic distillation combines teacher-student predicted-box IoU loss and binary classification logic loss. | `_cmdistill_output_loss()` uses `1 - IoU(decoded student box, decoded teacher box)` plus BCE from student logits to teacher sigmoid probabilities. |
| Total loss | `L_total = L_det + lambda1 L_fea + lambda2 L_rela + lambda3 L_log`. | Controlled comparison profile adds weighted CMDistill profile loss to student detection loss; weights are exposed as `CMDISTILL_FEATURE_WEIGHT`, `CMDISTILL_RELATION_WEIGHT`, and `CMDISTILL_LOGIT_WEIGHT`. |

## Known Adaptation Boundaries

- Original CMDistill uses IR teacher to RGB student for AAV inference with RGB-only input. Our OGSOD adaptation uses RGB teacher to SAR student for SAR-only inference.
- Original CMDistill experiments use YOLOv5s at `640 x 640`; our controlled LADD/OGSOD formal protocol uses YOLO11 variants at `256 x 256`, no mosaic, and SAR/RGB paired data.
- No official CMDistill code was found. This is a paper-aligned reimplementation, not a line-by-line reproduction.
- PKD is not a CMDistill replacement. It is only a reference for implementing Pearson feature normalization when mapping Eq. (1) to tensors.

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
