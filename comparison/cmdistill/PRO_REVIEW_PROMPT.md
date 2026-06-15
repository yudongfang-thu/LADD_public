# Prompt For External Pro Review

Copy the prompt below into the external review model.

```text
You are reviewing the CMDistill implementation in this GitHub repository:

https://github.com/yudongfang-thu/LADD_public

Focus on the directory:

comparison/cmdistill/

Important context:

1. CMDistill is now a high-priority comparison method for the LADD_public OGSOD project because the CCLKD paper reports CMDistill as a strong OGSOD/YOLO benchmark competitor.
2. The CMDistill paper is the source of truth for the method definition:
   - comparison/cmdistill/paper/CMDistill__2025_JSTARS__Cross_Modal_Distillation_Framework_for_AAV_Image_Object_Detection.pdf
   - DOI: 10.1109/JSTARS.2024.3479717
3. PKD is NOT the method being implemented. PKD is only a reference for the tensor-level PCC/Pearson feature normalization detail, because CMDistill describes PCCFD conceptually but does not provide official code. See:
   - comparison/cmdistill/references/PKD_REFERENCE.md
   - comparison/cmdistill/references/pkd_loss_mmrazor.py
4. No official CMDistill code was found. This should be reviewed as a paper-aligned reimplementation/adaptation, not a line-by-line reproduction.

Please review whether the implementation is faithful enough to run as a controlled comparison baseline. In particular, audit:

1. PCCFD:
   - Is the 1x1 adaptive layer use appropriate?
   - Is shallowest + deepest feature selection consistent with the paper?
   - Is channel-wise Pearson normalization + MSE/2 a defensible implementation of CMDistill PCCFD, using PKD only for the under-specified tensor reduction detail?

2. SLRD:
   - Does deepest-feature affinity matrix construction match the CMDistill paper?
   - Is L1 relation loss correct?
   - Are there missing normalization, sampling, or scaling issues?

3. IBCLD:
   - Does decoded box IoU loss correctly implement the paper's IoU logic distillation?
   - Does BCE from student logits to teacher sigmoid probabilities correctly implement binary classification logic distillation?
   - Are candidate selection and teacher-confidence filtering defensible, or should all predictions/assigned boxes be used?

4. Training integration:
   - Check the actual code in:
     - ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py
     - ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/loss.py
     - comparison/code/smoke_check_comparison_losses.py
     - comparison/code/launch_formal_from_yolo_kd_job.sh
     - comparison/code/launch_formal_transfer_kd_job.sh
     - ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh
   - Verify that CMDistill does not accidentally depend on unrelated LADD losses.
   - Verify that formal launchers enable KD_CALIBRATION_MODE=affine and use the intended weights.

5. Adaptation risks:
   - Original CMDistill is IR teacher -> RGB student, YOLOv5s, 640 input.
   - This project adapts it to RGB teacher -> SAR student on OGSOD, SAR-only inference, YOLO11 controlled comparison protocol, 256 input.
   - Identify which differences are acceptable for a controlled comparison and which require explicit reporting or code changes.

Please produce:

1. A severity-ordered list of implementation issues, with GitHub file/line references where possible.
2. A short verdict: "ready for smoke training", "needs code fix first", or "method mismatch too large".
3. Concrete next steps before launching GPU experiments.
4. Any suggested changes to the review packet documentation so the method cannot be misreported as official CMDistill.
```
