# Analysis Notes

## Key Findings

1. Wave 1 completed successfully for both controls. `det_only_same_trainer` ended at mAP50=0.33064 and mAP50-95=0.13490. `two_branch_no_kd` ended at mAP50=0.32165 and mAP50-95=0.13479.
2. The no-KD two-branch control is not collapsed relative to det-only at 80 epochs. Its mAP50-95 is effectively identical; mAP50 is modestly lower.
3. Wave 2 was still running at archive time. ATKD-only and full CCLKD diagnostics both show `feature_capture_ok=1.0` and `nan_or_inf_detected=0.0` in the latest recorded epoch.
4. Full CCLKD has active CCL (`ccl_loss` around 0.693 in the current snapshot) and higher weighted KD/detection ratio than ATKD-only, matching the intended mode difference.

## Caveats

- The archive is a mixed final/partial snapshot: wave 1 final, wave 2 partial.
- Single seed only (`seed=0`), so no variance estimate.
- The GPU server had other jobs active during parts of the queue; timing is not a clean throughput benchmark.
- The logs contain non-fatal Albumentations/Pydantic warnings about `size`; training continued and wrote metrics.

## Recommended Next Actions

1. After wave 2 finishes, refresh this archive or create a final `diagnostics_20260612_yolov5x_b32_e80_queue_final` package with final ATKD/full metrics.
2. Compare final `paper_full` against `paper_atkd_only` and the two controls using the same 80-epoch protocol before deciding whether CCLKD is underperforming due to CCL or due to online teacher coupling.
3. If full CCLKD remains weak, inspect `weighted_kd_to_student_det_ratio`, `cop_positive_ratio`, and class-wise COP candidate counts across epochs before changing loss weights.
