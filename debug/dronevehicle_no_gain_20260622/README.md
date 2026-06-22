# Debug Case: DroneVehicle No-Gain Behavior

Created: 2026-06-22 CST

## Problem Statement

On DroneVehicle under the CCLKD-aligned cross-dataset protocol, both LADD main/probe-style training and Dynamic LADD fail to outperform the RGB student baseline.

This is different from OGSOD, where LADD can improve over SAR/RGB-modality student baselines under the mosaic100 protocol. DroneVehicle therefore should not be used as positive cross-dataset evidence unless the method/protocol is redesigned.

## Protocol Context

| Item | Value |
|---|---|
| Dataset | DroneVehicle train/val, YOLO-HBB conversion |
| Student modality | RGB |
| Teacher modality | IR |
| Model | YOLO11n |
| Image size | 512 |
| Epochs | 200 |
| Batch | 16 |
| Mosaic | 0.0 |
| Close mosaic | 0 |
| Mixup | 0.1 |
| Optimizer family | CCLKD-aligned SGD-style cross-dataset protocol |

## Raw Evidence

| File | Meaning |
|---|---|
| `raw/student_rgb_baseline_results.csv` | Full 200 epoch RGB student baseline. |
| `raw/teacher_ir_baseline_results.csv` | Full 200 epoch IR teacher baseline. |
| `raw/ladd_main_results.csv` | LADD main/probe-style IR -> RGB result. |
| `raw/ladd_dynamic_results.csv` | Dynamic LADD IR -> RGB partial result. |
| `raw/ladd_wo_reach_results.csv` | No-reach LADD IR -> RGB partial result. |
| `raw/*_args.yaml` | Training args for each copied run. |
| `raw/ladd_*_diagnostics.csv` | LADD diagnostics for internal losses/distances. |

## Key Numbers

See `analysis/run_summary_all.csv` and `analysis/gain_gap_summary.csv` for the full tables.

| Run | Rows | Best AP50-95 | Last AP50-95 | Gain vs Student Best | Gap Closure Best |
|---|---:|---:|---:|---:|---:|
| RGB student baseline | 200 | 0.51053 | 0.51007 | 0.00000 | 0.000 |
| IR teacher baseline | 200 | 0.56964 | 0.56889 | 0.05911 | 1.000 |
| LADD main IR -> RGB | 200 | 0.50992 | 0.50374 | -0.00061 | -0.010 |
| Dynamic LADD IR -> RGB | 90 | 0.50545 | 0.50545 | -0.00508 | -0.086 |
| LADD no-reach IR -> RGB | 16 | 0.48553 | 0.44950 | -0.02500 | -0.423 |

## Working Hypotheses

1. The IR teacher is stronger than RGB student, but the learned transferable representation does not help the RGB detector under this protocol.
2. The modality direction is IR -> RGB, which may differ qualitatively from OGSOD's RGB teacher -> SAR student setting; IR-private information may not be learnable by RGB in the same way.
3. The DroneVehicle protocol uses `imgsz=512`, `epochs=200`, `batch=16`, `mosaic=0.0`, `mixup=0.1`, which is very different from OGSOD mosaic100 B=800. The method may need different balancing or longer training to show effect.
4. Internal losses are not exploding; the issue looks more like ineffective transfer / negative transfer than a numerical failure.
5. Because both main/probe-style LADD and Dynamic LADD underperform the student baseline, DroneVehicle is currently a negative or diagnostic result, not positive cross-dataset evidence.

## Files for External Analysis

- Summary: `analysis/run_summary_all.csv`
- Gain/gap table: `analysis/gain_gap_summary.csv`
- Internal loss comparison: `analysis/ladd_internal_best_final_compare.csv`, `analysis/loss_progress_summary.csv`
- Figures: `figures/dronevehicle_ladd_baseline_curves_20260622.png`, `figures/dronevehicle_dynamic_ladd_vs_baselines.png`

## Suggested Questions for a Reviewer Model

1. Is the failure mainly caused by modality direction, data protocol, insufficient epochs, or LADD loss/architecture mismatch?
2. Should DroneVehicle be removed from the main paper, kept as diagnostic appendix, or redesigned as a separate cross-dataset protocol?
3. Which minimal follow-up would best test whether this is a protocol issue rather than a method issue?
