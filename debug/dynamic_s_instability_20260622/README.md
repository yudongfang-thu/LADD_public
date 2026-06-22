# Debug Case: YOLO11s Dynamic Instability

Created: 2026-06-22 CST

## Problem Statement

The current Dynamic candidate (`LADD_A1B_MODE=dynamic`) looks attractive on some OGSOD seed0 evidence, but the YOLO11s Dynamic line has shown unstable behavior:

1. A 4090 partial run reached a strong best AP50-95 and then dropped sharply before completion.
2. A separate AutoDL run crashed/failed very early.
3. A current AutoDL2 rerun is still running and has not yet resolved whether Dynamic is stable through late training.
4. The dynprobe/main YOLO11s reference is much more stable near the end.

This is a candidate blocker for using Dynamic as the paper mainline unless the rerun shows the old collapse was machine/run-specific.

## Raw Evidence

| File | Meaning |
|---|---|
| `raw/s_dynamic_4090_partial_results.csv` | Old YOLO11s Dynamic partial run from 4090 evidence. Shows late AP drop. |
| `raw/s_main_dynprobe_autodl_results.csv` | Completed YOLO11s dynprobe/main reference run. |
| `raw/s_dynamic_autodl_early_crash_results.csv` | Old AutoDL Dynamic run that failed within 4 epochs. |
| `raw/s_dynamic_autodl2_current_a1_results.csv` | Current AutoDL2 Dynamic rerun A-stage snapshot. |
| `raw/s_dynamic_autodl2_current_b_results.csv` | Current AutoDL2 Dynamic rerun B-stage snapshot copied while running. |
| `raw/s_dynamic_autodl2_current_*.yaml/env/txt/log` | Current rerun args, metadata, manifest, and B-stage log. |

## Key Numbers

See `analysis/run_summary_all.csv` for the full table. Current highlights:

| Run | Rows | Best AP50-95 | Best Epoch | Last AP50-95 | Drop |
|---|---:|---:|---:|---:|---:|
| Dynamic s, 4090 partial | 712 | 0.63647 | 656 | 0.60079 | 0.03568 |
| Main dynprobe s, AutoDL | 800 | 0.63487 | 708 | 0.62764 | 0.00723 |
| Dynamic s, AutoDL early crash | 4 | 0.49673 | 1 | 0.15055 | 0.34618 |
| Dynamic s, AutoDL2 current B snapshot | 237 | 0.57290 | 237 | 0.57290 | 0.00000 |

## Working Hypotheses

1. Dynamic may be vulnerable to late detector/KD loss shocks when the trainable teacher-side core and reach/query pathway keep changing during B.
2. The 4090 partial collapse coincided with `train/kd_loss` rising from 0.05968 at epoch 656 to 0.17844 at epoch 712 and detector losses rising as AP fell.
3. Dynprobe/main stabilizes the B-stage target/probe path better: comparable epochs do not show the same KD/loss jump.
4. The current AutoDL2 rerun is still too early to clear the risk; the important window is after roughly epoch 600.

## Files for External Analysis

- Summary: `analysis/run_summary_all.csv`
- Epoch snapshots: `analysis/epoch_snapshots_all.csv`
- Previous focused analysis: `analysis/run_summary.csv`, `analysis/epoch_snapshots.csv`, `analysis/window_stats.csv`
- Figures: `figures/ogsod_s_dynamic_collapse_loss_vs_main_20260622.png`, `figures/ogsod_s_dynamic_collapse_reach_dist_vs_main_20260622.png`

## Suggested Questions for a Reviewer Model

1. Is the late AP drop better explained by KD target drift, detector loss shock, reach loss behavior, or training protocol?
2. Does dynprobe/main merely hide the instability, or does it provide a principled stabilization mechanism?
3. If Dynamic becomes the mainline, which Dynamic-aligned ablations must be rerun to avoid a mismatched ablation story?
