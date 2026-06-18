# Baseline Dual-Modality Gap Table

Interpretation: `gap` is `RGB best mAP50-95(B) - SAR best mAP50-95(B)` under the same capacity and protocol. The curves compare the two modality-specific single-modality baselines, not a fused two-input detector.

## nomosaic

| Model | SAR best AP @ epoch | RGB best AP @ epoch | RGB-SAR gap | SAR last AP | RGB last AP | SAR/RGB epochs | Status | Note |
|---|---:|---:|---:|---:|---:|---:|---|---|
| YOLO11n | 0.55654 @734 | 0.63018 @723 | 0.07364 | 0.55127 | 0.62737 | 800/800 | complete |  |
| YOLO11s | 0.62897 @729 | 0.65768 @647 | 0.02871 | 0.62233 | 0.64958 | 800/800 | complete |  |
| YOLO11m | 0.65580 @704 | 0.67909 @663 | 0.02329 | 0.64903 | 0.67159 | 800/800 | complete |  |
| YOLO11l | 0.65427 @735 | 0.68356 @618 | 0.02929 | 0.64892 | 0.66892 | 800/800 | complete |  |
| YOLO11x | 0.65867 @685 | 0.68284 @539 | 0.02417 | 0.64801 | 0.65820 | 800/800 | complete |  |

## mosaic100

| Model | SAR best AP @ epoch | RGB best AP @ epoch | RGB-SAR gap | SAR last AP | RGB last AP | SAR/RGB epochs | Status | Note |
|---|---:|---:|---:|---:|---:|---:|---|---|
| YOLO11n | 0.54091 @746 | 0.61610 @758 | 0.07519 | 0.53836 | 0.61345 | 800/800 | complete |  |
| YOLO11s | 0.61972 @770 | 0.66029 @679 | 0.04057 | 0.61570 | 0.65818 | 800/759 | partial | 90 remote confirmed 759 epochs |
| YOLO11m | 0.65092 @713 | 0.67340 @600 | 0.02248 | 0.64251 | 0.66845 | 793/680 | partial | 90 remote confirmed latest snapshot; 793 epochs; 90 remote confirmed stopped snapshot; 680 epochs |

## Outputs

- `figures/baseline_nomosaic_dual_modality_curves.png`
- `figures/baseline_mosaic100_dual_modality_curves.png`
- `figures/baseline_nomosaic_capacity_overlay.png`
- `figures/baseline_mosaic100_capacity_overlay.png`
- `figures/baseline_dual_modality_gap_by_protocol.png`
- `baseline_dual_modality_summary.csv`
- `baseline_dual_modality_gap_table.csv`
