# CCLKD YOLOv5x 400epoch Running Status

Snapshot time: `2026-06-14 08:05:35 +0800`.

This archive tracks the four YOLOv5x scaling-fix b32/s0/400ep main runs. No loss change, sweep, or stop action is implied by this generated report.

## Current Results

| run | epoch | AP50 | AP | same-epoch det AP | delta AP | KD/det | feature ok | NaN/Inf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full CCLKD | 278 | 0.66011 | 0.37985 | 0.37221 | 0.00764 | 0.48605 | 1.00000 | 0.00000 |
| ATKD-only | 200 | 0.61454 | 0.33383 | 0.32520 | 0.00863 | 0.05655 | 1.00000 | 0.00000 |
| CCL-only | 327 | 0.68563 | 0.40862 | 0.40040 | 0.00822 | 0.46716 | 1.00000 | 0.00000 |
| Det-only baseline | 368 | 0.70783 | 0.42550 | baseline |  | 0.00000 |  | 0.00000 |

## Loss Contribution

| run | epoch | student det sum | ATKD | CCL | KD total | ATKD share | CCL share | KD/det |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full CCLKD | 278 | 0.05149 | 0.09851 | 0.69224 | 0.79074 | 0.12458 | 0.87543 | 0.48605 |
| ATKD-only | 200 | 0.05811 | 0.10447 | 0.00000 | 0.10447 | 1.00000 | 0.00000 | 0.05655 |
| CCL-only | 327 | 0.04744 | 0.00000 | 0.69389 | 0.69389 | 0.00000 | 1.00000 | 0.46716 |
| Det-only baseline | 368 | 0.04372 | 0.00000 | 0.00000 | 0.00000 |  |  | 0.00000 |

## Milestone Readiness

| epoch | det AP | ATKD AP | CCL AP | Full AP | Full - ATKD | Full - CCL | note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 80 | 0.25132 | 0.25550 | 0.25521 | 0.25640 | 0.00090 | 0.00119 | pre_200_snapshot |
| 100 | 0.26324 | 0.26782 | 0.26603 | 0.26901 | 0.00119 | 0.00298 | pre_200_snapshot |
| 125 | 0.27807 | 0.28386 | 0.28164 | 0.28352 | -0.00034 | 0.00188 | pre_200_snapshot |
| 150 | 0.29367 | 0.30024 | 0.29892 | 0.29997 | -0.00027 | 0.00105 | pre_200_snapshot |
| 185 | 0.31722 | 0.32360 | 0.32023 | 0.32287 | -0.00073 | 0.00264 | pre_200_snapshot |
| 200 | 0.32520 | 0.33383 | 0.32936 | 0.33255 | -0.00128 | 0.00319 | aligned_snapshot |
| 250 | 0.35625 | pending | 0.36116 | 0.36360 | pending | 0.00244 | pending |
| 300 | 0.38416 | pending | 0.39227 | pending | pending | pending | pending |
| 350 | 0.41471 | pending | pending | pending | pending | pending | pending |
| 399 | pending | pending | pending | pending | pending | pending | pending |

## Current Decision

- Use exact same-epoch comparisons only.
- Exact epoch 200 is complete: ATKD delta AP = `+0.00863`, CCL delta AP = `+0.00416`, Full delta AP = `+0.00735`.
- At epoch 200, Full is still above det-only but below ATKD-only by `-0.00128` AP, so mark CCL weak/negative synergy candidate and continue to the 250 milestone unless a stop condition appears.
- CCL-only has high KD/det pressure with weak AP gain at epoch 200, so keep CCL low-efficiency candidate status.
- If a milestone row contains `pending`, do not make aligned milestone decisions from it.
- Continue the active main runs unless a documented stop condition is met.
- Do not modify CCLKD loss or launch sweeps before the 200/250 milestone evidence justifies it.
