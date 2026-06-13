# CCLKD YOLOv5x 400epoch Running Status

Snapshot time: `2026-06-14 03:41:44 +0800`.

This archive tracks the four YOLOv5x scaling-fix b32/s0/400ep main runs. No loss change, sweep, or stop action is implied by this generated report.

## Current Results

| run | epoch | AP50 | AP | same-epoch det AP | delta AP | KD/det | feature ok | NaN/Inf |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full CCLKD | 228 | 0.63087 | 0.34940 | 0.34332 | 0.00608 | 0.45916 | 1.00000 | 0.00000 |
| ATKD-only | 174 | 0.59179 | 0.31653 | 0.30917 | 0.00736 | 0.05941 | 1.00000 | 0.00000 |
| CCL-only | 279 | 0.65032 | 0.37938 | 0.37259 | 0.00679 | 0.42872 | 1.00000 | 0.00000 |
| Det-only baseline | 316 | 0.67844 | 0.39387 | baseline |  | 0.00000 |  | 0.00000 |

## Loss Contribution

| run | epoch | student det sum | ATKD | CCL | KD total | ATKD share | CCL share | KD/det |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full CCLKD | 228 | 0.05508 | 0.10236 | 0.69364 | 0.79600 | 0.12859 | 0.87141 | 0.45916 |
| ATKD-only | 174 | 0.05908 | 0.11176 | 0.00000 | 0.11175 | 1.00009 | 0.00000 | 0.05941 |
| CCL-only | 279 | 0.05176 | 0.00000 | 0.69390 | 0.69390 | 0.00000 | 1.00000 | 0.42872 |
| Det-only baseline | 316 | 0.04819 | 0.00000 | 0.00000 | 0.00000 |  |  | 0.00000 |

## Milestone Readiness

| epoch | det AP | ATKD AP | CCL AP | Full AP | Full - ATKD | Full - CCL | note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 80 | 0.25132 | 0.25550 | 0.25521 | 0.25640 | 0.00090 | 0.00119 | pre_200_snapshot |
| 100 | 0.26324 | 0.26782 | 0.26603 | 0.26901 | 0.00119 | 0.00298 | pre_200_snapshot |
| 125 | 0.27807 | 0.28386 | 0.28164 | 0.28352 | -0.00034 | 0.00188 | pre_200_snapshot |
| 150 | 0.29367 | 0.30024 | 0.29892 | 0.29997 | -0.00027 | 0.00105 | pre_200_snapshot |
| 185 | 0.31722 | pending | 0.32023 | 0.32287 | pending | 0.00264 | pending |
| 200 | 0.32520 | pending | 0.32936 | 0.33255 | pending | 0.00319 | pending |
| 250 | 0.35625 | pending | 0.36116 | pending | pending | pending | pending |
| 300 | 0.38416 | pending | pending | pending | pending | pending | pending |
| 350 | pending | pending | pending | pending | pending | pending | pending |
| 399 | pending | pending | pending | pending | pending | pending | pending |

## Current Decision

- Use exact same-epoch comparisons only.
- If a milestone row contains `pending`, do not make aligned milestone decisions from it.
- Continue the active main runs unless a documented stop condition is met.
- Do not modify CCLKD loss or launch sweeps before the 200/250 milestone evidence justifies it.
