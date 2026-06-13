# YOLOv5x CCLKD Milestone Component Comparison

| epoch | det_only_ap | atkd_ap | atkd_delta_ap | ccl_ap | ccl_delta_ap | full_ap | full_delta_ap | full_minus_atkd_ap | full_minus_ccl_ap | best_component_by_ap | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | 0.25132 | 0.25550 | 0.00418 | 0.25521 | 0.00389 | 0.25640 | 0.00508 | 0.00090 | 0.00119 | full | pre_200_snapshot |
| 100 | 0.26324 | 0.26782 | 0.00458 | 0.26603 | 0.00279 | 0.26901 | 0.00577 | 0.00119 | 0.00298 | full | pre_200_snapshot |
| 125 | 0.27807 | 0.28386 | 0.00579 | 0.28164 | 0.00357 | 0.28352 | 0.00545 | -0.00034 | 0.00188 | atkd | pre_200_snapshot |
| 150 | 0.29367 | 0.30024 | 0.00657 | 0.29892 | 0.00525 | 0.29997 | 0.00630 | -0.00027 | 0.00105 | atkd | pre_200_snapshot |
| 185 | 0.31722 | pending | pending | 0.32023 | 0.00301 | 0.32287 | 0.00565 | pending | 0.00264 | full | pending |
| 200 | 0.32520 | pending | pending | 0.32936 | 0.00416 | 0.33255 | 0.00735 | pending | 0.00319 | full | pending |
| 250 | 0.35625 | pending | pending | 0.36116 | 0.00491 | pending | pending | pending | pending | ccl | pending |
| 300 | 0.38416 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| 350 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| 399 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
