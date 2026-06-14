# YOLOv5x CCLKD aligned-to-ATKD comparison

Cutoff epoch: `287`. This table compares all four runs at exactly shared YOLOv5 CSV epochs.

|   epoch |   det_only_ap |   atkd_ap |   atkd_delta_ap_vs_det |   ccl_ap |   ccl_delta_ap_vs_det |   full_ap |   full_delta_ap_vs_det |   full_minus_atkd_ap |   full_minus_ccl_ap | best_by_ap   |
|--------:|--------------:|----------:|-----------------------:|---------:|----------------------:|----------:|-----------------------:|---------------------:|--------------------:|:-------------|
|       0 |       0.01713 |  0.015237 |              -0.001893 | 0.015237 |             -0.001893 |  0.015237 |              -0.001893 |              0       |             0       | det_only     |
|      50 |       0.21904 |  0.22407  |               0.00503  | 0.22388  |              0.00484  |  0.22553  |               0.00649  |              0.00146 |             0.00165 | full         |
|     100 |       0.26324 |  0.26782  |               0.00458  | 0.26603  |              0.00279  |  0.26901  |               0.00577  |              0.00119 |             0.00298 | full         |
|     150 |       0.29367 |  0.30024  |               0.00657  | 0.29892  |              0.00525  |  0.29997  |               0.0063   |             -0.00027 |             0.00105 | atkd         |
|     200 |       0.3252  |  0.33383  |               0.00863  | 0.32936  |              0.00416  |  0.33255  |               0.00735  |             -0.00128 |             0.00319 | atkd         |
|     250 |       0.35625 |  0.36522  |               0.00897  | 0.36116  |              0.00491  |  0.3636   |               0.00735  |             -0.00162 |             0.00244 | atkd         |
|     287 |       0.3768  |  0.38994  |               0.01314  | 0.38386  |              0.00706  |  0.38504  |               0.00824  |             -0.0049  |             0.00118 | atkd         |

## Notes

- Delta columns are computed against det-only at the exact same epoch.
- This avoids comparing ATKD epoch 287 with det-only/CCL/full near epoch 400.
- Positive late-epoch gaps should be interpreted together with the curve shape, because det-only can catch up later.
