# YOLOv5x CCLKD projection-to-400 diagnostic

YOLOv5 records 400 training epochs as CSV epochs `0..399`.

This is a diagnostic extrapolation, not a completed experimental result. Observed curves remain the authoritative evidence.

## Final estimates

| estimate                                           |   epoch |       ap |   delta_ap_vs_det |
|:---------------------------------------------------|--------:|---------:|------------------:|
| atkd_observed_epoch287                             |     287 | 0.38994  |          0.01314  |
| atkd_epoch399_flat_latest_delta_conservative       |     399 | 0.45302  |          0.01314  |
| atkd_epoch399_linear_delta_from_200_287_main       |     399 | 0.459537 |          0.019657 |
| atkd_epoch399_recent_delta_from_237_287_aggressive |     399 | 0.465013 |          0.025133 |
| full_epoch399_projected_delta_from_300_392         |     399 | 0.448195 |          0.008315 |
| ccl_observed_epoch399                              |     399 | 0.44348  |          0.0036   |
| det_only_observed_epoch399                         |     399 | 0.43988  |          0        |

## Milestones

|   epoch |   det_only_ap |   ccl_ap |   ccl_delta_ap |   atkd_ap |   atkd_delta_ap | atkd_status                  |   full_ap |   full_delta_ap | full_status                  |   best_ap | best_method   |
|--------:|--------------:|---------:|---------------:|----------:|----------------:|:-----------------------------|----------:|----------------:|:-----------------------------|----------:|:--------------|
|     287 |       0.3768  |  0.38386 |        0.00706 |  0.38994  |        0.01314  | observed                     |  0.38504  |        0.00824  | observed                     |  0.38994  | atkd          |
|     300 |       0.38416 |  0.39227 |        0.00811 |  0.397651 |        0.013491 | projected_delta_from_200_287 |  0.3934   |        0.00924  | observed                     |  0.397651 | atkd          |
|     350 |       0.41471 |  0.42223 |        0.00752 |  0.431315 |        0.016605 | projected_delta_from_200_287 |  0.42415  |        0.00944  | observed                     |  0.431315 | atkd          |
|     399 |       0.43988 |  0.44348 |        0.0036  |  0.459537 |        0.019657 | projected_delta_from_200_287 |  0.448195 |        0.008315 | projected_delta_from_300_392 |  0.459537 | atkd          |

## Projection rule

- ATKD main projection: completed det-only curve plus linear extrapolation of ATKD same-epoch AP delta fitted on epochs 200..287.
- ATKD conservative bound: keep epoch-287 delta constant to epoch 399.
- ATKD aggressive bound: extrapolate delta trend from epochs 237..287.
- Full projection only fills the small 392..399 gap from its post-300 delta trend.
