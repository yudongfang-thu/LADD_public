# Analysis Notes

## Raw Comparison

| comparison | AP50 delta | AP delta |
|---|---:|---:|
| standard `train.py` - custom det-only | +0.23992 | +0.17474 |
| two-branch no-KD - custom det-only | -0.00899 | -0.00011 |
| ATKD-only - custom det-only | +0.02528 | +0.01659 |
| full CCLKD - custom det-only | +0.01668 | +0.01030 |
| full CCLKD - ATKD-only | -0.00860 | -0.00629 |

## Findings

1. **Primary blocker: trainer alignment.** The custom trainer detector-only path is far below standard YOLOv5 `train.py`, so the next debugging target is not the CCLKD loss itself.
2. **Two-branch machinery is not the main failure mode.** Adding the teacher branch without KD leaves AP essentially unchanged relative to custom det-only.
3. **ATKD is positive relative to the custom trainer baseline.** This suggests LLD/FLD/RLD signals are numerically active and not causing collapse in the 80 epoch setting.
4. **CCL is active but not beneficial here.** Full CCLKD has a live CCL loss near `0.693`, but full is lower than ATKD-only.
5. **The result remains far from the CCLKD paper target.** CCLKD Table 5 reports YOLOv5 baseline `80.9/46.3` and full CCLKD `88.7/57.3`; this 80 epoch local standard YOLOv5 baseline reaches `57.06/30.96`, while the custom full CCLKD reaches `34.73/14.52`.

## Next Checks

1. Compare standard YOLOv5 `train.py` vs custom det-only for optimizer groups, warmup, EMA, augmentation, anchor/target assignment, loss scaling, and validation settings.
2. Re-run custom det-only after alignment before interpreting ATKD/full absolute AP.
3. After custom det-only is aligned, repeat ATKD-only and full CCLKD under the same settings.
