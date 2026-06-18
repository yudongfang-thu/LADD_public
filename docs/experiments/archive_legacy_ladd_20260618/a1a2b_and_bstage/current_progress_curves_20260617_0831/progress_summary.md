# Current Progress Curves Snapshot

Source: refreshed CSV snapshots under `raw/`.

Important caveat: LADD B-phase epoch is the local B-stage epoch. The same-epoch baseline columns are a reference on the plotted x-axis, not a claim that both runs have identical pretraining history.

| model | run | status | latest epoch | latest AP | best epoch | best AP | baseline AP at latest epoch | delta latest |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| n | SAR baseline s0 | done | 800 | 0.53836 | 746 | 0.54091 |  |  |
| n | clean A1B static s0 (AutoDL) | running | 527 | 0.53482 | 527 | 0.53482 | 0.50660 | 0.02822 |
| n | clean A1B dynamic s0 (AutoDL) | running | 442 | 0.51912 | 440 | 0.51914 | 0.48487 | 0.03425 |
| n | A-scheme dynprobe s0 (4090) | running | 67 | 0.38298 | 1 | 0.41367 | 0.24804 | 0.13494 |
| n | legacy A2last s123 (90) | running | 405 | 0.50725 | 402 | 0.50777 | 0.47417 | 0.03308 |
| s | SAR baseline s0 | done | 800 | 0.61570 | 770 | 0.61972 |  |  |
| s | clean A1B static s0 (4090) | running | 485 | 0.58027 | 485 | 0.58027 | 0.58490 | -0.00463 |
| s | clean A1B dynamic s0 (4090) | running | 376 | 0.59624 | 375 | 0.59628 | 0.55775 | 0.03849 |
| s | A-scheme dynprobe s0 (AutoDL) | running | 74 | 0.47144 | 1 | 0.49692 | 0.33250 | 0.13894 |
| s | legacy skipA2 s0 (90) | running | 257 | 0.57199 | 257 | 0.57199 | 0.50440 | 0.06759 |
| s | legacy A1A2B s123 (AutoDL) | running | 746 | 0.59928 | 712 | 0.60373 | 0.61766 | -0.01838 |
| m | SAR baseline s0 | running | 793 | 0.64251 | 713 | 0.65092 |  |  |
| m | RGB teacher baseline s0 (stopped) | early_stopped | 680 | 0.66845 | 600 | 0.67340 | 0.64895 | 0.01950 |

Stopped baseline check:

- `m / RGB teacher baseline s0` stopped at epoch 680 due Ultralytics EarlyStopping: no improvement in the last 80 epochs. Best epoch reported by the log is 600.

Generated figures:

- `figures/progress_ap_by_model.png` / `.pdf`
- `figures/progress_ap50_by_model.png` / `.pdf`
