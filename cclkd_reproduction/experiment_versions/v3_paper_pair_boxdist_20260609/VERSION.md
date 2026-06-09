# v3 paper-pair CCL + per-side DFL box-distribution snapshot

Snapshot time: `2026-06-09 16:50` on the dual-4090 server.

This version corresponds to the current paper-aligned diagnostic branch:

```text
CCLKD_FORMULATION=paper
CCLKD_CCL_MODE=paper_pair
CCLKD_CCL_SOURCE=box_distribution
CCLKD_RLD_MODE=paper_instance
```

The key implementation differences from previous archived versions are:

- CCL uses paper-pair semantics: positive pair `sim(S_pos, T_pos)`, negative pair `sim(S_neg, T_neg)`.
- `box_distribution` CCL uses per-side YOLO DFL distributions: reshape `[N, 4*reg_max] -> [N, 4, reg_max]`, softmax over bins, then flatten and normalize.
- Formal launcher defaults to the paper formulation and keeps `paper_pair` as the default CCL mode.
- The run protocol follows the current 400-epoch CCLKD comparison setup, not a final LADD mainline choice.

## Snapshot files

| File | Content |
|---|---|
| `cclkd_v3_snapshot.tgz` | Raw pulled artifact bundle from the 4090 server; contains `results.csv`, `args.yaml`, selected text logs, process status, and file manifest. No checkpoint weights are included. |
| `process_status.json` | Active process snapshot at pull time. |
| `current_status.json` | Compact parsed status for the six current runs. |
| `metrics_summary.csv` | Current epoch, best/latest metrics, baseline-at-same-epoch deltas, and ETA. |
| `milestone_comparison.csv` | Fixed-epoch ranking and baseline deltas. |
| `results/` | Copied `results.csv` for each current run. |
| `args/` | Copied `args.yaml` for each current run. |

## Current metrics

Baseline reference: `../baseline_reference/sar_yolo11n_400ep_laddproto_results.csv`.

| Run | Epoch | AP50 | AP50-95 | Same-epoch baseline AP50-95 | Delta AP50-95 | Note |
|---|---:|---:|---:|---:|---:|---|
| `lld` | 158 | 0.652329 | 0.401732 | 0.386820 | +0.014912 | running |
| `lld_fld` | 161 | 0.656575 | 0.407762 | 0.388560 | +0.019202 | running |
| `lld_fld_rld` | 164 | 0.657699 | 0.406629 | 0.391310 | +0.015319 | running |
| `ccl_only` | 357 | 0.773673 | 0.510307 | 0.502990 | +0.007317 | running |
| `atkd` | 392 | 0.787397 | 0.523595 | 0.514900 | +0.008695 | running |
| `full` | 359 | 0.776816 | 0.516802 | 0.503950 | +0.012852 | running |

## Fixed-epoch ranking

The main relation is directionally healthy but small:

```text
E200: full 0.433274 > atkd 0.430199 > ccl_only 0.428485
E250: full 0.464674 > atkd 0.460946 > ccl_only 0.458394
E300: full 0.492442 > atkd 0.490531 > ccl_only 0.485905
E350: full 0.513572 > atkd 0.512457 > ccl_only 0.508147
```

The early non-CCL branch relation is less clean:

```text
E150: lld_fld 0.400576 > full 0.400169 > atkd 0.398089 > lld 0.396877 > lld_fld_rld 0.396802 > ccl_only 0.392827
```

This means:

- CCL appears to be a positive add-on to ATKD at equal epoch.
- CCL-only is consistently weaker than full, as expected.
- `lld_fld` remains slightly stronger than `lld_fld_rld` in the early/mid phase, so RLD is not yet showing a clean positive contribution.
- The absolute gain is small: near late training, `full` is only about `+0.013` AP50-95 above the same-protocol SAR baseline.

## Why this still needs audit

The implementation direction is no longer obviously wrong, but the effect size is smaller than expected. The next audit should focus on mechanisms that can silently compress the gain:

1. **Online teacher definition**: confirm the online RGB teacher branch is trained/evaluated exactly as intended, and that teacher-side COP quality is strong enough throughout training.
2. **KD loss scale**: compare detection loss, LLD, FLD, RLD and CCL magnitudes at matched epochs; small effective gradients could explain the weak improvement.
3. **COP density**: log class-wise and epoch-wise COP token counts; sparse COP may make CCL/RLD underpowered.
4. **CCL source choice**: compare `box_distribution` with `roi_feature`; DFL localization distributions may be semantically weaker for category-constrained contrast than neck/ROI features.
5. **RLD contribution**: since `lld_fld_rld` is not clearly better than `lld_fld`, inspect RLD formulation/weight and whether its gradient is aligned with the current YOLO11 feature geometry.
6. **Protocol baseline strength**: the formal 400-epoch baseline is strong; small positive gains may be real, but this should be verified after full 400-epoch completion and with seed repeats.

## Validity status

This snapshot is valid as a debugging/evidence package for the current implementation, but it is **not** a final CCLKD claim. Use it to explain why another audit round is needed.
