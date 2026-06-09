# v3 paper-pair CCL + per-side DFL box-distribution snapshot

Initial raw-log snapshot time: `2026-06-09 16:50` on the dual-4090 server.
Latest lightweight CSV/status update: `2026-06-09 20:29`.

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
| `cclkd_v3_update.tgz` | Lightweight latest update bundle with current `results.csv`, `args.yaml`, process status, and manifests. No checkpoint weights are included. |
| `process_status.json` | Active process snapshot at pull time. |
| `current_status.json` | Compact parsed status for the six current runs. |
| `metrics_summary.csv` | Current epoch, best/latest metrics, baseline-at-same-epoch deltas, and ETA. |
| `milestone_comparison.csv` | Fixed-epoch ranking and baseline deltas. |
| `files_update_20260609_2029.txt` | File manifest for the lightweight update bundle. |
| `results/` | Copied `results.csv` for each current run. |
| `args/` | Copied `args.yaml` for each current run. |

## Current metrics

Baseline reference: `../baseline_reference/sar_yolo11n_400ep_laddproto_results.csv`.

| Run | Epoch | AP50 | AP50-95 | Same-epoch baseline AP50-95 | Delta AP50-95 | Note |
|---|---:|---:|---:|---:|---:|---|
| `lld` | 244 | 0.719204 | 0.463431 | 0.443510 | +0.019921 | running |
| `lld_fld` | 249 | 0.721280 | 0.465211 | 0.446960 | +0.018251 | running |
| `lld_fld_rld` | 256 | 0.726508 | 0.464087 | 0.450790 | +0.013297 | running |
| `ccl_only` | 400 | 0.785596 | 0.520551 | 0.515460 | +0.005091 | completed |
| `atkd` | 400 | 0.788921 | 0.524869 | 0.515460 | +0.009409 | completed |
| `full` | 400 | 0.787642 | 0.525314 | 0.515460 | +0.009854 | completed |

## Fixed-epoch ranking

The main relation is directionally healthy but small:

```text
E200: lld_fld 0.435608 > full 0.433895 > lld 0.431651 > atkd 0.430931 > lld_fld_rld 0.429893 > ccl_only 0.429273
E225: full 0.450408 > lld_fld 0.450250 > lld 0.449980 > atkd 0.446349 > lld_fld_rld 0.444728 > ccl_only 0.444627
E238: lld 0.459383 > lld_fld 0.458373 > full 0.457883 > atkd 0.454678 > lld_fld_rld 0.452986 > ccl_only 0.451762
E250: full 0.465338 > atkd 0.461998 > lld_fld_rld 0.460387 > ccl_only 0.459055
E300: full 0.492990 > atkd 0.490579 > ccl_only 0.486651
E350: full 0.514404 > atkd 0.512768 > ccl_only 0.508547
E400: full 0.525314 > atkd 0.524869 > ccl_only 0.520551
```

The early non-CCL branch relation is less clean:

```text
E150: lld_fld 0.401291 > full 0.400819 > atkd 0.398739 > lld_fld_rld 0.397772 > lld 0.397343 > ccl_only 0.393371
```

This means:

- CCL appears to be a positive add-on to ATKD at equal epoch.
- CCL-only is consistently weaker than full, as expected.
- `lld_fld` remains slightly stronger than `lld_fld_rld` in the early/mid phase, so RLD is not yet showing a clean positive contribution.
- `E250` does not yet include exact `lld`/`lld_fld` rows in the current pulled CSV, so it should not be treated as a complete non-CCL comparison until those runs pass 250.
- The final gain is small: at epoch 400, `full` is only `+0.009854` AP50-95 above the same-protocol SAR baseline and only `+0.000445` AP50-95 above `atkd`.

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
