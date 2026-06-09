# v3 paper-pair CCL + per-side DFL box-distribution snapshot

Initial raw-log snapshot time: `2026-06-09 16:50` on the dual-4090 server.
Latest lightweight CSV/status update: `2026-06-09 23:56`.

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
| `cclkd_v3_snapshot.tgz` | Original raw pulled artifact bundle. No checkpoint weights are included. |
| `cclkd_v3_update.tgz` | Earlier lightweight update bundle. No checkpoint weights are included. |
| `process_status.json` | Active process/GPU snapshot at latest pull time. |
| `current_status.json` | Compact parsed status for the six current runs. |
| `metrics_summary.csv` | Current epoch, best/latest metrics, baseline-at-same-epoch deltas, and ETA. |
| `milestone_comparison.csv` | Fixed-epoch ranking and baseline deltas. |
| `loss_component_summary.csv` / `.md` | Weighted KD component loss snapshots from `results.csv`. |
| `raw_runs/log_tail_*_20260609_2356.txt` | Tail excerpts from current outer logs. |
| `results/` | Copied `results.csv` for each current run. |
| `args/` | Copied `args.yaml` for each current run when available. |

## Current metrics

Baseline reference: `../baseline_reference/sar_yolo11n_400ep_laddproto_results.csv`.

| Run | Epoch | AP50 | AP50-95 | Same-epoch baseline AP50-95 | Delta AP50-95 | Note |
|---|---:|---:|---:|---:|---:|---|
| `lld` | 329 | 0.769286 | 0.508380 | 0.491870 | 0.016510 | running |
| `lld_fld` | 336 | 0.776800 | 0.511846 | 0.494710 | 0.017136 | running |
| `lld_fld_rld` | 349 | 0.781441 | 0.513163 | 0.500580 | 0.012583 | running |
| `ccl_only` | 400 | 0.785596 | 0.520551 | 0.515460 | 0.005091 | completed |
| `atkd` | 400 | 0.788921 | 0.524869 | 0.515460 | 0.009409 | completed |
| `full` | 400 | 0.787642 | 0.525314 | 0.515460 | 0.009854 | completed |

## Matched-epoch reading

At the latest common epoch for the three still-running non-CCL branches (`epoch 329`), all three are positive versus the same-protocol SAR baseline, but RLD is still weaker than the simpler branches:

- `lld`: AP50-95 `0.508380`, delta `0.016510`
- `lld_fld`: AP50-95 `0.508390`, delta `0.016520`
- `lld_fld_rld`: AP50-95 `0.504213`, delta `0.012343`

For the completed 400-epoch CCL branch runs, the final ordering remains `full > atkd > ccl_only > baseline`, but the full-vs-ATKD gain is only about `+0.000445` AP50-95.

## Loss observation

The late-stage weighted RLD term is extremely small: about `5e-5` to `8e-5` in `lld_fld_rld`. This explains why RLD does not show a clean positive contribution in the matched-epoch curves. The current running jobs were launched before `cclkd_diagnostics.csv` logging was added, so COP density and temperature diagnostics are not available for this batch.

## Validity status

This snapshot is valid as a debugging/evidence package for the current implementation, but it is **not** a final CCLKD claim. Use it to explain why the implementation trend is healthy while the effect size and RLD contribution remain under audit.
