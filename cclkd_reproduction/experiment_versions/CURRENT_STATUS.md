# Current CCLKD Training Status

Pulled from the dual-4090 server at `2026-06-09 23:56`.

Latest version directory: [`v3_paper_pair_boxdist_20260609`](v3_paper_pair_boxdist_20260609/VERSION.md).

## Current Implementation

```text
formulation = paper
ccl_mode = paper_pair
ccl_source = box_distribution
rld_mode = paper_instance
```

## Latest Curves

Same-epoch baseline uses `baseline_reference/sar_yolo11n_400ep_laddproto_results.csv`.

|run|status|latest_epoch|latest_AP50|latest_AP|same_epoch_baseline_AP|delta_AP|latest_KD|note|
|---|---|---:|---:|---:|---:|---:|---:|---|
|lld|running|329|0.769286|0.508380|0.491870|0.016510|0.417440|positive, still running|
|lld_fld|running|336|0.776800|0.511846|0.494710|0.017136|0.511550|positive, still running; late AP now slightly above lld|
|lld_fld_rld|running|349|0.781441|0.513163|0.500580|0.012583|0.446750|positive vs baseline, but weaker than lld/lld_fld at matched epoch|
|ccl_only|completed|400|0.785596|0.520551|0.515460|0.005091|0.513330|completed; positive but weak|
|atkd|completed|400|0.788921|0.524869|0.515460|0.009409|1.216540|completed|
|full|completed|400|0.787642|0.525314|0.515460|0.009854|1.748430|completed; full > atkd at 400, but additive CCL gain is tiny|

## Equal-Epoch Reading

The three still-running non-CCL branches can now be compared at `epoch 329`:

| run | AP50 | AP50-95 | delta AP vs baseline |
|---|---:|---:|---:|
| `lld` | 0.769286 | 0.508380 | 0.016510 |
| `lld_fld` | 0.774516 | 0.508390 | 0.016520 |
| `lld_fld_rld` | 0.773145 | 0.504213 | 0.012343 |

The current matched-epoch relationship is still not RLD-positive: `lld` and `lld_fld` are ahead of `lld_fld_rld` in AP50-95 at the same epoch. The latest values for `lld_fld_rld` are higher only because that run is further along in training.

At 400 epochs for the completed CCL branch runs:

```text
E300: full 0.492990 > atkd 0.490579 > ccl_only 0.486651
E350: full 0.514404 > atkd 0.512768 > ccl_only 0.508547
E400: full 0.525314 > atkd 0.524869 > ccl_only 0.520551
```

## Loss Components

See [`loss_component_summary.csv`](v3_paper_pair_boxdist_20260609/loss_component_summary.csv) and [`loss_component_summary.md`](v3_paper_pair_boxdist_20260609/loss_component_summary.md). The important late-stage observation is that RLD is only around `5e-5` to `8e-5`, so it is numerically tiny in the current formulation/weighting.

Status is a snapshot; the three non-CCL runs may continue past these epochs after this file is committed.
