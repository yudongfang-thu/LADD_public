# Current CCLKD Training Status

Pulled from the dual-4090 server at `2026-06-09 16:50`.

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
|lld|running|158|0.652329|0.401732|0.386820|+0.014912|0.847560|positive but small|
|lld_fld|running|161|0.656575|0.407762|0.388560|+0.019202|0.963410|strongest early non-CCL branch|
|lld_fld_rld|running|164|0.657699|0.406629|0.391310|+0.015319|0.972000|RLD not yet clearly positive over LLD+FLD|
|ccl_only|running|357|0.773673|0.510307|0.502990|+0.007317|0.505170|positive but weak|
|atkd|running|392|0.787397|0.523595|0.514900|+0.008695|1.212330|near completion|
|full|running|359|0.776816|0.516802|0.503950|+0.012852|1.749980|full > atkd at equal epoch, but gain is small|

## Equal-Epoch Reading

The main ordering is directionally reasonable:

```text
E200: full > atkd > ccl_only
E250: full > atkd > ccl_only
E300: full > atkd > ccl_only
E350: full > atkd > ccl_only
```

However, the gain is smaller than expected. The current snapshot should therefore be used as an audit package, not as a final CCLKD result. The next checks should focus on online teacher quality, COP density, KD loss scale, CCL source choice, and whether RLD is actually helping under the current YOLO11 protocol.

Status is a snapshot; runs may continue past these epochs after this file is committed.
