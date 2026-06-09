# Current CCLKD Training Status

Pulled from the dual-4090 server at `2026-06-09 20:29`.

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
|lld|running|244|0.719204|0.463431|0.443510|+0.019921|0.615520|positive, still running|
|lld_fld|running|249|0.721280|0.465211|0.446960|+0.018251|0.734070|positive, still running|
|lld_fld_rld|running|256|0.726508|0.464087|0.450790|+0.013297|0.666540|RLD not clearly positive over LLD+FLD|
|ccl_only|completed|400|0.785596|0.520551|0.515460|+0.005091|0.513330|positive but weak|
|atkd|completed|400|0.788921|0.524869|0.515460|+0.009409|1.216540|completed|
|full|completed|400|0.787642|0.525314|0.515460|+0.009854|1.748430|full > atkd, but additive CCL gain is tiny|

## Equal-Epoch Reading

The main ordering is directionally reasonable:

```text
E200: lld_fld > full > lld > atkd > lld_fld_rld > ccl_only
E225: full > lld_fld > lld > atkd > lld_fld_rld > ccl_only
E238: lld > lld_fld > full > atkd > lld_fld_rld > ccl_only
E250: full > atkd > lld_fld_rld > ccl_only
E300: full > atkd > ccl_only
E350: full > atkd > ccl_only
E400: full > atkd > ccl_only
```

The `E250` row does not yet include exact `lld`/`lld_fld` measurements in the current pulled CSV, so use `E238` as the latest complete six-way non-CCL/CCL comparison. At `E238`, `lld > lld_fld > lld_fld_rld`, so RLD still does not show a clean positive contribution at matched epoch.

At 400 epochs, `full > atkd > ccl_only > baseline`, but `full - atkd` is only about `+0.00045` AP50-95. The implementation trend is healthy, while the gain remains smaller than expected. The current snapshot should therefore be used as an audit package, not as a final CCLKD claim. The next checks should focus on online teacher quality, COP density, KD loss scale, CCL source choice, and whether RLD is actually helping under the current YOLO11 protocol.

Status is a snapshot; runs may continue past these epochs after this file is committed.
