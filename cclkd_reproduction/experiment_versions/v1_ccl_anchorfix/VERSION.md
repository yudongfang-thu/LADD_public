# v1 CCL anchor-direction fix

Commit/state: `9bc1b33 Fix paper CCL anchor direction`

Paper CCL branch changed to use the student target-category feature as the single anchor, contrasting teacher same-category and teacher non-target features. This fixes CCL-only. Full from this version was superseded by the later RLD/PATM fix.

## Runs

|run|status|latest_epoch|latest_AP50|latest_AP|latest_KD|epoch50_AP50|epoch50_AP|mtime|note|
|---|---|---|---|---|---|---|---|---|---|
|ccl_only_cclanchorfix|running|150|0.644947|0.397256|0.053320|0.542228|0.301586|2026-06-08 21:31:41|valid current CCL-only curve|
|full_cclanchorfix|stopped/superseded|3|0.048144|0.014841|8.331120|||2026-06-08 15:17:01|superseded: CCL anchor fixed; RLD/PATM were still incomplete|

## Files

- `metrics_summary.csv`: compact metrics for this version.
- `results/`: copied `results.csv` files from the 4090 server. No checkpoints or weights are included.
