# v0 initial paper-formulation ablation

Commit/state: `before 9bc1b33; launched from the first paper-formulation implementation`

Initial six-way paper ablation. Later audit found the paper CCL branch still used the wrong negative anchor, and RLD/PATM were incomplete. Only LLD and LLD+FLD remain useful as non-CCL/non-RLD diagnostic curves.

## Runs

|run|status|latest_epoch|latest_AP50|latest_AP|latest_KD|epoch50_AP50|epoch50_AP|mtime|note|
|---|---|---|---|---|---|---|---|---|---|
|lld|running|304|0.751658|0.493256|0.471530|0.555032|0.313466|2026-06-08 21:31:49|valid: LLD-only curve; no CCL/RLD/PATM dependency|
|lld_fld|running|287|0.743759|0.486046|0.642010|0.558326|0.311368|2026-06-08 21:32:45|valid: LLD+FLD fixed-temperature curve; no CCL/RLD/PATM dependency|
|lld_fld_rld_old|stopped/superseded|151|0.651822|0.398525|1.004960|0.555973|0.306399|2026-06-08 15:18:59|invalid: RLD used mean-MSE scale and no PATM T_j^2|
|atkd_old|stopped/superseded|149|0.647249|0.396648|2.046540|0.551512|0.305113|2026-06-08 15:18:50|invalid: PATM only affected LLD; FLD/RLD not aligned to Algorithm 1|
|ccl_only_old|stopped/superseded|148|0.645997|0.397882|0.216940|0.545892|0.302388|2026-06-08 15:08:28|invalid: CCL paper branch used student negative vs teacher negative|
|full_old|stopped/superseded|143|0.639091|0.393667|2.463390|0.550356|0.305264|2026-06-08 15:08:22|invalid: both CCL anchor and RLD/PATM issues|

## Files

- `metrics_summary.csv`: compact metrics for this version.
- `results/`: copied `results.csv` files from the 4090 server. No checkpoints or weights are included.
