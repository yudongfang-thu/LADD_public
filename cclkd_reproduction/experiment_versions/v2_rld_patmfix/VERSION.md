# v2 RLD and PATM alignment fix

Commit/state: `bf73697 Align CCLKD PATM and RLD losses`

RLD changed from mean MSE to squared Frobenius-scale self-correlation loss; PATM T_j^2 now affects FLD and RLD in the paper formulation. This is the current formal version for RLD, ATKD, and full.

## Runs

|run|status|latest_epoch|latest_AP50|latest_AP|latest_KD|epoch50_AP50|epoch50_AP|mtime|note|
|---|---|---|---|---|---|---|---|---|---|
|lld_fld_rld_rldpatmfix|running|146|0.646752|0.395538|1.245440|0.550384|0.306155|2026-06-08 21:31:48|valid current LLD+FLD+RLD curve|
|atkd_rldpatmfix|running|146|0.646546|0.390955|3.955070|0.554280|0.302668|2026-06-08 21:33:17|valid current ATKD curve|
|full_rldpatmfix|running|150|0.643886|0.392374|4.173890|0.549127|0.301626|2026-06-08 21:33:23|valid current full CCLKD curve|

## Files

- `metrics_summary.csv`: compact metrics for this version.
- `results/`: copied `results.csv` files from the 4090 server. No checkpoints or weights are included.
