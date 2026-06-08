# CCLKD Experiment Versions

This directory archives the CCLKD reproduction/debugging sequence on the 4090 server. It is intentionally organized by implementation version so invalid diagnostic runs are not mixed with current formal runs.

No checkpoint weights are included; only `results.csv`, compact summaries, and status metadata are tracked.

## Version Map

|version|meaning|commit/state|
|---|---|---|
|v0_initial_paper_ablation|v0 initial paper-formulation ablation|before 9bc1b33; launched from the first paper-formulation implementation|
|v1_ccl_anchorfix|v1 CCL anchor-direction fix|9bc1b33 Fix paper CCL anchor direction|
|v2_rld_patmfix|v2 RLD and PATM alignment fix|bf73697 Align CCLKD PATM and RLD losses|

## Top-Level Files

- `CURRENT_STATUS.md`: latest pulled live process/GPU status and current run metrics.
- `current_status.json`: raw live status snapshot from the 4090 server.
- `metrics_summary.csv`: all archived runs across all versions.
- `baseline_reference/sar_yolo11n_400ep_laddproto_results.csv`: same-protocol SAR baseline reference used for comparisons.

## Validity Rule

Use `v2_rld_patmfix` for current RLD/ATKD/full CCLKD claims, `v1_ccl_anchorfix` for CCL-only after anchor correction, and only the LLD/LLD+FLD curves from `v0_initial_paper_ablation` as still-valid early diagnostic curves. Other v0/v1 rows are preserved as evidence of the implementation audit, not as final results.
