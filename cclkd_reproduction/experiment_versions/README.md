# CCLKD Experiment Versions

This directory archives the CCLKD reproduction/debugging sequence on the 4090 server. It is intentionally organized by implementation version so invalid diagnostic runs are not mixed with current formal runs.

No checkpoint weights are included; only `results.csv`, compact summaries, and status metadata are tracked.

## Version Map

|version|meaning|commit/state|
|---|---|---|
|v0_initial_paper_ablation|v0 initial paper-formulation ablation|before 9bc1b33; launched from the first paper-formulation implementation|
|v1_ccl_anchorfix|v1 CCL anchor-direction fix|9bc1b33 Fix paper CCL anchor direction|
|v2_rld_patmfix|v2 RLD and PATM alignment fix|bf73697 Align CCLKD PATM and RLD losses|
|v3_paper_pair_boxdist_20260609|v3 paper-pair CCL + per-side DFL box-distribution snapshot|current 2026-06-09 implementation snapshot; stable but small gain, under audit|

## Top-Level Files

- `CURRENT_STATUS.md`: latest pulled live process/GPU status and current run metrics.
- `current_status.json`: raw live status snapshot from the 4090 server.
- `metrics_summary.csv`: all archived runs across all versions.
- `baseline_reference/sar_yolo11n_400ep_laddproto_results.csv`: same-protocol SAR baseline reference used for comparisons.

## Validity Rule

Use `v3_paper_pair_boxdist_20260609` as the latest debugging snapshot for the current paper-pair/box-distribution implementation. It shows stable but small positive gains and should be treated as evidence for further audit, not as a final CCLKD claim. Older v0/v1/v2 rows are preserved as implementation-history evidence and should not be mixed into final comparisons without checking their version notes.
