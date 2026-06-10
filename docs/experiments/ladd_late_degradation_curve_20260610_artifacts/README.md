# LADD Late Degradation Curve Artifacts

Lightweight derived records for the 2026-06-10 LADD late-degradation curve analysis.

Files:

- `curve_run_summary.csv`: best/last AP and late AP slope per run.
- `curve_best_late_window_delta.csv`: best-window versus late-window loss deltas.
- `curve_key_epoch_points.csv`: key epoch metric/loss/LR points.
- `h1_s_b400_diagnostic_keypoints.csv`: H1 s b400 BN, grad, KD, and NaN/Inf diagnostic points.
- `curve_analysis_command_log.txt`: compact analysis log and input list.

These files are derived from existing `results.csv` and `ladd_diagnostics.csv` artifacts. They do not include checkpoint weights or TensorBoard event files.
