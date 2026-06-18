# LADD Loss Audit Snapshot 2026-06-16

This directory archives compact evidence generated while auditing active LADD
mainline and skip-A2 experiments.

## Contents

- `tables/ladd_loss_contamination_args_audit_20260616.csv`
  - Run-level argument audit for LADD loss switches and active auxiliary losses.
- `tables/ladd_actual_extra_loss_audit_20260616.csv`
  - Results-file audit of nonzero extra loss columns by run and phase.
- `tables/ladd_loss_dynamics_summary_20260616.csv`
  - B-stage loss dynamics summary for key runs.
- `tables/ladd_loss_dynamics_windows_20260616.csv`
  - B-stage early/best/final window summaries.
- `tables/ladd_a2_loss_dynamics_summary_20260616.csv`
  - A2-stage loss dynamics summary for key runs.
- `tables/ladd_a2_loss_dynamics_windows_20260616.csv`
  - A2-stage early/best/final window summaries.
- `figures/ladd_b_loss_dynamics_key_runs_20260616.png`
  - B-stage curve comparison for key runs.
- `figures/ladd_a2_loss_dynamics_key_runs_20260616.png`
  - A2-stage curve comparison for key runs.

## Notes

The snapshot is intended as experiment evidence, not as a final method
definition. It records what was actually active in completed and running LADD
jobs so that later claims can be checked against logged arguments and loss
columns.

Temporary PDF text extracts under `tmp/pdfs/` were not committed because they
are literature-reading scratch files rather than experiment data.
