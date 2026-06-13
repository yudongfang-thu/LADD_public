# Scaling-Fix YOLOv5x CCLKD 400epoch Snapshot

Current snapshot: `2026-06-14 00:13:27 +08`.

This directory archives compact evidence for the active YOLOv5x CCLKD scaling-fix 400epoch runs on server 90.

Files:

- `summary.csv`: current metrics, diagnostics, exact same-epoch det-only deltas, and GPU status.
- `running_status.md`: human-readable status report.
- `milestone_component_comparison.csv`: fixed milestone exact-epoch component comparison.
- `milestone_component_comparison.md`: Markdown rendering of the milestone table.
- `process_gpu_snapshot.txt`: raw GPU/process snapshot.
- `runs/`: per-run CSVs, configs, metadata, log tails, and error keyword grep.

Large artifacts are excluded: checkpoint weights, TensorBoard event files, and full nohup logs.
