# Scaling-Fix YOLOv5x CCLKD 400epoch Snapshot

This directory archives the current 2026-06-13 running snapshot for YOLOv5x CCLKD scaling-fix 400epoch experiments on server 90.

Files:

- `summary.csv`: machine-readable current metrics, diagnostics, same-epoch det-only deltas, GPU status.
- `running_status.md`: human-readable current status report.
- `process_gpu_snapshot.txt`: raw GPU/process snapshot from server 90.
- `runs/`: compact per-run evidence with CSVs, configs, metadata, log tails, and error keyword grep.

Large artifacts are intentionally excluded: checkpoint weights, TensorBoard event files, and full nohup logs.
