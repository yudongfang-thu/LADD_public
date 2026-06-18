# LADD A2 Selection Diagnostics Evidence (2026-06-12)

This directory contains lightweight evidence synced from `/root/shared-nvme/LADD_public_p1` on the dual-4090 server.

Remote commit: `665cdb4871d9a5620befc46d93d693765779d60b`
Snapshot time: 2026-06-12 16:21-16:23 CST.

Included:
- per-phase `results.csv`, `args.yaml`, `manifest.txt` when present, and `ladd_diagnostics.csv`;
- chain logs and final model text files when present;
- compact outer log extracts;
- packaged run manifest and summary CSV.

Excluded:
- checkpoint weights (`*.pt`, `*.pth`);
- TensorBoard event files;
- wandb directories;
- complete run directories;
- full large logs.

Important: `s_A2_lr3e4_short13_Bdet200` is a running B200 snapshot in this package, not a completed B result.
