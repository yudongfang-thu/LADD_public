# LADD repair experiment evidence manifest 20260613

- Source servers: main dual-4090 `/root/shared-nvme/LADD_public_p1`, AutoDL `/root/autodl-tmp/LADD_public`.
- Evidence root: `ladd/results/repair_experiments_20260613/`.
- Files: 129.
- Size: 1.69 MiB.
- Included: `results.csv`, `args.yaml`, `ladd_diagnostics.csv`, source manifests, compact log tail extracts.
- Excluded: checkpoints (`.pt`, `.pth`), TensorBoard event files, `wandb`, full run directories, full large logs.

## Summary CSVs

- `summary/ladd_repair_experiments_summary_20260613.csv`: one row per completed/terminated run.
- `summary/ladd_repair_phase_summary_20260613.csv`: one row per phase, with key AP and loss fields.

## Evidence Subdirectories

- `evidence/main_4090/`: completed S1, S2, N1, M1, M2 compact evidence from the dual-4090 server.
- `evidence/autodl/`: completed M3/M1_auto compact evidence plus the terminated HalluciDet b8 note from AutoDL.
