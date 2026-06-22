# LADD Debug Evidence Pack

Created: 2026-06-22 CST

This directory is a compact evidence pack for the current method-risk analysis.
It intentionally stores raw CSV/log/provenance files, not checkpoint weights.

## Current Debug Questions

| Case | Directory | Question |
|---|---|---|
| YOLO11s Dynamic instability | `dynamic_s_instability_20260622/` | Why does `clean_a1b_dyn` show late collapse / unstable detector losses on YOLO11s, while dynprobe/main is much more stable? |
| DroneVehicle no-gain behavior | `dronevehicle_no_gain_20260622/` | Why do LADD main and Dynamic LADD fail to improve over the RGB student baseline on DroneVehicle under the CCLKD-aligned protocol? |

## File Conventions

- `raw/`: original `results.csv`, `args.yaml`, diagnostics, metadata, and selected logs copied from local evidence or remote servers.
- `analysis/`: derived summaries and snapshots generated from the raw files.
- `figures/`: copied plots that were already generated during analysis.
- `debug_manifest.csv`: file list and byte sizes for the entire debug pack.
- `source_manifest.tsv`: provenance mapping from renamed local raw files back to original run paths.

## Notes

- No `.pt`, `.pth`, TensorBoard event, W&B, ONNX, engine, or raw dataset files are included.
- Some runs are partial by design because the run was stopped after the failure mode became clear.
- The raw CSVs are the authoritative evidence; README interpretations are only working hypotheses.
