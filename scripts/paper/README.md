# Paper Launchers

These launchers are the recommended entrypoints for paper-facing OGSOD HBB mosaic100 experiments.

Run from the repository root:

```bash
bash scripts/paper/run_paper_baseline.sh <sar|rgb> <n|s|m|l|x> <seed> <gpu_id>
bash scripts/paper/run_paper_ladd.sh <n|s|m|l|x> <seed> <gpu_id>
bash scripts/paper/run_paper_comparison_kd.sh <fgd|ld|cmdistill> <n|s|m|l|x> <seed> <gpu_id>
bash scripts/paper/run_paper_hallucidet.sh <n|s|m> <seed> <gpu_id>
bash scripts/paper/run_paper_cclkd_online.sh <n|s> <seed> <gpu_id>
```

Optional cross-dataset extension, separate from OGSOD main-table runs:

```bash
bash scripts/paper/run_paper_cclkd_yolo11n_cross_baseline.sh <vedai|dronevehicle> <student|teacher> <seed> <gpu_id>
bash scripts/paper/run_paper_ladd_cclkd_yolo11n_cross_dataset.sh <vedai|dronevehicle> <seed> <gpu_id>
```

All paper launchers support:

```bash
DRY_RUN=1
EXIST_OK=1
PAPER_STRICT_GIT=1
```

Protocol invariants:

- `protocol=mosaic100`
- `imgsz=256`
- `epochs=800`
- `mosaic=1.0`
- `close_mosaic=700`
- seeds restricted to `0, 42, 123`
- dataset YAMLs restricted to `configs/paper/datasets/`

The VEDAI / DroneVehicle extension deliberately uses a different protocol:

- `protocol_id=cclkd_yolo11n_cross_dataset_20260619`
- `imgsz=512`
- `epochs=200`
- `batch=16`
- `optimizer=SGD`
- `model=YOLO11n`

Those rows belong in a cross-dataset or appendix table, not the OGSOD mosaic100 main table.

Validation:

```bash
bash scripts/paper/validate_engineering_cleanup.sh
```

FGD/LD/HalluciDet wrappers use the locked implementations in
`comparison/FINAL_LOCKED_METHODS_CN.md`. HalluciDet is P1-gated and should be
checked by dry-run/smoke before it enters a comparison table. CCLKD online is
optional and must not be mixed with the separate paper-protocol reproduction line.
