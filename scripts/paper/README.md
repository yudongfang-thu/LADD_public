# Paper Launchers

These launchers are the recommended entrypoints for paper-facing OGSOD HBB mosaic100 experiments.

Run from the repository root:

```bash
bash scripts/paper/run_paper_baseline.sh <sar|rgb> <n|s|m|l|x> <seed> <gpu_id>
bash scripts/paper/run_paper_ladd_probea.sh <n|s|m|l|x> <seed> <gpu_id>
bash scripts/paper/run_paper_comparison_kd.sh <fgd|ld|cmdistill> <n|s|m|l|x> <seed> <gpu_id>
bash scripts/paper/run_paper_hallucidet.sh <n|s|m> <seed> <gpu_id>
bash scripts/paper/run_paper_cclkd_online.sh <n|s> <seed> <gpu_id>
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

Validation:

```bash
bash scripts/paper/validate_engineering_cleanup.sh
```

HalluciDet is P1-gated because the current standalone trainer does not implement `close_mosaic=700`. CCLKD online is optional and must not be mixed with the separate paper-protocol reproduction line.
