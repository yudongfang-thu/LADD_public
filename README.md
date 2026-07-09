# LADD: Learnability-Aware Decomposition Distillation

This repository is the paper-facing implementation for cross-modal object detection distillation with learnability-aware feature decomposition. The main OGSOD track studies RGB-guided SAR detection, while the VEDAI and DroneVehicle extensions evaluate the same LADD idea under CCLKD-aligned visible/infrared transfer protocols. The repository keeps the current runnable implementation, experiment protocol, and compact result summaries. Historical raw logs, large diagnostic bundles, and deprecated-method archives have been removed from the public branch.

Last updated: 2026-07-09 CST.

## Current Debug Status (2026-07-09)

The active investigation has shifted from the earlier paper-facing `mosaic100`
mainline to a stricter `pure direct-400` rescue/evidence audit. The current
direct-400 evidence is not claim-ready: the best audited/pending LADD rescue
row is still below the FGD comparison floor, and running/progress rows are not
treated as final results.

For the latest compact handoff, current metrics, audit boundaries, and next
routes, read:

- [docs/PROJECT_STATUS_20260709_CN.md](docs/PROJECT_STATUS_20260709_CN.md)
- [docs/PROJECT_ONBOARDING_20260709_CN.md](docs/PROJECT_ONBOARDING_20260709_CN.md)
- [docs/experiments/DIRECT400_EXPERIMENT_LEDGER_20260709_CN.md](docs/experiments/DIRECT400_EXPERIMENT_LEDGER_20260709_CN.md)
- [docs/CODE_AND_DOC_CHANGE_MAP_20260709_CN.md](docs/CODE_AND_DOC_CHANGE_MAP_20260709_CN.md)

These public handoff pages intentionally record only compact public-facing
evidence and links. They exclude checkpoint weights, private server details,
raw logs, and large run directories.

## What To Read First

1. [docs/paper/PAPER_PROTOCOL_CN.md](docs/paper/PAPER_PROTOCOL_CN.md): paper-facing OGSOD HBB mosaic100 protocol and main-table gate.
2. [docs/paper/METHOD_NAME_WHITELIST_CN.md](docs/paper/METHOD_NAME_WHITELIST_CN.md): allowed paper method names and forbidden historical labels.
3. [paper_results/README.md](paper_results/README.md): canonical result schema and main-table candidate rules.
4. [scripts/paper/README.md](scripts/paper/README.md): paper launchers and validation commands.
5. [docs/ladd_method_definition.md](docs/ladd_method_definition.md): LADD method definition.
6. [docs/paper/CCLKD_YOLO11N_CROSS_DATASET_PROTOCOL_CN.md](docs/paper/CCLKD_YOLO11N_CROSS_DATASET_PROTOCOL_CN.md): optional VEDAI / DroneVehicle LADD extension aligned to the CCLKD YOLO11n protocol.

## Directory Map

| Directory | Content |
|---|---|
| `baseline/` | Formal OGSOD HBB baseline code and compact result summary. |
| `ladd/` | LADD HBB code snapshots, launch scripts, and compact mainline summary. |
| `comparison/` | Current FGD/LD/CMDistill/HalluciDet-YOLO/CCLKD controlled-method review and notes. |
| `cclkd_reproduction/` | Paper-aligned CCLKD reproduction protocol, paper PDF, online trainer, YOLO11n ablation plan, and compact diagnostics. |
| `docs/` | Method notes, experiment status, literature survey, and public-facing summaries. |
| `paper_results/` | Canonical paper-facing result schema and generated candidate tables. |
| `scripts/paper/` | Paper-gated launchers and engineering validation. |
| `shared/` | Dataset YAML templates, shared KD scaffolding, and vendored Ultralytics code. |

## Paper-Facing Protocol

The paper-facing main protocol is OGSOD HBB `mosaic100`: `imgsz=256`, `epochs=800`, `mosaic=1.0`, `close_mosaic=700`, deterministic training, and paired SAR/RGB baselines by capacity and seed. LADD main rows must be `LADD` with `A -> B` and no A2.

No-mosaic LADD is retained as verified fallback / robustness evidence. Historical A1-A2-B, BN-freeze, smoke, partial, 400ep, and diagnostic runs are retained only as archive/diagnostic evidence and are not used for main-table claims.

VEDAI and DroneVehicle are optional cross-dataset extensions, not OGSOD main-table rows. They use the separate CCLKD YOLO11n protocol (`imgsz=512`, `epochs=200`, `batch=16`, SGD) and compare LADD against reported CCLKD YOLO11n table numbers.

## Direct Runtime Layout

The repository is runnable from its root. The compatibility paths under `tools/`,
`scripts/ogsod_public/`, and `configs/datasets/` point to the paper-facing source
files above, so formal launchers do not depend on the former private workspace.

Before training, replace `/path/to/OGSOD-1.0` in the two public dataset YAMLs or
run `scripts/prepare_server_runtime.sh <dataset-root> <asset-root>`. Checkpoint
weights and generated runs remain untracked.

## Current High-Level Status

- Paper-facing baseline/LADD/comparison runs must use `scripts/paper/` and pass `paper_results/` validation.
- LADD mainline is `LADD`: `clean_a1b_dynprobe`, `A -> B`, SAR-only inference, no A2.
- Controlled comparisons are FGD-style, LD, CMDistill-style, HalluciDet-YOLO adaptation, and optional CCLKD online comparison. CCLKD paper-aligned reproduction lives separately in `cclkd_reproduction/`; frozen-teacher CCLKD loss components are not sufficient for official CCLKD reporting.
- VEDAI / DroneVehicle LADD extension runs are separate CCLKD YOLO11n protocol runs, not OGSOD mosaic100 paper-gate rows.

## Paper-Facing Entry Points

Read first:

1. [docs/paper/PAPER_PROTOCOL_CN.md](docs/paper/PAPER_PROTOCOL_CN.md)
2. [docs/experiments/LADD_MAINLINE_STANDARD_CN.md](docs/experiments/LADD_MAINLINE_STANDARD_CN.md)
3. [docs/ladd_method_definition.md](docs/ladd_method_definition.md)
4. [docs/experiments/PROJECT_EXPERIMENT_MAP_20260614_CN.md](docs/experiments/PROJECT_EXPERIMENT_MAP_20260614_CN.md)

Run paper experiments only through:

- [scripts/paper/run_paper_baseline.sh](scripts/paper/run_paper_baseline.sh)
- [scripts/paper/run_paper_ladd.sh](scripts/paper/run_paper_ladd.sh)
- [scripts/paper/run_paper_comparison_kd.sh](scripts/paper/run_paper_comparison_kd.sh)
- [scripts/paper/run_paper_cclkd_yolo11n_cross_baseline.sh](scripts/paper/run_paper_cclkd_yolo11n_cross_baseline.sh) for VEDAI / DroneVehicle extension baselines only
- [scripts/paper/run_paper_ladd_cclkd_yolo11n_cross_dataset.sh](scripts/paper/run_paper_ladd_cclkd_yolo11n_cross_dataset.sh) for VEDAI / DroneVehicle LADD extension only

Curate and validate paper table rows through:

- [tools/paper_collect_results.py](tools/paper_collect_results.py)
- [tools/paper_validate_main_table.py](tools/paper_validate_main_table.py)
- [paper_results/](paper_results/)

Raw historical run directories and legacy result summaries are not paper-table sources unless they pass paper validation.

## Privacy / Security

This package intentionally excludes authentication credentials, private keys, full connection commands, datasets, checkpoint weights, raw run directories, and server logs.
