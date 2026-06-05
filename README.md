# LADD: Learnability-Aware Decomposition Distillation

This repository is the paper-facing implementation and evidence package for RGB-guided SAR object detection distillation. It keeps the current runnable implementation, experiment protocol, diagnostic evidence, and clearly separated historical archives in one place.

Last updated: 2026-06-05 16:45 CST.

## What To Read First

1. [PACKAGE_AUDIT_CN.md](PACKAGE_AUDIT_CN.md): what is included, what is intentionally excluded, and where each evidence block lives.
2. [docs/experiments/PROTOCOL_AND_CCLKD_AUDIT_20260605_CN.md](docs/experiments/PROTOCOL_AND_CCLKD_AUDIT_20260605_CN.md): protocol error audit, invalid dual-4090 runs, and CCLKD rewrite notes.
3. [docs/experiments/BASELINE_LADD_STATUS_CN.md](docs/experiments/BASELINE_LADD_STATUS_CN.md): latest baseline and LADD status.
4. [comparison/IMPLEMENTATION_REVIEW_CN.md](comparison/IMPLEMENTATION_REVIEW_CN.md): comparison implementation audit, fixes, and remaining fidelity gaps.
4. [ladd/LADD_COLLAPSE_DEBUG_CN.md](ladd/LADD_COLLAPSE_DEBUG_CN.md): LADD collapse timeline and the code/config versions that were tested.
5. [comparison/archive/excluded_methods/README.md](comparison/archive/excluded_methods/README.md): excluded methods and invalid historical results.

## Directory Map

| Directory | Content |
|---|---|
| `baseline/` | Formal OGSOD HBB baseline code and copied result CSVs from the 90 server. |
| `ladd/` | LADD HBB code snapshots, mainline results, collapse diagnostics, and 90/4090D evidence. |
| `comparison/` | Current FGD/LD/HalluciDet review, CCLKD online-trainer gap notes, and a clearly separated excluded-method archive. |
| `docs/` | Method notes, experiment status, literature survey, and public-facing summaries. |
| `shared/` | Dataset YAML templates, shared KD scaffolding, and vendored Ultralytics code. |
| `server_logs/` | Selected experiment-server logs, compressed as `.log.gz`. No SSH credentials are included. |

## Direct Runtime Layout

The repository is runnable from its root. The compatibility paths under `tools/`,
`scripts/ogsod_public/`, and `configs/datasets/` point to the paper-facing source
files above, so formal launchers do not depend on the former private workspace.

Before training, replace `/path/to/OGSOD-1.0` in the two public dataset YAMLs or
run `scripts/prepare_server_runtime.sh <dataset-root> <asset-root>`. Checkpoint
weights and generated runs remain untracked.

## Current High-Level Status

- Baseline is stable: YOLO11n/s have SAR/RGB 3-seed formal no-mosaic baselines; YOLO11m/l/x have seed0 baselines.
- LADD main evidence is strongest on YOLO11n: seed0 and seed42 completed with positive gains; seed123 and some 4090D runs expose a B-stage collapse/BN-running-stat issue.
- Controlled comparisons are under audit. FGD/LD were corrected on 2026-06-04. CCLKD loss-level code was corrected on 2026-06-05, but it still needs an online teacher-student trainer before it can be treated as a CCLKD reproduction or main-table comparison. Dual-4090 smoke/formal partial runs with an erroneous `nc=5` yaml are invalid.
- CrossKD and CoLD are downgraded to historical archive material and are not active experiment lines.

## Privacy / Security

This package intentionally excludes SSH passwords, private keys, full connection commands, datasets, and checkpoint weights. Some copied `args.yaml` and compressed logs preserve absolute training paths such as `runs_public/...` or server-local project roots because they are useful provenance for debugging; these are not login credentials.
