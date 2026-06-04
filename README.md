# LADD Public Debug Package

This repository is a public evidence package for debugging RGB-guided SAR object detection distillation. It is not a polished paper-release repository yet.

Last updated: 2026-06-04 00:05 CST.

## What To Read First

1. [PACKAGE_AUDIT_CN.md](PACKAGE_AUDIT_CN.md): what is included, what is intentionally excluded, and where each evidence block lives.
2. [docs/experiments/BASELINE_LADD_STATUS_CN.md](docs/experiments/BASELINE_LADD_STATUS_CN.md): latest baseline and LADD status.
3. [comparison/IMPLEMENTATION_REVIEW_CN.md](comparison/IMPLEMENTATION_REVIEW_CN.md): comparison implementation audit, fixes, and remaining fidelity gaps.
4. [ladd/LADD_COLLAPSE_DEBUG_CN.md](ladd/LADD_COLLAPSE_DEBUG_CN.md): LADD collapse timeline and the code/config versions that were tested.
5. [comparison/archive/excluded_methods/README.md](comparison/archive/excluded_methods/README.md): excluded methods and invalid historical results.

## Directory Map

| Directory | Content |
|---|---|
| `baseline/` | Formal OGSOD HBB baseline code and copied result CSVs from the 90 server. |
| `ladd/` | LADD HBB code snapshots, mainline results, collapse diagnostics, and 90/4090D evidence. |
| `comparison/` | Current FGD/LD/CCLKD/HalluciDet review and a clearly separated excluded-method archive. |
| `docs/` | Method notes, experiment status, literature survey, and public-facing summaries. |
| `shared/` | Dataset YAML templates, shared KD scaffolding, and vendored Ultralytics code. |
| `server_logs/` | Selected experiment-server logs, compressed as `.log.gz`. No SSH credentials are included. |

## Current High-Level Status

- Baseline is stable: YOLO11n/s have SAR/RGB 3-seed formal no-mosaic baselines; YOLO11m/l/x have seed0 baselines.
- LADD main evidence is strongest on YOLO11n: seed0 and seed42 completed with positive gains; seed123 and some 4090D runs expose a B-stage collapse/BN-running-stat issue.
- Controlled comparisons are now FGD-style, true DFL-based LD, CCLKD-style, and HalluciDet-style. FGD/LD were corrected on 2026-06-04, so their old runs are historical only.
- CrossKD and CoLD are downgraded to historical archive material and are not active experiment lines.

## Privacy / Security

This package intentionally excludes SSH passwords, private keys, full connection commands, datasets, and checkpoint weights. Some copied `args.yaml` and compressed logs preserve absolute training paths such as `runs_public/...` or server-local project roots because they are useful provenance for debugging; these are not login credentials.
