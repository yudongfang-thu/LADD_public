# LADD: Learnability-Aware Decomposition Distillation

This repository is the paper-facing implementation for RGB-guided SAR object detection distillation. It keeps the current runnable implementation, experiment protocol, and compact result summaries. Historical raw logs, large diagnostic bundles, and deprecated-method archives have been removed from the public branch.

Last updated: 2026-06-16 CST.

## What To Read First

1. [PACKAGE_AUDIT_CN.md](PACKAGE_AUDIT_CN.md): what is included, what is intentionally excluded, and where each evidence block lives.
2. [docs/method/METHOD_DEFINITIONS_AND_IMPLEMENTATION_CN.md](docs/method/METHOD_DEFINITIONS_AND_IMPLEMENTATION_CN.md): current method names, runnable entries, and implementation boundaries.
3. [docs/experiments/LADD_MAINLINE_STANDARD_CN.md](docs/experiments/LADD_MAINLINE_STANDARD_CN.md): formal LADD mainline protocol.
4. [comparison/IMPLEMENTATION_REVIEW_CN.md](comparison/IMPLEMENTATION_REVIEW_CN.md): comparison implementation audit, fixes, and remaining fidelity gaps.
5. [cclkd_reproduction/README.md](cclkd_reproduction/README.md): CCLKD paper-protocol reproduction line.

## Directory Map

| Directory | Content |
|---|---|
| `baseline/` | Formal OGSOD HBB baseline code and compact result summary. |
| `ladd/` | LADD HBB code snapshots, launch scripts, and compact mainline summary. |
| `comparison/` | Current FGD/LD/CMDistill/HalluciDet-YOLO/CCLKD controlled-method review and notes. |
| `cclkd_reproduction/` | Paper-aligned CCLKD reproduction protocol, paper PDF, online trainer, YOLO11n ablation plan, and compact diagnostics. |
| `docs/` | Method notes, experiment status, literature survey, and public-facing summaries. |
| `shared/` | Dataset YAML templates, shared KD scaffolding, and vendored Ultralytics code. |

## Direct Runtime Layout

The repository is runnable from its root. The compatibility paths under `tools/`,
`scripts/ogsod_public/`, and `configs/datasets/` point to the paper-facing source
files above, so formal launchers do not depend on the former private workspace.

Before training, replace `/path/to/OGSOD-1.0` in the two public dataset YAMLs or
run `scripts/prepare_server_runtime.sh <dataset-root> <asset-root>`. Checkpoint
weights and generated runs remain untracked.

## Current High-Level Status

- Baseline is stable: YOLO11n/s have SAR/RGB 3-seed formal no-mosaic baselines; YOLO11m/l/x have seed0 baselines.
- LADD main evidence is strongest on YOLO11n: the current cap2 + B BN-freeze line has three positive seeds; larger capacities still have late-regression diagnostics.
- Controlled comparisons are under audit. The current method set is FGD-style, LD, CMDistill-style, HalluciDet-YOLO adaptation, and CCLKD online comparison. CCLKD paper-aligned reproduction lives separately in `cclkd_reproduction/`; frozen-teacher CCLKD loss components are not sufficient for official CCLKD reporting. Dual-4090 smoke/formal partial runs with an erroneous `nc=5` yaml are invalid.

## Privacy / Security

This package intentionally excludes SSH passwords, private keys, full connection commands, datasets, checkpoint weights, raw run directories, and server logs.
