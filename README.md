# LADD: Learnability-Aware Decomposition Distillation

This repository is the paper-facing implementation for RGB-guided SAR object detection distillation. It keeps the current runnable implementation, experiment protocol, and compact result summaries. Historical raw logs, large diagnostic bundles, and deprecated-method archives have been removed from the public branch.

Last updated: 2026-06-18 CST.

## What To Read First

1. [docs/paper/PAPER_PROTOCOL_CN.md](docs/paper/PAPER_PROTOCOL_CN.md): paper-facing OGSOD HBB mosaic100 protocol and main-table gate.
2. [docs/paper/METHOD_NAME_WHITELIST_CN.md](docs/paper/METHOD_NAME_WHITELIST_CN.md): allowed paper method names and forbidden historical labels.
3. [paper_results/README.md](paper_results/README.md): canonical result schema and main-table candidate rules.
4. [scripts/paper/README.md](scripts/paper/README.md): paper launchers and validation commands.
5. [docs/ladd_clean_a1b_method_definition.md](docs/ladd_clean_a1b_method_definition.md): LADD Probe-A / clean A1B method definition.

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

The paper-facing main protocol is OGSOD HBB `mosaic100`: `imgsz=256`, `epochs=800`, `mosaic=1.0`, `close_mosaic=700`, deterministic training, and paired SAR/RGB baselines by capacity and seed. LADD main rows must be `LADD Probe-A / clean_a1b_dynprobe` with `A1 -> B` and no A2.

Historical no-mosaic, A1-A2-B, BN-freeze, smoke, partial, close@100, 400ep, and diagnostic runs are retained only as archive/diagnostic evidence and are not used for main-table claims.

## Direct Runtime Layout

The repository is runnable from its root. The compatibility paths under `tools/`,
`scripts/ogsod_public/`, and `configs/datasets/` point to the paper-facing source
files above, so formal launchers do not depend on the former private workspace.

Before training, replace `/path/to/OGSOD-1.0` in the two public dataset YAMLs or
run `scripts/prepare_server_runtime.sh <dataset-root> <asset-root>`. Checkpoint
weights and generated runs remain untracked.

## Current High-Level Status

- Paper-facing baseline/LADD/comparison runs must use `scripts/paper/` and pass `paper_results/` validation.
- LADD mainline is `LADD Probe-A / LADD-clean A1B`: `clean_a1b_dynprobe`, `A1 -> B`, SAR-only inference, no A2.
- Controlled comparisons are FGD-style, LD, CMDistill-style, HalluciDet-YOLO adaptation, and optional CCLKD online comparison. CCLKD paper-aligned reproduction lives separately in `cclkd_reproduction/`; frozen-teacher CCLKD loss components are not sufficient for official CCLKD reporting.

## Privacy / Security

This package intentionally excludes SSH passwords, private keys, full connection commands, datasets, checkpoint weights, raw run directories, and server logs.
