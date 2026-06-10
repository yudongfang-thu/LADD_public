# CCLKD Full Regression Snapshot, 2026-06-10

## Current Decision

The current YOLOv5x `current_full` run should be stopped.

It is systematically below the SAR baseline from epoch 0 to 114. This is
treated as an implementation/protocol regression of the current proxy trainer,
not evidence against the original CCLKD paper.

Next, run only the minimal stop-loss diagnostics:

- D0: `det_only_same_trainer`
- D1: `two_branch_no_kd`

Implementation note after the 2026-06-10 audit patch:

- `current_full` is retained only as a legacy alias for `raw_proxy_full`.
- The raw proxy is not the default and is not treated as a verified CCLKD
  reproduction.
- True CCLKD-style YOLOv5 audit entry points are `paper_atkd_only`,
  `paper_ccl_only`, and `paper_full`.
- Formal `paper_*` runs should wait until D0/D1 show that the custom trainer and
  two-branch setup are not already causing the regression.

Snapshot time: `Wed Jun 10 12:52:42 CST 2026`

This directory contains a compact evidence package for the current YOLOv5-X
online CCLKD full run. Checkpoints and full logs are intentionally excluded.

## Included Runs

| Directory | Run |
|---|---|
| `full_cclkd_gpu1/` | YOLOv5-X online CCLKD full, SAR student + RGB teacher, batch 64, seed 0 |
| `sar_baseline_b64/` | YOLOv5-X SAR baseline, batch 64, seed 0 |
| `rgb_baseline_b64/` | YOLOv5-X RGB baseline, batch 64, seed 0, still running at snapshot time |
| `ladd_yolo11s_phase_b/` | Concurrent LADD YOLO11s phase-b reference curve |

Each run folder includes `results.csv` when available. YOLOv5 runs also include
`command.sh`, `run_meta.txt`, and a compact `nohup` tail.

## Same-Epoch Comparison

The relevant comparison is `full_cclkd_gpu1` vs `sar_baseline_b64`, because the
CCLKD student detector is SAR. At the latest full CCLKD snapshot:

| epoch | run | P | R | AP50 | AP50-95/AP |
|---:|---|---:|---:|---:|---:|
| 114 | SAR baseline | 0.66186 | 0.54314 | 0.54650 | 0.28094 |
| 114 | full CCLKD | 0.48072 | 0.33387 | 0.32089 | 0.12988 |
| 114 | RGB baseline | 0.79848 | 0.77743 | 0.79987 | 0.43032 |

Delta, full CCLKD minus SAR baseline at epoch 114:

| metric | delta |
|---|---:|
| AP50 | -0.22561 |
| AP50-95/AP | -0.15106 |

## Checkpoint Trend

| epoch | SAR AP50 | SAR AP | full AP50 | full AP | full - SAR AP50 | full - SAR AP |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.06706 | 0.01995 | 0.03496 | 0.01090 | -0.03210 | -0.00905 |
| 10 | 0.26270 | 0.10163 | 0.10609 | 0.02963 | -0.15661 | -0.07200 |
| 20 | 0.31800 | 0.12922 | 0.13540 | 0.04312 | -0.18260 | -0.08610 |
| 50 | 0.47173 | 0.22755 | 0.25990 | 0.09941 | -0.21183 | -0.12814 |
| 100 | 0.53581 | 0.27332 | 0.31257 | 0.12529 | -0.22324 | -0.14803 |
| 114 | 0.54650 | 0.28094 | 0.32089 | 0.12988 | -0.22561 | -0.15106 |

## Interpretation

This is not just a slow-start curve at the current snapshot. The full CCLKD run
is systematically behind the SAR baseline at the same epoch, and the gap grows
from early training through epoch 114.

The current online CCLKD trainer is a YOLOv5-adapted implementation rather than
a verified line-by-line reproduction of the original CCLKD training code. The
results therefore indicate a likely implementation/protocol issue in the current
adaptation, not a claim about the original CCLKD paper.

Immediate audit targets:

- Verify whether the online teacher should be trainable or frozen/warm-started
  differently for the intended CCLKD protocol.
- Re-check CCL positive/negative sampling and class-conditional feature pairing.
- Re-check LLD/FLD/RLD feature/logit tensor definitions against the original
  YOLOv5 detection head.
- Add `no_kd`, `det_only_same_trainer`, and component-only runs before drawing a
  method-level conclusion.

## Runtime Snapshot

`gpu_snapshot.csv` and `gpu_processes.csv` record GPU utilization at snapshot
time. At snapshot time, full CCLKD was still running on GPU1 and had not failed.
