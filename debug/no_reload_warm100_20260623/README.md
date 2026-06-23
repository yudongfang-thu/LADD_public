# No-Reload Warm100 Debug Line

Created: 2026-06-23 CST

This debug directory records the current response to the baseline-reload control
problem: the reload-only detector control appears to gain performance in a way
that can explain much of the apparent LADD gain. The goal of this line is to
test main methods that avoid loading a fully trained detector at B-stage start.

No checkpoint weights or datasets are stored here.

## Core Question

Can LADD still produce a real gain if B-stage does not start from a fully
trained SAR baseline checkpoint?

## Protocols Being Run On AutoDL Dual 4090

Timestamp: `20260623_151719`

1. `warm100` detector cache
   - Train SAR detector from `yolo11n.pt` for 100 epochs.
   - Mosaic enabled during these 100 epochs.
   - Output detector checkpoint: `weights/last.pt`.

2. `A1` decomposition cache
   - Train A1 decomposition for 10 epochs from the existing SAR baseline and
     RGB teacher.
   - Completed before B-stage launch.

3. `B700 after_e100` jobs
   - Wait until `warm100/results.csv` reaches epoch 100.
   - Then launch three B-stage jobs from the warm100 detector and A1 decomp:
     `detonly_after_e100`, `probeA_after_e100`, and `dynamic_after_e100`.
   - Important fix: the first orchestrator incorrectly treated an in-progress
     `last.pt` as completed warm100; those partial B runs were stopped and are
     not valid evidence.

4. Direct YOLO-init no-warm control
   - Started while GPU1 was idle.
   - B-stage starts directly from `yolo11n.pt`.
   - Decomposition modules are split-loaded from the A1 cache.
   - This is the cleanest "A1 -> B from YOLO init" comparison against the
     warm100 versions.

## Current AutoDL Screens

Snapshot time: 2026-06-23 15:44:32 CST

| Screen | Role | GPU | Status |
|---|---|---:|---|
| `nl_warm100_n_s0_g0_20260623_151719` | warm100 detector cache | 0 | running |
| `nl_orch_n_s0_20260623_151719` | fixed orchestrator waiting for epoch 100 | CPU | running |
| `nl_b700_probeA_yoloinit_n_s0_g1_20260623_154117` | direct YOLO-init B-stage Probe-A control | 1 | running |

GPU snapshot:

| GPU | Used MiB | Total MiB | Util % | Role |
|---:|---:|---:|---:|---|
| 0 | 3011 | 24564 | 13 | warm100 |
| 1 | 11789 | 24564 | 33 | direct YOLO-init Probe-A |

## Latest Metrics At Snapshot

| Run | Epoch | Total | mAP50 | mAP50-95 | Note |
|---|---:|---:|---:|---:|---|
| warm100 detector | 20 | 100 | 0.31685 | 0.14233 | running |
| A1 cache | 10 | 10 | 0.35138 | 0.13109 | completed |
| direct YOLO-init Probe-A B | 1 | 700 | 0.14983 | 0.05607 | running |

## Invalid Partial Runs

The first orchestrator launched B-stage too early because YOLO writes
`weights/last.pt` during training. These runs were stopped and should not be
used:

| Partial run | Stopped at |
|---|---:|
| `detonly_partial` | 5/700 |
| `probeA_partial` | 4/700 |
| `dynamic_partial` | 5/700 |

## Files

| File | Purpose |
|---|---|
| `autodl_no_reload_warm100_queue.sh` | AutoDL launcher and fixed orchestrator. |
| `STATUS_20260623_1544.md` | Raw remote status snapshot used for this README. |
| `CURRENT_ACTIONS_20260623.md` | Short action log and intended comparisons. |

