# Current Action Log

Date: 2026-06-23 CST

## Why This Exists

The baseline-reload control made the current mainline suspect: simply reloading
a trained baseline and continuing training can reproduce much of the apparent
method gain. The next useful experiments must reduce or remove that confound.

## Actions Taken

1. Stopped low-priority AutoDL dual-GPU jobs with user approval.
   - Stopped `dynamic_freeze_probe_only` on GPU0.
   - Stopped `yolo11m dynamic` on GPU1.

2. Started the no-reload warm100 line on AutoDL dual 4090.
   - `warm100`: SAR detector from `yolo11n.pt`, 100 epochs, mosaic enabled.
   - `A1 cache`: 10 epochs decomposition cache from existing SAR baseline/RGB
     teacher.
   - Intended B-stage: after warm100 reaches epoch 100, launch
     `detonly_after_e100`, `probeA_after_e100`, and `dynamic_after_e100`.

3. Found and fixed an orchestration bug.
   - Bug: B-stage was launched when `weights/last.pt` existed, but YOLO writes
     `last.pt` during training before epoch 100.
   - Fix: the orchestrator now also requires `warm100/results.csv` latest epoch
     to be at least 100 before launching B-stage.
   - The early B-stage partial runs were stopped and marked invalid.

4. Added direct YOLO-init B-stage control.
   - While GPU1 was idle, launched a no-warm Probe-A B-stage job.
   - Detector source: `yolo11n.pt`.
   - Decomposition source: completed A1 cache.
   - Purpose: compare against warm100 and test whether any gain survives when
     B-stage does not receive a trained/reloaded SAR detector.

## Intended Comparisons

Primary comparison group:

| Condition | Detector init for B | Decomp init | Purpose |
|---|---|---|---|
| direct YOLO-init Probe-A | `yolo11n.pt` | A1 cache | no detector reload / no warm control |
| detonly_after_e100 | warm100 `last.pt` | none/effectively det-only | warm100-only control |
| probeA_after_e100 | warm100 `last.pt` | A1 cache | current mainline without fully trained baseline reload |
| dynamic_after_e100 | warm100 `last.pt` | A1 cache | dynamic candidate without fully trained baseline reload |

Interpretation rule:

- If `probeA_after_e100` or `dynamic_after_e100` only matches
  `detonly_after_e100`, then the method still lacks evidence beyond detector
  continuation.
- If direct YOLO-init Probe-A is viable, it is stronger evidence against the
  reload confound, but it may be harder to optimize.

