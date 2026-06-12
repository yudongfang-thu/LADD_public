# YOLOv5x Train.py Scaling Fix Alignment Evidence

Date: 2026-06-12

This archive stores the 80-epoch batch32 YOLOv5x SAR runs used to verify the
loss-gain scaling fix in `train_yolov5_cclkd_full.py`.

## Summary

| Experiment | Mode | GPU | Epoch | AP50 | AP50:95 | Delta AP50 vs train.py | Delta AP vs train.py |
|---|---|---:|---:|---:|---:|---:|---:|
| standard train.py det-only | `det_only_same_trainpy` | 0 | 79 | 0.57056 | 0.30964 | 0.00000 | 0.00000 |
| custom det-only, scaled fix | `det_only_same_trainer` | 1 | 79 | 0.56807 | 0.30862 | -0.00249 | -0.00102 |
| two-branch no KD, scaled fix | `two_branch_no_kd` | 3 | 79 | 0.56609 | 0.30616 | -0.00447 | -0.00348 |

The custom detector-only trainer is now aligned with the standard YOLOv5
`train.py` baseline under the same b32/e80 protocol. The remaining AP gap is
0.00102, well below the 0.02 acceptance threshold.

The two-branch no-KD run is also close to the standard baseline, with AP gap
0.00348. This supports the diagnosis that the earlier custom-trainer collapse
was caused mainly by missing YOLOv5 loss-gain scaling, not by the two-branch
structure itself.

## Evidence Files

Each run subdirectory contains:

- `results.csv`: full 80-epoch metric curve.
- `run_meta.txt`: launcher metadata.
- `command.sh`: exact launch command.
- `hyp.yaml` and `opt.yaml`: training configuration snapshots.
- `nohup_tail.txt`: readable tail of the training log.
- `nohup.log.gz`: compressed full training log.
- `cclkd_yolov5_diagnostics.csv`: custom trainer diagnostics when applicable.

The standard train.py run also includes YOLOv5 plots (`results.png`, PR/F1/P/R
curves, confusion matrix).

Checkpoint weights and TensorBoard event files are intentionally excluded.

## Notes

- The train.py det-only run ended with a PIL/TensorBoard image logging
  compatibility error after metrics and checkpoints had already been written.
- The custom runs ended with a PyTorch 2.6 `torch.load(weights_only=True)`
  compatibility error during `strip_optimizer()`. The final metrics were already
  written; checkpoints were excluded from this archive.
- The two-branch run was requested for GPU2, but GPU2 had insufficient free VRAM.
  It was launched on GPU3 and this is recorded in its `run_meta.txt`.
