# CCLKD YOLOv5x GPU2 smoke and det-only evidence

Snapshot pulled from server `ladd90` (`/mnt/dataY/ydf/projects/LADD_public`) on 2026-06-12 CST.

This archive records the YOLOv5x CCLKD smoke and the first 80-epoch mechanism check that were run on GPU2. It is a compact evidence package: checkpoint weights and TensorBoard event files are intentionally excluded.

## Included runs

| Run directory | Mode | Batch | Epochs | GPU | Status |
|---|---|---:|---:|---:|---|
| `paper_full_smoke_b32_1e1b_final/` | `paper_full` | 32 | 1 | 2 | completed smoke; checker passed |
| `det_only_same_trainer_b32_e80_r1/` | `det_only_same_trainer` | 32 | 80 | 2 | completed 80 epochs |

## Key results

### paper_full smoke

The smoke checker passed:

| Metric | Value |
|---|---:|
| epoch | 0 |
| feature_capture_ok | 1.0 |
| student_feature_levels | 3.0 |
| teacher_feature_levels | 3.0 |
| cop_valid_candidates | 1046 |
| cop_positive_candidates | 466 |
| cop_positive_ratio | 0.4455066922 |
| weighted_kd_to_student_det_ratio | 0.0 |
| nan_or_inf_detected | 0.0 |

The weighted KD ratio is `0.0` because this smoke ran at epoch 0 with the configured KD warmup (`kd_warmup_epochs=3`).

### det_only 80 epoch mechanism check

Final validation row at epoch 79:

| P | R | AP50 | AP50-95/AP |
|---:|---:|---:|---:|
| 0.46574 | 0.34265 | 0.33064 | 0.13490 |

This run verifies the custom YOLOv5x trainer path without teacher/KD. It is not a CCLKD method result; it is a mechanism check before launching long `paper_full` runs.

## Current remote status

At pull time (`remote_status_20260612.txt`):

- No active `train_yolov5_cclkd_full.py` / YOLOv5x CCLKD process was found.
- GPU2 was occupied by non-LADD processes, not by this experiment.
- `screen` is unavailable on the server image, so runs were checked through `pid.txt`, `ps`, logs, and result files.

## Files

Each run subdirectory contains:

- `command.sh`: exact launch command.
- `run_meta.txt`: run metadata.
- `hyp.yaml` / `opt.yaml`: YOLOv5 training configuration snapshots.
- `results.csv`: validation and training metrics.
- `cclkd_yolov5_diagnostics.csv`: CCLKD diagnostic metrics.
- `pid.txt`: original launched PID.

Log evidence:

- `paper_full_smoke_b32_1e1b_final/nohup.log`: full smoke log, small enough to track.
- `det_only_same_trainer_b32_e80_r1/nohup_tail_500.log`: final 500 lines of the 80-epoch run. The full remote `nohup.log` was about 24 MB and was not tracked to keep the public evidence package compact.

Excluded artifacts:

- `weights/*.pt` checkpoint files.
- TensorBoard `events.out.tfevents.*` files.
- Full 24 MB det-only `nohup.log`.
