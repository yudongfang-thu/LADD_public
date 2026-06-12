# YOLOv5x CCLKD Batch32 80-Epoch Queue Diagnostics

Archive timestamp: 2026-06-12 11:52 +08. Remote server: 90 (`inspur-NF5468M5`).

This archive captures the four-run YOLOv5x CCLKD batch32/80epoch queue. Wave 1 finished before archiving; wave 2 was still running, so its metrics are a partial snapshot and must not be treated as final.

## Runs

| Wave | Mode | Status | GPU | Epochs recorded | Latest mAP50 | Latest mAP50-95 | Best mAP50 | KD ratio | COP+ ratio | Feature capture | NaN/Inf |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| wave1 | `det_only_same_trainer` | completed | 1 | 80 | 0.33064 | 0.13490 | 0.33064 | 0.00000 |  |  | 0.0 |
| wave1 | `two_branch_no_kd` | completed | 3 | 80 | 0.32165 | 0.13479 | 0.32165 | 0.00000 |  |  | 0.0 |
| wave2 | `paper_atkd_only` | running_snapshot | 1 | 18 | 0.15905 | 0.05288 | 0.18724 | 0.04873 | 0.96049 | 1.0 | 0.0 |
| wave2 | `paper_full` | running_snapshot | 3 | 18 | 0.13904 | 0.04236 | 0.15835 | 0.14797 | 0.96061 | 1.0 | 0.0 |

## Completed Wave-1 Comparison

- `det_only_same_trainer` final epoch 79: mAP50=0.33064, mAP50-95=0.13490, delta vs det-only mAP50=+0.00000, delta mAP50-95=+0.00000.
- `two_branch_no_kd` final epoch 79: mAP50=0.32165, mAP50-95=0.13479, delta vs det-only mAP50=-0.00899, delta mAP50-95=-0.00011.

Interpretation: the two-branch no-KD control is close to the det-only custom-trainer baseline at 80 epochs (mAP50-95 essentially tied, mAP50 lower by about 0.009). This suggests the two-branch machinery alone is not causing a severe collapse in this setting.

## Wave-2 Running Snapshot

- `paper_atkd_only` snapshot at epoch 17: mAP50=0.15905, mAP50-95=0.05288, weighted KD/student-det ratio=0.04873, COP positive ratio=0.96049, feature_capture_ok=1.0, nan_or_inf=0.0.
- `paper_full` snapshot at epoch 17: mAP50=0.13904, mAP50-95=0.04236, weighted KD/student-det ratio=0.14797, COP positive ratio=0.96061, feature_capture_ok=1.0, nan_or_inf=0.0.

Interpretation: both KD modes have valid feature capture and no NaN/Inf flag in the diagnostics. `paper_full` has an expected CCL component near 0.69 and a higher weighted KD/detection ratio than ATKD-only. These are mechanism checks, not final accuracy conclusions because wave 2 is incomplete in this snapshot.

## Included Evidence

- Per-run `run_meta.txt`, `command.sh`, `opt.yaml`, `hyp.yaml`.
- Per-run `results.csv` and `cclkd_yolov5_diagnostics.csv`.
- Per-run `nohup_head_220.log`, `nohup_tail_1000.log`, and `nohup_key_events.log`.
- Queue script/log and remote status snapshot.
- `summary.csv` and `summary.json` for quick parsing.

## Excluded Artifacts

- Checkpoint weights are excluded by repository policy and `.gitignore`.
- TensorBoard event files are excluded. They are small in this run but not needed for the evidence package.
- Full `nohup.log` files are excluded because completed runs are about 24-25 MB each; head/tail/key-event extracts are included instead.

