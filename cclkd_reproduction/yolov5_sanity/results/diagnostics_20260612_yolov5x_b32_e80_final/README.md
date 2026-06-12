# YOLOv5x CCLKD Batch32 80-Epoch Final Diagnostics

Snapshot time: 2026-06-12 16:20 CST on server 90.

This archive captures the completed batch32 / 80 epoch YOLOv5x CCLKD reproduction gate and mechanism-check runs. It supersedes the earlier partial queue archive:

`cclkd_reproduction/yolov5_sanity/results/diagnostics_20260612_yolov5x_b32_e80_queue/`

## Final Metrics

| run | trainer / mode | status | final epoch | AP50 | AP50-95/AP | delta AP vs custom det-only | delta AP vs standard train.py |
|---|---|---|---:|---:|---:|---:|---:|
| `standard_train_py` | standard YOLOv5 `train.py` SAR baseline | complete | 79 | 0.57056 | 0.30964 | +0.17474 | +0.00000 |
| `det_only_same_trainer` | custom trainer, detector only | complete | 79 | 0.33064 | 0.13490 | +0.00000 | -0.17474 |
| `two_branch_no_kd` | custom trainer, student+teacher branches, no KD | complete | 79 | 0.32165 | 0.13479 | -0.00011 | -0.17485 |
| `paper_atkd_only` | custom trainer, LLD+FLD+RLD / ATKD only | complete | 79 | 0.35592 | 0.15149 | +0.01659 | -0.15815 |
| `paper_full` | custom trainer, ATKD + CCL | complete | 79 | 0.34732 | 0.14520 | +0.01030 | -0.16444 |

## Mechanism Diagnostics

| run | LLD | FLD | RLD | CCL | feature capture | NaN/Inf | weighted KD / student det |
|---|---:|---:|---:|---:|---:|---:|---:|
| `det_only_same_trainer` | 0.00000 | 0.00000 | 0.00000 | 0.00000 | - | 0.0 | 0.00000 |
| `two_branch_no_kd` | 0.00000 | 0.00000 | 0.00000 | 0.00000 | - | 0.0 | 0.00000 |
| `paper_atkd_only` | 0.01268 | 0.14180 | 0.08659 | 0.00000 | 1.0 | 0.0 | 0.05749 |
| `paper_full` | 0.01283 | 0.14597 | 0.08838 | 0.69349 | 1.0 | 0.0 | 0.22570 |

## Key Observations

1. The standard YOLOv5 `train.py` baseline is much stronger than the custom trainer det-only control: AP50-95/AP is `0.30964` vs `0.13490`, a gap of `+0.17474`.
2. The two-branch no-KD control is essentially tied with custom det-only on AP (`0.13479` vs `0.13490`), so the online teacher branch itself is not the main source of the gap.
3. ATKD-only improves over custom det-only by `+0.01659` AP and is the best custom-trainer run in this 80 epoch batch32 set.
4. Full CCLKD improves over custom det-only by `+0.01030` AP, but is lower than ATKD-only by `-0.00629` AP.
5. Full CCLKD has an active CCL term (`0.69349`) and higher weighted KD / student detection ratio (`0.22570`) than ATKD-only (`0.05749`), so the mode difference is active.

## Interpretation

These results should be treated as a YOLOv5x trainer-alignment diagnostic rather than a final CCLKD method conclusion. The standard YOLOv5 `train.py` baseline is far above the custom det-only trainer under the same 80 epoch batch32 budget. Until the custom trainer baseline approaches standard `train.py`, low absolute AP for `paper_atkd_only` or `paper_full` cannot be attributed cleanly to the CCLKD loss design.

Relative to the custom trainer baseline, KD mechanisms are not collapsed: ATKD-only and full both improve AP. However, CCL does not add a positive gain on top of ATKD in this run.

## Included Evidence

Each run directory includes:

- `results.csv`
- `command.sh`
- `hyp.yaml`
- `opt.yaml`
- `run_meta.txt`
- `pid.txt`
- `nohup_head.txt`
- `nohup_tail.txt`
- `nohup_key_events.txt`
- `file_sizes.txt`
- `cclkd_yolov5_diagnostics.csv` for custom trainer runs

The standard YOLOv5 `train.py` run also includes lightweight result plots copied from YOLOv5 (`results.png`, `F1_curve.png`, `P_curve.png`, `R_curve.png`, `PR_curve.png`).

Excluded intentionally:

- checkpoint weights (`*.pt`)
- TensorBoard event files
- full `nohup.log` files

The full logs were excluded from Git because each is about 17-24 MB; compact head/tail/key-event extracts are included instead. Checkpoints and event files were also excluded to keep the public repository lightweight.

## Known Log Tail Issues

- Standard YOLOv5 `train.py` completed `results.csv` through epoch 79, then hit a `PIL.Image.Resampling` error during post-training plotting/saving. The final metrics are present.
- Custom trainer runs completed `results.csv` through epoch 79, then hit PyTorch `weights_only` checkpoint load/strip errors in the tail. The final metrics and diagnostics are present.
