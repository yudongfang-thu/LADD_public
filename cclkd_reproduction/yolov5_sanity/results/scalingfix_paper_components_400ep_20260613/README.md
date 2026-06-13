# CCLKD YOLOv5x 400epoch Running Status (2026-06-14)

90 服务器快照时间：`2026-06-14 02:33:14 +0800`。

本次更新继续跟踪 4 个 YOLOv5x scaling-fix b32/s0/400ep 主实验，并补充 Full run 的 online teacher RGB validation。当前未修改 loss、未启动 sweep、未停止主实验。

## 当前结果

| 实验 | GPU | 进度 | AP50 | AP | 同 epoch det-only AP | ΔAP vs det-only | KD/det ratio | 显存 used/free | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Full CCLKD | 0 | 216/399 | 0.62098 | 0.34303 | 0.33524 | 0.00779 | 0.45018 | 16275 / 7850 MiB | running |
| ATKD-only | 1 | 168/399 | 0.58778 | 0.31239 | 0.30484 | 0.00755 | 0.05871 | 20139 / 3986 MiB | running |
| CCL-only | 3 | 267/399 | 0.64386 | 0.37259 | 0.36548 | 0.00711 | 0.42223 | 23950 / 176 MiB | running |
| Det-only baseline | 5 | 303/399 | 0.66992 | 0.38651 | baseline |  | 0.00000 | 11851 / 12275 MiB | running |

## Loss Contribution

| 实验 | epoch | student det display sum | ATKD | CCL | KD total | ATKD share | CCL share | KD/det ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full CCLKD | 216 | 0.05553 | 0.09915 | 0.69220 | 0.79135 | 0.12529 | 0.87471 | 0.45018 |
| ATKD-only | 168 | 0.05956 | 0.11141 | 0.00000 | 0.11141 | 1.00000 | 0.00000 | 0.05871 |
| CCL-only | 267 | 0.05228 | 0.00000 | 0.69374 | 0.69374 | 0.00000 | 1.00000 | 0.42223 |
| Det-only baseline | 303 | 0.05008 | 0.00000 | 0.00000 | 0.00000 |  |  | 0.00000 |

## Online Teacher Evaluation

| model/branch | epoch | modality | AP50 | AP |
|---|---:|---|---:|---:|
| Full run online teacher | 214 | RGB | 0.80073 | 0.42392 |
| Full run student | 214 | SAR | 0.61927 | 0.34123 |
| Det-only student | 214 | SAR | 0.60733 | 0.33427 |
| Independent RGB YOLOv5x baseline | 399 | RGB | 0.86506 | 0.52414 |

- Online teacher 明显强于 SAR student：teacher-student AP gap = `0.08269`。
- Full student 相对同 epoch det-only 只提升 `0.00696` AP，约回收 teacher-det gap 的 `0.07763`。
- 原文 CCLKD 是 online dual-branch teacher-student 协议：teacher 使用 easy-to-detect modality 及检测损失训练，student 使用 hard-to-detect modality 并接收 ATKD/CCL。

## 当前判断

- Training is stable; four main 400epoch runs remain active.
- Feature capture works and no NaN/Inf is detected in paper runs.
- All paper runs still show positive same-epoch AP delta, but gains remain small.
- Full is positive at epoch 216, but Full is not yet clearly better than ATKD-only at common epochs.
- CCL dominates Full KD pressure: latest Full KD consists mostly of CCL, while AP gain remains small.
- The online teacher is not weak, so the main bottleneck is likely KD/CCL transfer efficiency rather than absence of teacher signal.
- Do not modify loss or launch sweep before the 200/250 aligned snapshots are archived.

## Artifacts

- `summary.csv`
- `milestone_component_comparison.csv/md`
- `loss_contribution_latest.csv/md`
- `teacher_eval_online_full_latest.csv/md`
- `figures/yolov5x_cclkd_*.png/pdf`
- Per-run `results.csv`, `cclkd_yolov5_diagnostics.csv`, `nohup_tail_300.log`, `nohup_error_grep_tail.log`.
