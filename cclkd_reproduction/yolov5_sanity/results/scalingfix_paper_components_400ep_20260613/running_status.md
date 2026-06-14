# CCLKD YOLOv5x 400epoch Running Status (2026-06-14)

90 服务器快照时间：`2026-06-14 18:27:45 +0800`。

本次更新继续跟踪 4 个 YOLOv5x scaling-fix b32/s0/400ep 主实验，并补充 ATKD/CCL component decomposition。当前未修改 loss、未启动 sweep、未停止主实验。

## 当前结果

| 实验 | GPU | 进度 | AP50 | AP | 同 epoch det-only AP | ΔAP vs det-only | KD/det ratio | 显存 used/free | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Full CCLKD | 0 | 392/399 | 0.727320 | 0.444830 | 0.436880 | 0.007950 | 0.613547 | 14626 / 9499 MiB | running |
| ATKD-only | 1 | 287/399 | 0.676370 | 0.389940 | 0.376800 | 0.013140 | 0.057651 | 21368 / 2757 MiB | running |
| CCL-only | 3 | 399/399 | 0.724110 | 0.443480 | 0.439880 | 0.003600 | 0.556275 | 18670 / 5455 MiB | completed |
| Det-only baseline | 5 | 399/399 | 0.722990 | 0.439880 | baseline |  | 0.000000 | 20281 / 3844 MiB | completed |

## Exact-Epoch Component Milestones

Exact epoch matches only; `pending` means at least one run has not reached that epoch in the local archive.

| epoch | det_only_ap | atkd_ap | atkd_delta_ap | ccl_ap | ccl_delta_ap | full_ap | full_delta_ap | full_minus_atkd_ap | full_minus_ccl_ap | best_component_by_ap | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | 0.25132 | 0.25550 | 0.00418 | 0.25521 | 0.00389 | 0.25640 | 0.00508 | 0.00090 | 0.00119 | full | pre_200_snapshot |
| 100 | 0.26324 | 0.26782 | 0.00458 | 0.26603 | 0.00279 | 0.26901 | 0.00577 | 0.00119 | 0.00298 | full | pre_200_snapshot |
| 125 | 0.27807 | 0.28386 | 0.00579 | 0.28164 | 0.00357 | 0.28352 | 0.00545 | -0.00034 | 0.00188 | atkd | pre_200_snapshot |
| 150 | 0.29367 | 0.30024 | 0.00657 | 0.29892 | 0.00525 | 0.29997 | 0.00630 | -0.00027 | 0.00105 | atkd | pre_200_snapshot |
| 185 | 0.31722 | 0.32360 | 0.00638 | 0.32023 | 0.00301 | 0.32287 | 0.00565 | -0.00073 | 0.00264 | atkd | pre_200_snapshot |
| 200 | 0.32520 | 0.33383 | 0.00863 | 0.32936 | 0.00416 | 0.33255 | 0.00735 | -0.00128 | 0.00319 | atkd | aligned_snapshot |
| 250 | 0.35625 | 0.36522 | 0.00897 | 0.36116 | 0.00491 | 0.36360 | 0.00735 | -0.00162 | 0.00244 | atkd | aligned_snapshot |
| 300 | 0.38416 | pending | pending | 0.39227 | 0.00811 | 0.39340 | 0.00924 | pending | 0.00113 | full | pending |
| 350 | 0.41471 | pending | pending | 0.42223 | 0.00752 | 0.42415 | 0.00944 | pending | 0.00192 | full | pending |
| 399 | 0.43988 | pending | pending | 0.44348 | 0.00360 | pending | pending | pending | pending | ccl | pending |

## Loss / Component Contribution

| 实验 | epoch | student det sum | LLD | FLD | RLD | CCL | KD total | ATKD share | CCL share | KD/det ratio | COP+ | T mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full CCLKD | 392 | 0.040215 | 0.006232 | 0.034696 | 0.033274 | 0.693717 | 0.767919 | 0.096627 | 0.903373 | 0.613547 | 0.992846 | 2.788691 |
| ATKD-only | 287 | 0.050776 | 0.008879 | 0.033811 | 0.050182 | 0.000000 | 0.092872 | 1.000000 | 0.000000 | 0.057651 | 0.988012 | 2.830132 |
| CCL-only | 399 | 0.040188 | 0.000000 | 0.000000 | 0.000000 | 0.693987 | 0.693987 | 0.000000 | 1.000000 | 0.556275 | 0.993671 | 2.784254 |
| Det-only baseline | 399 | 0.040102 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  |  | 0.000000 |  |  |

## Online Teacher Evaluation

| model/branch | epoch | modality | AP50 | AP |
|---|---:|---|---:|---:|
| Full run online teacher | 214 | RGB | 0.80073 | 0.42392 |
| Full run student | 214 | SAR | 0.61927 | 0.34123 |
| Det-only student | 214 | SAR | 0.60733 | 0.33427 |
| Independent RGB YOLOv5x baseline | 399 | RGB | 0.86506 | 0.52414 |

## 当前判断

- Training remains numerically stable; feature capture works and no NaN/Inf is detected in paper runs.
- Det-only and CCL-only have reached epoch 399 in the local archive; Full is near completion and ATKD-only is still running.
- ATKD-only shows weak but consistent positive same-epoch AP gain with low KD/det ratio, suggesting under-coupled but useful ATKD signal.
- CCL-only has very small final AP gain despite high KD/det ratio and CCL loss near log(2), suggesting low-efficiency CCL in the current implementation.
- Full CCLKD is positive vs det-only but is not consistently better than ATKD-only at common 200/250 epochs; CCL dominates Full weighted KD pressure.
- Do not modify loss or launch sweeps before offline component diagnostics, gradient/cosine probes, and CCL pos-vs-neg similarity probes support a specific intervention.

## Artifacts

- `summary.csv`
- `milestone_component_comparison.csv/md`
- `component_decomposition_timeseries.csv`
- `component_decomposition_snapshot.md`
- `loss_contribution_latest.csv/md`
- `teacher_eval_online_full_latest.csv/md`
- `figures/yolov5x_cclkd_*.png/pdf`
- Per-run `results.csv`, `cclkd_yolov5_diagnostics.csv`, `nohup_tail_300.log`, `nohup_error_grep_tail.log`.
