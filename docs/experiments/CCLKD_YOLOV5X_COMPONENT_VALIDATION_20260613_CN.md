# CCLKD YOLOv5x Scaling-Fix Component Validation (2026-06-13)

本记录整理 90 服务器上 YOLOv5x CCLKD scaling-fix 后的 80epoch 组件验证结果，并归档已经自动接上的 400epoch 占位/长跑实验快照。实验均为 OGSOD HBB SAR，`img=256`，`batch=32`，`seed=0`，YOLOv5x student/teacher 初始化为 `external/yolov5/yolov5x.pt`。

## 结论摘要

1. custom trainer 的 det-only scaling 修复后已经与 standard YOLOv5 `train.py` 对齐：custom det-only AP=0.30862，standard train.py AP=0.30964，差值约 +0.00102。
2. 80epoch 组件验证中，Full CCLKD 为 AP50=0.57566、AP=0.30965，相对 custom det-only 为 AP50 +0.00759、AP +0.00103；收益很小，但方向为正。
3. ATKD-only 为 AP50=0.57097、AP=0.30692；CCL-only 为 AP50=0.56441、AP=0.30391。单独看 CCL-only 低于 det-only，说明 CCL 单组件在 80epoch 不构成正收益。
4. 三个 80epoch run 的 `feature_capture_ok=1.0`，`nan_or_inf_detected=0.0`，末轮 `cop_positive_ratio` 约 0.990，未见数值异常。
5. 400epoch 的 Full/ATKD-only/CCL-only 已在 GPU0/GPU1/GPU3 上接续运行，用于观察长训练曲线；当前快照只作为进度记录，不作为最终结论。
6. 日志错误关键字检索只命中 Albumentations 初始化 warning，不是训练崩溃、OOM、NaN 或 Inf。

## 80epoch 主结果

| 实验 | mode | epoch | AP50 | AP | delta AP50 vs custom det-only | delta AP vs custom det-only |
|---|---|---:|---:|---:|---:|---:|
| Standard YOLOv5 train.py SAR baseline | train.py | 79 | 0.57056 | 0.30964 | 0.00249 | 0.00102 |
| Custom det-only scaled | det_only_same_trainer | 79 | 0.56807 | 0.30862 | 0.00000 | 0.00000 |
| Two-branch no KD scaled | two_branch_no_kd | 79 | 0.56609 | 0.30616 | -0.00198 | -0.00246 |
| ATKD-only | paper_atkd_only | 79 | 0.57097 | 0.30692 | 0.00290 | -0.00170 |
| CCL-only | paper_ccl_only | 79 | 0.56441 | 0.30391 | -0.00366 | -0.00471 |
| Full CCLKD | paper_full | 79 | 0.57566 | 0.30965 | 0.00759 | 0.00103 |

## 80epoch 诊断

| 实验 | mode | epoch/target | AP50 | AP | feature ok | NaN/Inf | COP positive ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| ATKD-only | paper_atkd_only | 79/80 | 0.57097 | 0.30692 | 1.0 | 0.0 | 0.99013 |
| CCL-only | paper_ccl_only | 79/80 | 0.56441 | 0.30391 | 1.0 | 0.0 | 0.99008 |
| Full CCLKD | paper_full | 79/80 | 0.57566 | 0.30965 | 1.0 | 0.0 | 0.99013 |

## 400epoch 当前快照

| 实验 | mode | epoch/target | AP50 | AP | feature ok | NaN/Inf | COP positive ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| Full CCLKD | paper_full | 5/400 | 0.15676 | 0.05156 | 1.0 | 0.0 | 0.95362 |
| ATKD-only | paper_atkd_only | 19/400 | 0.32452 | 0.12968 | 1.0 | 0.0 | 0.96894 |
| CCL-only | paper_ccl_only | 12/400 | 0.28832 | 0.11234 | 1.0 | 0.0 | 0.96836 |

这些 400epoch 数值是 2026-06-13 早间同步时的中途快照，不能与 80epoch 最终值直接比较。它们的作用是确认自动排队已经转入实际训练，并保留后续追踪的起点。

## 证据归档

- 机器可读汇总：`cclkd_reproduction/yolov5_sanity/results/diagnostics_20260613_yolov5x_scaledfix_components_e80_400queue/summary_metrics.csv`
- 文档侧汇总 CSV：`docs/experiments/cclkd_yolov5x_component_validation_20260613.csv`
- 每个 run 的 `results.csv`、`cclkd_yolov5_diagnostics.csv`、`hyp.yaml`、`opt.yaml`、`run_meta.txt`、`command.sh`、`pid.txt`、`nohup_tail_240.log`、`nohup_error_grep_tail.log` 已归档到：
  `cclkd_reproduction/yolov5_sanity/results/diagnostics_20260613_yolov5x_scaledfix_components_e80_400queue/runs/`
- 远程 GPU/进程快照：`cclkd_reproduction/yolov5_sanity/results/diagnostics_20260613_yolov5x_scaledfix_components_e80_400queue/remote_status_snapshot_20260613_0705.txt`

## 后续判断

当前证据支持“scaling-fix 后 custom trainer 已对齐，Full CCLKD 在 80epoch 单 seed 上不再出现明显负收益”。但 Full 的 AP 收益仅约 +0.001，不能作为强结论。后续需要看 400epoch Full 是否稳定超过 det-only/standard 基线，以及 CCL-only 是否持续低于基线。
