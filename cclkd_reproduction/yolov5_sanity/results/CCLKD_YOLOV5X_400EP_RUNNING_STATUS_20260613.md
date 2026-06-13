# CCLKD YOLOv5x 400epoch Running Status (2026-06-13)

90 服务器快照时间：`2026-06-13 23:42:58 +08`。

本次为 P0 running snapshot：Full 和 ATKD-only 均已超过 epoch 150；CCL-only 和 det-only 已早于本次快照超过 150。四条主实验均继续运行，未自动重启，未停止。

## 当前结果

| 实验 | GPU | 进度 | AP50 | AP | 同 epoch det-only AP | ΔAP vs det-only | KD/det ratio | 显存 used/free | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Full CCLKD | 0 | 185/399 | 0.59976 | 0.32287 | 0.31722 | 0.00565 | 0.43622 | 16275 / 7850 MiB | running |
| ATKD-only | 1 | 151/399 | 0.57164 | 0.30139 | 0.29451 | 0.00688 | 0.05804 | 20139 / 3986 MiB | running |
| CCL-only | 3 | 237/399 | 0.62667 | 0.35324 | 0.34817 | 0.00507 | 0.40220 | 23319 / 806 MiB | running |
| Det-only baseline | 5 | 268/399 | 0.64535 | 0.36632 | baseline |  | 0.00000 | 10715 / 13411 MiB | running |

## 诊断表

| 实验 | student box/obj/cls | teacher box/obj/cls | KD total | LLD | FLD | RLD | CCL | COP+ | temp | feature ok | NaN/Inf |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full CCLKD | 0.05083/0.00580/0.00170 | 0.04147/0.00509/0.00112 | 0.80010 | 0.01031 | 0.03280 | 0.06335 | 0.69364 | 0.98589 | 2.85704 | 1.0 | 0.0 |
| ATKD-only | 0.05259/0.00601/0.00187 | 0.04317/0.00531/0.00119 | 0.11144 | 0.01107 | 0.03155 | 0.06882 | 0.00000 | 0.98226 | 2.87130 | 1.0 | 0.0 |
| CCL-only | 0.04799/0.00559/0.00139 | 0.04014/0.00494/0.00085 | 0.69407 | 0.00000 | 0.00000 | 0.00000 | 0.69407 | 0.98798 | 2.84596 | 1.0 | 0.0 |
| Det-only baseline | 0.04633/0.00549/0.00130 | 0.00000/0.00000/0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |  |  |  | 0.0 |

## 当前判断

- Current training is numerically stable.
- Feature capture works and no NaN/Inf is detected.
- Same-epoch AP deltas are positive but still small: Full ΔAP=0.00565, ATKD-only ΔAP=0.00688, CCL-only ΔAP=0.00507.
- ATKD-only appears more efficient than CCL-only in terms of gain per KD/det ratio: ATKD-only ratio=0.05804, CCL-only ratio=0.40220.
- CCL-only has high KD pressure but very small AP gain, suggesting possible low-efficiency CCL.
- Do not modify loss or launch sweeps until the 200epoch aligned snapshot is available.
- 200epoch 规则 A 的早期信号：Full ΔAP >= +0.005；到 200 对齐前继续跑，不改代码。

## 日志关键字摘要

- `paper_full`：只有 Albumentations/Pydantic 初始化 warning
- `paper_atkd_only`：只有 Albumentations/Pydantic 初始化 warning
- `paper_ccl_only`：只有 Albumentations/Pydantic 初始化 warning
- `det_only_same_trainer`：只有 Albumentations/Pydantic 初始化 warning

## 证据归档

证据包路径：`cclkd_reproduction/yolov5_sanity/results/scalingfix_paper_components_400ep_20260613/`

每个 run 已更新：`results.csv`、`cclkd_yolov5_diagnostics.csv`、`nohup_tail_300.log`、`nohup_error_grep_tail.log`、`run_meta.txt`、`command.sh`、`opt.yaml`、`hyp.yaml`。

未上传内容：checkpoint 权重、TensorBoard event 文件、完整 nohup 大日志。
