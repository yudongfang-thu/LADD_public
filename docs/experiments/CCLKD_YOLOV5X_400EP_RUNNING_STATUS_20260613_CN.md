# CCLKD YOLOv5x 400epoch Running Status (2026-06-14)

90 服务器快照时间：`2026-06-14 00:13:27 +08`。

本次更新继续跟踪 4 个 YOLOv5x scaling-fix b32/s0/400ep 主实验。当前不改 loss、不启动 sweep、不新增实验、不停止主实验。

## 当前结果

| 实验 | GPU | 进度 | AP50 | AP | 同 epoch det-only AP | ΔAP vs det-only | KD/det ratio | 显存 used/free | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Full CCLKD | 0 | 190/399 | 0.60260 | 0.32503 | 0.32033 | 0.00470 | 0.44153 | 16275 / 7850 MiB | running |
| ATKD-only | 1 | 154/399 | 0.57474 | 0.30333 | 0.29610 | 0.00723 | 0.06036 | 20139 / 3986 MiB | running |
| CCL-only | 3 | 242/399 | 0.63011 | 0.35620 | 0.35128 | 0.00492 | 0.40563 | 23319 / 806 MiB | running |
| Det-only baseline | 5 | 273/399 | 0.64690 | 0.36961 | baseline |  | 0.00000 | 10719 / 13407 MiB | running |

## 诊断表

| 实验 | student box/obj/cls | teacher box/obj/cls | KD total | LLD | FLD | RLD | CCL | COP+ | temp | feature ok | NaN/Inf |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full CCLKD | 0.05034/0.00581/0.00160 | 0.04161/0.00514/0.00114 | 0.79898 | 0.01054 | 0.03292 | 0.06164 | 0.69388 | 0.98487 | 2.85794 | 1.0 | 0.0 |
| ATKD-only | 0.05346/0.00597/0.00177 | 0.04358/0.00529/0.00122 | 0.11693 | 0.01117 | 0.03840 | 0.06736 | 0.00000 | 0.98302 | 2.87086 | 1.0 | 0.0 |
| CCL-only | 0.04755/0.00561/0.00145 | 0.03948/0.00493/0.00094 | 0.69395 | 0.00000 | 0.00000 | 0.00000 | 0.69395 | 0.98741 | 2.84218 | 1.0 | 0.0 |
| Det-only baseline | 0.04563/0.00527/0.00126 | 0.00000/0.00000/0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |  |  |  | 0.0 |

## 当前判断

- Training is stable.
- Feature capture works and no NaN/Inf is detected.
- All paper runs show positive same-epoch AP delta.
- Gains are still small: Full ΔAP=0.00470, ATKD-only ΔAP=0.00723, CCL-only ΔAP=0.00492.
- Full is not yet clearly better than ATKD-only at common epochs. At exact epoch 150, `full_minus_atkd_ap=-0.00027`.
- ATKD-only appears more efficient than CCL-only in terms of gain per KD/det ratio: ATKD-only ratio=0.06036, CCL-only ratio=0.40563.
- CCL has high KD/det ratio but limited AP gain; keep it marked as possible low-efficiency CCL.
- Do not modify loss or launch sweep before 200/250 aligned snapshots.
- P0 已完成；下一次刷新节点是 Full 和 ATKD-only 都到 epoch 200 后的 200epoch 对齐快照。当前不应用 200/250 最终判断。

## Milestone Table

新增固定 epoch 对齐表，严格 exact epoch 对齐，缺失 epoch 写 `pending`，不使用 nearest epoch：

- `milestone_component_comparison.csv`
- `milestone_component_comparison.md`

## Planning Note

暂不启动 sweep。只有 200/250 对齐后同时出现 ATKD-only 明显高于 det-only、Full 低于或基本等于 ATKD-only、CCL-only weak gain、Full weighted KD/det ratio 明显高于 ATKD-only，才准备以下候选：CCL weight 0.25、CCL weight 0.5、KD warmup 10。

## 日志关键字摘要

- `paper_full`：只有 Albumentations/Pydantic 初始化 warning
- `paper_atkd_only`：只有 Albumentations/Pydantic 初始化 warning
- `paper_ccl_only`：只有 Albumentations/Pydantic 初始化 warning
- `det_only_same_trainer`：只有 Albumentations/Pydantic 初始化 warning

## 证据归档

证据包路径：`cclkd_reproduction/yolov5_sanity/results/scalingfix_paper_components_400ep_20260613/`

每个 run 已更新：`results.csv`、`cclkd_yolov5_diagnostics.csv`、`nohup_tail_300.log`、`nohup_error_grep_tail.log`、`run_meta.txt`、`command.sh`、`opt.yaml`、`hyp.yaml`。

未上传内容：checkpoint 权重、TensorBoard event 文件、完整 nohup 大日志。
