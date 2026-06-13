# CCLKD YOLOv5x 400epoch Running Status (2026-06-13)

90 服务器快照时间：`2026-06-13 19:10:41 +08`。

本快照覆盖 4 个正在运行的 YOLOv5x scaling-fix b32/s0/400ep 实验：`paper_full`、`paper_atkd_only`、`paper_ccl_only` 和 `det_only_same_trainer` baseline。

## 当前结果

| 实验 | GPU | 进度 | AP50 | AP | 同 epoch det-only AP | ΔAP vs det-only | 状态 | 显存 used/free |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Full CCLKD | 0 | 134/399 | 0.55657 | 0.28908 | 0.28303 | 0.00605 | running | 16275 / 7850 MiB |
| ATKD-only | 1 | 125/399 | 0.55339 | 0.28386 | 0.27807 | 0.00579 | running | 19429 / 4696 MiB |
| CCL-only | 3 | 188/399 | 0.59320 | 0.32147 | 0.31901 | 0.00246 | running | 22996 / 1129 MiB |
| Det-only baseline | 5 | 207/399 | 0.60098 | 0.32980 | baseline |  | running | 9061 / 15064 MiB |

## 诊断表

| 实验 | student box/obj/cls | KD total | LLD | FLD | RLD | CCL | COP+ | temp | KD/det ratio | feature ok | NaN/Inf |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full CCLKD | 0.05378/0.00599/0.00179 | 0.81213 | 0.01103 | 0.03806 | 0.06936 | 0.69369 | 0.98374 | 2.86880 | 0.41941 | 1.0 | 0.0 |
| ATKD-only | 0.05467/0.00607/0.00198 | 0.11507 | 0.01164 | 0.03375 | 0.06968 | 0.00000 | 0.98156 | 2.87636 | 0.05768 | 1.0 | 0.0 |
| CCL-only | 0.05051/0.00584/0.00174 | 0.69392 | 0.00000 | 0.00000 | 0.00000 | 0.69392 | 0.98573 | 2.85285 | 0.37916 | 1.0 | 0.0 |
| Det-only baseline | 0.04967/0.00574/0.00148 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |  |  | 0.00000 |  | 0.0 |

## 当前判断

1. 绝对指标上，`det_only_same_trainer` 当前最高，但它已经跑到 epoch 207，训练进度领先，所以正式比较必须看同 epoch delta。
2. 当前快照中，三条 paper run 在同 epoch AP 上都略高于 det-only：Full ΔAP=0.00605，ATKD-only ΔAP=0.00579，CCL-only ΔAP=0.00246。
3. 这些正 delta 仍然很小，不能提前宣称 400epoch 成功；需要继续按 epoch 150/200/250/300/350/400 节点观察。
4. 当前 paper_full 还未到 epoch 150；stop rule 中关于 epoch 150/200 的相对 AP 条件暂不触发。数值诊断上 feature_capture_ok=1.0，nan_or_inf_detected=0.0。
5. 日志关键字检索未发现训练崩溃、OOM、NaN 或 traceback；命中的内容是已知的 Albumentations/Pydantic 初始化 warning。

## 日志关键字摘要

- `paper_full`：只有 Albumentations/Pydantic 初始化 warning
- `paper_atkd_only`：只有 Albumentations/Pydantic 初始化 warning
- `paper_ccl_only`：只有 Albumentations/Pydantic 初始化 warning
- `det_only_same_trainer`：只有 Albumentations/Pydantic 初始化 warning

## 证据归档

证据包路径：`cclkd_reproduction/yolov5_sanity/results/scalingfix_paper_components_400ep_20260613/`

每个 run 已归档：

- `results.csv`
- `cclkd_yolov5_diagnostics.csv`
- `hyp.yaml`、`opt.yaml`、`run_meta.txt`、`command.sh`、`pid.txt`
- `nohup_tail_300.log`
- `nohup_error_grep_tail.log`

未上传内容：checkpoint 权重、TensorBoard event 文件、完整 nohup 大日志。

汇总 CSV：`cclkd_reproduction/yolov5_sanity/results/scalingfix_paper_components_400ep_20260613/summary.csv`

GPU/进程快照：`cclkd_reproduction/yolov5_sanity/results/scalingfix_paper_components_400ep_20260613/process_gpu_snapshot.txt`
