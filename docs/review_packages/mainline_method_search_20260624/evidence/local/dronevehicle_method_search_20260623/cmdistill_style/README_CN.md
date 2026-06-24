# CMDistill-Style

日期：2026-06-23

## 作用

预留给 DroneVehicle sub2k seed0 上的 CMDistill-style 对照。后续所有相关 run 放在本方法目录下，避免与 DSN / oldsplit 混表。

## 计划目录

```text
runs_public/dronevehicle_method_search/sub2k_seed0_fullval/cmdistill_style/ir_to_rgb/
logs/dronevehicle_method_search/sub2k_seed0_fullval/cmdistill_style/ir_to_rgb/
```

## 已启动 run

时间：2026-06-23 23:53 CST

远端：`ladd4090-zw1`，GPU1

目标：先验证 DroneVehicle sub2k 风洞是否存在“外部成熟 KD 方法能带来正向提升”的潜力，再决定新主线方案是否值得继续在该风洞迭代。

协议：

```text
student modality: RGB
teacher modality: IR
model: YOLO11n
imgsz: 512
batch: 64
epochs: 200
optimizer: SGD
lr0/lrf: 0.01 / 0.01
mosaic/close_mosaic/mixup: 0.0 / 0 / 0.1
seed: 0
student init: RGB baseline best
teacher weights: IR baseline best
```

CMDistill profile：

```text
COMPARISON_KD_PROFILE=cmdistill
PROFILE_KD_WEIGHT=1.0
PROFILE_KD_REPLACE_BASE=1
STUDENT_BRANCH_MODE=raw
TEACHER_FEATURE_MODE=raw
KD_CALIBRATION_MODE=affine
CMDISTILL_FEATURE_WEIGHT=1.0
CMDISTILL_RELATION_WEIGHT=1.0
CMDISTILL_LOGIT_WEIGHT=1.0
CMDISTILL_TEMPERATURE=4.0
CMDISTILL_MAX_TOKENS=512
CMDISTILL_MIN_CONFIDENCE=0.05
LAMBDA_REACH=0.0
LAMBDA_REC=0.0
LAMBDA_TASKL=0.0
ALPHA_S_REC=0.0
```

远端 run：

```text
runs_public/dronevehicle_method_search/sub2k_seed0_fullval/cmdistill_style/ir_to_rgb/cmdistill_ir2rgb_yolo11n_e200_b64_img512_mosaic0p0_mixup0p1_s0_20260623_235356_b
logs/dronevehicle_method_search/sub2k_seed0_fullval/cmdistill_style/ir_to_rgb/cmdistill_ir2rgb_yolo11n_e200_b64_img512_mosaic0p0_mixup0p1_s0_20260623_235356_gpu1
```

启动后检查：

- 进程 pid：`15467`
- 参数已确认：`imgsz=512`，`batch=64`，`mixup=0.1`，`comparison_kd_profile=cmdistill`，`profile_kd_replace_base=True`。
- 训练已进入 epoch 11；GPU1 峰值显存约 18GB（含并发 OGSOD RGB baseline），batch 64 可以运行。
- epoch 11 快照：当前 best 为 epoch 1，`mAP50=0.55672`，`mAP50-95=0.34968`。早期值低于 RGB baseline best `0.56886/0.36087`，暂不判定结果，后续以 best checkpoint 与同协议 det-only continued-training control 对照。
- epoch 18 快照：当前 best 仍为 `mAP50=0.55672`，`mAP50-95=0.34968`；latest `mAP50-95=0.30770`。
- 2026-06-24 00:37 CST 快照：epoch 118，best epoch 104，`mAP50=0.56267`，`mAP50-95=0.35382`；latest `mAP50=0.55864`，`mAP50-95=0.35042`。该 run 有一定恢复，但仍低于 RGB baseline best `0.36087` 和 high-LR det-only best `0.35876`，且 high-LR det-only 本身存在 reload 掉点混杂。因此该结果不能证明 CMDistill 正向，需等待 low-LR/no-warmup CMDistill 对照。
- 2026-06-24 00:58 CST 快照：epoch 184，best 仍为 epoch 154，`mAP50=0.56564`，`mAP50-95=0.35835`；latest `mAP50=0.55998`，`mAP50-95=0.35437`。仍低于 RGB baseline best `0.36087` 和 high-LR det-only best `0.35876`，所以 high-LR CMDistill 仍不算正向。

## Det-only / reload control

时间：2026-06-23 23:57 CST

远端：`ladd4090-zw1`，GPU0

目的：判定从 RGB baseline best 继续训练 200 epoch 本身是否会涨点，避免把 reload/continued-training 收益误判为 CMDistill 收益。

远端 run：

```text
runs_public/dronevehicle_method_search/sub2k_seed0_fullval/cmdistill_style/c0_detonly_reload/detonly_reload_rgb_yolo11n_e200_b64_img512_mosaic0p0_mixup0p1_s0_20260623_235746_b
logs/dronevehicle_method_search/sub2k_seed0_fullval/cmdistill_style/c0_detonly_reload/detonly_reload_rgb_yolo11n_e200_b64_img512_mosaic0p0_mixup0p1_s0_20260623_235746_gpu0
```

关键参数：

```text
COMPARISON_KD_PROFILE=none
PROFILE_KD_WEIGHT=0.0
ALPHA_KD=0.0
LAMBDA_REACH=0.0
LAMBDA_REC=0.0
LAMBDA_TASKL=0.0
ALPHA_S_REC=0.0
LADD_B_DET_ONLY=1
```

启动后检查：参数解析和数据集扫描正常，`imgsz=512`、`batch=64`、`mixup=0.1` 与 CMDistill run 一致。

早期快照：

- epoch 2：当前 best `mAP50=0.56705`，`mAP50-95=0.35876`，已经接近原 RGB baseline best `0.56886/0.36087`。
- 解释：若 det-only control 后续追平或超过 CMDistill，则 CMDistill 不算正向方法收益。

2026-06-24 00:58 CST 快照：high-LR det-only epoch 170，best 仍为 epoch 1 的 `0.56705 / 0.35876`，latest `mAP50=0.54488`，`mAP50-95=0.35197`，late5 AP50-95 `0.35134`。这进一步确认 high-LR continued-training protocol 不适合作为方法结论依据。

## 已启动：low-LR/no-warmup CMDistill

时间：2026-06-24 00:38 CST

目的：在 reload protocol 修正后重新验证 CMDistill 是否能成为“风洞有正向潜力”的 sanity check。该 run 必须与 low-LR/no-warmup det-only control 比较。

远端队列：

```text
logs/dronevehicle_method_search/sub2k_seed0_fullval/cmdistill_style/ir_to_rgb_lowlr_nowarmup/queue_cmdistill_lowlr_after_det_control_20260624.sh
logs/dronevehicle_method_search/sub2k_seed0_fullval/cmdistill_style/ir_to_rgb_lowlr_nowarmup/queue_cmdistill_lowlr_after_det_control_20260624.log
```

队列 PID：`21191`

触发条件：等待 low-LR/no-warmup det-only control 至少 20 epoch，且 `best AP50-95 >= 0.358`、`latest AP50-95 >= 0.340`。该条件已满足，2026-06-24 01:13:40 CST 启动：

```text
runs_public/dronevehicle_method_search/sub2k_seed0_fullval/cmdistill_style/ir_to_rgb_lowlr_nowarmup/cmdistill_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_011340_b
logs/dronevehicle_method_search/sub2k_seed0_fullval/cmdistill_style/ir_to_rgb_lowlr_nowarmup/cmdistill_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_011340_gpu0.outer.log
```

协议差异：

```text
lr0/lrf: 0.001 / 0.1
warmup_epochs: 0.0
warmup_bias_lr: 0.0
其他模型、数据、batch、augmentation 与 high-LR CMDistill 一致
```

2026-06-24 01:18 CST 快照：

```text
rows: 9
best AP50/AP50-95: 0.56594 / 0.36089
latest AP50/AP50-95: 0.56461 / 0.35824
late5 AP50-95: 0.35840
```

当前解释：CMDistill low-LR 暂时超过原 RGB baseline AP50-95 `0.36087` 的幅度只有 `+0.00002`，且低于同协议 det-only low-LR best `0.36279`。需要至少 20 rows 后再判定。

协议卫生：后续重启脚本已加 `STRICT_BATCH_SIZE=1`，防止 batch 64 在 OOM 时自动降 batch。当前 live run 启动时尚未带 strict 标志，但初始日志未见 OOM/fallback；若后续出现 fallback，该 run 只能作为诊断，不能作为正式对照。

2026-06-24 01:25 CST 快照：

```text
rows: 29
best AP50/AP50-95: 0.56913 / 0.36286
latest AP50/AP50-95: 0.56063 / 0.36027
late5 AP50-95: 0.35913
```

当前解释：CMDistill low-LR 已经跑过 20 rows，best AP50-95 比 det-only low-LR best `0.36279` 高 `+0.00007`，属于擦线正向；同 epoch 16 对照为 CMDistill `0.36286` vs det-only `0.35837`。这能初步证明风洞不是完全没有外部 KD 正信号，但幅度太小，不能直接作为主线。

2026-06-24 01:28 CST 快照：

```text
rows: 37
best AP50/AP50-95: 0.56913 / 0.36286
latest AP50/AP50-95: 0.55743 / 0.35434
late5/late10/late20 AP50-95: 0.35754 / 0.35663 / 0.35705
```

与 det-only low-LR 对照：

```text
best delta: +0.00007
latest delta: +0.00357
late5/late10/late20 delta: +0.00565 / +0.00643 / +0.00720
epoch 16 AP50-95: CMDistill 0.36286 vs det-only 0.35837
epoch 20 AP50-95: CMDistill 0.35721 vs det-only 0.35196
```

当前解释：CMDistill 的同 epoch 对照比 det-only 明显好，说明外部 KD 在这个风洞中确实有可观测正信号；但它的 best 只比 det-only 全局 best 高 `0.00007`，不能作为主线，只能作为 sanity positive。

2026-06-24 01:35 CST 快照：

```text
rows: 56
best AP50/AP50-95: 0.56913 / 0.36286
latest AP50/AP50-95: 0.56210 / 0.35745
late5/late10/late20 AP50-95: 0.35582 / 0.35380 / 0.35544
```

与 det-only low-LR 对照：

```text
best delta: +0.00007
latest delta: +0.01072
late5/late10/late20 delta: +0.00715 / +0.00589 / +0.00701
epoch 50 AP50-95: CMDistill 0.35170 vs det-only 0.34889
epoch 52 AP50-95: CMDistill 0.35458 vs det-only 0.34977
epoch 54 AP50-95: CMDistill 0.35223 vs det-only 0.34968
```

当前解释：CMDistill 在 50+ rows 仍保持 epoch-matched 优势，说明该风洞确实能观测到稳定一些的跨模态 KD 正信号。但由于全局 best 只擦线超过 det-only，且 CMDistill 是外部 baseline，不是 LADD 主线候选，下一步应把它作为风洞 sanity 和 protocol anchor。

2026-06-24 01:41 CST 快照：

```text
rows: 73
best AP50/AP50-95: 0.56913 / 0.36286
latest AP50/AP50-95: 0.54839 / 0.35023
late5/late10/late20 AP50-95: 0.35270 / 0.35333 / 0.35431
```

与 det-only low-LR 当前窗口对照：

```text
best delta: +0.00007
latest delta: +0.00447
late5/late10/late20 delta: +0.00534 / +0.00703 / +0.00841
```

当前解释：CMDistill 继续作为稳定 sanity-positive；但其 best 仍只是擦线，因此不能替代 LADD 主线候选。
