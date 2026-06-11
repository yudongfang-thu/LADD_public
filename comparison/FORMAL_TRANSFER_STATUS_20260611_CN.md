# Non-CCLKD formal transfer 状态记录（2026-06-11）

本文记录 `fgd`、`ld`、`hallucidet_style` 三个 frozen-teacher transfer
comparison run 的恢复、平台期和早停状态。范围只包含 non-CCLKD；CCLKD
不在本文统计范围内。

服务器：`ladd4090:/root/shared-nvme/LADD_public`

协议：formal no-mosaic，`imgsz=256`，`epochs=800`，same-capacity same-seed
RGB teacher，SAR-only inference。

## 1. 恢复与日志处理

2026-06-11 服务器共享盘曾触发 `Disk quota exceeded`，导致部分 run
在训练完成一个 epoch 后已经写入 `results.csv`，但 `last.pt` 尚未更新。恢复时采用：

1. 读取 `weights/last.pt` 中的 checkpoint epoch。
2. 对齐 `results.csv`，删除超过 checkpoint 可恢复范围的孤儿 epoch 行。
3. 使用 `RESUME_FROM=<run>/weights/last.pt`、`EXIST_OK=1`、`SAVE_PERIOD=-1`
   继续原 run directory。
4. 为避免共享盘日志再次触发配额错误，恢复日志写入 `/tmp/ladd_resume_logs/`；
   训练结果仍写入原 run directory 的 `results.csv`、`weights/last.pt` 和
   `weights/best.pt`。

该恢复方式保留模型权重、EMA、optimizer state 和训练 epoch 状态；恢复后
`kd_loss`、det loss 和 mAP 曲线未出现类似从头训练的断崖式波动。

## 2. 早停设置

已检查完成 run 的 `args.yaml`，当前 formal transfer comparison 设置为：

```text
epochs: 800
patience: 800
save_period: -1
cos_lr: true
close_mosaic: 0
```

因此 early stopping 实际等于关闭；完成的 run 均为跑满 800 epoch，不是早停结束。
后续若只做工程节省时间，可考虑对 HalluciDet-style 使用更短 patience；但当前正式
comparison 为保持协议一致，仍记录为 800 epoch formal run。

## 3. 2026-06-11 21:07 CST 快照

| Run | 状态 | latest epoch | latest mAP50-95 | best mAP50-95 | best epoch | latest kd_loss | 判断 |
|---|---|---:|---:|---:|---:|---:|---|
| `n_fgd` | running | 242 | 0.38095 | 0.52982 | 1 | 2.22953 | 运行中；恢复后缓慢回升，但远低于早期 best，FGD 速度和曲线都需单独标注 |
| `n_ld` | running | 644 | 0.56989 | 0.57035 | 640 | 0.00531 | 运行中；已接近平台，latest 与 best 差距极小 |
| `n_hallucidet_style` | complete | 800 | 0.57239 | 0.57365 | 785 | 0.25837 | 已完成；最后 50 epoch 基本平台，final 比 best 低 0.00126 |
| `s_fgd` | running | 227 | 0.58786 | 0.61673 | 1 | 0.75269 | 运行中；恢复后缓慢回升，但仍低于早期 best |
| `s_ld` | running | 613 | 0.64386 | 0.64390 | 612 | 0.00278 | 运行中；已接近平台，latest 与 best 基本一致 |
| `s_hallucidet_style` | complete | 800 | 0.63124 | 0.64310 | 639 | 0.17476 | 已完成；639 后进入平台并轻微退化，final 比 best 低 0.01186 |

## 4. HalluciDet-style 平台期现象

`n_hallucidet_style`：

| Window | 起止 epoch | mAP50-95 变化 | range | window best |
|---|---|---:|---:|---|
| last 20 | 781 -> 800 | -0.00052 | 0.00194 | 0.57365@785 |
| last 50 | 751 -> 800 | -0.00010 | 0.00309 | 0.57365@785 |
| last 100 | 701 -> 800 | +0.00103 | 0.00309 | 0.57365@785 |

结论：`n_hallucidet_style` 已到平台；800 epoch final 与 best 差距很小，使用
`best.pt` 或 final 都不会改变总体判断。

`s_hallucidet_style`：

| Window | 起止 epoch | mAP50-95 变化 | range | window best |
|---|---|---:|---:|---|
| last 20 | 781 -> 800 | -0.00199 | 0.00239 | 0.63363@786 |
| last 50 | 751 -> 800 | -0.00502 | 0.00531 | 0.63655@753 |
| last 100 | 701 -> 800 | -0.00905 | 0.00913 | 0.64037@702 |
| last 200 | 601 -> 800 | -0.00743 | 0.01186 | 0.64310@639 |

结论：`s_hallucidet_style` 在 epoch 639 达到 best，之后轻微退化；正式表格应使用
`best.pt` 对应 best mAP，不能只看最后一轮。

## 5. 轻量日志与结果位置

训练结果目录：

```text
runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/transferred_kd/yolo11n/fgd/transfer_fgd_hbb_ogsod11n_formal_nomosaic_yolo11n_fgd_v2_20260610_transfer_s0_b_e800_b64_s0_gpu1
runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/transferred_kd/yolo11n/ld/transfer_ld_hbb_ogsod11n_formal_nomosaic_yolo11n_ld_v2_20260610_transfer_s0_b_e800_b64_s0_gpu1
runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/transferred_kd/yolo11n/hallucidet_style/transfer_hallucidet_style_hbb_ogsod11n_formal_nomosaic_yolo11n_hallucidet_style_v2_20260610_transfer_s0_b_e800_b64_s0_gpu1
runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/transferred_kd/yolo11s/fgd/transfer_fgd_hbb_ogsod11s_formal_nomosaic_yolo11s_fgd_v2_20260610_transfer_s0_b_e800_b64_s0_gpu0
runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/transferred_kd/yolo11s/ld/transfer_ld_hbb_ogsod11s_formal_nomosaic_yolo11s_ld_v2_20260610_transfer_s0_b_e800_b64_s0_gpu0
runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/transferred_kd/yolo11s/hallucidet_style/transfer_hallucidet_style_hbb_ogsod11s_formal_nomosaic_yolo11s_hallucidet_style_v2_20260610_transfer_s0_b_e800_b64_s0_gpu0
```

恢复日志目录：

```text
/tmp/ladd_resume_logs/
```

大文件策略：checkpoint、完整训练日志和中间 epoch checkpoint 不进入 GitHub。本文只记录
轻量级数值、状态、路径和判断。

