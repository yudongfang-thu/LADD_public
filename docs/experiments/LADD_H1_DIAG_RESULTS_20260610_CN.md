# LADD H1 诊断结果记录

日期：2026-06-10

服务器：双卡 RTX 4090，远端路径 `/root/shared-nvme/LADD_public`

本记录只归档轻量证据文件，不包含 checkpoint 权重。原始证据副本见 `docs/experiments/ladd_h1_diag_20260610_artifacts/`。

## 1. 结果表

| Run | Epoch | last AP50-95 | best AP50-95 | best epoch | baseline | 判断 |
|---|---:|---:|---:|---:|---:|---|
| `diag_h1_n_s0_b100_smoke` | 100 | 0.55872 | 0.55872 | 100 | 0.55654 | smoke 通过，H1 未破坏 n 的健康训练 |
| `diag_h1_s_s0_b400_datafix` | 400 | 0.62036 | 0.63127 | 216 | 0.62897 | best 正向，但 last 退化，未通过 H1 冻结标准 |
| `online_cclkd_n_s0_e400` | 400 | 0.52793 | 0.52793 | 400 | 0.55654 | 显著低于 n SAR baseline |
| `yolov5_rgb_x_pretrained` | 400 | 0.52414 | 0.52414 | 400 | N/A | YOLOv5 sanity/gate 证据，不进入 LADD H1 判据 |
| `yolov5x_online_cclkd_full` | 262 rows | 0.12141 | 0.15081 | 169 | N/A | watchdog 曾重启，作为 diagnostic-only |

## 2. 关键结论

H1 的代码修复没有破坏 YOLO11n：`n b100 smoke` 高于同容量 SAR baseline `+0.00218`。

YOLO11s 的 `b400` 结果说明 H1 仍不能直接冻结为主线：best 高于 SAR baseline `+0.00230`，但最终 last 低于 baseline `-0.00861`，后期退化仍然存在。

因此下一轮如果继续非 m 方向，应优先进入 `P2_s`：

- `diag_h1_s_seed0_alpha_kd_0p5_b400`
- `diag_h1_s_seed0_alpha_kd_0p25_b400`
- `diag_h1_s_seed0_detonly_b400`

它们分别用于判断 s 的退化是否来自 KD 强度过大、capacity mismatch，还是检测训练协议本身。

## 3. 轻量证据

归档目录：

```text
docs/experiments/ladd_h1_diag_20260610_artifacts/
```

包含内容：

- H1 n/s 的 `results.csv`、`args.yaml`、chain manifest、phase manifest、master log、outer log。
- H1 s 的 `ladd_diagnostics.csv`。
- CCLKD n 400 的 `results.csv`、`args.yaml`。
- YOLOv5 sanity/CCLKD diagnostic 的 `results.csv`、`opt.yaml`、`hyp.yaml`、`command.sh`、`nohup_tail500.log` 和 watchdog log。

未归档内容：

- `weights/*.pt`
- `*.pth`
- TensorBoard event 文件
- 大体积完整 `nohup.log`；GitHub 只保留 tail 摘要，完整日志保留在远端 run 目录
- `runs_public/` 和 `logs/` 的整目录镜像

## 4. 读法

当前 H1 更像是“实现污染修复后，n 保持健康，但 s 仍有 late degradation”的证据。它支持继续做 P2_s 定位，不支持直接宣传为最终主线。

补充背景：90 服务器旧 `mosaic=1.0, close_mosaic=700` 收敛主线下，YOLO11n LADD legacy/cap2 六条 B 阶段 run 均未出现崩溃，并稳定高于同协议 SAR baseline。见 [LADD_MOSAIC90_MAINLINE_EVIDENCE_20260528_CN.md](LADD_MOSAIC90_MAINLINE_EVIDENCE_20260528_CN.md)。这说明 H1/no-mosaic 下的后期退化不能简单归因于 LADD 方法本身必然不稳定。

后期 loss 曲线补充分析见 [LADD_LATE_DEGRADATION_CURVE_ANALYSIS_20260610_CN.md](LADD_LATE_DEGRADATION_CURVE_ANALYSIS_20260610_CN.md)。当前读法是：退化 run 并非 loss 爆炸，而是 train/KD loss 继续下降，同时 validation loss，尤其 `val/cls_loss`，在 best epoch 后上升。
