# Reload / Continued-Training Controls

日期：2026-06-24

本目录记录 DroneVehicle sub2k 上的 reload / continued-training control。当前发现非常关键：从 RGB baseline best 继续训练时，如果直接使用从头训练协议的 `lr0=0.01` 和默认 warmup，det-only control 也会快速掉点。因此所有 B-only KD / LADD 结论都必须同时比较一个低 LR、无 warmup 的 reload control。

## 已运行：high-LR det-only control

远端 run：

```text
runs_public/dronevehicle_method_search/sub2k_seed0_fullval/cmdistill_style/c0_detonly_reload/detonly_reload_rgb_yolo11n_e200_b64_img512_mosaic0p0_mixup0p1_s0_20260623_235746_b
```

关键协议：

```text
student init: RGB baseline best
lr0/lrf: 0.01 / 0.01
warmup_epochs: default 3.0
warmup_bias_lr: default 0.1
KD/LADD losses: off
```

2026-06-24 00:07 CST 快照：

```text
best AP50/AP50-95: 0.56705 / 0.35876
latest epoch 21 AP50-95: 0.28183
late5 AP50-95: 0.28976
```

解释：best 仍接近 RGB baseline best `0.56886 / 0.36087`，但继续训练曲线快速下滑，说明高 LR + warmup 对已收敛 checkpoint 不稳定。

2026-06-24 00:31 CST 快照：

```text
det-only high-LR: epoch 83, best AP50/AP50-95 0.56705 / 0.35876, latest AP50-95 0.33277, late5 0.33100
CMDistill high-LR: epoch 101, best AP50/AP50-95 0.55672 / 0.34968, latest AP50-95 0.33151, late5 0.34057
raw feature KD high-LR: epoch 66, best AP50/AP50-95 0.56589 / 0.35732, latest AP50-95 0.31132, late5 0.32295
oldsplit C high-LR-ish: epoch 67, best AP50/AP50-95 0.56315 / 0.35439, latest AP50-95 0.33325, late5 0.33411
```

解释：所有 high-LR B/C continued-training 线仍然呈现 early-best、late-window 下滑的 reload 形态。当前还不能用这些 run 判定 CMDistill/rawKD/oldsplit 本身失败，必须等待 low-LR/no-warmup control。

2026-06-24 00:48 CST 快照：

```text
det-only high-LR: epoch 138, best AP50/AP50-95 0.56705 / 0.35876, latest 0.53394 / 0.33916, late5 0.34441
CMDistill high-LR: epoch 154, best AP50/AP50-95 0.56564 / 0.35835, latest 0.56564 / 0.35835, late5 0.35559
raw feature KD high-LR: epoch 120, best AP50/AP50-95 0.56589 / 0.35732, latest 0.53996 / 0.34428, late5 0.34568
oldsplit C high-LR-ish: epoch 109, best AP50/AP50-95 0.56315 / 0.35439, latest 0.54692 / 0.34850, late5 0.34789
```

解释：CMDistill high-LR 后期有所恢复，但 best 仍低于 RGB baseline `0.56886 / 0.36087` 与 high-LR det-only `0.56705 / 0.35876`。这仍然支持“先等 low-LR/no-warmup control，再判定方法”的策略。

2026-06-24 00:58 CST 快照：

```text
det-only high-LR: epoch 170, best AP50/AP50-95 0.56705 / 0.35876, latest 0.54488 / 0.35197, late5 0.35134
CMDistill high-LR: epoch 184, best AP50/AP50-95 0.56564 / 0.35835, latest 0.55998 / 0.35437, late5 0.35434
raw feature KD high-LR: epoch 151, best AP50/AP50-95 0.56589 / 0.35732, latest 0.54524 / 0.34956, late5 0.34976
oldsplit A2: completed, best AP50/AP50-95 0.56322 / 0.36326, latest 0.55846 / 0.36035, late5 0.36040
oldsplit C high-LR-ish: epoch 133, best AP50/AP50-95 0.55536 / 0.35518, latest 0.54902 / 0.35164, late5 0.35155
```

解释：`oldsplit A2` 是唯一超过 RGB baseline AP50-95 的短程信号，但它没有经过同结构 det-only control，且 C 阶段回落。因此它只作为 `oldsplit_a2only_controlled` 的动机，不作为已成立主线候选。

## 已启动：low-LR no-warmup controls

远端队列：

```text
logs/dronevehicle_method_search/sub2k_seed0_fullval/reload_protocol_controls/queue_lowlr_nowarmup_20260624_000832.sh
logs/dronevehicle_method_search/sub2k_seed0_fullval/reload_protocol_controls/queue_lowlr_nowarmup_20260624_000832.log
```

队列 pid：`17441`

触发条件：任意 GPU 显存低于 `15000 MB`。

2026-06-24 00:58 CST 快照：两张 GPU 仍约 `23.1GB / 23.9GB` 占用，队列还在等待显存空档，尚未启动 low-LR run。`nvidia-smi` 显示的 compute PIDs 在容器内 `ps` 不可见，这是当前 Paratera 容器/宿主 PID 映射现象；外层训练 shell 和 `results.csv` 仍在更新，暂不按孤儿显存处理。

2026-06-24 01:18 CST 快照：low-LR 队列已经启动并产出早期结果。

```text
det-only low-LR:
  run: runs_public/dronevehicle_method_search/sub2k_seed0_fullval/reload_controls/lr1e-3_nowarmup/detonly_reload_lowlr1e3_nowarmup_rgb_yolo11n_e200_b64_img512_s0_20260624_010438_b
  rows: 49
  best AP50/AP50-95: 0.56818 / 0.36279
  latest AP50/AP50-95: 0.55058 / 0.34978
  late5 AP50-95: 0.35064

raw feature KD low-LR:
  run: runs_public/dronevehicle_method_search/sub2k_seed0_fullval/raw_feature_kd/ir_to_rgb_lowlr_nowarmup/rawfeatkd_lowlr1e3_nowarmup_ir2rgb_yolo11n_a0p25_affine_e200_b64_img512_s0_20260624_010738_b
  rows: 38
  best AP50/AP50-95: 0.56875 / 0.36265
  latest AP50/AP50-95: 0.56165 / 0.35825
  late5 AP50-95: 0.35595
```

当前解释：low-LR det-only 本身已经超过原 RGB baseline best AP50-95 `0.36087`，所以后续所有方法必须优先超过同协议 det-only best `0.36279`，不能只和原 baseline 比。raw feature KD low-LR 早期 best 与 det-only 非常接近但略低 `0.00014` AP50-95，暂不视为正向。

2026-06-24 01:25 CST 快照：

```text
det-only low-LR: rows=71, best AP50/AP50-95 0.56818 / 0.36279, latest 0.54631 / 0.34951, late5 AP50-95 0.35141
raw feature KD low-LR: rows=59, best AP50/AP50-95 0.56875 / 0.36265, latest 0.54901 / 0.35048, late5 AP50-95 0.35323
```

同 epoch 对照提示：rawKD 在 epoch 53 的 AP50-95 为 `0.34809`，det-only epoch 53 为 `0.34645`，但 rawKD 的全局 best 仍略低于 det-only best。当前只说明 rawKD 不明显破坏训练，不能作为独立正向主线。

2026-06-24 01:28 CST 快照：

```text
det-only low-LR: rows=79, best AP50/AP50-95 0.56818 / 0.36279, latest 0.54695 / 0.35077, late5/late10/late20 AP50-95 0.35189 / 0.35020 / 0.34985
raw feature KD low-LR: rows=68, best AP50/AP50-95 0.56875 / 0.36265, latest 0.54976 / 0.35208, late5/late10/late20 AP50-95 0.34897 / 0.35022 / 0.35084
```

解释：rawKD 的 latest/late20 已略高于 det-only 当前窗口，但全局 best 仍未超过 det-only best。它更像“轻微稳定训练”而不是明确 KD 收益。

2026-06-24 01:35 CST 快照：

```text
det-only low-LR: rows=101, best AP50/AP50-95 0.56818 / 0.36279, latest 0.54235 / 0.34673, late5/late10/late20 AP50-95 0.34867 / 0.34791 / 0.34842
raw feature KD low-LR: rows=87, best AP50/AP50-95 0.56875 / 0.36265, latest 0.54428 / 0.34767, late5/late10/late20 AP50-95 0.34953 / 0.35030 / 0.35083
```

解释：det-only low-LR 后半程仍有缓慢下滑；rawKD 的当前窗口略好于 det-only，但全局 best 仍未超过 det-only best。后续比较要同时看 best 与同 epoch/late-window。

2026-06-24 01:41 CST 快照：

```text
det-only low-LR: rows=120, best AP50/AP50-95 0.56818 / 0.36279, latest 0.54018 / 0.34576, late5/late10/late20 AP50-95 0.34736 / 0.34630 / 0.34590
raw feature KD low-LR: rows=105, best AP50/AP50-95 0.56875 / 0.36265, latest 0.54433 / 0.34761, late5/late10/late20 AP50-95 0.34654 / 0.34813 / 0.34848
```

解释：det-only 与 rawKD 都仍在后半程缓慢下滑。rawKD 在 latest/late20 上略好，但 best 仍低于 det-only，不构成主线。

启动顺序：

```text
1. detonly_reload_lowlr1e3_nowarmup
2. rawfeatkd_lowlr1e3_nowarmup_a0p25_affine
```

共同协议：

```text
student init: RGB baseline best
teacher weights: IR baseline best
YOLO11n, imgsz=512, batch=64, epochs=200, seed=0
lr0/lrf: 0.001 / 0.1
warmup_epochs: 0.0
warmup_bias_lr: 0.0
warmup_momentum: 0.937
mosaic/close_mosaic/mixup: 0.0 / 0 / 0.1
```

预期解释：

- 若 low-LR det-only 能稳定保持 baseline，而 KD 不能超过它，则问题主要在跨模态监督。
- 若 high-LR 掉点、low-LR 稳定，则后续 DSN / fused shared 的 S2 必须采用 low-LR no-warmup protocol。
- 若 low-LR det-only 本身继续涨点，则所有方法收益必须超过该 reload gain，不能只和原 baseline 比。
