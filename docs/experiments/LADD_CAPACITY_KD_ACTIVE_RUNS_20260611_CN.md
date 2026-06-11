# LADD Capacity-aware KD 已启动实验记录

日期：2026-06-11

本文只记录已启动实验的配置、路径和当前运行状态，不作为最终结果结论。最终判据仍以 `B400` 或 `m A2 probe` 完成后的 `results.csv`、`ladd_diagnostics.csv` 和 summary 工具输出为准。

最新状态：`2026-06-11 20:20:25 CST`，两条 YOLO11s alpha sweep 已跑满 B400；`s B det-only r2` 已完成 A1/A2 并进入 B 阶段。最新结果入口见 [LADD_CAPACITY_KD_DIAG_RESULTS_20260611_CN.md](LADD_CAPACITY_KD_DIAG_RESULTS_20260611_CN.md)。

## 1. 代码版本

远端实验运行目录：

```text
/root/shared-nvme/LADD_public_p1
```

远端实验代码 commit：

```text
69166620f6bc8b5885cbf285ec3ef3a7242c7e6e
fix(ladd): refresh effective KD weights at train epoch start
```

本记录提交时，本地 GitHub 主线已继续包含后续 CCLKD 兼容性修复；这些后续提交不应被误认为本轮 LADD 实验实际运行代码。后续分析每个 run 时，以对应 `manifest.txt` 中的 `git_commit` 为准。

## 2. 当前 GPU 快照

快照时间：`2026-06-11 01:29:16 CST`

| GPU | 显存使用 | 剩余显存 | 利用率 | 说明 |
|---|---:|---:|---:|---|
| 0 | 23608 MiB | 474 MiB | 99% | 已有 s 对比任务 + `s alpha_kd=0.5` A2 |
| 1 | 19633 MiB | 4449 MiB | 99% | 已有 n 对比任务 + `m A2 probe` + `s alpha_kd=0.25` |

GPU1 后续不建议再继续叠加 LADD 任务，除非有已有任务结束或明确接受吞吐下降。

## 3. 已启动实验

| 优先级 | 实验 | GPU | 状态 | 目的 |
|---|---|---:|---|---|
| P1 | `s alpha_kd=0.5 B400` | 0/1 | B400 已完成 | 判断 YOLO11s late degradation 是否来自 KD 强度过大 |
| P1 | `m A2 probe, B=1` | 1 | 已完成 | 判断 YOLO11m 是否在 A2 阶段已经低于 baseline |
| P2 | `s alpha_kd=0.25 B400` | 1 | B400 已完成 | 与 `alpha_kd=0.5` 构成 KD 强度对照 |
| P2 | `s B det-only B400` | 0 | B 运行中 | 判断无 KD/aux 时 B 长训练是否仍退化 |

虽然 `s alpha_kd=0.25` 在原计划中属于 P2，但用户明确要求在 1 卡再加一个任务，因此已在 GPU1 启动。该实验应与 `s alpha_kd=0.5` 一起解释，不能单独作为容量结论。

2026-06-11 11:52 CST 曾启动第一版 `s B det-only B400`，随后因会话/资源中断需要重新排布。2026-06-11 16:03 CST 在 GPU0 启动 `r2` 链，A1 于 16:14 完成，A2 于 18:05 完成，随后进入 B400。该项与两个 alpha sweep run 配合解释，用于区分 B 长训练自身退化和 KD/aux 负迁移；但注意 `LADD_A2_DET_ONLY=0`，因此 B 阶段继承的是正常 A2 checkpoint。

## 4. 运行明细

### 4.1 YOLO11s alpha_kd=0.5 B400

Run tag：

```text
formal_nomosaic_yolo11s_cap2_s0_diag_capkd_s_s0_alphaKD0p5_b400
```

关键配置：

```text
gpu_id=0
seed=0
batch_size=64
epochs_a1=10
epochs_a2=50
epochs_b=400
alpha_kd=0.5
B_FREEZE_BN_STATS=1
LADD_B_DET_ONLY=0
LADD_A2_DET_ONLY=0
ladd_diag_log_bn=1
ladd_diag_log_grad=0
ladd_grad_clip_norm=0.0
effective_grad_clip_norm=10.0
```

日志目录：

```text
/root/shared-nvme/LADD_public_p1/logs/formal_nomosaic_20260528/ladd/formal_nomosaic_yolo11s_cap2_s0_diag_capkd_s_s0_alphaKD0p5_b400_gpu0
```

A1 实际 run 目录：

```text
/root/shared-nvme/LADD_public/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11s/cap2/ladd_hbb_ogsod11s_formal_nomosaic_yolo11s_cap2_s0_diag_capkd_s_s0_alphaKD0p5_b400_a1_e10_b64_s0_gpu0
```

当前进程快照：

```text
chain_pid=196537
current_phase=b
b_train_pid=199778
```

2026-06-11 11:56 CST 快照：

```text
B epoch=283/400
best=0.63074@218
last=0.62803
```

### 4.2 YOLO11m A2 probe, B=1

Run tag：

```text
formal_nomosaic_yolo11m_cap2_s0_diag_capkd_m_s0_a2probe_b1_retry1
```

关键配置：

```text
gpu_id=1
seed=0
batch_size=32
epochs_a1=10
epochs_a2=50
epochs_b=1
alpha_kd=1.0
B_FREEZE_BN_STATS=0
LADD_B_DET_ONLY=0
LADD_A2_DET_ONLY=0
ladd_diag_log_bn=1
ladd_diag_log_grad=0
ladd_grad_clip_norm=0.0
effective_grad_clip_norm=10.0
```

日志目录：

```text
/root/shared-nvme/LADD_public_p1/logs/formal_nomosaic_20260528/ladd/formal_nomosaic_yolo11m_cap2_s0_diag_capkd_m_s0_a2probe_b1_retry1_gpu1
```

完成状态：

```text
chain_complete=2026-06-11 04:20:49 CST
```

关键结果：

```text
A2 best=0.65026@10
A2 last=0.63725
B=1 last=0.62528
```

备注：最早一次不带 `retry1` 的 `m A2 probe` 在 AMP 权重缓存竞争中读取到未完整写入的 `yolo26n.pt`，报 `EOFError: Ran out of input`。当前有效实验为 `retry1`。

### 4.3 YOLO11s alpha_kd=0.25 B400

Run tag：

```text
formal_nomosaic_yolo11s_cap2_s0_diag_capkd_s_s0_alphaKD0p25_b400
```

关键配置：

```text
gpu_id=1
seed=0
batch_size=64
epochs_a1=10
epochs_a2=50
epochs_b=400
alpha_kd=0.25
B_FREEZE_BN_STATS=1
LADD_B_DET_ONLY=0
LADD_A2_DET_ONLY=0
ladd_diag_log_bn=1
ladd_diag_log_grad=0
ladd_grad_clip_norm=0.0
effective_grad_clip_norm=10.0
```

日志目录：

```text
/root/shared-nvme/LADD_public_p1/logs/formal_nomosaic_20260528/ladd/formal_nomosaic_yolo11s_cap2_s0_diag_capkd_s_s0_alphaKD0p25_b400_gpu1
```

当前进程快照：

```text
chain_pid=197800
current_phase=b
b_train_pid=200583
```

2026-06-11 11:56 CST 快照：

```text
B epoch=265/400
best=0.63027@199
last=0.62901
```

注意：该 run 在 A2 阶段触发一次 CUDA OOM，Ultralytics 自动将 batch 从 64 降到 32 后继续完成。B 阶段目前未见 OOM。与 `alpha_kd=0.5` 的严格可比性需要标记这一点。

### 4.4 YOLO11s B det-only B400

Run tag：

```text
formal_nomosaic_yolo11s_cap2_s0_diag_capkd_s_s0_bdetonly_b400_r2
```

关键配置：

```text
gpu_id=0
seed=0
batch_size=64
epochs_a1=10
epochs_a2=50
epochs_b=400
alpha_kd=1.0
B_FREEZE_BN_STATS=1
LADD_B_DET_ONLY=1
LADD_A2_DET_ONLY=0
ladd_diag_log_bn=1
ladd_diag_log_grad=0
ladd_grad_clip_norm=0.0
effective_grad_clip_norm=10.0
```

日志目录：

```text
/root/shared-nvme/LADD_public_p1/logs/formal_nomosaic_20260528/ladd/formal_nomosaic_yolo11s_cap2_s0_diag_capkd_s_s0_bdetonly_b400_r2_gpu0
```

2026-06-11 20:20 CST 快照：

```text
A1 complete: best=0.62878, last=0.62878
A2 complete: best=0.62400, last=0.62400
B running: epoch=74/400, best=0.62244@68, last=0.62191
```

注意：该有效 run 是 `r2`。第一版 `formal_nomosaic_yolo11s_cap2_s0_diag_capkd_s_s0_bdetonly_b400` 在早前调度中断后不作为当前有效证据。`r2` 只设置 `LADD_B_DET_ONLY=1`，`LADD_A2_DET_ONLY=0`，因此 B 阶段继承的是正常 A2 后的 checkpoint。

证据包路径：

```text
ladd/results/capacity_kd_20260611/bdetonly_b400_r2/
```

## 5. 后续读取建议

完成后优先读取：

```bash
python tools/summarize_ladd_capacity_diag.py \
  <s_alpha0p5_b400_b_run_dir> \
  <s_alpha0p25_b400_b_run_dir> \
  <m_a2_probe_a2_run_dir>
```

解释顺序：

1. 先看 `m A2 probe` 的 A2 best 是否低于 `0.65380`，决定是否暂停 m full B。
2. 再比较 `s alpha_kd=0.5` 与 `s alpha_kd=0.25` 的 B400 last AP50-95 是否接近或超过 `0.62697`。
3. 如果 `0.25` 明显稳于 `0.5`，再考虑将 capacity-aware KD 作为下一轮受控诊断，而不是直接改主线结论。
