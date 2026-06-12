# LADD A2 Damage Diagnostics Analysis 20260612

## 结论摘要

本轮诊断已经足够支持以下判断：

1. `s` 的问题不是单一 KD 权重问题。`alphaKD0.5/0.25` 都能在 B 阶段达到约 `0.630` 的 best，但 final 分别掉到 `0.61802` 和 `0.61719`，均低于 `s` SAR baseline final `0.62233`。
2. `s B det-only r2` 在 B 阶段关闭有效 KD/aux 后仍然从 `0.63025@226` 掉到 `0.61923@400`。这说明 B 阶段存在协议/继续训练导致的 late regression，不能只归因于 KD。
3. `s A2 lr3e-4` 明显提高 A2 peak：`0.63309@13`，但 50 epoch final 仍降到 `0.62443`。降低 A2 LR 有效，但 A2 后期仍会回落。
4. `s A2 short15` 和 `s A2 lr3e-4 short15` 都显示 peak 出现在 epoch 13 左右。`lr3e-4 short15` final `0.62383`，好于 SAR baseline final `0.62233`，是当前最合理的 `s` 修复候选，但 final 仍未达到 safe threshold `0.62697`。
5. `m A2 det-only` 已经失败：best `0.64521`、final `0.63892`，低于 `m` SAR baseline final `0.64903`。`m A2 lr3e-4 retry2` 也在 A2 epoch 40 前保持偏低，best `0.64707@4`，说明 `m` 的 A2 损伤不只是 A2 aux，也不只是 lr=1e-3。
6. 所有有效诊断行 `nan_or_inf_any=0`。当前没有 NaN/Inf 证据。`m A2 lr3e-4 retry2` 中断原因是保存 `last.pt` 时 `Disk quota exceeded`，不是训练数值崩溃或 OOM。

## Baseline 和阈值

| model | SAR baseline best | SAR baseline final | baseline drop | safe threshold |
|---|---:|---:|---:|---:|
| YOLO11s | 0.62897 | 0.62233 | 0.00664 | 0.62697 |
| YOLO11m | 0.65580 | 0.64903 | 0.00677 | 0.65380 |

## 主结果表

完整机器可读表见：

- `docs/experiments/ladd_a2_damage_summary_20260612.csv`
- `ladd/results/a2_damage_20260612/final_evidence/summary/ladd_a2_damage_phase_metrics_20260612.csv`

| experiment | phase | status | epochs | best | best epoch | last | last epoch | best-final drop | last - baseline final | excess drop |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| s alphaKD0.5 | A2 | complete | 50 | 0.62400 | 50 | 0.62400 | 50 | 0.00000 | 0.00167 | -0.00664 |
| s alphaKD0.5 | B | complete | 402 | 0.63074 | 218 | 0.61802 | 400 | 0.01272 | -0.00431 | 0.00608 |
| s alphaKD0.25 | A2 | complete | 50 | 0.62771 | 29 | 0.62532 | 50 | 0.00239 | 0.00299 | -0.00425 |
| s alphaKD0.25 | B | complete | 402 | 0.63027 | 199 | 0.61719 | 400 | 0.01308 | -0.00514 | 0.00644 |
| s B det-only r2 | A2 | complete | 50 | 0.62400 | 50 | 0.62400 | 50 | 0.00000 | 0.00167 | -0.00664 |
| s B det-only r2 | B | complete | 400 | 0.63025 | 226 | 0.61923 | 400 | 0.01102 | -0.00310 | 0.00438 |
| s A2 det-only | A2 | complete | 50 | 0.62795 | 13 | 0.62222 | 50 | 0.00573 | -0.00011 | -0.00091 |
| m A2 det-only | A2 | complete | 50 | 0.64521 | 4 | 0.63892 | 50 | 0.00629 | -0.01011 | -0.00048 |
| s A2 lr3e-4 | A2 | complete | 50 | 0.63309 | 13 | 0.62443 | 50 | 0.00866 | 0.00210 | 0.00202 |
| m A2 lr3e-4 retry2 | A2 | incomplete | 40 | 0.64707 | 4 | 0.64123 | 40 | 0.00584 | -0.00780 | -0.00093 |
| s A2 short15 | A2 | complete | 15 | 0.63025 | 13 | 0.62150 | 15 | 0.00875 | -0.00083 | 0.00211 |
| s A2 lr3e-4 short15 | A2 | complete | 15 | 0.63159 | 13 | 0.62383 | 15 | 0.00776 | 0.00150 | 0.00112 |

`excess drop = method best-final drop - SAR baseline best-final drop`。

## 诊断标志核对

关键 flags 已写入 summary CSV。需要特别注意：

- `s B det-only r2` 的 B 阶段：`effective_alpha_kd=0`，`effective_lambda_reach=0`，`effective_lambda_match_inner=0`，`effective_lambda_rank_inner=0`，`ladd_b_det_only=1`。该 run 可作为有效 B det-only 证据。
- `s/m A2 det-only` 的 A2 阶段：`ladd_a2_det_only=1`，有效 KD/reach/match/rank 均为 0。该 run 可作为 A2 aux 关闭证据。
- 其它 A2 正常目标 run：`effective_alpha_kd=1`，`effective_lambda_reach/match/rank=1`。
- 所有纳入表格的诊断 run：`nan_or_inf_any=0`。
- `m A2 lr3e-4 retry2`：中断在 A2 `40/50`，outer log 显示 `OSError: [Errno 122] Disk quota exceeded`，发生在保存 `weights/last.pt` 时。

## 解释

### s：A2 peak 可恢复，但 final 不稳定

`s A2 lr3e-4` 的 best 达到 `0.63309@13`，说明 A2 的目标并非完全不可用，降低 LR 后可以超过 SAR baseline best。问题在于 A2 后期训练仍然把 final 拉回 `0.62443`。

`s A2 short15` 只缩短 A2、不降 LR，best `0.63025@13`，但 final `0.62150`，低于 baseline final。说明短 A2 本身不足够。

`s A2 lr3e-4 short15` best `0.63159@13`，final `0.62383`，是当前 `s` 上最好的 practical candidate，但仍未达到 safe threshold `0.62697`。如果要作为主线修复，应继续考虑更短 A2，例如保存 epoch 13 best 进入 B，或 A2 early-stop/phase-stop 使用 best checkpoint，而不是固定使用 phase final。

### m：不是简单 aux 或 LR 问题

`m A2 det-only` 关闭 A2 aux 后仍明显失败，final 比 m SAR baseline final 低 `0.01011`。

`m A2 lr3e-4 retry2` 虽然只到 epoch 40，但 best `0.64707@4` 仍低于 m SAR baseline final `0.64903`，更低于 safe threshold `0.65380`。因此当前证据不支持继续开 `m full B/B400/B800`。

由于该 run 被磁盘配额中断，若需要严谨补完，应先释放 quota，然后只恢复或重跑 `m A2 lr3e-4`。但从已有 40 epoch 曲线看，补完剩余 10 epoch 大概率不会反转结论。

### B：det-only 也会 late-regress

`s B det-only r2` 的 B 阶段完全关闭有效 KD 和 reach/match/rank，仍从 best `0.63025@226` 掉到 final `0.61923@400`。这表明 B 阶段晚期退化不只来自 KD loss 本身，也可能来自继续训练协议、BN/statistics、优化 schedule、A2 checkpoint 状态或 detector fine-tune 本身。

`alphaKD0.5/0.25` 的 B 阶段 final 更低，说明 KD 设置会进一步加剧 final regression，但不是唯一原因。

## 当前建议

1. 暂停新实验，先解决远端 disk quota。否则新的 run 可能继续在保存 checkpoint 时失败。
2. 不启动 `m full B / m B400 / m B800`。m 的 A2 尚未通过。
3. 不继续 `s alpha_kd` sweep。B 阶段 KD sweep 已经证明 final 退化明显。
4. `s` 的下一步修复优先级应是：A2 使用 best checkpoint 或 epoch 13 左右 early-stop，再进入 B，而不是继续扩大 KD 配置矩阵。
5. `m` 的下一步若继续，应先做 `m A2 short15` 或 A2 best-checkpoint probe；但要在 quota 清理后进行。

## 证据位置

轻量证据已整理到：

- `ladd/results/a2_damage_20260612/final_evidence/`
- `ladd/results/a2_damage_20260612/final_evidence/runs/`
- `ladd/results/a2_damage_20260612/final_evidence/chain_logs/`
- `ladd/results/a2_damage_20260612/final_evidence/log_extracts/`
- `ladd/results/a2_damage_20260612/final_evidence/summary/`

本次证据包只包含 `results.csv`、`args.yaml`、`ladd_diagnostics.csv`、chain manifest、compact log extracts 和 summary。未包含 checkpoint、`.pt/.pth`、TensorBoard event、wandb、完整 run 目录或完整大日志。
