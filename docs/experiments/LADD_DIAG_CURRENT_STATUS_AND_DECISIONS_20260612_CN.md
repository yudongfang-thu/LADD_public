# LADD 当前诊断实验状态与继续/停止判断（2026-06-12）

## 1. 本次更新范围

本次更新同步了双卡 4090 服务器当前诊断实验的轻量证据：

```text
ladd/results/a2_damage_20260612/
```

包含：

- A2 damage 三条已完成链的 `results.csv`、`args.yaml`、`ladd_diagnostics.csv`。
- `s B det-only r2` 当前 `292/400` snapshot。
- 对应 `chain.log` 和关键 log extract。
- `summary/a2_damage_and_current_diag_summary_20260612.csv`。

远端主工作区：

```text
/root/shared-nvme/LADD_public_p1
```

本次仍然没有同步：

- checkpoint 权重
- 完整 run 目录
- TensorBoard event
- wandb
- 完整大日志

## 2. 当前诊断 run 状态

基准：

- YOLO11s SAR seed0 baseline best/final/drop：`0.62897 / 0.62233 / 0.00664`
- YOLO11m SAR seed0 baseline best/final/drop：`0.65580 / 0.64903 / 0.00677`
- s safe threshold：`0.62697`
- m safe threshold：`0.65380`

| run | phase | status | best AP50-95 | best epoch | last/latest AP50-95 | 读法 |
|---|---|---|---:|---:|---:|---|
| `s A2 det-only` | A2 | complete | 0.62795 | 13 | 0.62222 | best 过 s safe，但略低于 s baseline best；last 约等于 s baseline final |
| `m A2 det-only` | A2 | complete | 0.64521 | 4 | 0.63892 | best 和 last 均低于 m baseline final，det-only 也不能保住 m |
| `s A2 lr3e-4` | A2 | complete | 0.63309 | 13 | 0.62443 | best 明显超过 s baseline best；last 高于 s baseline final，但仍有 late drop |
| `s B det-only r2` | B | running snapshot | 0.63025 | 226 | 0.62594 @292 | 当前仍高于 s baseline final，但从 best 后下滑；建议跑完 400 |

三条 A2 damage 链的 B1 都只是占位，不解释 B1 AP：

- `s A2 det-only B1`: `0.60240`
- `m A2 det-only B1`: `0.62422`
- `s A2 lr3e-4 B1`: `0.60436`

这些 B1 只有 1 epoch，用于保持 chain 完整，不用于判断 B 阶段方法性能。

## 3. 当前证据说明了什么

### 3.1 s 模型：A2 能冲高，但 50 epoch 后段会回落

`s A2 det-only`：

- A2 best `0.62795@13`
- A2 last `0.62222`
- auxiliary/reach/rec/sep/task/private losses 全部有效关闭：
  - `effective_alpha_kd=0.0`
  - `effective_lambda_reach=0.0`
  - `effective_lambda_match_inner=0.0`
  - `effective_lambda_rank_inner=0.0`
  - `ladd_a2_det_only=1`

解释：

- 去掉 A2 auxiliary objectives 后，s 仍然能接近 baseline best，并且 last 接近 baseline final。
- 这说明 s 的 A2 损伤不是单纯由 auxiliary objectives 造成。
- 但 best 出现在 epoch 13，后面回落，说明 A2 的训练长度/后段稳定性仍然是问题。

`s A2 lr3e-4`：

- A2 best `0.63309@13`
- A2 last `0.62443`

解释：

- 降低 A2 lr 后，s 的 peak 明显恢复，超过 baseline best。
- 但 final 仍低于 peak `0.00866`，drop 大于 s baseline drop `0.00664`。
- 因此 s 更像是“可冲高，但 A2 50 epoch 后段不稳定”，不是“完全学不起来”。

### 3.2 m 模型：A2 det-only 也失败，不能直接进入 full B

`m A2 det-only`：

- A2 best `0.64521@4`
- A2 last `0.63892`
- m baseline best/final 是 `0.65580 / 0.64903`

解释：

- 即使 A2 只保留 detection loss，m 仍然明显低于 baseline final。
- 这说明 m 的问题不能简单归因于 A2 auxiliary objectives。
- 在 m A2 没有恢复之前，不应启动 m full B、m B400 或 m B800。

### 3.3 B 阶段 KD 不是唯一问题

`s alpha_kd=0.5/0.25 B400` 已完成，best 都略高于 s baseline best，但 final 均低于 s baseline final，并且 excess drop 大于 protocol baseline drop。

`s B det-only r2` 当前 `292/400`：

- B best `0.63025@226`
- latest `0.62594@292`
- B 阶段 diagnostics 确认：
  - `effective_alpha_kd=0.0`
  - `effective_lambda_reach=0.0`
  - `effective_lambda_match_inner=0.0`
  - `effective_lambda_rank_inner=0.0`

解释：

- det-only B 可以从退化 A2 里一度恢复到 baseline best 以上。
- 但它仍在 best 后下滑，必须等到 400 epoch 才能判断 det-only B 是否自然 late-regress。
- 由于该 run 继承了正常 A2，而该 A2 已经退化，因此它不能单独证明 B KD 是主因。

## 4. 哪些实验建议继续

### 继续等待完成

| 实验 | 建议 | 原因 |
|---|---|---|
| `s B det-only r2` | 继续跑到 400 | 已到 `292/400`，是判断 B det-only 是否也 late-regress 的关键证据；不要中断 |

### 可以作为下一步最小新增实验

| 实验 | 优先级 | 原因 |
|---|---:|---|
| `m A2 lr3e-4` | P1 | 原计划 P1-4 尚未完成；用于判断 m 是否主要对 A2 lr=1e-3 过敏 |
| `s A2 short/early-stop probe` | P1 | s 的 best 均在 epoch 13 左右，说明 shorter A2 或 early-stop 可能比继续 B 调参更直接 |
| `s A2 lr3e-4 short` | P2 | 若要验证 s 的 practical fix，可围绕 lr3e-4 + shorter A2 做最小链路 |

说明：

- `m A2 lr3e-4` 仍有价值，因为 `m A2 det-only` 说明“只关 aux”不够，但还没有排除“较低 A2 lr 能保住 m”的可能。
- `s A2 short/early-stop` 比继续 alpha sweep 更有信息量，因为 s 的 peak 已经清楚出现在 A2 早期。

## 5. 哪些实验暂时不需要继续

| 实验类型 | 判断 | 原因 |
|---|---|---|
| 新的 `s alpha_kd` sweep | 暂停 | `0.5` 和 `0.25` 都未解决 final 退化；继续扫 B KD 会混淆 A2 因果 |
| `m full B` / `m B400` / `m B800` | 不启动 | m A2 det-only 已低于 baseline final，B full 没有解释价值 |
| KD decay / KD stop / dynamic gap | 暂停 | 属于 B 阶段增强；A2 损伤未定位前不应扩展 |
| seed42/123 扩展 | 暂停 | 当前 seed0 因果尚未定位，扩 seed 会消耗 GPU 且解释不清 |
| 大矩阵实验 | 暂停 | 当前需要最小定位，不需要铺矩阵 |
| mosaic 新实验 | 暂停 | 当前问题限定在 formal no-mosaic；mosaic 只作为历史证据参照 |

## 6. 当前推荐决策

1. **不要继续做 B 阶段 KD sweep。**
   - 已有 `alpha_kd=0.5/0.25` 说明降低 KD 强度不能解决 final 退化。

2. **不要开 m full B。**
   - m 在 A2 det-only 下都未达到 baseline final，直接开 B 会把 A2 损伤带入 B。

3. **让 `s B det-only r2` 跑完。**
   - 它是当前唯一还在跑的诊断实验，已接近后段，能回答 B det-only 自然 late-regression 的程度。

4. **下一条如果要开，应优先是 `m A2 lr3e-4` 或 `s A2 short/early-stop`。**
   - `m A2 lr3e-4` 用于补齐原 P1-4。
   - `s A2 short/early-stop` 用于验证 s 的 A2 早期 peak 是否能作为 practical fix。

5. **A2 诊断以后必须同时报告 baseline best、baseline final、baseline drop。**
   - 只用 safe threshold 会漏掉 protocol late-regression 与额外退化的区别。

## 7. 当前远端 GPU 状态快照

时间：`2026-06-12 02:43:13 CST`

```text
GPU0: 17428 / 24564 MiB, util 99%
GPU1: 3844 / 24564 MiB, util 13%
```

诊断相关 active process：

```text
pid=216504 gpu=0 mem=5518MiB ladd_hbb_ogsod11s_formal_nomosaic_yolo11s_cap2_s0_diag_capkd_s_s0_bdetonly_b400_r2_b_e400_b64_s0_gpu0
```

GPU1 当前基本空闲；如果要补 `m A2 lr3e-4`，资源上可行。但从因果清晰度看，建议先把本次证据提交并明确下一步只开一个最小 P1，而不是继续铺多条。
