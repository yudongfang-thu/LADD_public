# LADD A2 Selection 诊断阶段性结果（2026-06-12）

## 1. 数据源与状态

本次同步的是双卡 4090 服务器 `/root/shared-nvme/LADD_public_p1` 上 2026-06-12 启动的 A2 / A2->短 B 诊断实验。远端代码 commit 为 `665cdb4871d9a5620befc46d93d693765779d60b`。

轻量证据目录：

```text
ladd/results/a2_selection_20260612/current_evidence/
```

机器可读汇总：

```text
docs/experiments/ladd_a2_selection_summary_20260612.csv
ladd/results/a2_selection_20260612/current_evidence/summary/ladd_a2_selection_summary_20260612.csv
```

本次只同步 `results.csv`、`args.yaml`、`manifest.txt`、`ladd_diagnostics.csv`、chain log 和关键 log extract；未同步 checkpoint 权重、TensorBoard event、wandb、完整 run 目录或完整大日志。

## 2. 汇总表

| Run | 状态 | A2 best | A2 last | B best | B last | 解释 |
|---|---|---:|---:|---:|---:|---|
| `s_A2_lr3e4_short13_B1` | complete | 0.63057@13 | 0.63057@13 | 0.60811@1 | 0.60811@1 | Best s A2 selection candidate so far; A2 final equals peak and clears s safe threshold. |
| `s_A2_lr1e4_short15_B1` | complete | 0.63051@13 | 0.62647@15 | 0.60908@1 | 0.60908@1 | Lower LR keeps best high but still drops by epoch 15; worse than exact short13 selection. |
| `s_A2_lr3e4_short13_Bdet200` | running | 0.63057@13 | 0.63057@13 | 0.62436@84 | 0.62326@117 | Running snapshot: B det-only has not improved over A2 best; current B best is below A2 best. |
| `m_A2_lr3e4_full50_retry3_B1` | complete | 0.64611@8 | 0.63911@50 | 0.62682@1 | 0.62682@1 | Low LR full50 does not recover m; A2 best and final remain below m baseline final/best. |
| `m_A2_short10_B1` | complete | 0.64411@4 | 0.64276@10 | 0.62642@1 | 0.62642@1 | Short normal-lr A2 is still weak for m. |
| `m_A2_lr3e4_short10_B1` | complete | 0.64929@4 | 0.63990@10 | 0.62898@1 | 0.62898@1 | Low-LR short10 m reaches near baseline final at epoch 4 but collapses by epoch 10; suggests very-short A2 if m is pursued. |

## 3. 关键结论

1. **s 的最佳方向进一步收敛到 `A2 lr3e-4 short13`。** 该 run 的 A2 best/last 均为 `0.63057@13`，超过 s SAR baseline best `0.62897` 和 safe threshold `0.62697`。相比 `short15`，short13 避免了最后两轮回落。

2. **更低 A2 LR 不是主要答案。** `s A2 lr1e-4 short15` 的 A2 best 仍有 `0.63051@13`，但 last 掉到 `0.62647@15`，略低于 safe threshold，说明继续降 LR 不如直接选定更短 A2 峰值。

3. **`s short13 + B det-only200` 当前没有显示 B 增益。** 该 run 仍在运行，snapshot 到 B epoch 117，B best 为 `0.62436@84`、last 为 `0.62326@117`，均低于 A2 起点 `0.63057@13`。这支持当前判断：短 B 是否有价值需要等完成，但截至目前 A2 best 本身更强。

4. **m 仍不适合作为当前主线容量。** `m A2 lr3e-4 full50 retry3` 完成后 A2 best `0.64611@8`、last `0.63911@50`，低于 m SAR baseline final `0.64903` 和 best `0.65580`。`m A2 lr3e-4 short10` 虽然 epoch 4 到 `0.64929`，但 epoch 10 已掉到 `0.63990`，说明 m 若继续定位也只能考虑 very-short A2，而不应启动 full B。

5. **B=1 结果只作为 chain 占位，不解释方法效果。** 所有 B1 的 AP 都很低，符合占位性质，不应与 A2 或正式 B 阶段比较。

## 4. 当前决策建议

- s：保留 `A2 lr3e-4 short13` 作为当前最干净的 A2 selection candidate。等待 `B det-only200` 完成；若 B best 仍低于 A2 best，则主线应直接围绕 A2 best/short A2 而不是 B 修复展开。
- m：暂停 full B / B400 / B800。若需要补证，只做 `A2 lr3e-4 short4/short5` 这类 very-short A2 定位。
- B 阶段：暂不继续 alpha sweep、KD decay、KD stop 或 dynamic gap，避免在 A2 损伤未完全解决前混淆因果。

## 5. Git command

```bash
git add docs/experiments/LADD_A2_SELECTION_ANALYSIS_20260612_CN.md \
  docs/experiments/ladd_a2_selection_summary_20260612.csv \
  ladd/results/a2_selection_20260612/current_evidence

git commit -m "docs(ladd): add A2 selection diagnostics" -m "Summarize the 2026-06-12 A2 selection batch on the dual-4090 server, including short13/short15 s runs, m low-LR/short-A2 probes, and the running s short13+B-det-only snapshot.

Key evidence: s A2 lr3e-4 short13 reaches 0.63057@13 and is the strongest current s candidate; s lr1e-4 short15 drops by epoch 15; m low-LR/full50 and short10 variants remain below m baseline, so m full B should stay paused. Validation: recomputed best/last AP from synced results.csv and scanned compact logs for fatal errors. Excluded checkpoint weights, TensorBoard events, wandb, full run directories, and full large logs.

git_command: git add docs/experiments/LADD_A2_SELECTION_ANALYSIS_20260612_CN.md docs/experiments/ladd_a2_selection_summary_20260612.csv ladd/results/a2_selection_20260612/current_evidence && git commit -m 'docs(ladd): add A2 selection diagnostics' && git push origin main"

git push origin main
```
