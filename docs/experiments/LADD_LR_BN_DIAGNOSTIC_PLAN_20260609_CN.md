# LADD LR / BN / Schedule 诊断计划

最后更新：2026-06-09

本文档只定义诊断实验，不选择、不改写 LADD 主线结论。当前目标是判断 A2/B 稳定修正是否因为学习率过低、BN-freeze 过强或 B 阶段 cosine decay 过长而损失最终性能。

远端连通与最小验证记录见：[LADD_LR_BN_DIAGNOSTIC_SMOKE_20260609_CN.md](LADD_LR_BN_DIAGNOSTIC_SMOKE_20260609_CN.md)。

## 1. 为什么暂不直接选择主线

当前证据同时指向“需要稳定修正”和“稳定修正可能偏保守”：

- 旧 A2 默认高学习率会在 YOLO11n seed0 的 A2 约 epoch 8 出现检测 loss NaN；
- 旧 B 默认高学习率和 warmup 会在 YOLO11n seed123 的 B 阶段约 epoch 429 出现检测 loss NaN；
- B `lr0=1e-3` 能防 NaN，但 seed123 出现 best 低、last 大退化；
- `B_FREEZE_BN_STATS=1` 能让 YOLO11n seed0/42/123 三 seed 稳定正收益，但 seed0 的峰值低于 no-freeze 健康 run；
- YOLO11s seed0 BN-freeze 版本在 B epoch 263 达到 best，后续到 epoch 800 明显退化，说明 BN-freeze 不是全容量通解，也可能存在 schedule / LR tail 问题。

因此当前不应把某个设置直接宣布为最终主线，而应先用受控 LR / BN / schedule 诊断回答下面三个问题：

```text
Q1: A2/B lr0=1e-3 是否过低？
Q2: BN-freeze 是否过强？
Q3: B 阶段 800ep cosine decay 是否导致后期退化？
```

## 2. 诊断假设

| 假设 | 内容 | 对应实验 |
|---|---|---|
| H1 | `B_LR0=0.001` 太低，限制峰值 | `lr2e3_freeze`, `lr3e3_freeze`, `lr5e3_freeze` |
| H2 | `B_LRF=0.01` 导致尾部 lr 太低，B 后半程退化或停滞 | `tail_lr1e3_lrf0p1`, `tail_lr2e3_lrf0p1` |
| H3 | 全程 BN-freeze 过强，降低峰值 | `nofreeze_lr1e3` |
| H4 | delayed BN-freeze 可以兼顾峰值和稳定 | `delayed_bn200_lr1e3`, `delayed_bn200_lr2e3`, `delayed_bn100_lr2e3`, `delayed_bn400_lr2e3` |
| H5 | B=800 过长，best 已在 250-400 附近出现，后续训练只引入退化 | `b400_lr1e3_freeze` |

`B_FREEZE_BN_AFTER_EPOCH=N` 使用内部 zero-based epoch 约定：B epoch `< N` 时 BN 正常更新 running stats；从内部 epoch `N` 起冻结。比如 `N=200` 表示前 200 个 B epoch 正常更新，从第 201 个记录行开始 freeze。

## 3. 评价指标

每条诊断 run 至少看：

- `best AP50-95`
- `last AP50-95`
- `drop_best_to_last = best - last`
- `best_epoch`
- 是否出现 `NaN/Inf`
- `bn_running_var_max_global`
- `bn_running_var_p95_max`
- AP at epoch 200 / 300 / 400 / 800
- 相对 SAR baseline 增益
- teacher gap recovery

新增 `ladd_diagnostics.csv` 会记录：

```text
epoch, stage, lr_pg0, lr_pg1, lr_pg2,
train_box_loss, train_cls_loss, train_dfl_loss,
kd_loss, reach_match_loss, reach_rank_loss,
bn_running_var_max, bn_running_var_mean, bn_running_var_p95,
bn_running_mean_abs_max, bn_num_layers, bn_stats_mode,
nan_or_inf_detected
```

默认记录 BN 诊断，不记录梯度诊断。若设置 `LADD_DIAG_LOG_GRAD=1`，额外记录：

```text
grad_norm_total, grad_norm_backbone, grad_norm_neck, grad_norm_head
```

## 4. 选择规则

| 观察 | 解释 | 后续选择 |
|---|---|---|
| `lr2e3` / `lr3e3` 在 seed0 提升且 seed123 不崩 | 当前低 LR 确实限制性能 | 把较高 LR 作为候选，补 seed42 |
| `lrf0p1` 提升 last 或减少 `drop_best_to_last` | cosine tail lr 太低 | 保留 `lrf=0.1` 作为候选 |
| `nofreeze` 提高 seed0 但 seed123 退化或 BN var 爆 | BN-freeze 是必要稳定项 | 不把 no-freeze 作为主线，只作诊断 |
| `delayed_bn200` 接近 no-freeze 峰值并保持 seed123 稳定 | delayed BN-freeze 兼顾峰值和稳定 | 作为新主线候选，补 seed42 |
| `b400` 与 `b800` best 接近且 last 更好 | B=400 / early stop 可作为论文训练预算候选 | 优先考虑 400ep 或强 early-stop |
| 所有高 LR / delayed / no-freeze 都不能稳定超过 current_stable | 当前稳定候选足够保守可靠 | 保留 current_stable |

## 5. 第一批实际执行建议

第一批只跑 YOLO11n seed0 和 seed123 的 smoke tier：

```bash
DRY_RUN=1 LAUNCH=0 ladd/scripts/launch_ladd_lr_bn_diag_matrix.sh n 0 0 smoke
DRY_RUN=1 LAUNCH=0 ladd/scripts/launch_ladd_lr_bn_diag_matrix.sh n 123 1 smoke
```

确认命令、路径、baseline 权重和 run tag 后，再按 GPU 分配手动启动。不要先跑 seed42，不要先跑 YOLO11s/m。等 n seed0/123 判断出 1-2 个候选后，再用 seed42 复核。最后再把最优候选扩展到 YOLO11s seed0。

## 6. Smoke Tier 配置

| Tag | 目的 |
|---|---|
| `current_stable` | 当前稳定候选参照 |
| `lr2e3_freeze` | 测试当前性能下降是否由 B lr0 过低导致 |
| `lr3e3_freeze` | 测试更高但仍低于旧 0.01 的学习率是否稳定 |
| `tail_lr1e3_lrf0p1` | 测试后期 lr 尾部过低是否导致 B 后半程退化或没收敛 |
| `nofreeze_lr1e3` | 隔离 BN-freeze 对峰值的影响 |
| `delayed_bn200_lr1e3` | 测试前期允许 BN 适应、后期冻结是否兼顾峰值和稳定 |
| `delayed_bn200_lr2e3` | 测试“稍高 LR + delayed BN-freeze”是否优于 current_stable |
| `b400_lr1e3_freeze` | 测试是否 B=800 太长导致后期退化 |

## 7. Full Tier 增补

full tier 在 smoke 基础上增加：

| Tag | 目的 |
|---|---|
| `lr5e3_freeze` | 探测接近旧高 LR 但仍温和的上界，风险较高 |
| `tail_lr2e3_lrf0p1` | 测试高一点的主 lr 和更高尾部 lr |
| `delayed_bn100_lr2e3` | 判断 delayed freeze 的较早时机 |
| `delayed_bn400_lr2e3` | 判断更晚 freeze 是否提高峰值但不退化 |

## 8. 汇总命令

诊断完成后汇总：

```bash
python ladd/tools/summarize_ladd_lr_bn_diagnostics.py \
  runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd \
  --output-csv summary.csv \
  --output-md summary.md
```

`summary.md` 会自动列出：

- best AP 排名前 20；
- `drop_best_to_last` 最大的 20；
- `bn_running_var_max_global` 最大的 20；
- 出现 NaN/Inf 的 run；
- 每个配置组的均值。
