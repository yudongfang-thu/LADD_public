# LADD 阶段与协议复盘分析 20260615

本文回答当前主线诊断里的三个问题：

1. 旧 Sixiang/90 时代是否有 A1 直接跳 B、A2 直接跳 B 的相似证据。
2. 当前 OGSOD/HBB mosaic 与 no-mosaic 的 A2 选择和 B 入口是否可比。
3. 下一步是否应继续围绕 A2 长短、B 入口冲击、BN freeze 和分解质量做分析。

结论先行：

- 旧 Sixiang 中确实存在 `A1 -> B1 -> B2` 的 no-A2 counterfactual，但它低于 SAR baseline；因此旧证据不支持“跳过 A2 直接 B 会自然变好”。
- 旧 Sixiang 也有 `A2 -> B` 分支，B 200 epoch 后仍低于 A2 起点；因此旧证据不支持“长 A2 后直接 B 一定可靠”。
- 当前 HBB mosaic seed0/123 更像“短 A1 + A2 epoch1 checkpoint -> B”，不是旧 Sixiang 的长 A1 直接 B。
- 当前 HBB no-BN-freeze 的 B 入口存在极深 early shock，BN-freeze 明显压低该 shock；这更像 BN running stats 或训练态切换造成的 detector 污染，而不是单纯 KD loss 量级异常。
- mosaic100 能恢复，可能与其 B 阶段 800 epoch 长调度和增强协议有关；不应把 B100/B120/B200 的前段现象直接外推到 B800。

## 1. 旧 Sixiang 阶段跳转证据

图：旧 Sixiang MAINLINE、NEWCHAIN、D1 修正线。

![old_sixiang_stage_jump](../figures/old_sixiang_stage_jump_20260615/old_sixiang_stage_jump_curves_20260615.png)

数据：`old_sixiang_stage_jump_summary_20260615.csv`

| 线 | 真实阶段 | 关键结果 | 解释 |
|---|---|---:|---|
| `MAINLINE_20260427` | A1 50 -> A2 100 det=0 -> C | A2 last 到 0.05873/0.01107/0.00014，C best 到 0.55716/0.55937/0.55897 | A2 无 detection 监督导致 backbone 崩塌，C 在恢复 |
| `NEWCHAIN_20260427` | A1 100 -> B1 det=0 KD -> B2 det=1 KD | B2 best 0.53972/0.54219/0.54461，低于 SAR baseline 0.55304 | 长 A1 直接进 B1/B2 并不好 |
| `D1_A2DET1` | A1 50 -> A2 100 det=1 -> C | A2 best 0.56331，C best 0.57272 | A2 保留 detection 后修复旧崩塌 |
| `G1_A2DET1_reach1` | D1 + reach=1.0 | best 0.58258 | 旧 Sixiang 健康线来自 A2 det=1 + reach 强度降低 |

图：更早的 A2 -> B 分支。

![old_sixiang_a2_to_b](../figures/old_sixiang_stage_jump_20260615/old_sixiang_a2_to_b_branch_20260615.png)

| 线 | 真实阶段 | 关键结果 | 解释 |
|---|---|---:|---|
| `fullchain_from_a1reach` A2 | 从较长 A1 reach-refine 后进 A2 | A2 best 0.53909，last 0.53586 | A2 本身略高于分支起点 |
| `fullchain_from_a1reach` B | 从 A2 last 进入 B, resaux=0.10/0.25/0.50 | B best 0.53145/0.53084/0.53250 | B 没有继续提升，反而低于 A2 |
| 后续 C | 从 B last 进入 C | C best 约 0.53898 | 后续 C 只能回到 A2 附近 |

这说明旧 Sixiang 的历史经验并不是“跳 B 或直接 B 更好”，而是：阶段入口的 checkpoint 状态会强烈影响后续优化；A2 如果被 reach-only 拉偏，会造成恢复型曲线；A2 如果保留 detection 并降低 reach，整体更健康。

## 2. 当前 HBB B 入口量化

图：当前 HBB n 模型 B 入口、A2 selected epoch 和 final gain。

![current_hbb_b_entrance](../figures/stage_forensics_20260615/current_hbb_b_entrance_forensics_20260615.png)

数据：

- `hbb_b_entrance_metrics_20260615.csv`
- `hbb_b_early_shock_loss_probe_20260615.csv`

### 2.1 B 入口 shock

| 组 | n | B ep1->min20 mean | B best gain | B final gain | 解释 |
|---|---:|---:|---:|---:|---|
| mosaic100 cap2 | 3 | -0.2444 | +0.0251 | +0.0250 | 入口明显下探，但 800ep 后能恢复并超过 baseline |
| no-mosaic cap2 no-BN-freeze complete | 2 | -0.4185 | +0.0182 | +0.0211 | 入口极深下探，健康 seed 可恢复 |
| no-mosaic cap2 BN-freeze | 2 | -0.0103 | +0.0148 | +0.0150 | 入口 shock 几乎消失，但 best/final gain 略低 |
| no-mosaic cap2 unstable | 2 | -0.2584 | -0.0196 | -0.2964 | early shock/late regression/NaN 路径，不可算性能均值 |

### 2.2 early shock 对应 loss

| run | B ep1 | min20 AP | drop | min20 val box | min20 val cls | min20 train kd | 解释 |
|---|---:|---:|---:|---:|---:|---:|---|
| no-mosaic cap2 s0 no-BN-freeze | 0.53675 | 0.04351 | -0.49324 | 2.71947 | 2.17732 | 2.96849 | detector loss 和 KD 同时异常大 |
| no-mosaic cap2 s42 no-BN-freeze | 0.52719 | 0.18343 | -0.34376 | 2.38625 | 1.46840 | 0.25908 | detector loss 大，KD 不一定异常 |
| no-mosaic cap2 s123 BN-freeze | 0.53054 | 0.52252 | -0.00802 | 1.23535 | 0.56194 | 0.22673 | BN-freeze 稳住 detector |
| no-mosaic cap2 s42 BN-freeze | 0.53850 | 0.52596 | -0.01254 | 1.27672 | 0.57859 | 0.32711 | BN-freeze 稳住 detector |
| old-B crash s123 | 0.52182 | 0.01673 | -0.50509 | 3.36041 | 19.60610 | 0.29566 | cls loss 爆炸，后续 NaN/collapse |

这个表支持一个更具体的判断：B 入口问题不是单纯 `alpha_kd` 太大。至少在 s42 no-BN-freeze 中，KD loss 不大但 AP 仍明显下探，val detector loss 明显恶化；BN-freeze 后相同类型入口几乎不下探。因此 BN running stats/训练态切换仍是当前最像主因的机制之一。

## 3. mosaic 与 no-mosaic 的 A2 选择差异

已有数据：`ladd_a2_bestpt_lineage_protocol_compare_20260614.csv`

| protocol | seed | A2 best epoch | A2 best | A2 last | selected reach | selected task |
|---|---:|---:|---:|---:|---:|---:|
| mosaic100 | 0 | 1 | 0.42623 | 0.40143 | 0.00731 | 1.05179 |
| mosaic100 | 42 | 50 | 0.40313 | 0.40313 | 0.00119 | 0.50336 |
| mosaic100 | 123 | 1 | 0.42489 | 0.40238 | 0.00682 | 1.05177 |
| no-mosaic | 0 | 49 | 0.56273 | 0.56200 | 0.00105 | 0.35099 |
| no-mosaic | 42 | 44 | 0.56198 | 0.56141 | 0.00108 | 0.36837 |
| no-mosaic | 123 | 50 | 0.56574 | 0.56574 | 0.00098 | 0.34693 |

这个差异很关键：

- mosaic seed0/123 的 B 入口来自 A2 epoch1 best，这不是“长 A2 收敛后的解耦网络”。
- no-mosaic 的 B 入口来自 A2 late best/last，reach 更收敛、task loss 更低，但 B 入口不一定更稳。
- 因此 mosaic 的成功可能不是因为解耦目标更符合设计，而可能是“选中了更早、更软、更未充分解耦的 A2 checkpoint”，再由 B800 长调度慢慢修复。

## 4. 当前不能直接完成的分析

权重级分解质量分析需要 checkpoint 和配对数据。当前本机状态：

数据：`decomposition_checkpoint_availability_20260615.csv`

| 范围 | 状态 | 说明 |
|---|---|---|
| HBB no-mosaic n/s A2/B | 多数可用 | 90 snapshot 中有 no-mosaic n/s A2/B 权重 |
| HBB mosaic100 n cap2 A2/B | 本机缺权重 | 只有 results.csv，没有 A2 epoch1/best/last 权重 |
| old Sixiang MAINLINE | 可用 | old OBB analyzer 可直接用于部分旧权重 |
| old D1/G1 | 部分缺原始 run 权重 | d1_data 有 CSV，但完整 run 权重未在当前拷贝中定位到 |

因此下一步如果要做 `z_t/u_t`、`z_s/r_s` 余弦相似度，应先做 HBB 版 analyzer，并优先比较：

1. no-mosaic n cap2 s0 A2 best vs last；
2. no-mosaic n cap2 s42 A2 best vs last；
3. no-mosaic s cap2 s0 A2 best vs last；
4. 如能从 90 再拉 mosaic100 权重，再比较 mosaic seed0/123 A2 epoch1/best 与 no-mosaic late A2。

## 5. 当前最强解释

我现在对主线问题的排序如下：

1. **B 入口 detector/BN 状态冲击是实证最强的现象**。no-BN-freeze 的 early shock 可达 -0.34 到 -0.49 AP，BN-freeze 后约 -0.01。
2. **A2 检测 AP 降低本身不是根因**。旧 Sixiang 已经说明 A2 可以低，但问题在于后续入口是否可恢复以及恢复是否带来真正增益。
3. **mosaic100 的稳定高增益不代表方法机制更正确**。它可能依赖 A2 early checkpoint + B800 长调度 + mosaic/close-mosaic 的联合路径。
4. **B100/B120/B200 诊断不能替代 B800**。cos 或长程 schedule 下，前 100 epoch 和一个独立 B100 实验不是同一个优化过程。
5. **单纯降低 B LR 或 KD warmup 不足以解释全部现象**。B-lr1e-3 能降低早期 shock，但 seed123 仍 late-regress；KD 不大时也可能 detector loss 下探。

## 6. 建议下一步

优先级从高到低：

1. 先不要继续大规模铺实验，先实现 HBB 分解质量 analyzer。
2. 在同一套 HBB analyzer 下比较 no-mosaic A2 best/last 的 `z_t-u_t cos`、`z_s-r_s cos`、`z_s-z_t cos`、norm ratio。
3. 如果可远程访问 90，补拉 mosaic100 cap2 s0/s123 的 A2/B 权重，尤其是 A2 epoch1/best/last。
4. 只做一条真正可解释的 N 模型 B800 复盘实验：BN-freeze + A2 selected checkpoint 固定，避免 B100 误导。
5. 对已经失败的 no-BN-freeze/old-B crash，优先当作 BN/入口异常证据，不进入性能均值。

