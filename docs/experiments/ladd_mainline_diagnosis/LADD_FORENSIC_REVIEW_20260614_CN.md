# LADD 主线 forensic review 初版（2026-06-14）

本文用于把近期大量 LADD 诊断实验收束成可解释证据。当前目标不是继续开实验，而是回答：

1. 历史健康 LADD 主线到底是什么；
2. 近期失败实验和历史主线差在哪；
3. 哪些实验能解释主线，哪些只能作为诊断；
4. 下一步如果要 replay，只允许 replay 什么。

## 1. 证据入口

| 类型 | 路径 |
|---|---|
| LADD 线 registry summary | [ladd_line_registry_summary_20260614.csv](ladd_line_registry_summary_20260614.csv) |
| 当前 B800 曲线分析 | [../LADD_B800_RESTART_CURVE_ANALYSIS_20260614_CN.md](../LADD_B800_RESTART_CURVE_ANALYSIS_20260614_CN.md) |
| 当前 baseline overlay | [../LADD_CURRENT_BASELINE_OVERLAY_ANALYSIS_20260614_CN.md](../LADD_CURRENT_BASELINE_OVERLAY_ANALYSIS_20260614_CN.md) |
| 历史收敛主线 | [../LADD_CONVERGED_MAINLINE_COMPARISON_20260613_CN.md](../LADD_CONVERGED_MAINLINE_COMPARISON_20260613_CN.md) |
| Mosaic vs no-mosaic 协议重合对比 | [LADD_PROTOCOL_MOSAIC_VS_NOMOSAIC_OVERLAP_20260614_CN.md](LADD_PROTOCOL_MOSAIC_VS_NOMOSAIC_OVERLAP_20260614_CN.md) |
| B100/B120 压缩 schedule 归档 | [../LADD_B_ENTRANCE_COMPRESSED_SCHEDULE_ARCHIVE_20260614_CN.md](../LADD_B_ENTRANCE_COMPRESSED_SCHEDULE_ARCHIVE_20260614_CN.md) |
| 全局 registry | [../registry/experiment_registry_20260614.csv](../registry/experiment_registry_20260614.csv) |

注意：registry 中的 `claim_usable=yes/partial/no` 是自动粗标，只用于筛查。LADD 主线结论必须以本文的人工分类为准。

## 2. 训练协议分层

| protocol_id | 当前含义 | 在 LADD 主线中的用途 |
|---|---|---|
| `formal_nomosaic_800` | no-mosaic, 计划 800 epoch | 主线、baseline、正式长程诊断的唯一核心层 |
| `formal_nomosaic_compressed` | no-mosaic, 但 B100/B120/B200/B400 等短程 | 入口/早期趋势诊断，不能直接判断 B800 后期 |
| `historical_mosaic_mainline` | 早期 mosaic/close-mosaic 主线 | 证明 LADD 曾经健康，但不能与 no-mosaic 主表直接混比 |
| `smoke_probe_partial` | smoke/probe/短 A2/B1/中断快照 | 工程验证或局部问题定位 |
| `unknown_protocol` | 缺 args 或协议不完整 | 暂不进入结论 |

当前绝大多数 LADD 证据属于 no-mosaic，但必须进一步区分 `formal_nomosaic_800` 和 compressed schedule。

## 3. 历史健康主线

历史 90 服务器 no-mosaic 主线说明：LADD 不是天然不收敛。

| 实验 | 协议 | 结果 | 读法 |
|---|---|---:|---|
| n cap2 seed0 no-BN-freeze | formal no-mosaic B800 | best `0.57662@725`, last `0.57504` | 健康长程主线，best 在后期 |
| n cap2 seed42 no-BN-freeze | formal no-mosaic B800 | best `0.57420@735`, last `0.57293` | 健康长程主线 |
| n original/no-cap2 seed0 | formal no-mosaic B800 | best `0.57821@730`, last `0.57517` | n no-mosaic 最高单点，作为消融/诊断 |
| n seed123 old-B | formal no-mosaic partial | best `0.52182@1`, final `0.00000` | 暴露旧 B 数值崩溃 |
| n seed123 bstable1e3 | formal no-mosaic B800 | best `0.56161@165`, last `0.52875` | 降 B LR 防 NaN 但不能防 late regression |
| n seed0/42/123 BN-freeze | formal no-mosaic B800 | best 约 `0.57269-0.57615` | 保守稳定修复，形成 n 三 seed 正收益闭环 |

历史 mosaic100/close@100 主线也很重要：六条 n run 都没有 collapse，best 相对同协议 SAR baseline 大约提升 `+0.02072` 到 `+0.02750`。这只能作为“方法潜力与历史稳定性”证据，不能直接和当前 no-mosaic 主表混用。

新增协议重合对比说明：在最对称的 n/cap2 对照里，性能均值只计算健康完成 run，collapse/late-regression 单独计入稳定性事件。按这个 clean mean 口径，no-mosaic 的 raw AP 往往高于 mosaic100，因为 no-mosaic SAR baseline 本身更强；但 no-mosaic 的相对 LADD gain 更小，且 seed123 normal-BN 暴露 collapse/late-regression。这个现象支持“训练协议改变了 LADD 的收益边界和稳定性余量”，但目前还不能外推到 s/m，因为缺少同容量 mosaic 对称组。

## 4. 当前 B800 重启批次

| 实验 | 入口 | 当前/停止结果 | Forensic 读法 |
|---|---|---:|---|
| N1 SAR-best det-only | SAR baseline best detector | best `0.57583@343`, last `0.57540@353` | detection-only continuation 正常，B800 schedule 不是根因 |
| N1 SAR-last det-only | SAR baseline last detector | best `0.57737@355`, last `0.57695@360` | 当前最强安全对照，说明 baseline continuation 很强 |
| N2 A2-best full LADD | A2 best checkpoint | best `0.55681@214`, NaN around 229 | full LADD B 有信号但数值不稳定 |
| N2 A2-last full LADD | A2 last checkpoint | best `0.56073@271`, final finite `0.46290@319`, NaN around 319 | A2-last 不自动救主线，且崩溃更晚 |
| N3 YOLO-init + A2 decomp | detector 从 YOLO init, decomp from A2 | stopped around `0.494` | 诊断 split-load，可排除为当前主线候选 |
| N4 YOLO-init + A2 decomp + KD warmup | YOLO-init split-load + KD ramp | stopped around `0.476` | KD-only warmup 未显示优势，且入口不等同原主线 |

关键结论：

1. N1 强而稳定，排除了“B800 schedule / BN-freeze / 训练框架本身必然导致失败”。
2. N2 的问题不是一开始就坏，而是在中期出现 NaN/Inf；因此更像 A2 checkpoint + full B LADD objective 的数值稳定问题。
3. N3/N4 的入口语义已经改变，不应被当作 LADD 主线失败证据；它们说明“弱 detector + decomp split-load”目前无价值。
4. B100/B120/B200 的 compressed schedule 只能解释入口和短期趋势，不能等价为 B800 的后期能力。

## 5. 当前最可疑的漂移轴

| 轴 | 需要核查的问题 |
|---|---|
| B 阶段 loss 组合 | 历史健康主线是否只开 detection + rec + KD + reach？近期是否误开 sep/residual aux/其他 optional loss？ |
| A2/B checkpoint lineage | 历史健康主线进入 B 的 checkpoint 与当前 N2 A2-best/A2-last 是否语义一致？ |
| BN 行为 | BN-freeze 是稳定修复，但它也改变峰值；需要区分“稳定修复”与“性能最优” |
| NaN recovery | 当前恢复逻辑本身会因 deepcopy 失败而退出，这是第二层工程问题；第一层仍是 loss 非有限 |
| schedule 语义 | compressed schedule 不能直接解释 B800；必须按计划 epoch 对齐 LR 曲线 |
| 服务器/代码 commit | 90 健康主线、4090 当前诊断、AutoDL shutdown 结果必须按 commit/args/manifest 做 diff |

## 6. 人工分类建议

| 类别 | 包含 | 是否可进主线结论 |
|---|---|---|
| `mainline evidence` | 历史 90 no-mosaic n B800、BN-freeze n 三 seed、必要 baseline | 可以，但要标注协议和服务器 |
| `current failure evidence` | N2 A2-best/A2-last B800 NaN/退化 | 可以用于解释当前问题，不能作为最终性能 |
| `control evidence` | N1 SAR-best/SAR-last det-only B800 | 可以作为 detector continuation 对照 |
| `diagnostic only` | N3/N4 yolo-init split-load、B100/B120 compressed、A2 smoke/probe | 不能作为主线性能 |
| `historical context` | mosaic100/close@100 | 说明方法潜力，不与 no-mosaic 主表直接比较 |
| `archive/invalid` | 旧错误 CCLKD/旧崩溃/错误协议 | 只保留追溯 |

## 7. 当前工作结论

当前不能说“LADD 一无所获”。更准确的结论是：

1. LADD 在历史 formal no-mosaic n 和 mosaic100 协议下都出现过健康、后期继续增长的主线。
2. 当前 B800 重启批次说明 detection-only continuation 很强，full LADD 的 A2 入口有短期信号但中期数值不稳定。
3. 最近的 yolo-init split-load 实验是诊断分支，不是原主线复现，不能用来否定 LADD 主方法。
4. 目前最需要的是配置/loss/lineage diff，而不是继续堆实验。

## 8. 下一步 replay 约束

在完成配置 diff 前，不建议启动新大实验。如果必须做 replay，只允许一个目标：

```text
严格复现历史健康 n formal no-mosaic LADD 主线
```

replay 必须满足：

1. 使用同一协议：`imgsz=256, formal no-mosaic, B800, cos LR`。
2. 明确记录 A1/A2/B 每阶段 loss 组合。
3. 不启用近期额外 split-load、core warmup、KD warmup、yolo-init decomp 等诊断改法。
4. 打开必要 diagnostics：effective weights、NaN flag、grad norm、BN stats。
5. 先 dry-run 保存 args/manifest，再决定是否训练。

只有这个 replay 能回答：当前失败是代码/配置漂移，还是历史健康结果依赖某个尚未复现的隐含条件。
