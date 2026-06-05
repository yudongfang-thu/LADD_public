# LADD 实验计划

最后更新：2026-06-05

> 2026-06-04 对比方法决策更新：controlled main table 改为
> `FGD / LD / CCLKD-style / HalluciDet-style`。CrossKD 已停止并淘汰；CoLD
> 已降级归档。FGD/LD 修复前结果作废。当前权威说明见
> [`COMPARISON_EXPERIMENTS_CN.md`](COMPARISON_EXPERIMENTS_CN.md) 和
> [`../../comparison/IMPLEMENTATION_REVIEW_CN.md`](../../comparison/IMPLEMENTATION_REVIEW_CN.md)；
> 本文后续旧对比矩阵仅保留为历史计划记录。

> 2026-06-05 协议审计更新：双卡 4090 部署时 OGSOD HBB dataset yaml 被错误迁移为
> `nc=5` 旧类别表，正式协议应为 `nc=3`。因此双卡 4090 上 2026-06-04 启动的
> comparison smoke/formal partial runs 和 LADD BN-freeze sweep 全部作废，不进入
> active result table。当前不启动新实验，先完成代码和记录全面复核。

## 1. Baseline

目标：完成 OGSOD formal no-mosaic 协议下 YOLO11n/s/m/l 四档容量的 SAR/RGB baseline（主对比容量轴），为 LADD、消融和对比实验提供同 seed 起点。YOLO11x 作为容量趋势补充，不进入主对比矩阵。

协议：`imgsz=256, 800ep, cos_lr, full no-mosaic, default Albumentations`

执行策略更新：

- `LADD_public` 冻结为论文代码与实验记录的唯一源；服务器旧代码树只归档，不再反向覆盖 public。
- 双卡 4090 旧 `/root/shared-nvme/ladd` 已发现 active yaml 协议错误；错误结果已归档。修正后不得直接启动正式队列，必须先通过协议校验和人工复核。
- 117 已完成最终四方法真实 GPU smoke，但当前 GPU 被其他用户占用；不启动新任务。
- 4090D 已以无卡模式开机，仅用于恢复和归档已有结果。
- 90 保留已有进行中/历史任务，暂不新增受控对比任务。
- 正式比较实验从 `yolo11*.pt` 初始权重启动，不从 SAR baseline `best.pt` 继续训练；训练长度不作为对比指标，统一按 formal no-mosaic 协议训练到收敛。
- 若使用跨机器结果，论文中需要标注 machine/checkpoint provenance，并保留同 seed/同 epoch 的 sanity comparison。

### 1.1 当前状态

SAR baseline:

| Model | seed0 | seed42 | seed123 |
|---|---:|---:|---:|
| YOLO11n | 0.55654 ✅ | 0.55794 ✅ | 0.56128 ✅ |
| YOLO11s | 0.62897 ✅ | 0.62879 ✅ | 0.62357 ✅ |
| YOLO11m | 0.65580 ✅ | — | — |
| YOLO11l | 0.65427 ✅ | — | — |
| YOLO11x | 0.65867 ✅ | — | — |

RGB baseline:

| Model | seed0 | seed42 | seed123 |
|---|---:|---:|---:|
| YOLO11n | 0.63018 ✅ | 0.62664 ✅ | 0.62933 ✅ |
| YOLO11s | 0.65768 ✅ | 0.66218 ✅ | 0.65987 ✅ |
| YOLO11m | 0.67909 ✅ | — | — |
| YOLO11l | 0.68356 ✅ | — | — |
| YOLO11x | 0.68284 ✅ | — | — |

> YOLO11x 为容量趋势补充，非主对比轴。RGB seed0 已跑满 800 epoch，best epoch 为 538，last AP 为 0.65820。

### 1.2 当前可行性总览

主对比容量轴：n/s/m/l。✅ = 已完成/就绪，⏳ = 待完成，— = 不适用。

| Model | seed | SAR baseline | RGB teacher | LADD 可启动 | 对比实验可启动 |
|---|---:|---|---|---|---|
| YOLO11n | 0 | ✅ | ✅ | ✅ | ✅ |
| YOLO11n | 42 | ✅ | ✅ | ✅ | ✅ |
| YOLO11n | 123 | ✅ | ✅ | ✅ | ✅ |
| YOLO11s | 0 | ✅ | ✅ | ✅ | ✅ |
| YOLO11s | 42 | ✅ | ✅ | ✅ | ✅ |
| YOLO11s | 123 | ✅ | ✅ | ✅ | ✅ |
| YOLO11m | 0 | ✅ | ✅ | ✅ | ✅ |
| YOLO11m | 42 | ⏳ | ⏳ | ⏳ | ⏳ |
| YOLO11m | 123 | ⏳ | ⏳ | ⏳ | ⏳ |
| YOLO11l | 0 | ✅ | ✅ (90) | ⏳ 需执行机复核 | ⏳ 需执行机复核 |
| YOLO11l | 42 | ⏳ | ⏳ | ⏳ | ⏳ |
| YOLO11l | 123 | ⏳ | ⏳ | ⏳ | ⏳ |

> YOLO11x seed0 SAR/RGB 已完成，仅作容量趋势补充，不进入主对比矩阵。
> YOLO11s 三 seed SAR/RGB baseline 在源端已齐；117 旧权重 md5 曾与 90 不一致，已从 90 重拷贝并复核为源端记录 md5。

待补齐关键项：

| 缺口 | 影响 | 优先级 |
|---|---|---|
| YOLO11m seed42/123 SAR+RGB baseline | m 多 seed 验证 | P1 |
| YOLO11l seed42/123 SAR+RGB baseline | l 多 seed 验证 | P2 |
| 双卡 4090 YOLO11s teacher seed42/123 | s 多 seed 对比 | P1 |
| 双卡 4090 YOLO11m/l pretrain 与 teacher | m/l 容量扩展 | P2 |

执行路径：双卡 4090 `/root/shared-nvme/ladd`；数据 `/root/shared-nvme/OGSOD-1.0`。旧代码与已有结果归档到同盘带时间戳目录，并作为私有 checkpoint/result asset root 接入 public 运行目录。

### 1.3 待完成

| 优先级 | 实验 | 目的 |
|---|---:|---|
| ✅ | 归档双卡 4090 旧代码并部署 `LADD_public` | 已归档至 `/root/shared-nvme/archive/ladd_pre_public_20260604_220559` |
| 作废 | 双卡 4090 旧四方法目标机 smoke | 使用了错误 `nc=5` yaml，不作为 smoke 证据 |
| 作废 | 双卡 4090 YOLO11n 四方法正式三 seed 队列 | 已停止并归档，不能进入主表 |
| P0 | 公开仓库协议与代码审计 | 修正 yaml、CCLKD 实现和文档状态；推送后人工复核 |
| P1 | YOLO11m SAR/RGB seed42,123 | m 多 seed 验证 |
| P1 | YOLO11l seed0 SAR/RGB 可用性复核 | l 实验可在 90/4090D 启动 |
| P2 | YOLO11l SAR/RGB seed42,123 | l 多 seed 验证 |
| P3 | YOLO11x 多 seed | 容量趋势补充，非主表必须 |

### 1.4 判据

- n 三 seed 完成，gap 最大 → 主机制实验对象，同时也是所有对比方法的统一容量
- s 三 seed 完成 → 容量趋势验证，LADD 主表第二个容量点
- m/l 补齐多 seed → 证明方法在更大容量上仍然有效，LADD 主表第三、第四个容量点
- m 以上 gap 稳定在 ~0.02-0.03，LADD 涨点会变小，但多容量正趋势比单容量大涨幅更有说服力

---

## 2. LADD 主线

目标：在 formal no-mosaic 协议下，完成 LADD cap2 的多容量、多 seed 验证。

> 当前不新增 LADD 主线 run。先完成冻结对比实现的正式结果；已有 LADD 结果继续用于主线选择与崩溃分析。

主线配置：`A1=10 → A2=50 → B=800, cap2 reach-rank, A2/B MuSGD lr=0.001`

B 阶段温和修正：90 服务器上 YOLO11n cap2 seed123 的旧 B run 使用 `optimizer=auto, lr0=0.01`，前几轮 `lr/pg0` 到约 0.03，epoch 429 开始检测 loss NaN，最终 `last.pt` 被判定为 NaN/Inf 权重。同期 `reach_*` 为 0，`kd_loss` 仍有限，现象更像检测分支/优化器冲击，而不是 reach 或 cap2 单独爆炸。因此后续正式 LADD 默认采用 B 稳定设置：

```text
B_OPTIMIZER=MuSGD
B_LR0=0.001
B_LRF=0.01
B_WARMUP_EPOCHS=0
B_WARMUP_BIAS_LR=0.001
```

### 2.1 当前状态

| Model | seed0 | seed42 | seed123 |
|---|---:|---:|---:|
| YOLO11n cap2 `a2mu1e3` | 0.57662@725 ✅ | 0.57420@735 ✅ | old B 崩溃；`bstable1e3` 0.56161@165 后期退化 |
| YOLO11n cap2 BN-freeze | 0.57276@793 ✅ | 待补 | 0.57269@779 ✅ |
| YOLO11s cap2 | 0.63551@605，未满 800 | 4090D r2 0.60838@638，低于 baseline | 4090D r2 0.60849@513，低于 baseline |
| YOLO11m cap2 | B 异常，best 0.59796@1 | — | — |

补充说明：

- YOLO11n cap2 seed123 早先 B run 跑到 483 epoch 后出现 last AP 为 0 的异常；`bstable1e3` 完整跑满但 best 仅 `0.56161@165`，后期退化到 0.52875。
- 4090D 入口在关闭本地 tun 后恢复；HBB/HalluciDet-style 关键代码已同步并通过 hash、`py_compile`、`--help` 和 launcher `bash -n` 检查。4090D 当前 YOLO11n cap2 seed0 B run 在 epoch 342-346 连续 mAP50-95=0，best 0.54925@227，已停止以避免继续占用 4090D。
- 塌缩排查结论：YOLO11n seed0/123 坏 run 权重无 NaN/Inf，但 `last.pt` 的 BN `running_mean/running_var` 被污染；seed0 BN max running_var 到 1726，seed123 到 1333，而健康 seed42 约 47.7。
- `--freeze-bn-stats` / `FREEZE_BN_STATS=1` 已在 90 上完成 seed0/123 B 阶段，分别得到 `0.57276@793` 与 `0.57269@779`，说明 BN-freeze 是当前最可信的 B 稳定修复。
- 双卡 4090 已启动诊断实验 `bnfreeze_highlr_oldb_4090dual_diag_v1`：YOLO11n seed0/42/123 均从 90 的 `a2mu1e3_a2_e50` best 启动，B 阶段使用 old B 高学习率设置 `optimizer=auto -> MuSGD(lr=0.01)`、默认 warmup，同时启用 `FREEZE_BN_STATS=1`。该实验只回答“BN-freeze 能否单独救高学习率 B”，不作为正式主表协议。
- YOLO11s/m 当前数值不进入最终主表：s seed0 未满 800，m seed0 B 异常。

### 2.2 待完成

| 优先级 | 实验 | 前置 |
|---|---:|---|
| P0 | 补 YOLO11n seed42 BN-freeze | 形成严格同协议三 seed 稳定主线 |
| P1 | 在 public 最终代码上重跑 YOLO11s cap2 seed0 | s 容量验证 |
| P1 | 复盘 4090D YOLO11s 低结果 | 判断是否为协议/代码/环境差异 |
| P2 | 暂停 YOLO11m/l LADD 扩展 | 等 n/s 主线稳定后再排 |

### 2.3 预期主表

| Model | SAR baseline | LADD cap2 | Delta |
|---|---:|---:|---:|
| YOLO11n | 0.55859 ± 0.00216 | BN-freeze 2/3 done: 0.57273 | BN-freeze 2/3 done: +0.01381 |
| YOLO11s | 0.62779 ± 0.00289 | 0.63551@605 seed0 | +0.00654 seed0 |
| YOLO11m | 0.65580 | B 异常，暂停 | TBD |
| YOLO11l | 0.65427 | not started | TBD |

---

## 3. 消融实验

目标：证明 LADD 各组件的必要性。全部使用 YOLO11n seed0，单 seed。

| 优先级 | 消融 | 回答的问题 |
|---|---:|---|
| P1 | no cap2 (original rank loss) | cap2 反坍缩是否必要？性能代价多大？ |
| P1 | no teacher decomposition | 教师分解是否必要？ |
| P2 | single_proj vs split | student branch 设计选择 |
| P2 | lambda reach sweep {0.5, 1.0, 2.0} | reach 权重敏感度 |
| P3 | no reachability (reach_only_no_kd) | reach 独立于 KD 的贡献 |
| P3 | skip-B (A2→C) vs A2→B | B 阶段是否必要 |

前提：11n cap2 seed0 主结果已确认；正式批量启动建议等 11n cap2 三 seed 闭环后再排。

---

## 4. 对比实验

目标：与同类方法在同协议下公平对比。参见 [COMPARISON_EXPERIMENTS_CN.md](COMPARISON_EXPERIMENTS_CN.md)。

### 4.1 当前主表矩阵

| 类别 | 方法 | 当前状态 |
|---|---|---|
| 通用 KD | FGD-style | 双卡 4090 旧 smoke/formal 作废；待修正协议后重新 smoke |
| 通用 KD | LD | 双卡 4090 旧 smoke/formal 作废；待修正协议后重新 smoke |
| 跨模态 KD | CCLKD paper-structured reimplementation | 2026-06-05 重写为 COP/ATKD/CCL/RLD 结构；待人工复核与重新 smoke |
| 跨模态 KD | HalluciDet-style | 双卡 4090 旧 smoke/formal 作废；待修正协议后重新 smoke |

此前四方法 smoke 记录中，双卡 4090 部分因 `nc=5` yaml 错误作废；117 smoke 仅能证明
旧代码路径可运行，不能证明当前 public 修正版实现。2026-06-05 当前状态是：不启动新实验，
先完成 public 审计、CCLKD 代码重写和人工复核。

容量优先级：先 YOLO11n 三 seed闭环，同时保证 YOLO11s seed0 跑通；再扩展到
s/m/l。CrossKD、CoLD 与无效旧结果统一归档到
[`../../comparison/archive/excluded_methods/`](../../comparison/archive/excluded_methods/)。

### 4.2 当前条件：可做与需补

| 容量 | baseline 条件 | 可做项 | 需补 |
|---|---|---|---|
| YOLO11n | SAR/RGB 0/42/123 已齐 | 修正后可重新 smoke | 人工复核通过后再决定是否启动 |
| YOLO11s | SAR/RGB 0/42/123 已齐 | 当前可跑四方法 seed0 | 双卡 4090 缺 teacher seed42/123 |
| YOLO11m | seed0 SAR/RGB 已完成 | 四方法 seed0 可排队 | 补 SAR/RGB seed42/123 |
| YOLO11l | seed0 baseline 已完成 | seed0 可作为后续容量点 | 补 SAR/RGB seed42/123 |

## 5. 降级归档

CoLD、CrossKD、修复前 FGD 与旧 soft-logit LD 均不再独立追踪或继续运行。
完整材料见 [`../../comparison/archive/excluded_methods/`](../../comparison/archive/excluded_methods/)。

---

## 6. 执行顺序

```
Phase 0 (作废): 双卡 4090 旧四方法 smoke/formal partial runs，原因是 `nc=5` yaml
Phase 1 (当前): public 协议审计、CCLKD 重写、错误文档修正、推送 GitHub 供人工复核
Phase 2:        人工复核通过后，先只做协议校验和短 smoke
Phase 3:        smoke 通过后再决定是否启动 YOLO11n 正式对比
Phase 4:        根据 n 结果决定是否扩展到 s/m/l
```

当前明确不启动：任何新训练、LADD 主线、消融、CoLD、CrossKD/MGD/MMANet。关键决策点是
人工复核是否通过。
