# LADD 实验计划

最后更新：2026-06-06

> 2026-06-06 重要更新：当前优先级临时收束到 **CCLKD paper reproduction
> + CCLKD 原文消融**。LADD 主线继续自然完成已有 run，但学习率/BN 细节后置；
> CoLD 继续降级。此前双卡上 2026-06-06 16:01 之前启动的 CCLKD paper
> reproduction 使用旧实现，不能进入正式结论；已停止未完成的旧实现 CCLKD 进程。

## 1. Baseline

目标：完成 OGSOD formal no-mosaic 协议下 YOLO11n/s/m/l 四档容量的 SAR/RGB baseline（主对比容量轴），为 LADD、消融和对比实验提供同 seed 起点。YOLO11x 作为容量趋势补充，不进入主对比矩阵。

协议：`imgsz=256, 800ep, cos_lr, full no-mosaic, default Albumentations`

执行策略更新（2026-06-06）：

- **双卡 4090 为主执行节点**：两台 RTX4090 正在承接 LADD BN-freeze、CCLKD paper-protocol baseline、以及最新实现 CCLKD 复现/关键消融。路径为 `/root/shared-nvme/LADD_public`，数据为 3 类正确 OGSOD YAML。
- **4090D 降为结果恢复**：在线但无 GPU 模式，仅用于拉取已有结果，不启动新训练。
- **117 GPU 被占用**：其他用户进程占用 GPU，不可用。
- **90 作为并行执行节点**：继续跑已有正式对比方法，同时承接最新实现的 YOLO11n seed0 CCLKD 原文消融 7 项。
- **CoLD 降级为外部诊断**：不再占用 controlled 执行资源（4090/4090D/117），仅在 90 空闲时慢跑。
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
| 双卡 4090 m/l 权重与 teacher 同步 | n/s 对比实验已可跑；m/l 扩展前需同步对应 pretrain 与 teacher | P1 |
| 双卡 4090 s teacher seed42/123 同步 | s 对比实验扩展到多 seed 的前提 | P1 |

90 路径：`/mnt/dataY/ydf/projects/LADD_public`（最新公开仓库执行目录）和 `/mnt/dataY/ydf/projects/LADD_og`（旧结果/权重来源）。4090D 路径：`/root/autodl-tmp/LADD`（在线无 GPU 模式，仅结果恢复）。双卡 4090 路径：`/root/shared-nvme/LADD_public`，数据 `/root/shared-nvme/OGSOD-1.0`。

### 1.3 待完成

| 优先级 | 实验 | 目的 |
|---|---:|---|
| P1 | 同步双卡 4090 的 YOLO11s teacher seed42/123 | s 对比实验多 seed 前提 |
| P1 | 同步双卡 4090 的 YOLO11m/l 权重与 teacher | 扩展到 m/l 对比容量点 |
| P1 | YOLO11m SAR/RGB seed42,123 | m 多 seed 验证 |
| P2 | YOLO11l SAR/RGB seed42,123 | l 多 seed 验证 |
| P2 | YOLO11l seed0 SAR/RGB 可用性复核 | l 实验启动前提 |
| P3 | YOLO11x 多 seed | 容量趋势补充，非主表必须 |

### 1.4 判据

- n 三 seed 完成，gap 最大 → 主机制实验对象，同时也是所有对比方法的统一容量
- s 三 seed 完成 → 容量趋势验证，LADD 主表第二个容量点
- m/l 补齐多 seed → 证明方法在更大容量上仍然有效，LADD 主表第三、第四个容量点
- m 以上 gap 稳定在 ~0.02-0.03，LADD 涨点会变小，但多容量正趋势比单容量大涨幅更有说服力

---

## 2. LADD 主线

目标：在 formal no-mosaic 协议下，完成 LADD cap2 的多容量、多 seed 验证。

> **2026-06-06：LADD 主线继续自然完成已有 run，但不新增学习率 sweep。**
> 当前执行窗口优先跑 CCLKD paper reproduction 与原文消融。LADD 的
> BN-freeze/lr 细节确认后置。

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
| YOLO11n cap2 | 0.57662 ✅ | 0.57420 ✅ | B 稳定重跑中，best 0.56089@139/144 |
| YOLO11s cap2 | B 阶段运行中，best 0.60675@0/362 | — | — |
| YOLO11m cap2 | A2 阶段运行中，best 0.64011@20/26 | — | — |

补充说明：

- YOLO11n cap2 seed123 早先 B run 跑到 483 epoch 后出现 last AP 为 0 的异常，已启动 `bstable1e3` 稳定重跑。
- seed123 B 稳定重跑当前无 NaN，`lr0=0.001 + MuSGD + no warmup` 初步有效。
- 4090D 入口在关闭本地 tun 后恢复；HBB/HalluciDet-style 关键代码已同步并通过 hash、`py_compile`、`--help` 和 launcher `bash -n` 检查。4090D 当前 YOLO11n cap2 seed0 B run 在 epoch 342-346 连续 mAP50-95=0，best 0.54925@227，已停止以避免继续占用 4090D。
- 塌缩排查结论：YOLO11n seed0/123 坏 run 权重无 NaN/Inf，但 `last.pt` 的 BN `running_mean/running_var` 被污染；seed0 BN max running_var 到 1726，seed123 到 1333，而健康 seed42 约 47.7。已新增 `--freeze-bn-stats` / `FREEZE_BN_STATS=1`，训练时冻结 BN running stats 但保留 BN affine 参数梯度。
- 90 GPU7 已启动 YOLO11n seed0/123 B 修正版：`formal_nomosaic_yolo11n_cap2_s0_bnfreeze1e3_90_gpu7`、`formal_nomosaic_yolo11n_cap2_s123_bnfreeze1e3_90_gpu7`，均从 `a2mu1e3_a2_e50` best 启动，`MuSGD lr0=0.001 no warmup + FREEZE_BN_STATS=1`，已进入 epoch 1。
- 5 类错误配置只影响更早一批已作废的双卡诊断；90 上 YOLO11n seed123 旧 B 崩溃、`bstable1e3` 后期退化、`bnfreeze lr=0.005` 退化均发生在 3 类正确配置下。
- 当前有效 LADD 候选为 `cap2 + A2/B MuSGD lr=0.001 + B FREEZE_BN_STATS=1`。YOLO11n seed0/123 BN-freeze 已完成且正向；双卡 YOLO11n seed42 BN-freeze 正在跑，当前 best 约 `0.57615@400`，已对齐旧无 BN 版本。
- YOLO11s/m 当前数值仅为中途 best，不进入最终主表。

### 2.2 待完成（暂不启动）

LADD 主线已冻结，以下为方法冻结后待办清单，不在本波次执行：

| 优先级 | 实验 | 前置 |
|---|---:|---|
| P0 | 等 90 上 11n cap2 seed123 稳定 B 重跑完成（已有 run 自然完成） | — |
| P0 | 11n cap2 三 seed 闭环，使用 A2/B 温和修正 + BN freeze | n baseline 已齐，待解冻后启动 |
| P0 | YOLO11s cap2 seed0 B 或全链重跑 | s baseline 已齐 |
| P0 | YOLO11m cap2 seed0 | m seed0 SAR/RGB 已齐 |
| P1 | YOLO11s cap2 seed42,123 | s baseline 三 seed 齐 |
| P1 | YOLO11m cap2 seed42,123 | 需先补 m seed42,123 SAR/RGB baseline |
| P1 | YOLO11l cap2 seed0 | 需先有可用执行机 + l seed0 baseline |
| P2 | YOLO11l cap2 seed42,123 | 需先补 l seed42,123 SAR/RGB baseline |

### 2.3 预期主表

| Model | SAR baseline | LADD cap2 | Delta |
|---|---:|---:|---:|
| YOLO11n | 0.55859 ± 0.00216 | 2/3 done: 0.57531 | 2/3 done: +0.01804 |
| YOLO11s | 0.62779 ± 0.00289 | B running | TBD |
| YOLO11m | 0.65580 | A2 running | TBD |
| YOLO11l | 0.65427 | not started | TBD |

---

## 3. 消融实验

目标：证明 LADD 各组件的必要性。全部使用 YOLO11n seed0，单 seed。

> **2026-06-04：消融实验全部延后。** 待对比实验 11n 三 seed 闭环 + LADD 主线解冻后再排。

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

### 4.1 冻结的 controlled comparison 方法

经过实现可行性筛选，**四方法已冻结**，不再增删：

| 类别 | 方法 | 机制简述 | 实现来源 |
|---|---|---|---|
| 通用 KD #1 | FGD | focal/granularity distillation | 新实现，HBB profile `fgd` |
| 通用 KD #2 | true LD over DFL distributions | logit distribution distillation over DFL bins | 新实现，HBB profile `ld` |
| 通用 KD #3 | CCLKD-style | LLD/FLD/RLD/PATM/CCL online KD | 新实现，paper reproduction 与 comparison 分开记录 |
| 跨模态 | HalluciDet-style | training-time RGB privileged information, SAR-only inference | 新实现，HBB profile `hallucidet` |

**已从 controlled main table 移除的方法：**

| 方法 | 处置 | 原因 |
|---|---|---|
| CrossKD-style | 归档，仅保留代码审计 | 实现偏差大、与 FGD 机制重叠度高，不作为独立主表方法 |
| MGD | 归档，仅保留代码审计 | 实现成本 vs 区分度不划算 |
| MMANet-style | 归档，仅保留灵感参考 | 非检测原生方法，适配成本过高 |

**CoLD**：降级为外部报告/复现诊断方法，不进入 controlled comparison 主表。其角色是提供同任务（OGSOD）的外部参考锚点，但受限于无可靠公开代码、复现偏差大、收敛慢，不再占用 controlled 执行资源。详见 [第 5 节](#5-cold-复现单列)。

### 4.2 旧 run 与正式实现的区分

**非最终实现（仅作历史证据，不进入主表）：**

| 旧 run | 问题 |
|---|---|
| 4090D 上 FGD YOLO11n seed0/seed42 | 早期实现，未对齐最终 HBB profile |
| 4090D 上 CrossKD-style YOLO11n seed0 | 已归档，且方法本身退出 controlled 主表 |
| 90 上 LD/HalluciDet-style n/s seed0 旧 run | HBB profile 实现迭代前的早期版本，结果不可直接引用 |

**正式实现：** 四个方法均在 HBB `src/teacher_student_decomposition_kd_hbb/` 框架下以独立 `profile` 实现，统一使用 formal no-mosaic 协议 (`imgsz=256, 800ep, cos_lr, from-yolo-pretrain`)。所有主表结果必须来自正式实现的新 run。

### 4.3 执行优先级

执行机器：**90 + 双卡 4090 并行**。90 承接 YOLO11n seed0 CCLKD 原文消融；双卡承接最新实现的 CCLKD paper reproduction fixed run 与 YOLO11n seed42 关键消融。

| 阶段 | 内容 | 说明 |
|---|---|---|
| **Phase 0: GPU smoke** | 四个冻结方法各跑 5-10 epoch GPU smoke | 验证代码无 NaN/OOM/精度异常，**smoke 未通过不启动正式 run** |
| **Phase 1: YOLO11n 三 seed** | n seed0/42/123 四个方法正式 run | n gap 最大，主机制验证容量 |
| **Phase 2: YOLO11s** | s seed0 四个方法正式 run，随后补 seed42/123 | 容量趋势验证 |
| **Phase 3: m/l 扩展** | 待 s 双 seed 完成后再决策 | 需要先补 m/l teacher 权重 |

**不在本波次启动：**

- **LADD 主线**：尚未冻结，不在此波次与对比实验混合执行
- **消融实验**：全部延后至对比实验 11n 闭环后再排
- **CoLD**：降级为外部诊断，不占用 4090 执行时段
- **CrossKD/MGD/MMANet 新 run**：不启动

### 4.4 当前服务器资源与限制

| 服务器 | 状态 | 用途 |
|---|---|---|
| 双卡 4090 | 两台 RTX4090 正在运行 | **主执行节点**，承接 CCLKD fixed reproduction、关键消融、已有 LADD/baseline |
| 4090D | 在线，无 GPU 模式 | **仅结果恢复**，不启动新训练 |
| 117 | GPU 被其他用户进程占用 | 不可用 |
| 90 | 多任务运行中 | 最新 CCLKD YOLO11n seed0 全消融 + 对比方法三 seed |

### 4.5 双卡 4090 当前可用权重

| 资源 | 状态 |
|---|---|
| YOLO11n teacher seed0/42/123 | ✅ 已同步 |
| YOLO11n SAR baseline seed0/42/123 | ✅ 已同步 |
| YOLO11s teacher seed0 | ✅ 已同步 |
| YOLO11s SAR baseline seed0/42/123 | ✅ 已同步 |
| YOLO11s teacher seed42/123 | ❌ 缺失，需从 90 同步 |
| YOLO11m/l 全部权重 | ❌ 缺失，m/l teacher 与 pretrain 均未同步 |

### 4.6 容量优先级与预期主表

容量轴：**n/s/m/l**。先闭环 n 三 seed，再扩展到 s。

| 容量 | seed 策略 | 方法覆盖 | 前置条件 |
|---|---|---|---|
| YOLO11n | 0/42/123 | FGD / LD / CCLKD / HalluciDet | smoke 通过，权重已齐 |
| YOLO11s | 0 → 42/123 | 同上 | n 三 seed 闭环后启动；需先同步 s teacher seed42/123 |
| YOLO11m | 0 → 42/123 | 同上 | 需先同步 m 全部 teacher + pretrain 权重 |
| YOLO11l | 0 → 42/123 | 同上 | 需先同步 l 全部 teacher + pretrain 权重 |

预期主表结构（同协议、同 seed、from-yolo-pretrain）：

| 类别 | 方法 | 主表角色 |
|---|---|---|
| 通用 KD | FGD | controlled #1 |
| 通用 KD | LD (true DFL distribution) | controlled #2 |
| 通用 KD | CCLKD-style | controlled #3 |
| 跨模态 | HalluciDet-style | controlled #4 |
| 跨模态 | CoLD | 外部参考（非 controlled，见第 5 节） |
| — | LADD cap2 | 本文方法（不在本波次执行） |

### 4.7 HalluciDet-style 迁移说明

- 不直接声称复现官方 HalluciDet；在文中和代码中称为 `HalluciDet-style privileged modality hallucination`。
- 训练阶段使用 paired RGB/SAR：RGB teacher 或 RGB feature extractor 提供 privileged supervision，SAR student 学检测，同时用辅助分支/投影头对齐 RGB 中间特征。
- 推理阶段只输入 SAR，teacher 和 RGB 输入全部移除。
- 正式 run 统一使用 HBB `hallucidet` profile，from-yolo-pretrain 启动。

### 4.8 CCLKD 当前执行状态（2026-06-06）

CCLKD 现在分三类结果，不能混表：

| 类别 | 目的 | 协议 | 状态 |
|---|---|---|---|
| paper reproduction | 回答“CCLKD 原文方法是否复现” | online teacher-student, 400ep, mosaic=1.0, mixup=0.1, SGD lr=0.01, imgsz=256 | 旧实现结果仅诊断；fixed run 正在重启 |
| paper ablation | 对齐原文 Table 12 的模块趋势 | YOLO11n, seed0/42, 同 paper reproduction 协议 | seed0 全 7 项在 90 跑；seed42 关键项在双卡跑 |
| LADD-protocol comparison | 回答“在本文 formal no-mosaic 协议下 CCLKD 与 LADD/FGD/LD/HalluciDet 谁好” | formal no-mosaic, 800ep, from-yolo-pretrain | 等 paper 实现趋势通过后再决定是否重启正式对比 |

必须排除的旧结果：

| 旧结果 | 处置 | 原因 |
|---|---|---|
| 双卡 16:01 前启动的 `cclkd_paper_repro_yolo11*` | 仅诊断，不进正式结论 | Python 进程加载的是旧实现：CCL 使用 DFL logits，负样本相似度退化为均值标量 |
| 90 旧 `formal_nomosaic/comparisons/online_cclkd` | 仅诊断，不进正式结论 | 早期实现与当前 paper reproduction 代码不同 |

当前正在运行的最新实现：

| 服务器 | 实验 | 说明 |
|---|---|---|
| 90 `/mnt/dataY/ydf/projects/LADD_public` | YOLO11n seed0: `full / atkd / ccl_only / lld / lld_fld / lld_fld_rld / full_ccl05` | 全量原文消融；`full_ccl05` 只解释旧权重，不进主表 |
| 双卡 `/root/shared-nvme/LADD_public` | YOLO11n seed42: `full fixed / atkd / ccl_only / lld_fld_rld` | 关键消融补充，加速判断趋势 |
| 双卡 `/root/shared-nvme/LADD_public` | YOLO11s seed0: `full fixed` | 替代旧实现的 YOLO11s seed0 paper reproduction |

下一步判据：

- `full` 应高于或至少不低于 `atkd` 与 `ccl_only` 的中后期 best。
- `lld_fld_rld` 应给出稳定正向基础；若低于 SAR-only paper baseline，需要检查 COP/feature indexing。
- 若 90 seed0 与双卡 seed42 的模块排序一致，再补 fixed paper reproduction 的 n/s 三 seed；若趋势明显不符，先回到实现审计。

---

## 5. CoLD 复现（降级：外部参考 / 复现诊断）

CoLD **不进入 controlled comparison 主表**，降级为外部参考锚点 + 复现偏差诊断。详见 [../cold_repro/COLD_REPRO_FINAL_CN.md](../cold_repro/COLD_REPRO_FINAL_CN.md)。

### 5.1 降级原因

- 论文无可靠公开代码，当前复现方向对但趋势与原文不一致
- 收敛慢、脏活多、CPU-bound 瓶颈严重
- 不再占用 controlled 执行资源（4090/4090D/117），仅在 90 空闲时慢跑或诊断

### 5.2 当前状态

- YOLOv5x candidate CPM 在线 NCLD 50ep：相对 baseline +30%，方向对但趋势与论文相反
- 117 上 YOLOv5x RGB teacher e100 已完成，AP50-95 约 0.44790
- 最新 offline CoLD run 有 34 epoch results，但 detached log 显示 `Terminated`，117 当前无训练进程
- 离线 frozen teacher 无效
- 速度瓶颈：candidate 模式 Python 循环 ~2 s/it

### 5.3 后续

| 优先级 | 事项 | 说明 |
|---|---|---|
| P2 | 90 空闲时慢跑 CoLD 诊断 | 不占用 4090/4090D/117 的正常任务时段 |
| P2 | YOLOv5x → YOLO11 移植探索 | 对齐 formal no-mosaic 协议，仅诊断 |
| P3 | 解决 candidate 速度问题 | 向量化或切 matched 模式 |

CoLD 结果最终以外部报告形式进入论文（如 `External reference: CoLD (reproduction, YOLOv5x)`），不与 controlled FGD/LD/CCLKD/HalluciDet 共享主表。

---

## 6. 执行顺序（2026-06-06 更新）

### 6.1 MUST-RUN 队列

执行机器：**90 + 双卡 4090 并行**。当前以 CCLKD 复现/消融为最高优先级；LADD
已有 run 自然完成，不新增 LADD 学习率 sweep。

```
Phase C0 (done):          修复 CCLKD paper implementation
                          LLD 去掉 cls KL；FLD 改 MSE；RLD 改 C^T C；
                          CCL 改 neck feature + per-token negative similarity

Phase C1 (running):       CCLKD paper ablation
                          90: YOLO11n seed0 全 7 项
                          双卡: YOLO11n seed42 key ablations

Phase C2 (running):       CCLKD fixed paper reproduction
                          双卡: YOLO11n seed42 full fixed, YOLO11s seed0 full fixed
                          旧实现结果保留为 diagnostics，不进入正式表

Phase C3 (pending):       若 C1/C2 趋势合理，补 fixed paper reproduction n/s 三 seed
                          并重启 LADD-protocol CCLKD controlled comparison

Phase M1 (background):    FGD / LD / HalluciDet 继续已有正式 run
                          CCLKD 暂不加入 LADD-protocol 主表，直到 paper implementation 通过趋势验证
```

### 6.2 不在本波次启动

| 项目 | 原因 |
|---|---|
| 新 LADD 主线 / LR sweep | 已有 BN-freeze run 自然完成；当前算力优先 CCLKD |
| LADD 消融实验 | 延后至 CCLKD 复现/消融趋势确认后 |
| CoLD | 降级为外部诊断，不占用 4090 执行时段 |
| CrossKD / MGD / MMANet | 已归档，不启动新 run |
| m/l 容量扩展 | 需先同步 m/l teacher + pretrain 权重（当前缺失） |

### 6.3 关键决策点

- **C1/C2 趋势通过** → fixed paper reproduction 补 n/s 三 seed
- **CCLKD paper reproduction 有正向趋势** → 再重启 LADD-protocol CCLKD controlled comparison
- **FGD/LD/HalluciDet 完成 n 三 seed** → 与 LADD 主线候选做阶段性对照
- 所有 CCLKD 正式结果必须来自 2026-06-06 16:01 后启动的新实现 run
