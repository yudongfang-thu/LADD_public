# LADD 实验计划

最后更新：2026-06-04

> 2026-06-04 对比方法决策更新：controlled main table 改为
> `FGD / LD / CCLKD-style / HalluciDet-style`。CrossKD 已停止并淘汰；CoLD
> 已降级归档。FGD/LD 修复前结果作废。当前权威说明见
> [`COMPARISON_EXPERIMENTS_CN.md`](COMPARISON_EXPERIMENTS_CN.md) 和
> [`../../comparison/IMPLEMENTATION_REVIEW_CN.md`](../../comparison/IMPLEMENTATION_REVIEW_CN.md)；
> 本文后续旧对比矩阵仅保留为历史计划记录。

## 1. Baseline

目标：完成 OGSOD formal no-mosaic 协议下 YOLO11n/s/m/l 四档容量的 SAR/RGB baseline（主对比容量轴），为 LADD、消融和对比实验提供同 seed 起点。YOLO11x 作为容量趋势补充，不进入主对比矩阵。

协议：`imgsz=256, 800ep, cos_lr, full no-mosaic, default Albumentations`

执行策略更新：

- 117 暂停作为当前受控实验主力：该机文件 IO/网络过慢，先不继续消耗调度精力。
- 90 和 4090D 各准备一套代码树；哪个有空余 GPU 就优先塞正式比较实验。90 当前已同步 HalluciDet-style/LD 代码并启动可跑项；4090D 作为候选执行机，待 SSH/文件传输入口恢复后同步同一套代码。
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
| 4090D HBB LADD 代码树同步 | 4090D 空闲时无法接管当前受控实验 | P0 |

90 路径：`/mnt/dataY/ydf/projects/LADD_og`。4090D 路径：`/root/autodl-tmp/LADD`。4090D 当前 SSH 短命令偶尔可用，但访问 `/root/autodl-tmp`、`nvidia-smi` 或上传小文件会被入口断开；恢复后优先同步 `src/teacher_student_decomposition_kd_hbb/`、`tools/train_ladd_hbb.py`、`scripts/ogsod_public/formal_nomosaic_20260528/`、`configs/` 和 `yolo/ultralytics/` 本地 patch。

### 1.3 待完成

| 优先级 | 实验 | 目的 |
|---|---:|---|
| P0 | 同步完整 HBB LADD 代码到 4090D | 让 4090D 空闲时能接管主实验/消融/对比 |
| P0 | 90 上继续塞可跑的当前受控对比 | 避免 GPU 空闲，先保证 n/s seed0 跑通 |
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
- YOLO11s/m 当前数值仅为中途 best，不进入最终主表。

### 2.2 待完成

| 优先级 | 实验 | 前置 |
|---|---:|---|
| P0 | 在 90/4090D 重跑 11n cap2 三 seed，使用 A2/B 温和修正 | 目标执行机代码同步 + n baseline 已齐 |
| P0 | 等 90 上 11n cap2 seed123 稳定 B 重跑完成 | 用于判断 B 温和修正 |
| P0 | 在 90/4090D 跑 YOLO11s cap2 seed0 B 或全链重跑 | 目标执行机代码同步 + s baseline 已齐 |
| P0 | 在 90/4090D 跑 YOLO11m cap2 seed0 | 目标执行机代码同步 + m seed0 SAR/RGB 已齐 |
| P1 | YOLO11s cap2 seed42,123 | s baseline 三 seed 齐 |
| P1 | YOLO11m cap2 seed42,123 | 需先补 m seed42,123 SAR/RGB baseline |
| P1 | YOLO11l cap2 seed0 | 需先复核 l seed0 SAR/RGB baseline 在执行机可用 |
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
| 通用 KD | FGD-style | 修正版已实现；旧结果归档，待 smoke/重跑 |
| 通用 KD | LD | 真正 DFL localization KD 已实现；旧 soft-logit 结果归档，待 smoke/重跑 |
| 跨模态 KD | CCLKD-style | portable profile 已实现，待 smoke |
| 跨模态 KD | HalluciDet-style | 已实现；候选运行需跑满 |

容量优先级：先 YOLO11n 三 seed闭环，同时保证 YOLO11s seed0 跑通；再扩展到
s/m/l。CrossKD、CoLD 与无效旧结果统一归档到
[`../../comparison/archive/excluded_methods/`](../../comparison/archive/excluded_methods/)。

### 4.2 当前条件：可做与需补

| 容量 | baseline 条件 | 可做项 | 需补 |
|---|---|---|---|
| YOLO11n | SAR/RGB 0/42/123 已齐 | 四方法 smoke 与正式三 seed | 先验证修正版 loss/显存 |
| YOLO11s | SAR/RGB 0/42/123 已齐 | 四方法 seed0，稳定后补三 seed | 先完成 seed0 |
| YOLO11m | seed0 SAR/RGB 已完成 | 四方法 seed0 可排队 | 补 SAR/RGB seed42/123 |
| YOLO11l | seed0 baseline 已完成 | seed0 可作为后续容量点 | 补 SAR/RGB seed42/123 |

## 5. 降级归档

CoLD、CrossKD、修复前 FGD 与旧 soft-logit LD 均不再独立追踪或继续运行。
完整材料见 [`../../comparison/archive/excluded_methods/`](../../comparison/archive/excluded_methods/)。

---

## 6. 执行顺序

```
Phase 0 (当前): 90 已同步 HBB LADD 代码；4090D 待 SSH/上传入口恢复后同步同一套代码
Phase 1:        四个受控对比方法完成 smoke；LADD 保持运行
Phase 2:        11n seed0 跑通后补三 seed；空闲 GPU 优先塞 FGD/LD/CCLKD-style/HalluciDet-style
Phase 3:        补齐 m/l baseline 与迁移；扩展 LADD/对比到 n/s/m/l
Phase 4:        根据结果决定是否补 YOLO11x 或 PFGF/TIRDet-style 附录
```

关键决策点：Phase 2 结束时，如果 11n 三 seed LADD 正向且消融趋势合理 → 论文主表成立，后续只补证据。
