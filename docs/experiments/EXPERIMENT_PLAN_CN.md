# LADD 实验计划

最后更新：2026-06-03

## 1. Baseline

目标：完成 OGSOD formal no-mosaic 协议下 YOLO11n/s/m/l 四档容量的 SAR/RGB baseline（主对比容量轴），为 LADD、消融和对比实验提供同 seed 起点。YOLO11x 作为容量趋势补充，不进入主对比矩阵。

协议：`imgsz=256, 800ep, cos_lr, full no-mosaic, default Albumentations`

执行策略更新：

- 117 暂停作为非 CoLD 主力：该机文件 IO/网络过慢，先不继续消耗调度精力。
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
| 4090D HBB LADD 代码树同步 | 4090D 空闲时无法接管非 CoLD 实验 | P0 |

90 路径：`/mnt/dataY/ydf/projects/LADD_og`。4090D 路径：`/root/autodl-tmp/LADD`。4090D 当前 SSH 短命令偶尔可用，但访问 `/root/autodl-tmp`、`nvidia-smi` 或上传小文件会被入口断开；恢复后优先同步 `src/teacher_student_decomposition_kd_hbb/`、`tools/train_ladd_hbb.py`、`scripts/ogsod_public/formal_nomosaic_20260528/`、`configs/` 和 `yolo/ultralytics/` 本地 patch。

### 1.3 待完成

| 优先级 | 实验 | 目的 |
|---|---:|---|
| P0 | 同步完整 HBB LADD 代码到 4090D | 让 4090D 空闲时能接管主实验/消融/对比 |
| P0 | 90 上继续塞可跑的非 CoLD 对比 | 避免 GPU 空闲，先保证 n/s seed0 跑通 |
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

### 4.1 已完成

| 方法 | Model | best AP | vs baseline |
|---|---|---|---|
| FGD | YOLO11n | 0.55867 | -0.00049 |
| CrossKD-style | YOLO11n | 0.55764 | -0.00152 |

4090D 已完成结果已拉回本地：

- 摘要：`comparison/fgd/results/4090d_formal_kd_20260602/SUMMARY_CN.md`
- 归档根目录：`comparison/fgd/results/4090d_formal_kd_20260602/root/autodl-tmp/LADD`
- 范围：YOLO11n/s SAR/RGB baseline、YOLO11n FGD、YOLO11n CrossKD-style 的 `results.csv`、`best.pt`、`last.pt`、日志和环境元数据；未拉取数据集、cache、图片和 optimizer 中间 checkpoint。

### 4.2 新主表矩阵

对比实验共保留 5 个方法：

| 类别 | 方法 | 服务器 | seed 策略 | 当前状态 |
|---|---|---|---|---|
| 通用 KD #1 | FGD | 90/4090D | 3 seed | YOLO11n seed0 已完成；YOLO11n seed42 已在 4090D 启动正式 from-yolo-pretrain |
| 通用 KD #2 | CrossKD-style | 90/4090D | 3 seed | YOLO11n seed0 已完成，正式 from-yolo-pretrain 结果可用 |
| 通用 KD #3 | LD | 90/4090D | 3 seed | HBB `ld` profile 已实现；90 上 n/s seed0 from-yolo-pretrain 正在跑 |
| 跨模态 #1 | CoLD | 90 | 尽力对齐原文 | 主表保留；接受慢跑和复现偏差诊断 |
| 跨模态 #2 | HalluciDet-style | 90/4090D | 3 seed | HBB `hallucidet` profile 已实现；90 上 n/s seed0 from-yolo-pretrain 正在跑 |

非 CoLD 的四个方法（FGD、CrossKD-style、LD、HalluciDet-style）统一按 from-yolo-pretrain 正式协议运行，90/4090D 哪台有空就用哪台。容量优先级：先 YOLO11n 三 seed闭环，同时保证 YOLO11s seed0 跑通；再按 baseline 完整度扩展到 s/m/l。最终主表以 n/s/m/l 为目标容量轴。

### 4.3 当前条件：可做与需补

| 容量 | baseline 条件 | 90/4090D 非 CoLD 对比可做项 | 需补 |
|---|---|---|---|
| YOLO11n | SAR/RGB 0/42/123 已齐 | FGD/CrossKD-style seed0 已完成；FGD seed42 已在 4090D 跑；LD/HalluciDet-style seed0 正在 90 跑；其余 seed 可排队 | 继续补三 seed |
| YOLO11s | SAR/RGB 0/42/123 已齐 | LD/HalluciDet-style seed0 正在 90 跑；FGD/CrossKD-style 可排队 | 继续补 seed0 和三 seed |
| YOLO11m | seed0 SAR/RGB 已完成 | seed0 FGD/CrossKD-style/LD/HalluciDet-style 可排队 | 补 SAR/RGB seed42,123 |
| YOLO11l | seed0 baseline 已完成 | seed0 可作为后续容量点 | 补 SAR/RGB seed42,123 |
| CoLD | 不依赖 baseline 矩阵 | 不放非 CoLD 主队列 | 90 上尽力对齐原文复现 |

### 4.4 第二个跨模态方法筛选结论

筛选硬条件：

- 必须有当前可访问的公开代码仓库，不能是 redacted、404、仅口头声称开源。
- 尽量是 object detection，或者至少检测适配成本可控。
- 推理阶段必须能保持 SAR-only；若方法要求 RGB+SAR/thermal 双模态输入推理，则不适合作为我们的主表核心对比。
- 机制需要能解释为跨模态知识迁移，而不是只做普通 feature/logit KD。

候选判断：

| 方法 | 代码状态 | 任务/机制 | 当前判断 |
|---|---|---|---|
| CoLD | 论文无可靠代码，已按原文复现诊断 | OGSOD optical->SAR detection，类别定位蒸馏 | 必须保留，作为同任务 anchor |
| HalluciDet | GitHub 可访问：`heitorrapela/HalluciDet` | WACV 2024，训练期 RGB privileged information，测试期 IR-only detection | 推荐作为第二个跨模态主表候选；实现时标为 `HalluciDet-style` |
| PFGF | GitHub 可访问：`liting1018/PFGF` | CVPR 2025 thermal detection，pseudo visible feature / fine-grained fusion | 很新且强，但更像 pseudo-visible/translation pipeline，不是严格 KD，工程较重 |
| TIRDet | GitHub 可访问：`zeyuwang-zju/TIRDet` | ACM MM 2023 thermal detection，T2V translation + cross-modality aggregation | 可作为轻量备选；机制比 HalluciDet 更偏翻译/聚合 |
| ModTr | GitHub 可访问：`heitorrapela/ModTr` | ECCV 2024 IR->RGB modality translator for detection | 开源且较简单，但不是 KD，适合作为附录/备选诊断 |
| AMFD | GitHub 可访问：`bigD233/AMFD` | adaptive multimodal fusion distillation | 虽然是真蒸馏，但原实现推理依赖 RGB+IR 六通道双模态输入，不满足 SAR-only 主表约束 |
| DecomKD | 论文声称开源，但 `lyf0801/DecomKD` 当前不可访问 | RGB-T teacher -> thermal-only student，机制非常贴近 | 暂不进入主表；不能把不可复现方法作为核心候选 |
| Thermal OD via Cross-Modal KD from RGB | 页面代码仍为 redacted | RGB teacher -> thermal detector | 暂不进入主表；可复现性不满足要求 |
| C2KD / MMANet-style | 有思想或代码，但非 detection-ready / 非跨模态检测主任务 | modality gap / incomplete modality | 不作为第二个跨模态主表核心，可保留为实现灵感 |

当前推荐主表结构：

| 类别 | 主方法 | 备注 |
|---|---|---|
| 通用 KD #1 | FGD | 已完成 YOLO11n seed0，需在 90/4090D 补三 seed |
| 通用 KD #2 | CrossKD-style | 已完成 YOLO11n seed0，需在 90/4090D 补三 seed |
| 通用 KD #3 | LD | 经典 logit KD，HBB profile 已实现，90 上 n/s seed0 正式 run 运行中 |
| 跨模态 #1 | CoLD | 同任务，必须保留 |
| 跨模态 #2 | HalluciDet-style | 开源、检测任务、SAR-only 推理可成立；HBB profile 已实现，90 上 n/s seed0 正式 run 运行中 |

HalluciDet-style 迁移建议：

- 不直接声称复现官方 HalluciDet；在文中和代码中称为 `HalluciDet-style privileged modality hallucination`。
- 训练阶段使用 paired RGB/SAR：RGB teacher 或 RGB feature extractor 提供 privileged supervision，SAR student 学检测，同时用辅助分支/投影头对齐 RGB 中间特征。
- 推理阶段只输入 SAR，teacher 和 RGB 输入全部移除；若辅助分支仅用于训练，可不改变 YOLO11 SAR detector 的推理结构。
- 优先完成 YOLO11n 三 seed controlled comparison；当前先保证 YOLO11n/YOLO11s seed0 正式 run 跑通，若训练稳定，再随 baseline 补齐扩展到 s/m/l。

---

## 5. CoLD 复现（单列）

复现难度大，独立追踪。详见 [../cold_repro/COLD_REPRO_FINAL_CN.md](../cold_repro/COLD_REPRO_FINAL_CN.md)。

### 5.1 当前状态

- YOLOv5x candidate CPM 在线 NCLD 50ep：相对 baseline +30%，方向对但趋势与论文相反
- 117 上 YOLOv5x RGB teacher e100 已完成，AP50-95 约 0.44790
- 最新 offline CoLD run 有 34 epoch results，但 detached log 显示 `Terminated`，117 当前无训练进程
- 离线 frozen teacher 无效
- 速度瓶颈：candidate 模式 Python 循环 ~2 s/it

### 5.2 待完成

| 优先级 | 实验 | 说明 |
|---|---:|---|
| P1 | 迁移到 90 服务器跑 400ep | CPU-bound 任务，不占用 117/5880 Ada |
| P2 | YOLOv5x → YOLO11 移植 | 对齐 formal no-mosaic 协议 |
| P2 | 解决 candidate 速度问题 | 向量化或切 matched 模式 |

---

## 6. 执行顺序

```
Phase 0 (当前): 90 已同步 HBB LADD 代码；4090D 待 SSH/上传入口恢复后同步同一套代码
Phase 1:        90/4090D 跑 11n/11s LADD 与非 CoLD 对比 seed0；90 慢跑 CoLD
Phase 2:        11n seed0 跑通后补三 seed；空闲 GPU 优先塞 FGD/CrossKD-style/LD/HalluciDet-style
Phase 3:        补齐 m/l baseline 与迁移；扩展 LADD/对比到 n/s/m/l
Phase 4:        根据结果决定是否补 YOLO11x 或 PFGF/TIRDet-style 附录
```

关键决策点：Phase 2 结束时，如果 11n 三 seed LADD 正向且消融趋势合理 → 论文主表成立，后续只补证据。
