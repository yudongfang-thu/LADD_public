# LADD 消融实验计划

最后更新：2026-06-09

状态：**筹备中，暂不抢占当前 CCLKD 4090 资源**。本文档把后续消融分成两类：

1. CCLKD 对比方法的 Table 12 式组件消融，用于证明 comparison 方法实现语义正确。
2. LADD 主方法消融，用于支撑论文核心 claim。

当前执行优先级仍是先让新版 CCLKD 6 条消融自然跑到 250/300 epoch 形成趋势判断；**LADD 不应立刻铺消融，必须先冻结主线**。此前主线持续变化主要是为了处理 A2/B 阶段崩溃、BN running stats 污染和后期退化；这些稳定修正可能影响峰值性能，因此消融必须围绕最终冻结主线执行。

## 0. 当前判据

### CCLKD 消融判据

新版 CCLKD 不追求严格复刻原文绝对值，而是在 LADD formal no-mosaic baseline 协议下要求：

```text
full >= atkd >= ccl_only > baseline
```

并且 `lld`、`lld_fld`、`lld_fld_rld` 不出现明显反常。如果满足，则认为 CCLKD online paper-aligned 实现可作为受控对比方法。

### LADD 主线冻结判据

LADD 主线优先遵循“最小必要修正”原则：

- A2/B 默认高学习率导致 NaN 或后期退化的证据明确，因此 A2/B 使用 `MuSGD lr0=0.001 no warmup` 是稳定性修正，不再回退到默认高学习率作为主线。
- `B_FREEZE_BN_STATS=1` 已在 YOLO11n seed0/42/123 上形成正向证据，但 seed0 峰值低于非 freeze 健康 run，因此它应被视为稳定修正，而不是不经比较的性能增强模块。
- `cap2` 固定为 LADD 主线组件。它的价值是修正 reach-rank 几何、避免继续奖励反平行，并强化方法故事；`no_cap2` 只作为诊断/消融对照，不再作为主线候选。

主线冻结前必须完成：

| 必须项 | 目的 |
|---|---|
| YOLO11n seed0/42/123 `cap2 + B_FREEZE_BN_STATS=1` 汇总 | BN-freeze 已完成三 seed闭环，判断 freeze 是否稳定但过强 |
| 汇总 seed0/42/123 的 `a2mu1e3`、`bstable1e3`、`bnfreeze` | 分清崩溃修正、后期退化和方法收益 |
| 对 seed0 original vs cap2 做性能 + reach 几何对照 | 量化 cap2 的几何约束价值；不用于重新选择主线 |

若 BN-freeze 三 seed 均稳定正向，但平均 AP 明显低于健康非 freeze run，则论文表述应是：

```text
主线采用最小稳定修正版；BN-freeze 是防止后期退化的训练稳定设置，
其收益不是提高峰值，而是避免 seed-dependent collapse。
```

### LADD 消融判据

LADD 消融使用 YOLO11n seed0 单 seed，协议与主线一致：

```text
formal no-mosaic
A1=10 -> A2=50 -> B=800
A2/B MuSGD lr0=0.001, lrf=0.01, no warmup
B_FREEZE_BN_STATS=1
batch=64
```

核心判断不是每个消融都必须显著掉点，而是要回答 reviewer 会问的三个问题：

- 分解是否真的必要？
- reach / cap2 是否真的防止几何坍缩？
- student split / residual 私有分支是否比简单投影更合理？

## 1. 已在跑的 CCLKD 组件消融

入口：

```bash
comparison/code/launch_formal_online_cclkd_ablation_job.sh n <ablation> 0 <gpu>
```

统一设置：

```text
EPOCHS=400
BATCH_SIZE=64
CCLKD_FORMULATION=paper
CCLKD_CCL_MODE=paper_pair
CCLKD_CCL_SOURCE=box_distribution
CCLKD_RLD_MODE=paper_instance
```

| 消融 | 目的 | 当前状态 |
|---|---|---|
| `lld` | 检查 logit/localization KD 单项贡献 | 新版已启动 |
| `lld_fld` | 检查区域特征蒸馏增益 | 新版已启动 |
| `lld_fld_rld` | 检查关系蒸馏在 fixed T 下是否有效 | 新版已启动 |
| `ccl_only` | 检查 CCL 单独是否有效 | 新版运行中 |
| `atkd` | 检查 LLD+FLD+RLD+PATM 是否有效 | 新版运行中 |
| `full` | 检查 ATKD 与 CCL 是否互补 | 新版运行中 |

当前观察到的早期趋势：`ccl_only`、`atkd`、`full` 均稳定高于 baseline，且 `full` 在 100 epoch 后基本高于 `atkd`。若 250/300 epoch 保持该趋势，则 CCLKD 实现可视为正确，最终增益小可解释为 LADD baseline 协议较强。

## 2. LADD 主线冻结优先级

在正式 LADD 消融前，先做主线选择/冻结，不新增结构消融：

| 优先级 | 实验/整理项 | 回答的问题 | 当前证据 |
|---|---|---|---|
| P0 | seed0/42/123 主线候选汇总表 | BN-freeze 是否能作为三 seed 统一主线，是否压低峰值？ | seed0 BN-freeze 低于 seed0 非 freeze，seed42/123 BN-freeze 正向 |
| P0 | cap2 vs original 几何诊断 | cap2 带来的几何约束是否符合故事？ | seed0 original AP 略高，但 cap2 几何更合理 |
| P1 | s seed0 后期退化排查 | n 上有效是否能扩展到 s？ | 90 s seed0 中途正向；双卡 4090 s seed0 BN-freeze 跑满但 last 低于 baseline |

主线冻结后，才启动下面的主方法消融。

## 3. LADD 主方法消融总表

| 优先级 | 消融 | 配置变化 | 回答的问题 | 预期 |
|---|---|---|---|---|
| P0 | `main_cap2_bnfreeze` | 当前主线 | 主线参照 | 应稳定高于 SAR baseline |
| P1 | `no_cap2_original_rank` | `RANK_D_NEG_CAP=4.0` | cap2 是否必要，还是只是调参？ | 性能可接近，但 reach 几何更差或稳定性更弱 |
| P1 | `no_teacher_decomposition` | `TEACHER_FEATURE_MODE=raw` | RGB teacher 分解是否必要？ | 低于主线，说明直接模仿 RGB feature 会混入私有信息 |
| P1 | `single_proj_student` | `STUDENT_BRANCH_MODE=single_proj` | split/residual 设计是否必要？ | 低于或接近主线；若接近，说明结构贡献有限 |
| P1 | `no_reach_loss` | `LAMBDA_REACH=0.0` | reach ranking 是否贡献核心可达性约束？ | 性能下降，reach 诊断指标退化 |
| P2 | `no_residual_aux` | `LAMBDA_RESIDUAL_AUX=0.0` | SAR residual 辅助是否防止私有分支塌缩？ | 小幅下降或 residual 指标变差 |
| P2 | `no_teacher_private_aux` | `TEACHER_PRIVATE_AUX_MODE=none`, `LAMBDA_TEACHER_PRIVATE_AUX=0.0` | teacher private 分支监督是否必要？ | 可能小幅下降；若无影响，可简化方法叙述 |
| P2 | `lambda_reach_0p5` | `LAMBDA_REACH=0.5` | reach 权重敏感度下界 | 与主线接近或略低 |
| P2 | `lambda_reach_2p0` | `LAMBDA_REACH=2.0` | reach 权重敏感度上界 | 过强可能伤害检测 |
| P3 | `no_a2_det_supervision` | `A2_DET_LOSS_SCALE=0.0` | A2 阶段检测监督是否必要？ | 预期明显下降，仅在资源允许时跑 |
| P3 | `no_b_bnfreeze` | `B_FREEZE_BN_STATS=0` | BN freeze 是方法必要项还是稳定 workaround？ | 可能后期退化；需要重点监控 BN running stats |

## 4. 首批建议执行清单

首批只跑 5 条，全部 YOLO11n seed0，单 seed：

| 顺序 | 消融 | 原因 |
|---:|---|---|
| 1 | `main_cap2_bnfreeze` | 统一参照。如果已有同协议 seed0 主线可直接复用，不重复跑。 |
| 2 | `no_cap2_original_rank` | 最直接回答 cap2 是否必要。 |
| 3 | `no_teacher_decomposition` | 最核心的 LADD claim。 |
| 4 | `single_proj_student` | 回答 student split 是否只是复杂化。 |
| 5 | `no_reach_loss` | 回答 reach ranking 是否真正贡献。 |

这 5 条足够覆盖 reviewer 最可能追问的主机制。P2/P3 等首批趋势稳定后再排。

## 5. 启动模板

当前 launcher 只显式支持 `original|cap2`，但底层 chain 支持关键环境变量。因此建议用 `cap2` 入口加 `RUN_TAG_SUFFIX` 区分消融。

### 主线参照

```bash
B_OPTIMIZER=MuSGD B_LR0=0.001 B_LRF=0.01 \
B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.001 B_FREEZE_BN_STATS=1 \
RUN_TAG_SUFFIX=_bnfreeze1e3 \
  bash ladd/scripts/launch_formal_ladd_job.sh cap2 n 0 0
```

### no cap2

```bash
B_OPTIMIZER=MuSGD B_LR0=0.001 B_LRF=0.01 \
B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.001 B_FREEZE_BN_STATS=1 \
RUN_TAG_SUFFIX=_ab_no_cap2_bnfreeze1e3 \
  bash ladd/scripts/launch_formal_ladd_job.sh original n 0 0
```

### no teacher decomposition

```bash
TEACHER_FEATURE_MODE=raw \
B_OPTIMIZER=MuSGD B_LR0=0.001 B_LRF=0.01 \
B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.001 B_FREEZE_BN_STATS=1 \
RUN_TAG_SUFFIX=_ab_teacher_raw_bnfreeze1e3 \
  bash ladd/scripts/launch_formal_ladd_job.sh cap2 n 0 0
```

### single projection student

```bash
STUDENT_BRANCH_MODE=single_proj \
B_OPTIMIZER=MuSGD B_LR0=0.001 B_LRF=0.01 \
B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.001 B_FREEZE_BN_STATS=1 \
RUN_TAG_SUFFIX=_ab_single_proj_bnfreeze1e3 \
  bash ladd/scripts/launch_formal_ladd_job.sh cap2 n 0 0
```

### no reach loss

```bash
LAMBDA_REACH=0.0 \
B_OPTIMIZER=MuSGD B_LR0=0.001 B_LRF=0.01 \
B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.001 B_FREEZE_BN_STATS=1 \
RUN_TAG_SUFFIX=_ab_no_reach_bnfreeze1e3 \
  bash ladd/scripts/launch_formal_ladd_job.sh cap2 n 0 0
```

启动前必须先 `DRY_RUN=1` 检查命令中的数据集、baseline 权重、BN freeze、lr、mosaic 设置。
注意：`TEACHER_FEATURE_MODE`、`STUDENT_BRANCH_MODE`、`LAMBDA_REACH` 等底层变量通过 shell 环境继承给 chain 脚本，launcher 的 `Command:` 行不一定显式打印；正式启动后需检查 chain log 的 manifest / phase log。

## 6. 资源安排

### 当前窗口

- 双卡 4090：优先跑完新版 CCLKD 6 条消融，不再插入 LADD 主消融。
- 117：可连接但当前不作为主执行节点，除非确认 GPU 空闲且 IO 稳定。
- 90：若恢复可控连接，可承接 LADD 消融，因为 LADD 单条 800 epoch 更长，不适合频繁中断。

### 建议启动时机

1. 新版 CCLKD 6 条到 250/300 epoch，确认趋势正确。
2. 回收并复核 YOLO11n LADD 主线冻结证据，尤其是 BN-freeze 三 seed 汇总和 cap2/original 几何诊断。
3. 主线冻结后，再清理 4090 上的 CCLKD 进程并一次性开 4-6 条 LADD 消融，按显存实际占用分配。

## 7. 记录与归档要求

每个消融必须记录：

- 启动命令和 git commit。
- 使用的 SAR baseline / RGB teacher 权重路径。
- A1/A2/B 三阶段实际 run 目录。
- best AP50-95、best epoch、last AP50-95。
- reach 几何诊断：正负距离、cap 命中情况、是否出现反平行或 BN running stats 污染。
- 若中途失败，归档失败原因，不覆盖 run 目录。

结果进入表格时必须标注：

- `single seed ablation, YOLO11n seed0`
- 是否使用 `B_FREEZE_BN_STATS=1`
- 是否与主线同协议
