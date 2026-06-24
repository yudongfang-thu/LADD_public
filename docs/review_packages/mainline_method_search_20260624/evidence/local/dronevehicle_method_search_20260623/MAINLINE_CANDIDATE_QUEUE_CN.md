# DroneVehicle 主线方案搜索队列

日期：2026-06-23

目标：在 DroneVehicle sub2k seed0 full-val 风洞里找到一个稳定超过同协议 detector-only/reload control 的主线候选方案。正式比较统一使用 `YOLO11n, imgsz=512, batch=64, seed=0`；若某个结构因显存失败，只记录 OOM/诊断，不进入主线候选比较。

## 0. 判定规则

正向候选必须同时满足：

1. 超过 RGB student baseline best：`AP50=0.56886`，`AP50-95=0.36087`。
2. 超过同初始化、同 schedule 的 detector-only continued-training / reload control。
3. 超过 shuffled-pair distill control，避免只是在训练更久或数据增强噪声里涨点。
4. 曲线不能只靠最后一两个 epoch 偶然跳高；优先看 best、final、late-window mean 三个指标。

## 1. 当前优先级队列

| 优先级 | 方法目录 | 核心想法 | 先跑什么 | 通过条件 | 当前状态 |
|---:|---|---|---|---|---|
| P0 | `cmdistill_style` | 成熟 KD profile：feature + relation + logit distillation，验证风洞是否有正向潜力 | high-LR 已跑；low-LR/no-warmup CMDistill 已启动 | best AP50-95 超过 RGB baseline 与同协议 det-only control | low-LR 73 rows best AP50-95 `0.36286`，窗口稳定高于 det-only；sanity positive，不是主线 |
| P0.5 | `reload_controls` | 修正从 baseline best 继续训练时 `lr0=0.01 + warmup` 导致的 reload 掉点混淆 | low-LR no-warmup det-only + raw KD control | det-only 稳定后，后续方法必须超过它 | det-only low-LR 120 rows best `0.36279`；rawKD low-LR 105 rows best `0.36265` |
| P1 | `raw_feature_kd` | 去掉 LADD 分解，仅做 raw teacher feature KD，定位“分解/可达”是否引入负迁移 | B-only raw feature KD + logit off/on ablation | 至少不低于 baseline，且优于 shuffled-pair | low-LR rawKD 已启动，暂未超过 det-only low-LR |
| P2 | `dsn_shared_private` | 冻结已收敛 RGB/IR detector，先学习 shared/private latent，再用 shared latent 蒸馏 student | S1 projector 训练 -> S2 student distill | S2 超过 det-only reload，并且 S1 retrieval 明显非随机 | S1 已完成；S2 73 rows best AP50-95 `0.36435`，窗口均值高于 det-only但稳定性未证 |
| P2.1 | `dsn_shared_private/s2_refine_variants` | 保留 DSN shared latent，但降低后期干扰 | `w0p25_nodecay` + `w1p0_decay60_160_final0` | 50+/final/late-window 稳定超过 det-only，且后续 shuffled 不涨 | 等待型队列 pid `33404`，仍等待 primary >=100 rows 与空卡 |
| P2.2 | `dsn_shared_private/s2_shuffled_controls` | 破坏 batch 内 teacher/student pairing，验证 DSN shared latent 不是 reload/噪声 | `w1p0_shuffle_roll1` | shuffled 不应达到 matched DSN 的收益 | 后置等待队列已准备；等 P2.1 两个 refine 变体完成后再启动 |
| P3 | `reachable_fused_shared` | 两模态 share 先融合，再用 cap2/reachable 约束 fusion 接近两个 share、远离两个 private，并由 fusion 蒸馏新 student | nofusion split-rec control + sum/concat fused shared B-only distill，先等 low-LR controls 放行 | 超过全局 det-only、P3 nofusion split-rec control，且 shuffled-pair 失效 | 首次 P3c0 因并发 OOM/batch fallback 无效；队列已停止，待空卡 strict-batch 重启 |
| P4 | `oldsplit_90_hbb` | 把 90 服务器旧 Sixiang split/reach 方案搬到 DroneVehicle HBB | A1/A2/C adaptation | 若超过 det-only，说明旧主线仍有可迁移部分 | 已运行 GPU0 |
| P5 | `object_aware_dsn` | 只在 foreground/object tokens 上建 shared/private，减少背景跨模态错配 | S1 object-token projector -> S2 distill | 低于全图 DSN 的背景敏感性，mAP 提升更稳定 | 待启动 |
| P6 | `shared_only_ablation` | 不建 private，只蒸馏 shared bottleneck，测试 private 分支是否必要 | S2 distill only | 若接近 P2/P3，说明分解过重 | 待启动 |
| P7 | `teacher_conf_gated_kd` | 用 teacher confidence / objectness gate 过滤错误跨模态监督 | raw KD + gate | 优于 P1 且稳定性更好 | 待启动 |
| P8 | `late_kd_decay` | KD 只在早中期作用，后期回到 detector self-training，规避 late collapse | P1/P3 的 KD decay variant | final 不明显回落，late-window mean 提升 | 待启动 |
| P9 | `bn_freeze_controls` | 针对 reload/BN 污染，冻结 BN stats 或 teacher eval 强约束 | 对 P1/P3 加 BN freeze | 若只靠 BN freeze 修复，主线需先稳定训练协议 | 待启动 |
| P10 | `reachability_weighted_kd` | 不把 reach 当强损失，而是作为 KD token 权重；可达性只决定“哪些 token 值得蒸馏” | split/decomposed + `KD_WEIGHT_MODE=reachability_gap` | 优于 raw KD 与 teacher-conf gate，且不引起 split 结构掉点 | 待启动 |
| P11 | `oldsplit_a2only_controlled` | A2 出现过轻微正向，单独验证 A2-as-final，而不进入会回落的 C | A1 shared init -> A2 det-only split control -> A2 reach/KD | A2 reach/KD 超过全局 low-LR det-only 与同结构 A2 det-only | ready 队列已准备 |
| P12 | `object_prototype_kd` | 从 teacher foreground token 建 class/object prototype，student 对齐 prototype 而不是逐样本 feature | prototype memory + student feature KD | 对 pair 顺序不敏感，shuffled-pair 不应带来同等提升 | 待实现 |
| P13 | `box_logit_only_kd` | 避免 feature 空间硬对齐，只蒸馏 box distribution / class logits / VLR token | LD/CCLKD-style logit-only transfer | 若优于 feature KD，说明跨模态 feature 对齐过强 | 待启动或轻改 |
| P14 | `ema_fusion_teacher` | teacher decomposition/fusion target 使用 EMA，减少 B-stage target 漂移 | `TEACHER_TARGET_MODE=ema` + P3/P10 | late-window 更稳，BN/reload 诊断不过度恶化 | launcher 已支持，待启动 |
| P15 | `bidirectional_cycle_distill` | 同时训练 IR->RGB 与 RGB->IR 的 shared latent，用 cycle consistency 约束 common feature | 双向 S1/S2 或 alternating KD | 两个方向都优于各自 det-only，才算强证据 | 待实现 |
| P16 | `learnability_as_weight` | 把 LADD 的 learnable/unlearnable 判别从特征分支改成 sample/token weighting，减轻结构改动 | reach / task confidence / teacher margin 合成权重 | 低于 P3 风险，但能稳定超过 reload control | 待实现 |

## 2. `reachable_fused_shared` 设计草案

目录：

```text
runs_public/dronevehicle_method_search/sub2k_seed0_fullval/reachable_fused_shared/
logs/dronevehicle_method_search/sub2k_seed0_fullval/reachable_fused_shared/
docs/experiments/dronevehicle_method_search_20260623/reachable_fused_shared/
```

结构：

```text
f_rgb -> share_rgb, private_rgb
f_ir  -> share_ir,  private_ir

fusion = MLP(sum_or_concat(proj_rgb(share_rgb), proj_ir(share_ir)))

student f_s -> share_s, private_s
distill: share_s <- fusion
```

S1 损失：

```text
L = L_cap2(fusion, share_rgb, share_ir, private_rgb, private_ir)
  + lambda_task * L_weak_task(fusion)
  + lambda_rec  * L_rec([share, private], frozen_feature)
  + lambda_sep  * L_sep(share, private)
```

其中 `L_cap2` 沿用旧 cap2/reach 思路，但 anchor 改为 fused shared：拉近 `fusion <-> share_rgb/share_ir`，拉远 `fusion <-> private_rgb/private_ir`。融合方式先做两个版本：

1. `sum_mlp`：两个 share 先投影到同维度后相加，再过 MLP。
2. `concat_mlp`：两个 share 拼接后过 MLP。

S2 蒸馏：

```text
L_student = L_det(student)
          + alpha_kd * L_kd(share_s, stopgrad(fusion))
          + lambda_sep_s * L_sep(share_s, private_s)
```

必要控制：

```text
c0_detonly_reload: 同 student init、同 epochs、无 KD
c1_shuffled_pair: teacher/fusion pairing 打乱
c2_no_cap2: 去掉 reachable，只保留 fusion KD
c3_sum_vs_concat: sum_mlp 与 concat_mlp 对照
c4_no_private: 去掉 private 分支
```

2026-06-24 状态：当前实现为 HBB 最小侵入版，不新开完整 S1 预训练脚本，而是使用旧 A2 checkpoint 中已有的 `teacher_decomposition/teacher_decoder/teacher_task_heads/student_split`，再从 DroneVehicle RGB baseline split-load detector。新增 `fused_shared_mode=sum|concat`、CAP2、weak-task、fusion-KD 开关；首次远端队列在 GPU 并发下启动的 `c0_nofusion_splitrec` 出现 OOM 和 batch fallback，结果无效，队列已停止。后续重启必须使用 strict batch64，并先跑 `c0_nofusion_splitrec` 作为 P3 自身架构对照，再跑 `sum_mlp_cap2` 和 `concat_mlp_cap2`。

## 3. 低风险参数化候选

这些方案不需要再改核心网络，适合在 low-LR/no-warmup control 解释清楚后快速排队：

| 方案 | 可用开关 | 首发目录 | 备注 |
|---|---|---|---|
| teacher confidence gate | `KD_WEIGHT_MODE=teacher_task_conf` | `teacher_conf_gated_kd/teacher_conf_gate` | 复用 A2 teacher task head，仅给 KD token 加权 |
| raw KD late decay | `LADD_KD_DECAY_MODE=linear` | `teacher_conf_gated_kd/rawkd_late_decay` | epoch 60-160 把 KD 从 1.0 衰减到 0 |
| reachability weighted KD | `KD_WEIGHT_MODE=reachability_gap` | `reachability_weighted_kd/` | 需要 split/decomposed，可达性只做权重，不做强 reach loss |
| teacher EMA target | `TEACHER_TARGET_MODE=ema` | `ema_fusion_teacher/` | 可叠加到 P3/P10，目标是减少 target drift |

已准备但暂不抢卡的队列脚本：

```text
docs/experiments/dronevehicle_method_search_20260623/teacher_conf_gated_kd/queue_teacher_conf_and_late_decay_after_primary_20260624.sh
```

该脚本默认等 low-LR det-only、low-LR raw KD、low-LR CMDistill 都至少 20 epoch 后再发 `teacher_conf_gate` 与 `rawkd_late_decay`。

2026-06-24 00:50 CST 追加 ready 脚本：

```text
docs/experiments/dronevehicle_method_search_20260623/reachability_weighted_kd/queue_reachability_weighted_after_controls_20260624.sh
```

它同样等待 low-LR det-only/rawKD/CMDistill 至少 20 rows，再发 `splitkd_unweighted` 与 `reachgap_weighted`。当前仅同步准备，不启动。

另一个从当前结果派生出的 ready 队列：

```text
docs/experiments/dronevehicle_method_search_20260623/oldsplit_a2only_controlled/queue_oldsplit_a2only_after_controls_20260624.sh
```

原因：`oldsplit_a2` 曾达到 `AP50-95=0.36326`，略高于 RGB baseline，但 C 阶段回落，且没有同结构 control。该队列会复用 A1 后分叉出 `a2_detonly_split_control` 与 `a2_reach_kd_lowlr`，用于判断 A2 正信号是否真实。

## 4. 自动监控节奏

已创建 30 分钟 heartbeat 自动化，关注：

- CMDistill-style 是否超过 baseline / det-only control。
- DSN S1 retrieval 是否继续提升，是否可进入 S2。
- oldsplit 90 HBB A2/C 是否 collapse 或正向。
- 空闲 GPU 上是否可以补发下一个候选或必要 control。

如果 P0 在前 30-50 epoch 已明显低于 baseline 且无回升迹象，仍保留到至少中期观察；但可以并行启动 P1/P2 control，不等待 P0 完全结束。
