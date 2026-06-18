# LADD-clean / LADD-A1B 方法定义与实现口径

最后更新：2026-06-18

本文档定义当前 LADD 主线方法。自本口径起，主线不再使用 `A1 -> A2 -> B`，而使用更干净的：

```text
LADD-clean / LADD-A1B = A1 teacher decomposition warmup -> B SAR detector training + RGB teacher distillation
```

A2 只保留为历史诊断和消融入口，不再作为主线阶段，不应把旧 A1-A2-B 结果写作 clean A1B 结果。

当前主表协议固定为 `mosaic100`：800 epoch 中前 100 epoch 开 mosaic，后 700 epoch 关闭。全程 no-mosaic 只作为鲁棒性/附录协议，不与 mosaic100 主表直接混合。

## 1. 方法名称

| 名称 | 推荐用途 |
|---|---|
| `LADD-clean` | 论文/报告主方法名，强调去掉历史辅助损失和 A2 |
| `LADD-A1B` | 实验标签和代码标签，强调阶段链路 |
| `LADD Probe-A` | 当前固定主线的内部简称 |
| `clean_a1b_dynprobe` | 当前固定主线 run tag，表示 dynamic teacher core + frozen reach probe |
| `clean_a1b` | Static 消融标签，B 阶段冻结 teacher decomposition |
| `clean_a1b_dyn` | Dynamic 消融标签，B 阶段动态 teacher decomposition 且 reach probe 也继续训练 |

核心定义：训练时使用 paired RGB/SAR。RGB teacher 的中间特征被分解为 task-relevant common representation `z_t` 与 private/unlearnable component `u_t`；SAR student 学到 `z_s`，并在 B 阶段用 `z_s -> z_t` KD 辅助 SAR-only detector 训练。推理时仍是 SAR-only detector。

### 当前固定主线口径

当前只考虑 SAR baseline 初始化版本，不考虑 YOLO-init 版本。也就是说，A1 的 `MODEL` 是同协议收敛 SAR baseline `best.pt`，RGB teacher 是同协议收敛 RGB baseline `best.pt`。

| 线 | 标签 | B 阶段 teacher decomposition | B 阶段 reach probe | 推荐用途 |
|---|---|---|---|---|
| **Probe-A** | `clean_a1b_dynprobe` | train，继续打开 `t_rec/reach/taskL` | frozen/eval，且 reach loss 中 `q_s` detach | **固定主线** |
| Static | `clean_a1b` | frozen/eval | frozen/eval | 消融：验证 B 阶段不动态更新 teacher core 的影响 |
| Dynamic | `clean_a1b_dyn` | train，继续打开 `t_rec/reach/taskL` | train | 消融/诊断：验证完全动态 reach probe 是否带来不稳定 |

三条线都不包含 A2，也不恢复已经删除的历史 auxiliary/debug loss。主方法报告中默认使用 Probe-A；Static/Dynamic 只作为 ablation 或 diagnostic curve。

## 2. 现有代码实际阶段行为

代码入口：

| 文件 | 责任 |
|---|---|
| `ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py` | phase freeze、phase loss scale、paired SAR/RGB loader |
| `ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py` | detector loss、teacher/student reconstruction、reach、taskL、LADD KD，以及 FGD/LD/CMDistill/CCLKD comparison profile |
| `ladd/code/src/teacher_student_decomposition_kd_hbb/model.py` | teacher decomposition、student split、task/reach heads |
| `ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/` | 当前部署快照；本次已同步，和 `ladd/code/src` 保持一致 |

### A1 的实际 freeze / train

`trainer.py::_apply_manual_phase()` 中，`phase=a1` 的实际行为是：

| 模块 | A1 状态 | clean A1B 解释 |
|---|---|---|
| detector backbone/head `model.model` | frozen | A1 不更新 SAR detector，默认 `DET_LOSS_SCALE=0` |
| `student_split` | frozen | A1 不训练学生分解 |
| `teacher_decomposition` | train | 学习 `f_t -> z_t, u_t` |
| `teacher_decoder` | train | 支持 teacher reconstruction / decoded `z_t` |
| `student_reachability` | train if `reach_input_mode=adapter` | 学 reach query `q_s`，用于可达性约束 |
| `teacher_task_heads` | train | 在 `decoded_z_t` 上做 task-discriminative supervision |

### B_static 的实际 freeze / train

Static 消融使用 `LADD_B_A2_CORE=0`，对应 `trainer.py::_apply_manual_phase()` 的默认 B 逻辑：

| 模块 | B 状态 | clean A1B 解释 |
|---|---|---|
| detector backbone/head `model.model` | train | 正常 SAR detector 训练 |
| `student_split` | train if branch uses `z_s` | 学 `f_s -> z_s, r_s`，供 KD 和 student reconstruction 使用 |
| `teacher_decomposition` | frozen/eval | 固定 A1 学到的 teacher common space |
| `teacher_decoder` | frozen/eval | 固定 A1 teacher reconstruction path |
| `student_reachability` | frozen/eval | B 默认不继续训练 reach adapter |
| `teacher_task_heads` | frozen/eval | taskL 不在 B 中继续更新 teacher core |

因此 Static B 的长期主项是 detector loss 和 `z_s -> z_t` KD。`task_loss/reach/t_rec` 在 B 中为 0 是 phase scale 关闭的结果，不是代码没有这些实现。

### B_dynamic / B_probeA 的实际 freeze / train

Dynamic 和 Probe-A 都使用 `LADD_B_A2_CORE=1`。这不是恢复 A2，而是在 B 阶段继续训练 A1 初始化的 teacher decomposition core。二者的差异只在 student reachability probe 是否继续训练：

| 模块 | Dynamic 状态 | Probe-A 状态 | 解释 |
|---|---|---|
| detector backbone/head `model.model` | train | train | 正常 SAR detector 训练 |
| `student_split` | train | train | 学 `f_s -> z_s, r_s` |
| `teacher_decomposition` | train | train | `z_t/u_t` 随 B 阶段动态适配 |
| `teacher_decoder` | train | train | 支持 B 中的 teacher reconstruction |
| `student_reachability` | train if `reach_input_mode=adapter` | frozen/eval | Probe-A 把 reach query `q_s` 作为 A1 学到的固定 probe |
| `teacher_task_heads` | train | train | B 中继续 task-discriminative supervision |
| reach loss 中的 `q_s` | no detach | detach | Probe-A 避免 reach loss 在 B 阶段继续拉动学生 reach probe |

因此 Probe-A 保留 dynamic teacher core 的适配能力，但冻结 A1 学到的 student reachability probe，避免 B 阶段 reach loss 与 detector/KD 对学生侧目标产生额外拉扯。当前曲线证据显示 Probe-A 比完全 Dynamic 更稳定，因此固定为主线。

## 3. clean A1B objective 定义

clean A1B 主方法的 objective 只包含下面这些真正进入训练 loss 的项：

```text
L_A1 = lambda_rec * L_t_rec
     + lambda_reach * (lambda_match_inner * L_reach_match
                     + lambda_rank_inner * L_reach_rank_cap)
     + lambda_taskL * L_task

L_B_static  = L_det
            + alpha_kd * L_KD(z_s, stopgrad(z_t))
            + alpha_s_rec * L_s_rec

L_B_dynamic = L_det
            + alpha_kd * L_KD(z_s, stopgrad(z_t))
            + alpha_s_rec * L_s_rec
            + lambda_rec * L_t_rec
            + lambda_reach * (lambda_match_inner * L_reach_match
                            + lambda_rank_inner * L_reach_rank_cap)
            + lambda_taskL * L_task

L_B_probeA  = L_det
            + alpha_kd * L_KD(z_s, stopgrad(z_t))
            + alpha_s_rec * L_s_rec
            + lambda_rec * L_t_rec
            + lambda_reach * (lambda_match_inner * L_reach_match(stopgrad(q_s), z_t, u_t)
                            + lambda_rank_inner * L_reach_rank_cap(stopgrad(q_s), z_t, u_t))
            + lambda_taskL * L_task
```

其中 foreground mask / teacher mask 是作用域和统计机制，不是 clean 主方法里的额外 loss。当前代码中已移除 mask regularization loss，只保留 `mask_mean/mask_std/mask_fg_mean/mask_bg_mean` 诊断统计。

### A1 objective 项

| loss 字段 | clean A1B 状态 | 数学/语义作用 | 梯度流 |
|---|---:|---|---|
| detector loss | off | A1 不优化 detector AP，只保持 baseline 检测头不被扰动 | 无 |
| `t_rec_loss` | on | `L1(recon_t, f_t)`，约束 `z_t/u_t` 可重建 teacher feature | `teacher_decomposition`, `teacher_decoder` |
| `reach_match_loss` | on | 让 SAR reach query `q_s` 接近 teacher common `z_t` | `teacher_decomposition`, `student_reachability` |
| `reach_rank_loss` | on, cap2 | `softplus(delta + d_pos - min(d_neg, cap))`；默认 `rank_d_neg_cap=2.0`，避免继续奖励反平行坍缩 | `teacher_decomposition`, `student_reachability` |
| `task_loss` / taskL | on | `teacher_task_heads(decoded_z_t)` 对齐 YOLO assigner 的 `target_scores`，正面定义为 task-discriminative common representation supervision | `teacher_decomposition`, `teacher_decoder`, `teacher_task_heads` |

### A1/B 使用但不是 objective 的机制

| 机制 | clean A1B 状态 | 作用 |
|---|---:|---|
| foreground mask for reach | on | `USE_FG_MASK_FOR_REACH=1`，reach 只在有检测监督的 foreground token 上计算 |
| teacher decomposition mask | on | `USE_MASK=1` 使 teacher decomposition 产生 mask 和统计；clean 不启用 mask regularization loss |
| foreground mask for reconstruction | off | `USE_FG_MASK_FOR_REC=0`，teacher/student reconstruction 默认不只限 foreground |

### B_probeA objective 项（当前固定主线）

| loss 字段 | Probe-A 状态 | 数学/语义作用 | 梯度流 |
|---|---:|---|---|
| detector loss | on | 正常 YOLO HBB `box/cls/dfl` 检测训练 | SAR detector |
| `kd_loss` | on | foreground token 上让 `z_s` 对齐当前 `z_t`，默认 MSE/代码默认 KD 模式 | `student_split` 和相关 SAR feature path |
| `s_rec_loss` | on | student split reconstruction，`student_recon` 对齐 `student_raw`，防止 `z_s/r_s` 退化成任意拆分 | `student_split` |
| teacher `t_rec_loss` | on | B 中继续约束 teacher decomposition 可重建 | `teacher_decomposition`, `teacher_decoder` |
| reach losses | on, `q_s` detached | B 中继续维护 teacher common/private 几何，但不继续训练 student reach probe | `teacher_decomposition` 为主；`student_reachability` frozen |
| taskL | on | B 中继续保持 `z_t` 的 task-discriminative 语义 | `teacher_decomposition`, `teacher_decoder`, `teacher_task_heads` |

这个选择来自当前曲线证据：Static 的变量更少但收益较弱；完全 Dynamic 的 B 阶段变量最多，s 模型曲线出现过明显不稳定；Probe-A 在保留动态 `z_t` 适配能力的同时固定 reach probe，表现更稳定，叙事也更集中。

### B_static objective 项（消融）

| loss 字段 | clean A1B 状态 | 数学/语义作用 | 梯度流 |
|---|---:|---|---|
| detector loss | on | 正常 YOLO HBB `box/cls/dfl` 检测训练 | SAR detector |
| `kd_loss` | on | foreground token 上让 `z_s` 对齐 frozen `z_t`，默认 MSE/代码默认 KD 模式 | `student_split` 和相关 SAR feature path |
| `s_rec_loss` | on | student split reconstruction，`student_recon` 对齐 `student_raw`，防止 `z_s/r_s` 退化成任意拆分 | `student_split` |
| teacher `t_rec_loss` | off in B | A1 已固定 teacher decomposition；B 不继续改 teacher core | 无 |
| reach losses | off in B | B 默认不继续训练 reach adapter；避免把 B 重新变成 A2-core | 无 |
| taskL | off in B | task-discriminative `z_t` supervision 已在 A1 完成；B 使用 frozen `z_t` 作为 KD target | 无 |

Static 用于验证“固定 A1 teacher target，只靠 detector/KD/student-rec 是否足够”。它不是当前固定主线。

### B_dynamic objective 项（消融/诊断）

| loss 字段 | dynamic 状态 | 数学/语义作用 |
|---|---:|---|
| detector loss | on | 正常 YOLO HBB `box/cls/dfl` 检测训练 |
| `kd_loss` | on | foreground token 上让 `z_s` 对齐当前 `z_t` |
| `s_rec_loss` | on | student split reconstruction |
| teacher `t_rec_loss` | on | B 中继续约束 teacher decomposition 可重建 |
| reach losses | on | B 中继续维护 SAR reach query 与 teacher common space 的可达性 |
| taskL | on | B 中继续保持 `z_t` 的 task-discriminative 语义 |

Dynamic 的潜在优点是 `z_t/u_t` 和 reach probe 都可以随 SAR detector 和 KD 过程共同适配；风险是变量更多，且当前 s 模型曲线有不稳定现象。因此 Dynamic 只作为 Probe-A 的对照消融，不再作为主线。

### 已从当前 LADD 实现移除的历史 loss

这些项不再通过“设权重为 0”关闭，而是已从当前 HBB LADD 代码面删除：CLI 参数不再暴露、trainer 不再配置、loss vector 不再返回、model 不再实例化对应辅助 head。

| 历史项 | 当前状态 | 移除原因 |
|---|---|---|
| `t_sep_loss`, `s_sep_loss` | loss 定义和 CLI 权重已移除 | separation/decorrelation 是历史调试项，A1/B 主线不依赖 |
| `r_aux_loss` | residual aux head、mode、loss 已移除 | B 中长期量级很小，不作为主方法核心 |
| `u_aux_loss` | teacher private auxiliary loss 已移除 | teacher private auxiliary 叙事复杂，clean A1 用 reconstruction/reach/taskL 即可 |
| `mask_reg_loss` | mask target/sparse/smooth loss 已移除 | mask 只作为 decomposition 机制和统计，不作为 objective |
| `r_obb_loss`, `r_sar_loss` | residual OBB/SAR head 与 loss 已移除 | residual OBB/SAR auxiliary 是历史探索 |
| `proto_cls_loss`, `dkd_loss` | prototype/DKD loss 已移除 | 与当前固定方法集合无关；CCLKD 保留为独立 comparison profile |
| `s_repel_loss`, `path_b_loss`, `rs_comp_loss`, `recon_task_loss` | Path/repel/rs_comp/recon-task loss 已移除 | debug/ablation history，不进入主方法 |
| A2 | clean launcher 不运行 | 不稳定且叙事成本高 |

## 4. 为什么去掉 A2

当前实验事实支持把 A2 从主线中移除：

| 事实 | 结论 |
|---|---|
| A1 稳定：`task_loss` 约 `20 -> 0.30`，`reach_match_loss` 约 `0.56-0.62 -> 0.0018`，`reach_rank_loss` 约 `0.15 -> 0.022`，`t_rec_loss` 约 `0.022 -> 0.0077` | A1 足够承担 teacher common representation warmup |
| A1 后检测 AP 基本保持 SAR baseline | A1 不破坏 detector，是干净 warmup |
| AutoDL seed123 A2 完整 50 epoch：AP `0.53528 -> 0.50241`，best epoch 1 | A2 loss 下降不等于检测改善 |
| 90 seed42 A2 best epoch 1，后续 AP 崩溃并 NaN | A2 有数值稳定风险 |
| 90 seed0 retry A2 reach loss 失控并出现 NaN | A2 会引入额外不稳定源 |
| B 若使用 A2 `best.pt`，经常等价于 A2 epoch 1 | 实际效果接近跳过 A2 |

因此 clean 主线直接使用 `A1 best.pt -> B`。旧 A2 结果只能作为 diagnostic/ablation history，不进入 clean A1B 主表。

## 5. 与旧 A1-A2-B 实现的区别

| 项 | 旧主线 | clean A1B |
|---|---|---|
| 阶段 | `A1 -> A2 -> B` | `A1 -> B` |
| A2 | 主线必要阶段 | 不运行；历史诊断/消融 |
| A1 sep/private aux | 旧代码可算 | 当前代码已移除 |
| B sep/residual aux | 旧配置中可能非零 | 当前代码已移除 |
| B core | 可出现 `B_A2_CORE` 诊断 | Probe-A 主线使用 `LADD_B_A2_CORE=1` 且 `LADD_B_FROZEN_REACH_PROBE=1` |
| B task/reach/t_rec | 若 `B_A2_CORE=1` 会启用 | Probe-A 打开 teacher core/reach/taskL，但冻结 student reach probe；Static/Dynamic 必须单独标记为消融 |
| run tag | 历史 tag 含 A2/cap2 等混杂语义 | 必须包含 `clean_a1b` |
| 结果口径 | 可作为历史或附录 | 主表候选 |

## 6. 推荐实验协议

主线协议采用 mosaic-first100-close700，并要求 baseline、LADD、comparison methods 同协议重跑：

```text
dataset = OGSOD-1.0 HBB
imgsz = 256
epochs_B = 800
mosaic = 1.0
close_mosaic = 700   # 800 epoch 中前 100 epoch mosaic on，后 700 epoch mosaic off
cos_lr = true
optimizer = auto
lr0 = 0.01
lrf = 0.01
warmup_epochs = 3.0
warmup_bias_lr = 0.1
deterministic = true
batch = n/s:64, m/l:32, x:16
rank_d_neg_cap = 2.0
```

旧 formal no-mosaic 结果和旧 close@100/400ep 结果可以作为历史或附录，不应和 mosaic clean A1B 结果混在主表中直接比较。当前 no-mosaic Probe-A 可以作为“同方法在旧 formal 协议下是否仍稳定”的鲁棒性证据，但不能替代 mosaic100 主表。

当前优先级是先补齐 Probe-A 的 `n/s/m` 容量曲线。Static/Dynamic 已经不再是必须铺开的主线，只保留为消融证据：

| 优先级 | 容量 | 主线 | 前置条件 |
|---|---|---|---|
| P0 | `n` | Probe-A `clean_a1b_dynprobe` | `n` SAR/RGB mosaic100 baseline 已可用；当前 4090 run 中断，需补齐 |
| P0 | `s` | Probe-A `clean_a1b_dynprobe` | `s` SAR/RGB mosaic100 baseline 已可用；mosaic100 已完成一条 |
| P1 | `m` | Probe-A `clean_a1b_dynprobe` | 等 `m` SAR/RGB mosaic100 baseline 完成 |
| P2 | `n/s` | Static/Dynamic 消融 | 只在需要分析 B teacher core / reach probe 贡献时补充 |
| P3 | `n/s` | no-mosaic Probe-A | 旧 formal 协议鲁棒性，不进 mosaic100 主表 |

## 7. clean launcher profile

Probe-A 主线入口：

```bash
LADD_A1B_MODE=dynamic_probe bash ladd/scripts/launch_ladd_clean_a1b_job.sh <n|s|m|l|x> <seed> <gpu_id>
```

Static 消融入口：

```bash
bash ladd/scripts/launch_ladd_clean_a1b_job.sh <n|s|m|l|x> <seed> <gpu_id>
```

Dynamic 消融入口：

```bash
LADD_A1B_MODE=dynamic bash ladd/scripts/launch_ladd_clean_a1b_job.sh <n|s|m|l|x> <seed> <gpu_id>
```

该入口直接调用 `ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh`，顺序只跑 A1 和 B：

```text
A1: MODEL = SAR_BASELINE
B:  MODEL = A1 weights/best.pt
```

关键 clean profile：

| 变量 | 值 |
|---|---|
| `RANK_D_NEG_CAP` | `2.0` |
| `LAMBDA_REC` | `0.1` |
| `LAMBDA_TASKL` | `1.0` |
| `LAMBDA_REACH`, `LAMBDA_MATCH_INNER`, `LAMBDA_RANK_INNER` | `1.0` |
| `ALPHA_KD` | `1.0` |
| `ALPHA_S_REC` | `0.1` |
| `USE_MASK`, `USE_FG_MASK_FOR_REACH`, `USE_FG_MASK_FOR_REC` | `1`, `1`, `0` |
| `LADD_B_DET_ONLY`, `LADD_A2_DET_ONLY` | `0`, `0` |

三条 profile 差异为：

| `LADD_A1B_MODE` | run tag | `PROJECT_DIR` key | `LADD_B_A2_CORE` | `LADD_B_FROZEN_REACH_PROBE` | 用途 |
|---|---|---|---:|---:|---|
| `dynamic_probe` | `clean_a1b_dynprobe_*` | `ladd_clean_a1b_dynamic_probe` | `1` | `1` | 当前固定主线 |
| `static` | `clean_a1b_*` | `ladd_clean_a1b` | `0` | `0` | 消融 |
| `dynamic` | `clean_a1b_dyn_*` | `ladd_clean_a1b_dynamic` | `1` | `0` | 消融/诊断 |

每次运行会在对应 `logs/ladd_clean_a1b*/*/run_meta_clean_a1b.env` 写入 profile、loss 权重、协议参数、A1/B run name 和 checkpoint 链接；各阶段仍保留 `run_ladd_phase.sh` 自己的 `manifest.txt`。

### 当前防误开方式

当前防误开方式是代码层面删除历史 loss surface：

1. `train_ladd_hbb.py` / `current_hbb/tools/train_ladd_hbb.py` 不再接受 sep/aux/debug/proto/DKD/mask-reg 相关 CLI 参数。
2. `trainer.py` 的 `loss_names` 只包含 `box/cls/dfl/angle/t_rec/reach_match/reach_rank/task/kd/s_rec` 和 mask/reach 统计。
3. `loss.py` 的 loss tensor 只返回 10 个 loss 项；comparison profile 只保留 `fgd/ld/cmdistill/cclkd`。
4. `model.py` 不再实例化 residual aux、recon-task、r_obb、r_sar 辅助 head。
5. `launch_ladd_clean_a1b_job.sh` 不传任何历史 loss 参数，且固定只跑 A1/B。

## 8. clean 主表准入与 diagnostic 边界

可进入 clean A1B 主表的结果必须同时满足：

1. run tag 与 profile 必须可追溯；主方法行使用 `clean_a1b_dynprobe`，Static/Dynamic 只用于 ablation；
2. phase chain 是 `A1 -> B`，没有 A2；
3. 使用当前 cleaned LADD 代码；结果 CSV 不应出现 `t_sep/s_sep/r_aux/u_aux/mask_reg/recon_task/rs_comp/r_obb/s_repel/path_b/r_sar/dkd/proto_cls` 字段；
4. 主方法行必须是 `clean_a1b_dynprobe` / `LADD_A1B_MODE=dynamic_probe`；Static/Dynamic 只能进入 ablation 表；
5. baseline、LADD、comparison methods 使用同一个 mosaic/close_mosaic 协议；
6. 不包含 checkpoint 权重或其他大文件进入 public 包。

只能作为 diagnostic 的结果：

| 结果类型 | 处理 |
|---|---|
| 旧 A1-A2-B full chain | 历史/附录/消融，不写作 clean A1B |
| A2 best/last 选择实验 | A2 稳定性诊断 |
| no-mosaic formal LADD | 鲁棒性/附录；不能与 mosaic 主表直接比较 |
| Static `clean_a1b` | 消融：固定 teacher target 的影响 |
| Dynamic `clean_a1b_dyn` | 消融：完全动态 teacher core + reach probe 的影响 |
| 未标记 `clean_a1b_dynprobe` 的主方法 run | 不能写作最终 LADD Probe-A 主线 |
| sep/aux/debug loss 非零 run | auxiliary ablation，不是 clean 主线 |
| mosaic 与 no-mosaic 混合表 | 不作为同协议主表 |
