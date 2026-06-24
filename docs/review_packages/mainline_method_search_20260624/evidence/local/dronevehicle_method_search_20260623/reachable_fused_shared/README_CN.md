# Reachable Fused Shared

日期：2026-06-23

这是用户提出的新主线候选：双模态 backbone 的特征仍拆为 shared/private，但两个 shared 先通过额外投影融合成一个 fused shared；fused shared 同时受 reachable/cap2 约束、弱任务约束，并蒸馏给一个从头或从 RGB baseline 继续训练的 student shared 分支。

## 当前设计选择

- 冻结或半冻结已收敛双模态 detector/decomposition 作为 feature source。
- 当前 HBB 最小侵入实现使用旧 A2 checkpoint 提供 `z_t/u_t` 和 `z_s/r_s`，检测器从 DroneVehicle RGB baseline best split-load。
- fusion 先试 `sum_mlp`，再试 `concat_mlp`。
- reachable/cap2 的 anchor 使用 `fusion`，正样本为两个 modality share，负样本为两个 modality private。
- student 仍拆成 `share_s/private_s`，只让 `share_s` 接受 fused shared 蒸馏，`private_s` 用 separation 保持互补。

## 2026-06-24 接入状态

已在当前 HBB 主线中加入默认关闭的 fused shared 开关：

```text
ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/model.py
ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/loss.py
ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/trainer.py
ladd/code_versions/current_hbb/tools/train_ladd_hbb.py
ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh
```

新增参数：

```text
--fused-shared-mode none|sum|concat
--fused-shared-align-weight
--fused-shared-reach-weight
--fused-shared-kd-weight
--fused-shared-task-weight
```

loss 形式：

```text
fusion = MLP(z_t + z_s)       # sum, initialized near 0.5 * (z_t + z_s)
fusion = MLP(concat(z_t,z_s)) # concat

L_cap2 = pull(fusion, z_t/z_s) + rank(fusion closer to z_t/z_s than u_t/r_s)
L_task = weak_task_head(teacher_decoder(fusion), y)
L_kd   = KD(z_s, stopgrad(fusion))
```

远端 `ladd4090-zw1` 已通过 `py_compile` 与 `bash -n`。P3 队列已启动：

```text
logs/dronevehicle_method_search/sub2k_seed0_fullval/reachable_fused_shared/queue_reachable_fused_shared_after_controls_20260624.sh
logs/dronevehicle_method_search/sub2k_seed0_fullval/reachable_fused_shared/queue_reachable_fused_shared_after_controls_20260624.log
```

队列条件：等待 low-LR/no-warmup `detonly_reload` 与 `raw_feature_kd` control 都至少 20 epoch，且 det-only control 不出现明显 reload collapse；随后依次启动 `c0_nofusion_splitrec`、`sum_mlp_cap2` 与 `concat_mlp_cap2`。

2026-06-24 00:33 CST 更新：为了避免把 split-load/student-rec 架构收益误判为 fused shared 收益，队列脚本已更新并重启。新版队列 PID 为 `20241`，仍在等待 low-LR control 结果文件。

2026-06-24 01:18 CST 更新：`c0_nofusion_splitrec` 曾在 01:13:55 被队列启动，但当时 GPU0 已有 rawKD low-LR 与 CMDistill low-LR 并发，触发 OOM 和 Ultralytics 自动 batch fallback：

```text
batch 64 -> 32 -> 16 -> 8 -> torch.OutOfMemoryError
```

该 P3 c0 run 不生成有效结果，不能作为架构 control。为了避免继续启动 `sum_mlp_cap2` / `concat_mlp_cap2` 并混入错误协议，已停止队列 shell `pid=20241`；没有停止训练进程。后续 P3 必须在空卡或可确认显存充足时，用 `STRICT_BATCH_SIZE=1` 重新启动；若 batch64 不能成立，则直接失败，不允许降 batch 后纳入正式比较。

## 第一批实验

```text
P3a: sum_mlp + cap2 + weak_task + rec + sep -> S2 distill
P3b: concat_mlp + cap2 + weak_task + rec + sep -> S2 distill
P3c0: no fusion / no KD, same split-load + student-rec architecture control
P3c: sum_mlp, no cap2 -> isolate cap2 contribution
P3d: sum_mlp, shuffled teacher pairing -> rule out schedule gain
P3e: detector-only continued-training / reload control
```

## 成败标准

只有当 S2 超过 RGB baseline、超过全局 detector-only continued-training、超过 P3 自身 `c0_nofusion_splitrec` 架构对照、且 shuffled-pair 不涨点时，才把该方案视为主线候选。否则它只能作为机制诊断：例如说明 fusion/shared-private 本身可学，但不一定能带来 detector AP。
