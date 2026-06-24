# Teacher-Confidence Gated KD

日期：2026-06-24

## 目的

这条线测试一个更保守的主线改法：不先改变 detector 主干，也不强制分解 student feature，而是把跨模态 KD 只放大在 teacher task head 认为可靠的 foreground token 上。

关键问题：

```text
如果 raw feature KD 受跨模态噪声拖累，那么 teacher-confidence weighting 是否能优于 low-LR raw KD 与 det-only reload control？
```

## 设计

首发版本复用 oldsplit A2 checkpoint 中的 teacher decomposition / decoder / task heads，只用它们产生 `z_t` 和 token confidence；student 仍从 RGB baseline best 继续，检测分支保持 raw。

关键开关：

```text
TEACHER_FEATURE_MODE=decomposed
STUDENT_BRANCH_MODE=raw
KD_WEIGHT_MODE=teacher_task_conf
ALPHA_KD=0.25
LAMBDA_REACH=0.0
LAMBDA_REC=0.0
LAMBDA_TASKL=0.0
KD_CALIBRATION_MODE=affine
LR0=0.001
WARMUP_EPOCHS=0.0
```

解释：

- `teacher_task_conf` 使用 teacher task head 在正样本 token 上的类别置信度作为 KD token 权重。
- `LAMBDA_TASKL=0` 表示不再训练 teacher task head，只把 A2 checkpoint 里的 task head 当作一个可靠性估计器。
- 与 P3 fused shared 不同，这条线不引入新的 fusion module，主要验证“跨模态监督是否需要门控”。

## 队列脚本

本地脚本：

```text
docs/experiments/dronevehicle_method_search_20260623/teacher_conf_gated_kd/queue_teacher_conf_and_late_decay_after_primary_20260624.sh
```

远端建议路径：

```text
logs/dronevehicle_method_search/sub2k_seed0_fullval/teacher_conf_gated_kd/queue_teacher_conf_and_late_decay_after_primary_20260624.sh
```

2026-06-24 已同步到该远端路径，并通过 `bash -n` 检查；当前只是 ready 状态，尚未启动，避免与 low-LR controls / DSN S2 / P3 队列抢同一张刚释放的 GPU。

默认不抢当前主队列。启动后它会等待：

1. low-LR det-only control 至少 `MIN_ROWS=20` 且不崩。
2. low-LR raw KD 至少 `MIN_ROWS=20`。
3. low-LR CMDistill 至少 `MIN_ROWS=20`，确认 P0 sanity 有初步读数。

然后依次发：

```text
teacher_conf_gate/
rawkd_late_decay/
```

`rawkd_late_decay` 是 P8 的最小版本，用来判断 KD 后期衰减是否比持续 KD 更稳。
