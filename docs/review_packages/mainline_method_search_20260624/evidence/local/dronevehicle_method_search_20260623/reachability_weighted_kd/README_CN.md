# Reachability-Weighted KD

日期：2026-06-24

## 目的

这条线把 LADD 的“可达性”从强约束改成 token weighting：

```text
不再强行优化 L_reach(q_s, z_t, u_t)
只用 reachability gap = d(q_s, u_t) - d(q_s, z_t)
给 feature KD 的 foreground token 加权
```

这样可以验证一个关键假设：

```text
LADD 旧主线的问题可能不是“可达性没有信息”，而是把可达性作为强损失会扰乱 detector。
```

## 首发 run

队列脚本：

```text
docs/experiments/dronevehicle_method_search_20260623/reachability_weighted_kd/queue_reachability_weighted_after_controls_20260624.sh
```

远端 ready 路径：

```text
logs/dronevehicle_method_search/sub2k_seed0_fullval/reachability_weighted_kd/queue_reachability_weighted_after_controls_20260624.sh
```

队列默认等待：

1. low-LR det-only control 至少 20 rows。
2. low-LR raw KD control 至少 20 rows。
3. low-LR CMDistill sanity 至少 20 rows。

然后依次启动：

```text
splitkd_unweighted/
reachgap_weighted/
```

## 关键配置

共同配置：

```text
student init: RGB baseline best
teacher weights: IR baseline best
decomp init: oldsplit A2 last.pt
YOLO11n, imgsz=512, batch=64, epochs=200
lr0/lrf: 0.001 / 0.1
warmup: 0
STUDENT_BRANCH_MODE=split
TEACHER_FEATURE_MODE=decomposed
B_LOAD_STUDENT_SPLIT=1
B_LOAD_STUDENT_REACHABILITY=1
LAMBDA_REACH=0.0
LAMBDA_REC=0.0
LAMBDA_TASKL=0.0
ALPHA_S_REC=0.0
ALPHA_KD=0.25
KD_CALIBRATION_MODE=affine
```

区别：

```text
splitkd_unweighted: KD_WEIGHT_MODE=none
reachgap_weighted:  KD_WEIGHT_MODE=reachability_gap
```

## 判定

P10 只有同时满足下面条件才算正向：

1. `reachgap_weighted` 超过 low-LR det-only reload control。
2. `reachgap_weighted` 超过 low-LR raw KD control。
3. `reachgap_weighted` 超过 `splitkd_unweighted`。
4. late-window mean 不明显低于 best，避免只靠偶然 early best。
