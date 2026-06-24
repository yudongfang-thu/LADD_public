# Late KD Decay

日期：2026-06-24

## 目的

如果 B-only KD 的早期 best 接近 baseline，但后期持续下滑，那么一个自然的修正是让 KD 只在早中期提供跨模态初始化信号，后期逐渐回到 detector 自身的 supervised training。

本方案不主张单独作为新方法，只作为稳定化组件：

```text
L = L_det + alpha(t) * L_KD
alpha(t): 1.0 -> 0.0, epoch 60 到 160 线性衰减
```

## 首发配置

队列脚本与 P7 共用：

```text
docs/experiments/dronevehicle_method_search_20260623/teacher_conf_gated_kd/queue_teacher_conf_and_late_decay_after_primary_20260624.sh
```

首发 run：

```text
runs_public/dronevehicle_method_search/sub2k_seed0_fullval/teacher_conf_gated_kd/rawkd_late_decay/
```

关键开关：

```text
ALPHA_KD=0.25
LADD_KD_DECAY_MODE=linear
LADD_KD_DECAY_START_EPOCH=60
LADD_KD_DECAY_END_EPOCH=160
LADD_KD_FINAL_MULT=0.0
LR0=0.001
WARMUP_EPOCHS=0.0
```

判定时必须比较：

- low-LR det-only reload control。
- low-LR raw KD no-decay control。
- late-window mean，而不只看 best。
