# Oldsplit A2-Only Controlled

日期：2026-06-24

## 背景

当前 `oldsplit_90_hbb` 的 A2 阶段出现了一个有价值但不能直接宣称的信号：

```text
oldsplit_a2 best AP50/AP50-95 = 0.56322 / 0.36326
RGB baseline best AP50/AP50-95 = 0.56886 / 0.36087
```

但是：

- A2 后续 C 阶段回落，没有形成稳定最终方案。
- 该 run 使用的是 high-LR-ish 旧协议、batch 32。
- 没有同结构、同 schedule 的 A2 det-only split control。

因此不能把 A2 当作主线成功，只能把它当作“旧 split/reach 方案可能有短程正信号”的线索。

## 目的

构建一个最小 controlled run，回答：

```text
如果不进入 C，只把 A2 作为最终候选，且采用 low-LR/no-warmup continued-training 协议，
oldsplit reach/KD 是否仍能超过同结构 A2 det-only split control 和全局 low-LR reload control？
```

## 队列脚本

本地：

```text
docs/experiments/dronevehicle_method_search_20260623/oldsplit_a2only_controlled/queue_oldsplit_a2only_after_controls_20260624.sh
```

远端 ready 路径：

```text
logs/dronevehicle_method_search/sub2k_seed0_fullval/oldsplit_a2only_controlled/queue_oldsplit_a2only_after_controls_20260624.sh
```

默认等待：

1. low-LR det-only control 至少 20 rows。
2. low-LR raw KD control 至少 20 rows。
3. low-LR CMDistill sanity 至少 20 rows。

然后启动：

```text
a1_shared_init
a2_detonly_split_control
a2_reach_kd_lowlr
```

## 判定

`a2_reach_kd_lowlr` 必须同时超过：

1. RGB baseline best `0.36087`。
2. 全局 low-LR det-only reload control。
3. `a2_detonly_split_control`。

并且 late-window mean 不能明显低于 best。否则 A2 正信号仍然视作 reload/schedule 或结构噪声。
