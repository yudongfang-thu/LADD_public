# LADD 曲线按容量与协议分组对比

日期：2026-06-17

## 画图规则

本目录用于避免把不同容量、不同训练协议的曲线混在一起比较。当前固定采用：

- 先按模型容量分开：`YOLO11n`、`YOLO11s`。
- 再按训练协议分开：`formal no-mosaic`、`mosaic100 / close@100`。
- 每张图都必须包含同容量、同协议的 SAR baseline。
- LADD-like 曲线如果含有额外 `sep/aux` loss，需要在图例里标注 `(sep/aux)`。
- 协议不一致的曲线不放进主图。

## 图表

| 图 | 说明 |
|---|---|
| `n_nomosaic_ap_with_same_protocol_baseline.png` | YOLO11n no-mosaic：SAR baseline vs no-mosaic LADD-like/CMDistill |
| `n_mosaic100_ap_with_same_protocol_baseline.png` | YOLO11n mosaic100：SAR baseline vs historical/current mosaic LADD-like |
| `s_nomosaic_ap_with_same_protocol_baseline.png` | YOLO11s no-mosaic：SAR baseline vs no-mosaic LADD-like/CMDistill |
| `s_mosaic100_ap_with_same_protocol_baseline.png` | YOLO11s mosaic100：SAR baseline vs currently running s skipA2 LADD-like |

每张图左栏是绝对 AP50-95，右栏是 `method AP50-95 - same-protocol SAR baseline AP50-95`。右栏用于观察同协议曲线差距，但注意 LADD B 阶段通常从 A 阶段 checkpoint 启动，而 baseline 从初始检测模型训练，因此早期同 epoch 差值不能直接解释为最终公平增益。

## 当前摘要

| group | run | rows | best AP | last AP | note |
|---|---|---:|---:|---:|---|
| n no-mosaic | SAR baseline | 800 | 0.55654 | 0.55127 | 同协议 baseline |
| n no-mosaic | LADD-like yolo-init A1->B_A2core | 684 | 0.54678 | 0.54206 | sep/aux 污染 |
| n no-mosaic | CMDistill | 800 | 0.56777 | 0.56459 | 对比方法 |
| n mosaic100 | SAR baseline | 800 | 0.54091 | 0.53836 | 同协议 baseline |
| n mosaic100 | LADD-like A2last s0 | 800 | 0.56563 | 0.56076 | sep/aux 污染 |
| n mosaic100 | LADD-like skipA2 s42 | 800 | 0.56835 | 0.56430 | sep/aux 污染 |
| n mosaic100 | LADD-like A2last s123 | 309 | 0.48414 | 0.48414 | running, sep/aux 污染 |
| s no-mosaic | SAR baseline | 800 | 0.62897 | 0.62233 | 同协议 baseline |
| s no-mosaic | LADD-like yolo-init | 177 | 0.28862 | 0.28862 | sep/aux 污染，崩溃/极差 |
| s no-mosaic | CMDistill | 800 | 0.62716 | 0.61904 | 对比方法 |
| s mosaic100 | SAR baseline | 800 | 0.61972 | 0.61570 | 同协议 baseline |
| s mosaic100 | LADD-like skipA2 s0 | 52 | 0.46954 | 0.46159 | running, sep/aux 污染 |

## 被排除的曲线

- `autodl_yolo11s_mosaic_A1A2B_s123_b.csv`：命令记录显示 `mosaic=1.0, close_mosaic=0`，不是 `mosaic100 / close@100`，因此不放进 `s_mosaic100` 主图。它可以单独作为协议异常/全程 mosaic 的参考，但不能和 mosaic100 baseline 直接比较。

## 生成命令

```bash
python3 docs/experiments/figures/ladd_capacity_protocol_split_20260617/plot_capacity_protocol_split.py
```

输出：

- `capacity_protocol_split_summary_20260617.csv`
- `*_ap_with_same_protocol_baseline.png`
- `*_ap_with_same_protocol_baseline.pdf`
