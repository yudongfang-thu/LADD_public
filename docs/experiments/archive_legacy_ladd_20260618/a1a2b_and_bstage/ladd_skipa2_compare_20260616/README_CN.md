# LADD skipA2 对比记录

日期：2026-06-16

## 目的

这次整理用于回答一个具体决策问题：`skipA2` 方案是否应该被排除，还是应该和之前完整 A2+B 的 LADD 方案放在同一张曲线中继续比较。

这里的 `skipA2` 指当前已看到的实际运行方式：从 A1 best checkpoint 进入 B 阶段，而不是先完成 A2 分解预训练。当前没有在 90 服务器上找到 “LADD B 阶段直接加载 `yolo11*.pt`” 的 run；如果要评估这个更激进的 direct-YOLO 初始化方案，需要单独启动。

## 协议边界

需要严格区分两个 mosaic 协议：

- `mosaic_first100_close700`：Ultralytics 参数为 `mosaic=1.0, close_mosaic=700`，即 800 epoch 中前 100 epoch 开 mosaic，后 700 epoch 关闭 mosaic。
- `close_mosaic=100`：800 epoch 中前 700 epoch 开 mosaic，最后 100 epoch 关闭 mosaic。

当前 YOLO11n 的 skipA2 对比属于 `mosaic_first100_close700`，可以和旧 full B 以及当前 A2last 曲线放在一起看。

当前 YOLO11s 的 skipA2 B run 使用的是 `close_mosaic=100`，不能直接和 `mosaic_first100_close700` baseline 或旧 LADD 结果比较。

## 关键结果

| 组别 | run | latest epoch | latest AP50 | latest AP | best AP50 | best AP |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n | SAR baseline | 801 | 0.81755 | 0.53836 | 0.81941 | 0.54091 |
| YOLO11n | old cap2 full B s0 | 801 | 0.84888 | 0.56792 | 0.84962 | 0.56841 |
| YOLO11n | old cap2 full B s42 | 801 | 0.84345 | 0.56044 | 0.84864 | 0.56799 |
| YOLO11n | old cap2 full B s123 | 801 | 0.83592 | 0.56163 | 0.83626 | 0.56163 |
| YOLO11n | current A2last s0 | 654 | 0.83819 | 0.55858 | 0.83902 | 0.55884 |
| YOLO11n | current A2last s123 | 220 | 0.72482 | 0.45682 | 0.72482 | 0.45682 |
| YOLO11n | current skipA2 from A1 s42 | 660 | 0.83618 | 0.55900 | 0.83700 | 0.55900 |
| YOLO11s | SAR baseline | 801 | 0.90576 | 0.61570 | 0.90837 | 0.61972 |
| YOLO11s | RGB teacher | 760 | 0.94509 | 0.65818 | 0.94702 | 0.66029 |
| YOLO11s | current skipA2 from A1 s42 | 12 | 0.51148 | 0.23997 | 0.81994 | 0.48902 |

完整 CSV：

- `skipa2_compare_summary.csv`

曲线：

- `figures/yolo11n_skipa2_vs_full_b.png`
- `figures/yolo11s_current_skipa2_early.png`

## 当前判断

不建议现在排除 `skipA2`。YOLO11n 的同协议曲线显示，`current skipA2 from A1 s42` 到 epoch 660 的 AP 为 0.55900，和 `current A2last s0` 到 epoch 654 的 AP 0.55858 基本持平。

但也不建议直接把 `skipA2` 定为最终主方案。理由是：

1. 目前同协议下只有 YOLO11n 的一个 skipA2 seed，证据量不足。
2. 旧 full B 三个 seed 的最终 best AP 仍然更高，约 0.56163 到 0.56841。
3. 当前 YOLO11s skipA2 B 使用 `close_mosaic=100`，和我们要比较的 `mosaic_first100_close700` 不一致。
4. 当前没有 direct-YOLO 初始化的 LADD B run，因此不能把 “skipA2 from A1” 和 “direct YOLO weights” 混为同一个方案。

## 建议

短期决策应改成：

- 保留 `skipA2 from A1` 作为主候选之一。
- 重新开 YOLO11s 的 `skipA2 from A1`，使用 `mosaic=1.0, close_mosaic=700`，否则无法和 baseline / 旧 LADD 曲线公平比较。
- 如果要验证用户提出的更简洁方案，需要新增 direct-YOLO 初始化的 LADD B run，并在 run name 中显式标注 `direct_yolo_B`。
- 判断标准建议为：同协议、同模型容量、至少 s0/s42 两个 seed 下，`skipA2` 若与 full A2+B 的 AP 差距小于 0.005，且曲线更稳定，可以优先采用 `skipA2`，因为叙事更简洁、训练链更短。

