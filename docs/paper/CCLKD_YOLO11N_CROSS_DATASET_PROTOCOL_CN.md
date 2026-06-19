# CCLKD YOLO11n Cross-Dataset Protocol

最后更新：2026-06-19 CST

本文档固定 LADD 在 VEDAI / DroneVehicle 上的外部泛化实验口径。该实验只用于回答：

```text
LADD Probe-A 在 CCLKD 论文的 YOLO11n 跨数据集协议下是否仍能取得有效增益？
```

它不替代 OGSOD mosaic100 主线，也不进入 OGSOD 主表。

## 1. 当前决定

外部数据集实验直接对齐 CCLKD 的 YOLO11n extension 表，而不是重新定义一套 LADD 协议。

| 数据集 | CCLKD 对齐表 | 方向 | student / inference | teacher |
|---|---|---|---|---|
| VEDAI | Table 9, YOLO11n | visible -> infrared | IR | RGB / visible |
| DroneVehicle | Table 10, YOLO11n | infrared -> visible | RGB / visible | IR |

对比方法使用 CCLKD 论文 reported results。我们自己的实验只报告同协议下的 student baseline、teacher baseline 和 LADD Probe-A。

## 2. 固定协议

| 项 | 值 |
|---|---|
| model | YOLO11n |
| input size | 512 |
| baseline epochs | 200 |
| LADD chain | A1 -> B, no A2 |
| A1 epochs | 10 |
| B epochs | 200 |
| batch | 16 |
| optimizer | SGD |
| lr | 0.01 |
| momentum | 0.937 |
| weight decay | 0.0005 |
| mosaic | 1.0 |
| baseline close mosaic | 10 |
| A1 close mosaic | 0 |
| B close mosaic | 10 |
| mixup | 0.1 |
| seed | 0 first |

配置文件：`configs/paper/cclkd_yolo11n_cross_dataset.yaml`。

说明：A1 是 LADD 内部 warmup，只有 10 epoch；如果对 A1 也设置 `close_mosaic=10`，Ultralytics 会让 A1 全程关闭 mosaic。因此 A1 使用 `close_mosaic=0`，B 阶段和 baseline 使用 CCLKD-aligned `close_mosaic=10`。

## 3. 报告边界

可以写：

```text
We follow the CCLKD YOLO11n cross-dataset protocol on VEDAI and DroneVehicle.
Other method numbers are reported from CCLKD, while LADD is rerun under the same
dataset direction, backbone family, input size, epoch budget, optimizer, and batch size.
```

不要写：

```text
All comparison methods were rerun by us.
```

除非后续真的重跑了全部对比方法。

## 4. OGSOD 为什么不切换

OGSOD 是当前论文主线受控实验，已经按 LADD no-mosaic 协议完成或接近完成 baseline / LADD 主结果。重新切换 OGSOD 到 CCLKD protocol 会重开整套主表，且会混淆已经稳定的 LADD 主线。

因此：

- OGSOD：保留 LADD no-mosaic 主线。
- VEDAI / DroneVehicle：作为外部泛化表，对齐 CCLKD YOLO11n 协议。

## 5. 最小实验矩阵

| 优先级 | 数据集 | 运行 | 目的 |
|---|---|---|---|
| P0 | VEDAI | IR student baseline | LADD 下界 |
| P0 | VEDAI | RGB teacher baseline | teacher upper bound |
| P0 | VEDAI | LADD Probe-A | 对齐 Table 9 YOLO11n |
| P1 | DroneVehicle | RGB student baseline | LADD 下界 |
| P1 | DroneVehicle | IR teacher baseline | teacher upper bound |
| P1 | DroneVehicle | LADD Probe-A | 对齐 Table 10 YOLO11n |

先跑 VEDAI，因为本仓已有 VEDAI 512 数据准备记录。DroneVehicle 需要先确认官方数据下载、pair、label 和 split。
