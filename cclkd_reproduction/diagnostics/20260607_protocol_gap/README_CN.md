# CCLKD 复现协议差距审计

最后更新：2026-06-07

本目录保存 CCLKD 原文复现相关的审计材料。目标不是给出正式结论，而是把原文协议、其他论文参照、我们当前 90/双卡服务器上的实验结果和数据统计放在一起，便于再次排查“为什么 paper-protocol baseline 与 CCLKD 原文数字对不齐”。

## 1. 已拉回的材料

所有材料位于：

```text
cclkd_reproduction/diagnostics/20260607_protocol_gap/
```

核心文件：

| 文件/目录 | 内容 |
|---|---|
| `README_CN.md` | 本审计说明 |
| `RESULT_SUMMARY.csv` | 自动汇总所有拉回 `results.csv` 的最后指标、best epoch、关键训练参数 |
| `server_artifacts/server90/` | 90 服务器拉回的 CCLKD 消融结果、数据 YAML、诊断文档 |
| `server_artifacts/dual4090/` | 双卡 4090 服务器拉回的 baseline / CCLKD / ablation / smoke 结果、日志、数据 YAML |
| `server_artifacts/*/DATASET_STATS.txt` | 服务器上实际数据规模与标签类别分布 |

已排除内容：

```text
weights/
*.pt
*.pth
*.onnx
*.engine
*.torchscript
*.tflite
```

同时删除了 Ultralytics 自动保存的 batch 可视化图片和曲线图片，只保留 CSV/YAML/log/文本统计，避免仓库体积失控。
超过 1MB 的训练日志以 `.log.gz` 形式保存，内容未裁剪。

## 2. CCLKD 原文协议与结果

论文：

```text
Cross-modal contrastive learning-based object detection under incomplete modalities
Geo-spatial Information Science, 2026
DOI: 10.1080/10095020.2026.2633014
```

本地 PDF：

```text
cclkd_reproduction/paper/CCLKD__2026_GIS__Cross_Modal_Contrastive_Learning_Incomplete_Modalities.pdf
```

### 2.1 OGSOD-1.0 数据集

CCLKD 原文描述：

| 项 | 原文信息 |
|---|---|
| 数据集 | OGSOD-1.0 |
| 类别 | bridge / port(or harbor) / oil tank |
| train | 14,665 |
| test | 3,666 |
| patch size | 256 x 256 |
| 任务 | Optical to SAR，训练期 optical teacher + SAR student，推理期 SAR-only |

### 2.2 训练设置

CCLKD 原文 Table 2：

| Dataset | Input Size | Optimizer | Epoch | Batch | LR | Momentum |
|---|---:|---|---:|---:|---:|---:|
| OGSOD-1.0 | 256 x 256 | SGD | 400 | 32 | 0.01 | 0.937 |

原文还写到使用 standard image augmentation 和 MixUp。未公开的细项需要在复现中明确标注为近似设置，不能与 LADD 的 no-mosaic 800ep 协议混用。

### 2.3 原文主要数值

CCLKD Table 4/5/8/12/13 中与 OGSOD 相关的关键结果：

| 来源 | 方法 | AP50 (%) | AP (%) |
|---|---|---:|---:|
| Table 4 | Baseline | 80.9 | 46.3 |
| Table 4 | ATKD only | 87.1 | 55.4 |
| Table 4 | LCC only | 85.8 | 54.5 |
| Table 4 | Both ATKD and LCC | 88.7 | 57.3 |
| Table 5 | YOLOv5 | 80.9 | 46.3 |
| Table 5 | CoLD | 86.5 | 55.4 |
| Table 5 | CMDistill | 87.5 | 56.2 |
| Table 5 | CCLKD | 88.7 | 57.3 |
| Table 8 | CMDistill + YOLO11n | 86.3 | 53.3 |
| Table 8 | CCLKD + YOLO11n | 86.8 | 53.7 |
| Table 8 | CMDistill + YOLO11s | 87.1 | 53.8 |
| Table 8 | CCLKD + YOLO11s | 87.5 | 55.1 |
| Table 12 | ATKD LLD only | 83.4 | 48.5 |
| Table 12 | ATKD LLD+FLD | 84.2 | 49.3 |
| Table 12 | ATKD LLD+FLD+RLD | 84.9 | 50.1 |
| Table 12 | ATKD LLD+FLD+RLD+PATM | 87.0 | 55.1 |
| Table 12 | CCL only | 85.9 | 54.4 |
| Table 12 | full | 88.7 | 57.3 |
| Table 13 | Teacher YOLOv5X upper bound | 90.9 | - |
| Table 13 | Student baseline | 78.9 | - |
| Table 13 | Naive distillation baseline | 80.2 | - |
| Table 13 | CCLKD | 88.7 | - |

注意：原文的 YOLO11 表只报告 `CMDistill + YOLO11*` 和 `CCLKD + YOLO11*`，没有给出 YOLO11n/s 的 SAR-only baseline。

## 3. YOLO-CMFM 论文的外部参照

另一个本地 PDF：

```text
/Users/yudongfang/Desktop/光sar/YOLO-CMFM__2026_RemoteSensing__Visible_SAR_Multimodal_Object_Detection.pdf
```

该文同样使用 OGSOD-1.0，并明确写出：

| 项 | YOLO-CMFM |
|---|---|
| train/test | 14,665 / 3,666 |
| 类别 | Bridge / Harbor / Oil Tank |
| 原始 patch | 256 x 256 |
| 训练输入 | 640 x 640 |
| epoch | 400 |
| batch | 16 |
| optimizer | SGD lr=0.01, momentum=0.937, weight decay=5e-4 |
| augmentation | default Ultralytics YOLOv11, Mosaic + MixUp |

YOLO-CMFM Table 1/3 给出的 SAR-only YOLO11 baseline：

| 方法 | 模态 | AP50 (%) | AP (%) |
|---|---|---:|---:|
| YOLOv11n | SAR | 85.5 | 55.0 |
| YOLOv11s | SAR | 92.5 | 63.3 |
| YOLOv11m | SAR | 95.6 | 68.7 |

这个参照不能严格替代 CCLKD 原文 baseline，因为它的输入尺寸是 640 而不是 256，batch 也是 16 而不是 32。但它说明：在同一个 OGSOD-1.0 数据规模上，YOLO11 SAR-only 的合理训练结果应该显著高于我们当前 256 协议 baseline。

## 4. 我们服务器上的数据统计

90 与双卡服务器的数据统计一致，均为 3 类、同样数量：

| 服务器 | train images | test images | train labels | test labels |
|---|---:|---:|---:|---:|
| 90 | 14664 | 3667 | 14664 | 3667 |
| dual4090 | 14664 | 3667 | 14664 | 3667 |

标签实例分布：

| split | class 0 bridge | class 1 harbor | class 2 storage_tank |
|---|---:|---:|---:|
| train | 25533 | 3306 | 10136 |
| test | 6389 | 803 | 2422 |

备注：图片数量是 14664/3667，而不是论文写的 14665/3666。这可能来自文件组织或计数口径差异，但规模已经与 CCLKD/CMFM 论文描述基本一致，不再是早期误判的 2870/1162 小数据问题。

## 5. 我们当前 paper-protocol SAR-only baseline

来自双卡服务器：

```text
server_artifacts/dual4090/runs_public/ogsod/hbb/cclkd_reproduction_baselines/sar/
```

共同设置：

```text
imgsz=256
epochs=400
batch=32
optimizer=SGD
lr0=0.01
momentum=0.937
mosaic=1.0
mixup=0.1
data=configs/datasets/ogsod_hbb_sar.yaml
```

| 模型 | seed | epoch | AP50 | AP |
|---|---:|---:|---:|---:|
| YOLO11n | 0 | 400 | 59.74 | 35.67 |
| YOLO11n | 42 | 400 | 60.93 | 36.95 |
| YOLO11n | 123 | 400 | 60.42 | 36.48 |
| YOLO11s | 42 | 400 | 72.98 | 46.11 |
| YOLO11s | 123 | 400 | 72.27 | 45.43 |
| YOLO11s | 0 | 110 | 0.00 final / 23.44 best | 0.00 final / 9.68 best |

均值：

| 模型 | AP50 mean | AP mean | 说明 |
|---|---:|---:|---|
| YOLO11n | 60.36 | 36.37 | 3 seeds |
| YOLO11s | 72.63 | 45.77 | 排除异常 seed0 |

这些 baseline 与 CCLKD 原文 YOLOv5 baseline `80.9/46.3`、YOLO-CMFM YOLO11 SAR-only baseline `85.5/55.0` 或 `92.5/63.3` 都存在明显差距。

## 6. 我们当前 CCLKD 复现与消融结果

完整机器可读汇总见：

```text
RESULT_SUMMARY.csv
```

### 6.1 双卡 full online CCLKD

目录：

```text
server_artifacts/dual4090/runs_public/ogsod/hbb/cclkd_reproduction/
```

| 模型 | seed | epoch | AP50 | AP | 状态 |
|---|---:|---:|---:|---:|---|
| YOLO11n | 0 | 400 | 62.45 | 38.53 | 完成 |
| YOLO11n | 42 | 120 | 51.08 | 28.25 | 未完成 |
| YOLO11n | 123 | 67 | 47.71 | 25.55 | 未完成 |
| YOLO11s | 0 | 400 | 75.43 | 48.46 | 完成 |
| YOLO11s | 42 | 95 | 60.31 | 35.23 | 未完成 |
| YOLO11s | 123 | 60 | 56.24 | 31.67 | 未完成 |

与同/近似 baseline 的直接观察：

- YOLO11n seed0：从 SAR-only `59.74/35.67` 到 CCLKD `62.45/38.53`，约 `+2.85 AP50 / +2.86 AP`。
- YOLO11s seed0 的 SAR-only baseline 异常，因此不能用 seed0 做直接增益判断。若参考 s42/s123 baseline 均值，CCLKD s0 的 `75.43/48.46` 仍然只比当前 256 baseline 高约 `+2.8 AP50 / +2.7 AP`。

### 6.2 双卡 fixed20260606 目录

目录：

```text
server_artifacts/dual4090/runs_public/ogsod/hbb/cclkd_reproduction_fixed20260606/
```

| 模型 | seed | epoch | AP50 | AP | 状态 |
|---|---:|---:|---:|---:|---|
| YOLO11n | 42 | 400 | 62.48 | 38.47 | 完成 |
| YOLO11s | 0 | 238 | 58.87 final / 61.51 best | 36.52 final / 36.62 best | 未完成 |

### 6.3 YOLO11n 消融

90 服务器 seed0：

| 组合 | epoch | AP50 | AP |
|---|---:|---:|---:|
| full | 400 | 61.84 | 38.15 |
| ATKD only | 400 | 62.30 | 38.60 |
| LLD only | 400 | 62.37 | 38.72 |
| LLD+FLD | 400 | 62.74 | 38.56 |
| CCL only | 368 | 61.51 | 37.83 |
| full ccl=0.5 | 375 | 60.79 | 37.06 |
| LLD+FLD+RLD | 371 | 61.50 | 37.41 |

双卡 seed42：

| 组合 | epoch | AP50 | AP |
|---|---:|---:|---:|
| ATKD only | 400 | 62.24 | 38.38 |
| CCL only | 400 | 62.52 | 39.04 |
| LLD+FLD+RLD | 400 | 63.50 | 38.95 |

当前消融有正向信号，但绝对值仍远低于 CCLKD 原文 Table 12。尤其是原文 `CCL only = 85.9/54.4`，而我们当前 CCL-only 约 `62.5/39.0`。

## 7. 当前判断

1. 数据规模和类别数现在基本对齐：两台服务器都是 `14664/3667`、`nc=3`。
2. 当前 256 paper-protocol baseline 远低于 CCLKD 原文和 YOLO-CMFM 的外部参照。
3. CCLKD 相对我们当前 baseline 有约 2-3 个点收益，但这不能说明复现成功，因为绝对指标仍严重偏低。
4. 当前最可疑的不是单个 CCLKD loss，而是训练/评估协议仍有未对齐处，包括但不限于：
   - 输入尺寸 256 是否足以复现 YOLO11 体系，尤其与 YOLO-CMFM 的 640 差异；
   - Ultralytics YOLO11 默认增强项是否与 CCLKD 原文 “standard augmentation + MixUp” 一致；
   - YOLOv5 原文 baseline 与 YOLO11 迁移实验是否使用了不同 hyp；
   - 评估 split 是否与论文 test set 完全一致；
   - 类名 `port/harbor`、`oil tank/storage_tank` 只是命名差异还是数据映射差异；
   - 训练是否仍在 400 epoch 上升，是否需要查看 learning curve 而非只看 final。

## 8. 建议的下一步排查

优先级从高到低：

1. 跑一个 YOLO-CMFM-style sanity baseline：`YOLO11n/s SAR-only, imgsz=640, batch=16, 400ep, default YOLO11 augmentation, SGD lr=0.01`。目标是确认能否接近 YOLO-CMFM 的 `YOLO11n SAR 85.5/55.0` 和 `YOLO11s SAR 92.5/63.3`。
2. 若 640 baseline 仍明显偏低，优先检查数据读取、标签坐标、评估 split 和类别映射。
3. 若 640 baseline 接近外部参照，则说明 256 协议下 YOLO11 baseline 弱是关键原因，需要重新解释 CCLKD 原文 YOLO11 表的训练输入/增强细节。
4. 在 baseline 对齐前，不应把当前 CCLKD 结果作为正式复现或正式对比结论。
