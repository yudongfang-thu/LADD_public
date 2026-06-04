# CoLD 原文 baseline 复现重试方案

日期：2026-05-28  
目的：只复现 CoLD 原文 Table I 中的 SAR-only YOLOv5 baseline 坐标，暂不复现 CoLD 双分支蒸馏。

> 2026-05-30 审计更新：旧文档中 `sar_yolov5x_hbb_paper_s0 = AP50:95 0.61727 / AP50 0.86487` 的解释已判定为无效。`0.61727` 可追溯到 CSV 中的 recall 列，`0.86487` 可追溯到训练损失列或非目标指标列，不应作为 YOLOv5x mAP。详见 `docs/archive/invalid_metric_claims_20260530_CN.md`。本文保留旧问题背景，但所有基于 `0.61727 / 0.86487` 的结论均作废。

## 1. 当前结论

值得再试，但这次要把目标收窄成：

```text
OGSOD-1.0 HBB + SAR-only + 原生 YOLOv5x Detect head
目标坐标：AP50 = 80.9, AP50:95 = 46.3
```

不要先跑 CoLD loss。原因是 CoLD 原文只公开了部分训练协议，且我们必须先确认 SAR-only YOLOv5 baseline 的可复现坐标。旧文档曾引用的 `AP50:95=0.61727 / AP50=0.86487` 已经作废，不能再作为“baseline 过高”的证据。

## 2. CoLD 原文明确给出的设置

从本地 CoLD PDF §V-A 可确认：

| 项 | 原文设置 |
|---|---|
| 数据 | OGSOD-1.0 |
| 任务 | HBB object detection |
| 输入 | SAR-only inference；训练 CoLD 时 optical branch + SAR branch |
| baseline | YOLOv5 |
| backbone/容量 | 文献整理中记为 CSPDarkNet-X / 约 86.23M |
| image size | `256 x 256` |
| batch | `64` |
| epoch | `400` |
| optimizer | SGD |
| lr | `1e-2` |
| momentum | `0.937` |
| weight decay | `5e-4` |
| augmentation | Mosaic + Mixup |
| baseline result | `AP50=80.9`, `AP=46.3` |
| CoLD result | `AP50=87.6`, `AP=56.7` |

原文没有说清楚的关键项：

- 是否使用 COCO-pretrained `yolov5x.pt`；
- `Mixup` 的具体概率；
- “Mosaic + Mixup” 是否保留 YOLOv5 默认 `hsv/translate/scale/fliplr`；
- 使用的 YOLOv5 代码 tag 虽引用为 `Ultralytics/YOLOv5: V5.0--YOLOv5-P6 1280`，但表中模型参数更像普通 `yolov5x` 坐标，需要实际核对。

## 3. 旧复现为什么不能直接解释原文

本地文档里已有两类结果，其中第一行旧 baseline 记录已作废：

| 实验 | AP50:95 | AP50 | 问题 |
|---|---:|---:|---|
| 旧记录：原生 YOLOv5x SAR baseline | ~~0.61727~~ | ~~0.86487~~ | **INVALIDATED**：CSV 列误读，不能作为 mAP |
| 严格 v5.0 SAR baseline, COCO, mixup=0.1 | 0.4450 | 0.7412 | 2026-05-30 5090D 完整 400ep 重跑；AP 接近原文 baseline 0.463，AP50 仍偏低 |
| 严格 v5.0 SAR baseline, scratch, mixup=0.1 | 0.4170 | 0.7133 | 2026-05-30 5090D 完整 400ep 重跑 |
| CoLD YOLOv5x old/broken | 0.45232 | 0.69670 | 低于原文 CoLD，且不是 baseline |
| CoLD YOLOv5x fixed | 约 0.44 | 约 0.70 | 修增强和 KD scaling 后仍低 |
| CoLD YOLO11s paper reimplementation | 0.5743 | - | 与原文 CoLD 同量级，但 backbone 已换成 YOLO11s |

真正要先排查的是严格 v5.0 baseline 为什么 `AP50:95` 接近原文 `0.463`，但 `AP50` 只有 `0.7412`，低于原文 `0.809`。

优先怀疑顺序：

1. **评估/数据口径差异**：严格 v5.0 COCO 重跑的 AP 接近原文，但 AP50 差距较大，优先检查 label/split、val 实现和 NMS/IoU 统计口径。
2. **YOLOv5 版本差异**：原文引用 YOLOv5 v5.0；现在若使用新版 YOLOv5/Ultralytics，训练细节和 hyp 可能已变。
3. **数据 split / label 处理差异**：原文写 train/test 为 `14665/3666`，我们 prepared 数据是 `14664/3667`，且过滤过 14 个退化 OBB 目标。这个差异不一定能解释 15 AP，但必须记录。
4. **hyp 解释差异**：原文只写 Mosaic + Mixup，不足以唯一确定完整 YOLOv5 hyp。
5. **评价脚本差异**：应使用 YOLOv5 自带 val 结果，同时记录 AP50 和 AP50:95，避免混用 Ultralytics 新版指标字段。

## 4. 推荐重试矩阵

第一轮只跑 SAR baseline，不跑 RGB teacher，不跑 CoLD：

| run | YOLOv5 tag | init | hyp | 目标 |
|---|---|---|---|---|
| A | `v5.0` | `--weights ''` scratch | v5.0 default scratch hyp + `mixup=0.1` + `img=256` | 已完成：AP50:95 0.4170，AP50 0.7133 |
| B | `v5.0` | `yolov5x.pt` COCO pretrained | 同 A | 已完成：AP50:95 0.4450，AP50 0.7412 |
| C | `v5.0` | scratch | 只保留 Mosaic/Mixup，关闭 hsv/translate/scale/fliplr | 检查“literal Mosaic+Mixup”是否是原文坐标 |
| D | `v5.0` | scratch | v5.0 default scratch hyp, `mixup=0.0` | 检查 Mixup 本身影响 |
| E | 当前/旧本地 YOLOv5 | scratch | 与 A 相同 | 隔离 YOLOv5 版本差异 |

决策：

- 如果 A/B 的 AP 接近 `0.463` 但 AP50 仍明显低于 `0.809`，优先排查评估和数据口径，而不是继续解释旧 `0.617` 记录。
- 如果 A/B 都高于 `0.55`，则原文 `0.463` 可能还受数据处理、split 或 hyp 影响，需要继续查数据。
- 如果 C 才接近 `0.463`，说明原文的 “Mosaic + Mixup” 可能真的意味着关闭其他增强，但这种解释需要在论文里谨慎说明。

## 5. 远端建议命令

以下命令是给 90 服务器或远端 agent 的启动模板。先确认 prepared YAML 实际路径：

```bash
find /mnt/dataY/ydf/projects -path '*yamls/ogsod_hbb_sar.yaml' -print
```

准备独立 YOLOv5 v5.0 工作区：

```bash
cd /mnt/dataY/ydf/projects
git clone --branch v5.0 https://github.com/ultralytics/yolov5 yolov5_cold_v5p0
cd yolov5_cold_v5p0
python3 -m pip install -r requirements.txt
```

把 OGSOD HBB SAR YAML 复制或软链到 YOLOv5 工作区，确保 `path/train/val/names` 指向 prepared YOLO labels：

```bash
cp /path/to/ogsod_hbb_sar.yaml data/ogsod_hbb_sar.yaml
```

scratch baseline：

```bash
python3 train.py \
  --img-size 256 \
  --batch-size 64 \
  --epochs 400 \
  --data data/ogsod_hbb_sar.yaml \
  --cfg models/yolov5x.yaml \
  --weights '' \
  --hyp data/hyp.cold_paper.yaml \
  --device 0 \
  --name cold_anchor_sar_yolov5x_v5p0_scratch_s0 \
  --project runs/ogsod_cold_anchor
```

COCO-pretrained baseline：

```bash
python3 train.py \
  --img-size 256 \
  --batch-size 64 \
  --epochs 400 \
  --data data/ogsod_hbb_sar.yaml \
  --cfg models/yolov5x.yaml \
  --weights yolov5x.pt \
  --hyp data/hyp.cold_paper.yaml \
  --device 1 \
  --name cold_anchor_sar_yolov5x_v5p0_coco_s0 \
  --project runs/ogsod_cold_anchor
```

建议额外保存：

```bash
python3 val.py \
  --img-size 256 \
  --batch-size 64 \
  --data data/ogsod_hbb_sar.yaml \
  --weights runs/ogsod_cold_anchor/<run>/weights/best.pt \
  --task val
```

## 6. 复现成功/失败后的用法

如果 `scratch + v5.0` 能回到 `AP≈0.463`，就可以把 CoLD 原文表当成可对齐 anchor，再考虑把 LADD 放到同一 YOLOv5x/scratch 协议下做一版展示。

如果所有严格 baseline 都明显高于 `0.463`，就不要硬对齐 CoLD 原文表。论文里应写成：

```text
We additionally attempted to reproduce the original CoLD YOLOv5 protocol.
The original paper under-specifies pretraining and augmentation details.
Our native YOLOv5x baseline reaches XX.X AP, substantially different from
the reported 46.3 AP, so we report CoLD original numbers as external anchors
and use a unified protocol for controlled comparisons.
```

这个结果反而能保护主表：我们不是不想复现，而是证明了 CoLD 公开协议不足以唯一确定 baseline。
