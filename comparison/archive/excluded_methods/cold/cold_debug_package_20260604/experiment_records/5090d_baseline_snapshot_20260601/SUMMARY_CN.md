# 5090D YOLOv5x CoLD Baseline Snapshot

来源：5090D 镜像历史快照。连接地址和本地认证信息已从 public 包中脱敏；该快照仅保留结果、配置和必要说明。

初次快照保存了 `results.txt`、`opt.yaml`、`hyp.yaml`，以及已有的 `results.png`；未保存权重。

2026-06-02 重新连接 5090D 后，已补充保存主 SAR baseline 权重：

```text
cold_anchor/runs/ogsod_cold_anchor/cold_anchor_sar_yolov5x_v5p0_coco_5090d_full_coco_mixup010_r2/weights/best.pt
size = 699261781 bytes
```

注意：该权重是 SAR baseline，不是 RGB/optical teacher。5090D 当前只找到 `cold_anchor_sar_yolov5x...` 系列权重和原始 `yolov5x.pt`，未找到可直接用于 offline CoLD 的 OGSOD RGB/optical YOLOv5x teacher `best.pt`。

## 结果汇总

| run | epochs | best epoch | best AP50 | best AP | final AP50 | final AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cold_anchor_sar_yolov5x_v5p0_coco_5090d_full_coco_mixup000` | 254 | 253 | 0.7151 | 0.4136 | 0.7151 | 0.4136 |
| `cold_anchor_sar_yolov5x_v5p0_coco_5090d_full_coco_mixup010_r2` | 400 | 399 | 0.7412 | 0.4450 | 0.7412 | 0.4450 |
| `cold_anchor_sar_yolov5x_v5p0_scratch_5090d_full_scratch_mixup000` | 254 | 253 | 0.6506 | 0.3653 | 0.6506 | 0.3653 |
| `cold_anchor_sar_yolov5x_v5p0_scratch_5090d_full_scratch_mixup010_r1` | 400 | 399 | 0.7133 | 0.4170 | 0.7133 | 0.4170 |

## 主 baseline 对齐点

主 baseline：`cold_anchor_sar_yolov5x_v5p0_coco_5090d_full_coco_mixup010_r2`

| epoch | P | R | AP50 | AP |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.4890 | 0.1768 | 0.0870 | 0.0210 |
| 10 | 0.6717 | 0.2122 | 0.1968 | 0.0639 |
| 20 | 0.4264 | 0.3092 | 0.2964 | 0.1197 |
| 24 | 0.4890 | 0.3113 | 0.3033 | 0.1201 |
| 30 | 0.5632 | 0.3734 | 0.3714 | 0.1580 |
| 31 | 0.5585 | 0.3658 | 0.3534 | 0.1470 |
| 32 | 0.5593 | 0.3685 | 0.3649 | 0.1552 |
| 33 | 0.5037 | 0.3865 | 0.3555 | 0.1519 |
| 40 | 0.5541 | 0.3744 | 0.3692 | 0.1618 |
| 49 | 0.5754 | 0.4357 | 0.4294 | 0.1998 |
| 50 | 0.5726 | 0.4442 | 0.4355 | 0.2050 |
| 100 | 0.6927 | 0.4559 | 0.4739 | 0.2318 |
| 200 | 0.7535 | 0.5907 | 0.6063 | 0.3237 |
| 399 | 0.8283 | 0.7207 | 0.7412 | 0.4450 |

## 本地路径

- `cold_anchor/runs/ogsod_cold_anchor/cold_anchor_sar_yolov5x_v5p0_coco_5090d_full_coco_mixup000/results.txt`
- `cold_anchor/runs/ogsod_cold_anchor/cold_anchor_sar_yolov5x_v5p0_coco_5090d_full_coco_mixup010_r2/results.txt`
- `cold_anchor/runs/ogsod_cold_anchor/cold_anchor_sar_yolov5x_v5p0_scratch_5090d_full_scratch_mixup000/results.txt`
- `cold_anchor/runs/ogsod_cold_anchor/cold_anchor_sar_yolov5x_v5p0_scratch_5090d_full_scratch_mixup010_r1/results.txt`
