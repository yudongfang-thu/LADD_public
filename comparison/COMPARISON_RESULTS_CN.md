# 对比实验结果

最后更新：2026-06-03 23:50 CST

协议：`from-yolo-pretrain`, formal no-mosaic, 800 epoch, same-capacity same-seed RGB teacher, SAR-only inference.

## 已完成

| 方法 | Model/seed | 服务器 | epoch | best AP | vs same-seed SAR baseline | 判断 |
|---|---|---|---:|---:|---:|---|
| FGD | YOLO11n seed0 | 4090D | 800 | 0.55867@749 | -0.00049 | 基本打平，无正向 |
| CrossKD-style | YOLO11n seed0 | 4090D | 800 | 0.55764@737 | -0.00152 | 基本打平，无正向 |

## 正在运行

| 方法 | Model/seed | 服务器 | epoch | current/best AP |
|---|---|---|---:|---:|
| FGD | YOLO11n seed42 | 4090D | 35+ | 早期运行，CSV 已复制 |
| LD | YOLO11n seed0 | 90 | 236 | 0.42748@236 |
| LD | YOLO11s seed0 | 90 | 266 | 0.54401@266 |
| HalluciDet-style | YOLO11n seed0 | 90 | 61 | 0.33079@61 |
| HalluciDet-style | YOLO11s seed0 | 90 | 62 | 0.40823@62 |

## 结论

FGD/CrossKD-style 的完整 seed0 结果说明普通检测 KD 在当前 formal no-mosaic 协议下没有明显吃到 RGB teacher 的收益。LD 和 HalluciDet-style 还在训练早期/中期，不能提前下结论。

完整数据见各方法 `results/` 子目录。
