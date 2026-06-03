# 对比方法

同类检测 KD 方法在 OGSOD formal 协议下的受控对比。训练入口均在 `../ladd/code/train_ladd_hbb.py`。

| 方法 | 来源 | 类型 | 代码位置 |
|---|---|---|---|
| FGD | CVPR 2022 | 通用检测 feature KD | `../ladd/code/src/.../loss.py` — `fgd` profile |
| CrossKD-style | CVPR 2024 port | prediction-level KD | 同上 — `mgd`/style profile |
| LD | 经典 logit KD | baseline KD | 同上 |
| HalluciDet-style | — | 跨模态 privileged KD | 同上 |

CoLD 也属于对比方法，但因复现难度大单独成目录。

## 当前结论

FGD/CrossKD-style YOLO11n seed0 800ep 均未超越 baseline。LD/HalluciDet-style 的 n/s seed0 正在 90 服务器运行，结果 CSV 已复制到各自 `results/90_formal_nomosaic_20260528/` 目录。
