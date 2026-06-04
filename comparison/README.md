# 对比方法

同类检测 KD 方法在 OGSOD formal 协议下的受控对比。训练入口均在 `../ladd/code/train_ladd_hbb.py`，实现是统一的 KD profile 系统，不是四套彼此独立的代码库。代码映射见 [`METHOD_CODE_MAP_CN.md`](METHOD_CODE_MAP_CN.md)。

| 方法 | 来源 | 类型 | 代码位置 |
|---|---|---|---|
| FGD | CVPR 2022 | 通用检测 feature KD | `../ladd/code/src/.../loss.py` - `fgd` profile |
| CrossKD-style | CVPR 2024 port | prediction-level KD | 同上 - `crosskd` profile |
| LD | 经典 logit KD | baseline KD | 同上 - `ld` profile |
| HalluciDet-style | WACV 2021 inspiration | 跨模态 privileged KD | 同上 - `hallucidet` profile |

CoLD 也属于对比方法，但因复现难度大单独成目录。

## 当前结论

FGD/CrossKD-style YOLO11n seed0 800ep 均未超越 baseline。4090/4090D 正在补 FGD 与 CrossKD-style 的 n/s 其他 seed；LD/HalluciDet-style 仍主要依赖 90 服务器记录，未进入最终多 seed 主表。
