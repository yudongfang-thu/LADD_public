# 对比方法

同类检测 KD 方法在 OGSOD formal 协议下的受控对比。训练入口均在
`../ladd/code/train_ladd_hbb.py`，实现是统一的 KD profile 系统。实现边界与
本次修复见 [`IMPLEMENTATION_REVIEW_CN.md`](IMPLEMENTATION_REVIEW_CN.md)。
第二、三轮外部复核意见的逐条响应及官方证据见
[`REVIEW_FEEDBACK_RESPONSE_CN.md`](REVIEW_FEEDBACK_RESPONSE_CN.md)。
老师确认后的四方法最终实现 smoke 结果见
[`FINAL_IMPLEMENTATION_SMOKE_CN.md`](FINAL_IMPLEMENTATION_SMOKE_CN.md)。

| 方法 | 来源 | 类型 | 代码位置 |
|---|---|---|---|
| FGD-style | CVPR 2022 | teacher-attention weighted feature KD | `../ladd/code/src/.../loss.py` - `fgd` profile |
| LD | CVPR 2022 / TPAMI 2023 | DFL localization KD | 同上 - `ld` profile |
| CCLKD-style | GIS 2026 | 跨模态类别约束 KD | 同上 - `cclkd` profile |
| HalluciDet-style | WACV 2024 inspiration | 跨模态 privileged KD | 同上 - `hallucidet` profile |

CrossKD/MGD/MMANet/C2KD profile 保留用于历史审计，但 formal launcher 已禁止启动。
CoLD、CrossKD 与无效旧结果已统一移至
[`archive/excluded_methods/`](archive/excluded_methods/)，不再作为活跃实验线。

## 当前结论

FGD/LD 在 2026-06-04 修复了实现语义，修复前结果不能代表当前实现，必须重跑。
CCLKD-style 已接入 public 代码但尚未完成 smoke；HalluciDet-style 仍需跑满。
