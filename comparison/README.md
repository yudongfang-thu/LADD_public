# 对比方法

同类检测 KD 方法在 OGSOD formal 协议下的受控对比。训练入口均在
`../ladd/code/train_ladd_hbb.py`，实现是统一的 KD profile 系统。实现边界与
本次修复见 [`IMPLEMENTATION_REVIEW_CN.md`](IMPLEMENTATION_REVIEW_CN.md)。
第二、三轮外部复核意见的逐条响应及官方证据见
[`REVIEW_FEEDBACK_RESPONSE_CN.md`](REVIEW_FEEDBACK_RESPONSE_CN.md)。
旧 smoke 记录已降级为历史说明；双卡 4090 部分已因 `nc=5` yaml 错误作废。

| 方法 | 来源 | 类型 | 代码位置 |
|---|---|---|---|
| FGD-style | CVPR 2022 | teacher-attention weighted feature KD | `../ladd/code/src/.../loss.py` - `fgd` profile |
| LD | CVPR 2022 / TPAMI 2023 | DFL localization KD | 同上 - `ld` profile |
| CCLKD | GIS 2026 | 跨模态类别约束 KD | loss 组件保留为 `cclkd` profile；正式入口等待 online trainer |
| HalluciDet-style | WACV 2024 inspiration | 跨模态 privileged KD | 同上 - `hallucidet` profile |

CrossKD/MGD/MMANet/C2KD profile 保留用于历史审计，但 formal launcher 已禁止启动。
CoLD、CrossKD 与无效旧结果不再作为活跃实验线，原始归档数据不随精简 public 分支发布。

## 当前结论

FGD/LD 在 2026-06-04 修复了实现语义，修复前结果不能代表当前实现，必须重跑。
CCLKD 在 2026-06-05 修正 loss 级 LLD/FLD/RLD，但当前 frozen-teacher trainer
不符合原文 online 方法定义。双卡 4090 旧结果因 `nc=5` yaml 错误作废。当前阶段是人工复核，不启动 CCLKD 正式实验。
