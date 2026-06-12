# 对比方法

同类检测 KD 方法在 OGSOD formal 协议下的受控对比。FGD/LD 训练入口在
`../ladd/code/train_ladd_hbb.py`，实现是统一的 KD profile 系统。HalluciDet
已改为 standalone image/representation hallucination 协议，入口在
[`hallucidet/train_hallucidet.py`](hallucidet/train_hallucidet.py)。实现边界与
本次修复见 [`IMPLEMENTATION_REVIEW_CN.md`](IMPLEMENTATION_REVIEW_CN.md)。
非 CCLKD 对比方法的独立审查见
[`NON_CCLKD_IMPLEMENTATION_AUDIT_CN.md`](NON_CCLKD_IMPLEMENTATION_AUDIT_CN.md)。
第二、三轮外部复核意见的逐条响应及官方证据见
[`REVIEW_FEEDBACK_RESPONSE_CN.md`](REVIEW_FEEDBACK_RESPONSE_CN.md)。
2026-06-11 formal transfer 恢复、平台期与早停状态见
[`FORMAL_TRANSFER_STATUS_20260611_CN.md`](FORMAL_TRANSFER_STATUS_20260611_CN.md)。
旧 smoke 记录已降级为历史说明；双卡 4090 部分已因 `nc=5` yaml 错误作废。

| 方法 | 来源 | 类型 | 代码位置 |
|---|---|---|---|
| FGD-style | CVPR 2022 | fg/bg feature + attention mask KD | `../ladd/code/src/.../loss.py` - `fgd` profile |
| LD | CVPR 2022 / TPAMI 2023 | DFL localization KD + VLR-style candidate KD | 同上 - `ld` profile |
| CCLKD | GIS 2026 | 跨模态类别约束 KD | `../ladd/code/src/.../loss.py` - `cclkd` profile；原文复现入口见 [`../cclkd_reproduction/`](../cclkd_reproduction/) |
| HalluciDet-YOLO adaptation | WACV 2024 adaptation | SAR -> hallucinated 3-channel representation -> frozen RGB YOLO detector | [`hallucidet/train_hallucidet.py`](hallucidet/train_hallucidet.py) |

旧 `hallucidet_style` feature/response/margin KD profile 已从 launcher 和
`--comparison-kd-profile` choices 中移除，避免与 standalone HalluciDet 协议混淆。
历史 `hallucidet_style` 结果只能作为 archived diagnostic，不作为当前正式方法发布。

## 当前结论

FGD/LD 在 2026-06-04 修复了实现语义，并在 2026-06-10 更新为
FGD-YOLO focal+attention-mask adaptation 与 LD-YOLO main+VLR-style adaptation；
此前结果不能代表当前实现，必须重跑。
CCLKD 在 2026-06-05 修正 loss 级 LLD/FLD/RLD，但当前 frozen-teacher trainer
不符合原文 online 方法定义。CCLKD 原文复现入口见 [`../cclkd_reproduction/`](../cclkd_reproduction/)，与受控对比分离。
旧 `hallucidet`/`hallucidet_style` 运行只能作为历史参考，不能写作 HalluciDet
official reproduction。双卡 4090 旧结果因 `nc=5` yaml 错误作废。当前阶段是人工复核，不启动 CCLKD 正式实验。
