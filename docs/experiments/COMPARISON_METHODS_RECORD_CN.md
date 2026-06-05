# 对比方法记录

最后更新：2026-06-05

用途：给导师说明当前 controlled main table 的方法选择、实现边界和实验有效性。
详细代码复核见 [`../../comparison/IMPLEMENTATION_REVIEW_CN.md`](../../comparison/IMPLEMENTATION_REVIEW_CN.md)，
第二轮意见响应见
[`../../comparison/REVIEW_FEEDBACK_RESPONSE_CN.md`](../../comparison/REVIEW_FEEDBACK_RESPONSE_CN.md)。

## 1. 当前四方法

所有方法统一使用 `from-yolo-pretrain`、formal no-mosaic、同容量同 seed RGB
teacher、800 epoch 和 SAR-only inference。

| 类别 | 方法 | 来源 | DOI | 当前实现与状态 |
|---|---|---|---|---|
| 通用检测 KD | FGD-style | CVPR 2022, Focal and Global Knowledge Distillation for Detectors | [`10.1109/CVPR52688.2022.00460`](https://doi.org/10.1109/CVPR52688.2022.00460) | 官方 softmax attention 形式 + GT fg/bg + relation 近似；旧结果作废 |
| 通用输出 KD | LD | CVPR 2022, Localization Distillation for Dense Object Detection | [`10.1109/CVPR52688.2022.00919`](https://doi.org/10.1109/CVPR52688.2022.00919) | YOLO DFL regression KL，T=10，错形直接失败；旧 soft-logit 结果作废 |
| 跨模态 KD | CCLKD paper-structured reimplementation | GIS 2026, Cross-modal contrastive learning-based object detection under incomplete modalities | [`10.1080/10095020.2026.2633014`](https://doi.org/10.1080/10095020.2026.2633014) | COP + entropy temperature + LLD/FLD/RLD + CCL；待人工复核和重新 smoke，不是官方严格复现 |
| 跨模态 / privileged modality | HalluciDet-style | WACV 2024, HalluciDet | [`10.1109/WACV57701.2024.00147`](https://doi.org/10.1109/WACV57701.2024.00147) | 已接入；无显式 hallucination module，需写明 `-style` |

## 2. 选择逻辑

- FGD-style 检验普通 feature KD 是否足够。
- LD 检验只蒸馏定位输出分布是否足够。
- CCLKD paper-structured reimplementation 检验 category-constrained cross-modal contrastive KD。
- HalluciDet-style 检验训练期 RGB privileged information 对 SAR-only detector 的帮助。

## 3. 淘汰与外部报告

| 方法 | 处置 | 原因 |
|---|---|---|
| CrossKD | 代码保留，formal launcher 禁止启动 | 当前 YOLO port 没有真正 cross-head routing |
| MGD | 代码保留，禁止启动 | 无可训练 generator，不是完整 MGD |
| MMANet/C2KD | 代码保留，禁止启动 | 与原方法机制/任务存在较大差距 |
| CoLD | 降级并统一归档 | YOLOv5x 协议、容量和复现状态不适合 controlled main table |

## 4. 当前实验有效性

FGD、LD 在 2026-06-04 修复实现语义，因此修复前正在运行或已经完成的结果都
不能进入修正版主表。2026-06-05 发现双卡 4090 `nc=5` yaml 错误，因此双卡 4090
smoke/formal partial runs 也作废。当前不启动新实验；先完成人工复核，再重新做短 smoke。
