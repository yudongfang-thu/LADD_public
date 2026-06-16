# 对比方法记录

最后更新：2026-06-16

用途：给导师说明当前 controlled main table 的方法选择、实现边界和实验有效性。
当前方法定义和入口以
[`../method/METHOD_DEFINITIONS_AND_IMPLEMENTATION_CN.md`](../method/METHOD_DEFINITIONS_AND_IMPLEMENTATION_CN.md)
为准；详细代码复核见
[`../../comparison/IMPLEMENTATION_REVIEW_CN.md`](../../comparison/IMPLEMENTATION_REVIEW_CN.md)。

## 1. 当前方法状态

FGD/LD/CMDistill 统一使用 frozen-teacher KD profile：`from-yolo-pretrain` 或
`transfer`、formal no-mosaic、同容量同 seed RGB teacher、800 epoch 和 SAR-only
inference。HalluciDet-YOLO 使用 standalone hallucination trainer。CCLKD 不适用
frozen-teacher 入口，原文复现与 LADD formal online comparison 必须分开。

| 类别 | 方法 | 来源 | DOI | 当前实现与状态 |
|---|---|---|---|---|
| 通用检测 KD | FGD-style | CVPR 2022, Focal and Global Knowledge Distillation for Detectors | [`10.1109/CVPR52688.2022.00460`](https://doi.org/10.1109/CVPR52688.2022.00460) | 官方 softmax attention 形式 + GT fg/bg + relation 近似；旧结果作废 |
| 通用输出 KD | LD | CVPR 2022, Localization Distillation for Dense Object Detection | [`10.1109/CVPR52688.2022.00919`](https://doi.org/10.1109/CVPR52688.2022.00919) | YOLO DFL regression KL，T=10，错形直接失败；旧 soft-logit 结果作废 |
| 跨模态 KD | CMDistill-style | JSTARS 2025, Cross-Modal Distillation Framework for AAV Image Object Detection | [`10.1109/JSTARS.2024.3479717`](https://doi.org/10.1109/JSTARS.2024.3479717) | 非官方 paper-aligned adaptation：PCCFD + SLRD + IBCLD；需写明 `CMDistill-style` |
| 跨模态 KD | CCLKD online | GIS 2026, Cross-modal contrastive learning-based object detection under incomplete modalities | [`10.1080/10095020.2026.2633014`](https://doi.org/10.1080/10095020.2026.2633014) | online teacher-student：student det + teacher det + CCLKD loss；原文复现和受控对比两条线分开 |
| 跨模态 / privileged modality | HalluciDet-YOLO adaptation | WACV 2024, HalluciDet | [`10.1109/WACV57701.2024.00147`](https://doi.org/10.1109/WACV57701.2024.00147) | standalone SAR -> hallucination net -> frozen RGB YOLO detector；不是旧 `hallucidet_style` |

## 2. 选择逻辑

- FGD-style 检验普通 feature KD 是否足够。
- LD 检验只蒸馏定位输出分布是否足够。
- CMDistill-style 检验更强跨模态 detector KD 组件组合是否足够。
- HalluciDet-YOLO 检验显式 hallucinated representation + frozen RGB detector 监督是否足够。
- CCLKD 需要先 smoke online teacher-student trainer；原文协议复现和 LADD formal 受控对比分开汇报。

## 3. 当前实验有效性

FGD、LD 在 2026-06-04 修复实现语义，因此修复前正在运行或已经完成的结果都
不能进入修正版主表。2026-06-05 发现双卡 4090 `nc=5` yaml 错误，因此双卡 4090
smoke/formal partial runs 也作废。旧 `hallucidet_style` profile 已移除，只能作为
历史诊断。CMDistill 和 HalluciDet-YOLO 的有效 run 必须分别满足当前 profile/standalone
入口规则。
