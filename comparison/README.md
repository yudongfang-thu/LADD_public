# 对比方法

同类检测 KD 方法在 OGSOD formal 协议下的受控对比。方法名、有效入口和实现边界以
[`../docs/method/METHOD_DEFINITIONS_AND_IMPLEMENTATION_CN.md`](../docs/method/METHOD_DEFINITIONS_AND_IMPLEMENTATION_CN.md)
为准；代码速查见 [`METHOD_CODE_MAP_CN.md`](METHOD_CODE_MAP_CN.md)，历史实现审查见
[`IMPLEMENTATION_REVIEW_CN.md`](IMPLEMENTATION_REVIEW_CN.md)。

FGD、LD、CMDistill 使用 `../ladd/code/train_ladd_hbb.py` 的 frozen-teacher KD
profile；HalluciDet 使用 standalone image/representation hallucination 入口；
CCLKD 使用 online teacher-student 入口。旧 smoke 记录已降级为历史说明；双卡 4090
部分已因 `nc=5` yaml 错误作废。

| 方法 | 来源 | 类型 | 代码位置 |
|---|---|---|---|
| FGD-style | CVPR 2022 | fg/bg feature + attention mask KD | `../ladd/code/src/.../loss.py` - `fgd` profile |
| LD | CVPR 2022 / TPAMI 2023 | DFL localization KD + VLR-style candidate KD | 同上 - `ld` profile |
| CMDistill-style | JSTARS 2025 | PCCFD + SLRD + IBCLD | 同上 - `cmdistill` profile |
| HalluciDet-YOLO adaptation | WACV 2024 adaptation | SAR -> hallucinated 3-channel representation -> frozen RGB YOLO detector | [`hallucidet/train_hallucidet.py`](hallucidet/train_hallucidet.py) |
| CCLKD online comparison | GIS 2026 | online RGB teacher + SAR student CCLKD | [`code/launch_formal_online_cclkd_job.sh`](code/launch_formal_online_cclkd_job.sh)；原文复现见 [`../cclkd_reproduction/`](../cclkd_reproduction/) |

旧 `hallucidet_style` feature/response/margin KD profile 已从 launcher 和
`--comparison-kd-profile` choices 中移除，避免与 standalone HalluciDet 协议混淆。
历史 `hallucidet_style` 结果只能作为 archived diagnostic，不作为当前正式方法发布。

## 当前结论

FGD/LD 在 2026-06-04 修复了实现语义，并在 2026-06-10 更新为
FGD-YOLO focal+attention-mask adaptation 与 LD-YOLO main+VLR-style adaptation；
此前结果不能代表当前实现，必须重跑。CMDistill 是 2026-06-15 新增的非官方
paper-aligned adaptation，必须标注为 `CMDistill-style`。

CCLKD 原文复现入口见 [`../cclkd_reproduction/`](../cclkd_reproduction/)，与受控
对比分离；受控对比必须使用 online launcher，不能把 frozen-teacher loss 组件单独
写成 CCLKD 复现。旧 `hallucidet`/`hallucidet_style` 运行只能作为历史参考，不能
写作 HalluciDet official reproduction。双卡 4090 旧结果因 `nc=5` yaml 错误作废。
