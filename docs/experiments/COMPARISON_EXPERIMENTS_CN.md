# OGSOD 正式对比实验

最后更新：2026-06-04

## 1. 受控协议

```text
OGSOD-1.0 HBB, YOLO11 同容量, SAR-only inference
student init = yolo11*.pt
teacher = same-capacity same-seed RGB baseline best.pt
800ep, cos_lr, full no-mosaic, default Albumentations
```

训练长度不是指标；只有跑到收敛或明确异常退出的实验才能进入统计。

## 2. 正式四方法

| 方法 | 角色 | 当前代码状态 | 当前实验状态 |
|---|---|---|---|
| FGD-style | 通用 feature KD | 2026-06-04 加入 teacher attention，保留 GT fg/bg 与 relation | 修复前结果作废，待 smoke/重跑 |
| LD | 通用 localization-output KD | 2026-06-04 改为前景 DFL regression KL | 修复前 soft-logit 结果作废，待 smoke/重跑 |
| CCLKD-style | 跨模态 category-constrained KD | portable profile 已接入；缺完整 relationship-level 项 | 待 smoke/正式启动 |
| HalluciDet-style | privileged-modality KD | 已接入，无显式 hallucination module | 候选运行需跑满 |

实现边界见 [`../../comparison/IMPLEMENTATION_REVIEW_CN.md`](../../comparison/IMPLEMENTATION_REVIEW_CN.md)。

## 3. 启动入口

```bash
bash comparison/code/launch_formal_from_yolo_kd_job.sh fgd n 0 0
bash comparison/code/launch_formal_from_yolo_kd_job.sh ld n 0 0
bash comparison/code/launch_formal_from_yolo_kd_job.sh cclkd n 0 0
bash comparison/code/launch_formal_from_yolo_kd_job.sh hallucidet n 0 0
```

Formal launcher 会拒绝 `crosskd/mgd/c2kd/mmanet`，防止历史 profile 被误启动。

## 4. 结果口径

- CrossKD 历史结果只用于说明为什么淘汰，不进入主表。
- FGD 旧结果缺少 teacher attention，不代表修正版。
- LD 旧结果是分类 soft-logit KD，不代表 Localization Distillation。
- CoLD 已降级并统一归档，不再作为当前实验线。

## 5. 当前执行顺序

1. 分别对 FGD/LD/CCLKD-style 做 1 个短 smoke，验证 loss、shape、显存和数值稳定性。
2. 先跑 YOLO11n seed0；确认有效后补 n seed42/123。
3. 同时保证 YOLO11s seed0 跑通，再扩展三 seed。
4. m/l 仅在对应 baseline 和 n/s 结论稳定后扩展。
