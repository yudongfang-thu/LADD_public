# OGSOD 正式对比实验

最后更新：2026-06-05

## 1. 受控协议

FGD、LD、HalluciDet-style 当前使用 frozen-teacher 受控协议：

```text
OGSOD-1.0 HBB, YOLO11 同容量, SAR-only inference
student init = yolo11*.pt
teacher = same-capacity same-seed RGB baseline best.pt
800ep, cos_lr, full no-mosaic, default Albumentations
```

训练长度不是指标；只有跑到收敛或明确异常退出的实验才能进入统计。

CCLKD 是单独的 online teacher-student 方法，必须先在
[`../../cclkd_reproduction/`](../../cclkd_reproduction/) 中完成原文协议复现：
YOLO11s/YOLO11n、400 epoch、paper-matched augmentation。复现通过后，再回到
本目录按统一受控协议做主表对比。

## 2. 当前候选方法

| 方法 | 角色 | 当前代码状态 | 当前实验状态 |
|---|---|---|---|
| FGD-style | 通用 feature KD | 2026-06-04 加入 teacher attention，保留 GT fg/bg 与 relation | 双卡 4090 旧 smoke/formal 因 `nc=5` 作废；待重 smoke |
| LD | 通用 localization-output KD | 2026-06-04 改为前景 DFL regression KL | 双卡 4090 旧 smoke/formal 因 `nc=5` 作废；待重 smoke |
| CCLKD | 跨模态 category-constrained KD | loss 级实现已修正；online trainer 已补，待 GPU smoke | 先进入 `cclkd_reproduction/` 做原文复现 |
| HalluciDet-style | privileged-modality KD | 已接入，无显式 hallucination module | 双卡 4090 旧 smoke/formal 因 `nc=5` 作废；待重 smoke |

实现边界见 [`../../comparison/IMPLEMENTATION_REVIEW_CN.md`](../../comparison/IMPLEMENTATION_REVIEW_CN.md)。

## 3. 启动入口

```bash
bash comparison/code/launch_formal_from_yolo_kd_job.sh fgd n 0 0
bash comparison/code/launch_formal_from_yolo_kd_job.sh ld n 0 0
bash comparison/code/launch_formal_from_yolo_kd_job.sh hallucidet n 0 0
```

当前 frozen-teacher launcher 只用于 FGD/LD/HalluciDet-style。CCLKD 使用
`cclkd_reproduction/code/launch_cclkd_paper_repro_job.sh` 做 online teacher-student
原文复现。

## 4. 结果口径

- FGD 旧结果缺少 teacher attention，不代表修正版。
- LD 旧结果是分类 soft-logit KD，不代表 Localization Distillation。
- CCLKD frozen-teacher 结果不代表原文方法。

## 5. 当前执行顺序

1. 当前不启动新实验。
2. 双卡 4090 旧 smoke/formal partial runs 因 `nc=5` yaml 作废。
3. 先完成 public diff 人工复核。
4. 人工复核通过后，FGD/LD/HalluciDet-style 可先做协议校验和短 smoke；CCLKD 必须先 smoke online trainer，并在 `cclkd_reproduction/` 做原文条件复现。
