# 受控对比方法代码映射

最后更新：2026-06-18

本文档说明 public 包中当前对比方法的代码位置。FGD/LD 在同一个 HBB
trainer 中用 `--comparison-kd-profile` 切换；HalluciDet 使用 standalone
image/representation hallucination 入口。严格方法定义见
[`../docs/method/METHOD_DEFINITIONS_AND_IMPLEMENTATION_CN.md`](../docs/method/METHOD_DEFINITIONS_AND_IMPLEMENTATION_CN.md)，
历史实现边界见 [`IMPLEMENTATION_REVIEW_CN.md`](IMPLEMENTATION_REVIEW_CN.md)。

## 1. 入口

| 文件 | 作用 |
|---|---|
| `../ladd/code/train_ladd_hbb.py` | 统一训练入口，暴露 `--comparison-kd-profile`、`--freeze-bn-stats`；FGD/LD 实现已锁定，不再暴露内部变体参数 |
| `../ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py` | LADD 主 loss 和所有 comparison KD profile |
| `../ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py` | 阶段控制、teacher/student 前向、BN-freeze 逻辑 |
| `code/launch_formal_from_yolo_kd_job.sh` | FGD/LD/CMDistill 的 from-YOLO frozen-teacher 正式启动脚本 |
| `code/launch_formal_transfer_kd_job.sh` | FGD/LD/CMDistill 的 transfer frozen-teacher 启动脚本 |
| `hallucidet/train_hallucidet.py` | standalone HalluciDet-YOLO adaptation：SAR -> hallucination net -> frozen RGB YOLO detector detection loss |
| `code/launch_formal_online_cclkd_job.sh` | CCLKD 的 LADD 统一协议 online teacher-student 受控对比启动脚本 |
| `../cclkd_reproduction/` | CCLKD 原文协议复现目录；已包含 online trainer / launcher / protocol checker，GPU smoke 前不启动正式 CCLKD |

## 2. Profile 对应关系

| 方法 | 启动 profile | 核心函数/逻辑 | 当前说明 |
|---|---|---|---|
| FGD-style | `--comparison-kd-profile fgd` | `TSKDDetectionLossHBB._fgd_style_loss()` | fg/bg feature + attention mask；GT-box mask 默认启用；official trainable global relation 默认不启用 |
| LD | `--comparison-kd-profile ld` | `_ld_style_loss()` | foreground/main YOLO DFL KL + teacher-quality VLR-style candidate LD；错形直接失败 |
| CMDistill reimplementation | `--comparison-kd-profile cmdistill` | `_cmdistill_style_loss()` | 非官方 paper-aligned adaptation；PCCFD shallow/deep feature + deepest SLRD affinity + IBCLD decoded-box IoU / binary classification logic |
| CCLKD online comparison | `launch_formal_online_cclkd_job.sh` / `train_cclkd_online_hbb.py` | online teacher detection loss + SAR student detection loss + CCLKD loss | COP + localization-only LLD + FLD-MSE + RLD feature-correlation + CCL；原文复现见 `../cclkd_reproduction/` |
| HalluciDet-YOLO adaptation | `python comparison/hallucidet/train_hallucidet.py` | `HalluciDetTrainer` / `OfficialStyleHallucinationNetwork` | detection-loss-only standalone protocol；固定 official-style U-Net；不是旧 feature/response/margin KD profile |

旧 `hallucidet_style` KD profile 已移除，不再作为可启动方法发布。`--comparison-kd-profile cclkd`
仍保留为 loss 级兼容部件，但不是当前 CCLKD 正式入口。

## 3. 代码新鲜度

2026-06-18 已锁定 FGD/LD/HalluciDet 最终实现：FGD 为 GT-box mask + spatial/channel
attention，无 legacy relation；LD 为 YOLO DFL main + VLR-style candidate KL；
HalluciDet 为 official-style U-Net(resnet34 ImageNet) + replicate3 input。旧 custom
U-Net、normalization sweep、assigner fallback 和可调 VLR 分支均从 active 代码面移除。
2026-06-15 新增非官方 CMDistill controlled comparison profile，
并更新为 `v2_strict_20260615`：PCCFD + SLRD + IBCLD，launcher 默认启用
`KD_CALIBRATION_MODE=affine` 作为学生 adaptive layer。
2026-06-05 CCLKD loss 级实现已修正，并补齐
`cclkd_reproduction/code/` online teacher-student 复现入口；当前仍等待 GPU smoke。
`ladd/code/` 与
`ladd/code_versions/current_hbb/` 应保持字节一致；任何实验启动前先执行同步检查。

## 4. 结果口径

所有 controlled comparison 都应使用：

```text
formal no-mosaic
imgsz=256
800 epochs
same-capacity same-seed RGB teacher
SAR-only inference
```

训练长度不是对比指标，必须训练到收敛或明确异常退出。无效旧结果不进入当前主表。
