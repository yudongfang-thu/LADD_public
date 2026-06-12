# 受控对比方法代码映射

最后更新：2026-06-13

本文档说明 public 包中当前对比方法的代码位置。FGD/LD 在同一个 HBB
trainer 中用 `--comparison-kd-profile` 切换；HalluciDet 使用 standalone
image/representation hallucination 入口。严格实现边界见
[`IMPLEMENTATION_REVIEW_CN.md`](IMPLEMENTATION_REVIEW_CN.md)，第二轮复核响应见
[`REVIEW_FEEDBACK_RESPONSE_CN.md`](REVIEW_FEEDBACK_RESPONSE_CN.md)。

## 1. 入口

| 文件 | 作用 |
|---|---|
| `../ladd/code/train_ladd_hbb.py` | 统一训练入口，暴露 `--comparison-kd-profile`、`--freeze-bn-stats` 以及各 profile 超参数 |
| `../ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py` | LADD 主 loss 和所有 comparison KD profile |
| `../ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py` | 阶段控制、teacher/student 前向、BN-freeze 逻辑 |
| `code/launch_formal_from_yolo_kd_job.sh` | FGD/LD 的 from-YOLO frozen-teacher 正式启动脚本 |
| `code/launch_formal_transfer_kd_job.sh` | FGD/LD 的 transfer frozen-teacher 启动脚本 |
| `hallucidet/train_hallucidet.py` | standalone HalluciDet-YOLO adaptation：SAR -> hallucination net -> frozen RGB YOLO detector detection loss |
| `code/launch_formal_online_cclkd_job.sh` | CCLKD 的 LADD 统一协议 online teacher-student 受控对比启动脚本 |
| `../cclkd_reproduction/` | CCLKD 原文协议复现目录；已包含 online trainer / launcher / protocol checker，GPU smoke 前不启动正式 CCLKD |

## 2. Profile 对应关系

| 方法 | 启动 profile | 核心函数/逻辑 | 当前说明 |
|---|---|---|---|
| FGD-style | `--comparison-kd-profile fgd` | `TSKDDetectionLossHBB._fgd_style_loss()` | fg/bg feature + attention mask；GT-box mask 默认启用；official trainable global relation 默认不启用 |
| LD | `--comparison-kd-profile ld` | `_ld_style_loss()` | foreground/main YOLO DFL KL + teacher-quality VLR-style candidate LD；错形直接失败 |
| CCLKD | `launch_formal_online_cclkd_job.sh` / `train_cclkd_online_hbb.py` | online teacher detection loss + SAR student detection loss + CCLKD loss | COP + localization-only LLD + FLD-MSE + RLD feature-correlation + CCL；原文复现见 `../cclkd_reproduction/` |
| HalluciDet-YOLO adaptation | `python comparison/hallucidet/train_hallucidet.py` | `HalluciDetTrainer` / `HallucinationNetwork` | detection-loss-only standalone protocol；不是旧 feature/response/margin KD profile |

旧 `hallucidet_style` KD profile 已移除，不再作为可启动方法发布。

## 3. 代码新鲜度

2026-06-04 已修复 LD/FGD 语义。2026-06-10 FGD/LD 进一步更新为
FGD-YOLO focal+attention-mask adaptation 与 LD-YOLO main+VLR-style adaptation，
2026-06-13 移除旧 HalluciDet-style KD profile，保留 standalone HalluciDet-YOLO
adaptation。2026-06-05 CCLKD loss 级实现已修正，并补齐
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
