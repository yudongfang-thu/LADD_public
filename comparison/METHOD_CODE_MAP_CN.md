# 受控对比方法代码映射

最后更新：2026-06-04 08:55 CST

本文档说明 public 包中四个正式候选方法的代码位置。当前实现是在同一个 HBB
trainer 中用 `--comparison-kd-profile` 切换。严格实现边界见
[`IMPLEMENTATION_REVIEW_CN.md`](IMPLEMENTATION_REVIEW_CN.md)，第二轮复核响应见
[`REVIEW_FEEDBACK_RESPONSE_CN.md`](REVIEW_FEEDBACK_RESPONSE_CN.md)。

## 1. 入口

| 文件 | 作用 |
|---|---|
| `../ladd/code/train_ladd_hbb.py` | 统一训练入口，暴露 `--comparison-kd-profile`、`--freeze-bn-stats` 以及各 profile 超参数 |
| `../ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py` | LADD 主 loss 和所有 comparison KD profile |
| `../ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py` | 阶段控制、teacher/student 前向、BN-freeze 逻辑 |
| `code/launch_formal_from_yolo_kd_job.sh` | FGD/LD/HalluciDet-style 的 from-YOLO frozen-teacher 正式启动脚本 |
| `code/launch_formal_transfer_kd_job.sh` | FGD/LD/HalluciDet-style 的 transfer frozen-teacher 启动脚本 |
| `../cclkd_reproduction/` | CCLKD 原文协议复现目录；online trainer 补齐前不启动正式 CCLKD |

## 2. Profile 对应关系

| 方法 | 启动 profile | 核心函数/逻辑 | 当前说明 |
|---|---|---|---|
| FGD-style | `--comparison-kd-profile fgd` | `TSKDDetectionLossHBB._fgd_style_loss()` | 官方 softmax attention 形式 + GT fg/bg weighting + batch relation 近似 |
| LD | `--comparison-kd-profile ld` | `_ld_style_loss()` | foreground YOLO DFL regression-distribution KL；错形直接失败 |
| CCLKD | `--comparison-kd-profile cclkd` | `_cclkd_style_loss()` | COP + localization-only LLD + FLD-MSE + RLD feature-correlation + CCL；原文复现见 `../cclkd_reproduction/` |
| HalluciDet-style | `--comparison-kd-profile hallucidet` | `_hallucidet_style_loss()` | privileged RGB-to-SAR hallucination idea 的轻量移植 |

当前 public 对比方法只发布上述四项；其他历史候选不作为当前代码映射的一部分。

## 3. 代码新鲜度

2026-06-04 已修复 LD/FGD 语义。2026-06-05 CCLKD loss 级实现已修正，但仍等待
online teacher-student trainer。`ladd/code/` 与
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
