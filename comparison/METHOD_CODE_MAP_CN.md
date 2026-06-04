# 受控对比方法代码映射

最后更新：2026-06-04 08:55 CST

本文档说明 public 包中四个正式候选方法的代码位置。当前实现是在同一个 HBB
trainer 中用 `--comparison-kd-profile` 切换。严格实现边界见
[`IMPLEMENTATION_REVIEW_CN.md`](IMPLEMENTATION_REVIEW_CN.md)。

## 1. 入口

| 文件 | 作用 |
|---|---|
| `../ladd/code/train_ladd_hbb.py` | 统一训练入口，暴露 `--comparison-kd-profile`、`--freeze-bn-stats` 以及各 profile 超参数 |
| `../ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py` | LADD 主 loss 和所有 comparison KD profile |
| `../ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py` | 阶段控制、teacher/student 前向、BN-freeze 逻辑 |
| `code/launch_formal_from_yolo_kd_job.sh` | 从同容量同 seed YOLO RGB teacher 初始化/蒸馏的正式启动脚本 |
| `code/launch_formal_transfer_kd_job.sh` | transfer teacher 形式的正式四方法启动脚本 |

## 2. Profile 对应关系

| 方法 | 启动 profile | 核心函数/逻辑 | 当前说明 |
|---|---|---|---|
| FGD-style | `--comparison-kd-profile fgd` | `TSKDDetectionLossHBB._fgd_style_loss()` | teacher spatial/channel attention × GT fg/bg weighting + batch relation |
| LD | `--comparison-kd-profile ld` | `_ld_style_loss()` | foreground YOLO DFL regression-distribution KL |
| CCLKD-style | `--comparison-kd-profile cclkd` | `_cclkd_style_loss()` | adaptive feature/logit KD + category-constrained contrastive KD |
| HalluciDet-style | `--comparison-kd-profile hallucidet` | `_hallucidet_style_loss()` | privileged RGB-to-SAR hallucination idea 的轻量移植 |

`crosskd/mgd/c2kd/mmanet` 仍可在底层 CLI 中用于审计，但 formal launcher 会拒绝运行。

## 3. 代码新鲜度

2026-06-04 已修复 LD/FGD 语义并接入 CCLKD-style。`ladd/code/` 与
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

训练长度不是对比指标，必须训练到收敛或明确异常退出。CoLD/CrossKD 与无效旧结果只在 `archive/excluded_methods/` 中保留历史证据。
