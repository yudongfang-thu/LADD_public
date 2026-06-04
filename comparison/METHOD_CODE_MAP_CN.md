# 非 CoLD 对比方法代码映射

最后更新：2026-06-04 08:55 CST

本文档说明 public 包中除 CoLD 外四个对比方法的代码在哪里。当前实现不是把 FGD、CrossKD、LD、HalluciDet 各自放在独立源码目录，而是在同一个 HBB trainer 中用 `--comparison-kd-profile` 切换。

## 1. 入口

| 文件 | 作用 |
|---|---|
| `../ladd/code/train_ladd_hbb.py` | 统一训练入口，暴露 `--comparison-kd-profile`、`--freeze-bn-stats` 以及各 profile 超参数 |
| `../ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py` | LADD 主 loss 和所有 comparison KD profile |
| `../ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py` | 阶段控制、teacher/student 前向、BN-freeze 逻辑 |
| `code/launch_formal_from_yolo_kd_job.sh` | 从同容量同 seed YOLO RGB teacher 初始化/蒸馏的正式启动脚本 |
| `code/launch_formal_transfer_kd_job.sh` | transfer teacher 形式的启动脚本，主要给 LD/CrossKD sanity 使用 |

## 2. Profile 对应关系

| 方法 | 启动 profile | 核心函数/逻辑 | 当前说明 |
|---|---|---|---|
| FGD | `--comparison-kd-profile fgd` | `TSKDDetectionLossHBB._fgd_style_loss()` | foreground/background weighted feature MSE + batch relation |
| CrossKD-style | `--comparison-kd-profile crosskd` | `_crosskd_style_loss()` | prediction KD + feature KD 的本项目 port，不是官方逐行复现 |
| LD | `--comparison-kd-profile ld` | `_ld_style_loss()` | logit distillation baseline |
| HalluciDet-style | `--comparison-kd-profile hallucidet` | `_hallucidet_style_loss()` | privileged RGB-to-SAR hallucination idea 的轻量移植 |

## 3. 代码新鲜度

2026-06-04 已将 `ladd/code/` 同步到当前主工作区和 `ladd/code_versions/current_hbb/`，因此 public 中两处 LADD/HBB 代码现在一致。此前 `ladd/code/` 缺少 HalluciDet-style profile、`--freeze-bn-stats` 和相关超参数，会误导外部排查；这个问题已经修正。

## 4. 结果口径

所有非 CoLD 对比都应使用：

```text
formal no-mosaic
imgsz=256
800 epochs
same-capacity same-seed RGB teacher
SAR-only inference
```

训练长度不是对比指标，必须训练到收敛或明确异常退出。CoLD 因复现协议和源码差异较大，单独记录在 `cold/`。
