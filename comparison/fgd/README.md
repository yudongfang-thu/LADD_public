# FGD

Focal and Global Knowledge Distillation for Detectors, CVPR 2022.

## 方法

Foreground/background weighted feature KD + batch-level global relation matrix MSE。feature level distillation 经典方法。

## 代码位置

FGD 在本 public 包中不是独立训练器，使用统一入口：

```text
../ladd/code/train_ladd_hbb.py --comparison-kd-profile fgd
../ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py::_fgd_style_loss
```

## 结果

| Model/seed | 服务器 | epoch | best/current AP50-95 | vs SAR baseline | 状态 |
|---|---|---:|---:|---:|---|
| YOLO11n seed0 | 4090D | 800 | 0.55867 best | -0.00049 | 完成，基本打平 |
| YOLO11n seed42 | 4090D | 343 | 0.46993 current | 未完成 | 运行中 |
| YOLO11n seed123 | 4090 | 134 | 0.37024 current | 未完成 | validation OOM，不计完成 |
| YOLO11s seed0 | 4090 | 192 | 0.50358 current | 未完成 | 运行中 |

seed0 完整结果等于 baseline，无正向提升。其他 seed 仍未完成，不能进入最终统计。

完整训练日志见 `results/4090d_formal_kd_20260602/`
