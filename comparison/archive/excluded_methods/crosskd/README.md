# CrossKD-style

CrossKD: Cross-Head Knowledge Distillation for Object Detection, CVPR 2024.

## 方法

Prediction-level KD：student head 读取 teacher intermediate features，在 prediction 空间做蒸馏。当前为 style port 而非官方逐行复现。

## 代码位置

CrossKD-style 在本 public 包中使用统一入口：

```text
../ladd/code/train_ladd_hbb.py --comparison-kd-profile crosskd
../ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py::_crosskd_style_loss
```

当前是 prediction-level KD + feature KD 的本项目 port，不是官方逐行复现。

## 处置

YOLO11 当前检测头没有暴露官方 CrossKD 所需的 cross-head routing 接口。现有
profile 只是 prediction KL + foreground feature MSE，与原方法机制差异较大。

代码保留用于审计，但 formal launcher 已禁止启动；论文和 controlled main table
不再使用 CrossKD。

## 历史结果

| Model/seed | 服务器 | epoch | best/current AP50-95 | vs SAR baseline | 状态 |
|---|---|---:|---:|---:|---|
| YOLO11n seed0 | 4090D | 800 | 0.55764 best | -0.00152 | 完成，基本打平 |
| YOLO11n seed42 | 4090 | 195 | 0.40349 current | 未完成 | 运行中 |
| YOLO11n seed123 | 4090 | 186 | 0.39508 current | 未完成 | 运行中 |

这些结果仅作为淘汰方法的历史记录，不进入正式统计。
