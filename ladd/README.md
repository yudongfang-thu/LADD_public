# LADD — Learnability-Aware Decomposition Distillation

RGB-guided SAR object detection via teacher-student decomposition KD。

## 方法

教师（RGB）和学生（SAR）共享 YOLO 骨干。分解块将特征分为可学习部分（z_t/z_s）和私有部分（u_t/r_s），通过三阶段课程训练（A1→A2→B）：

- A1 (10ep)：训练教师分解 + reachability adapter
- A2 (50ep)：加入学生骨干，检测 loss=1.0，MuSGD lr=0.001
- B (800ep)：训练学生分解 + cap2 反坍缩蒸馏

cap2：移除 reach rank loss 中反平行方向的奖励，修正几何目标。

## 代码

- `code/train_ladd_hbb.py` — 训练入口（也支持对比方法）
- `code/src/teacher_student_decomposition_kd_hbb/` — HBB 实现（model/loss/trainer）
- `scripts/launch_formal_ladd_job.sh` — 标准启动脚本

依赖 `shared/teacher_student_decomposition_kd/` 基础框架和 `shared/yolo/` vendored YOLO。

## 结果

详见 `results/LADD_RESULTS_CN.md`

| Model | seed0 | seed42 | vs baseline |
|---|---|---:|---:|
| YOLO11n cap2 | 0.57662 | 0.57420 | +0.016-0.020 |
| YOLO11s cap2 | 进行中 | — | — |
| YOLO11m cap2 | 进行中 | — | — |
