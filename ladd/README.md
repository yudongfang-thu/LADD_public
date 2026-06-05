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
- `code/README.md` — 当前代码快照说明
- `scripts/launch_formal_ladd_job.sh` — 标准启动脚本

依赖 `shared/teacher_student_decomposition_kd/` 基础框架和 `shared/yolo/` vendored YOLO。

## 结果

详见 `results/LADD_RESULTS_CN.md`

崩溃与修复结论已经收敛进 `results/LADD_RESULTS_CN.md` 和 `../docs/experiments/LADD_MAINLINE_STANDARD_CN.md`；原始诊断包不随精简 public 分支发布。

| Model | seed0 | seed42 | vs baseline |
|---|---|---:|---:|
| YOLO11n cap2 `a2mu1e3` | 0.57662@725 | 0.57420@735 | +0.016-0.020 |
| YOLO11n cap2 BN-freeze | 0.57276@793 | 待补 | seed123 0.57269@779 |
| YOLO11s cap2 | 0.63551@605 | — | seed0 +0.00654，未满 800 |
| YOLO11m cap2 | 0.59796@1 | — | 异常，不进主表 |
