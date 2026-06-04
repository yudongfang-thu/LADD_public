# Attempt 3: Frozen Teacher (Offline KD)

服务器：117 (RTX 5880 Ada)
配置：`TEACHER_DET_WEIGHT=0`, `COLD_LOSS_MODE=candidate`, `COLD_TERMS=ncld`
Teacher：YOLOv5x 在 OGSOD RGB 上预训练 100ep (AP=0.448)

## 动机

排除在线 OKD 中 teacher 可能通过 CoLD loss 收到梯度的问题。用完全冻结的预训练 teacher 做纯离线蒸馏。

## 结果

NCLD 50ep 表现低于 baseline。冻结 teacher 不仅没有解决 TCLD > NCLD 的反转，反而让 NCLD 完全失效。

## 结论

OKD（在线教师训练）是 CoLD 的关键组件。论文 Table IV 也印证了这一点：OKD alone = +0.056 AP，甚至大于 CPM alone = +0.034 AP。
