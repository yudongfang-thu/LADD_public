# CCLKD LADD 协议排查与重启记录（2026-06-08）

## 1. 结论

2026-06-07 晚启动的 YOLO11n CCLKD LADD-protocol 400 epoch 消融不能作为有效对比结果。主要原因不是输入尺寸，也不是模型规模或数据集版本，而是 formal LADD baseline 与 CCLKD 消融之间存在训练动力学不一致：

1. SAR baseline 命令未显式指定 optimizer，实际为 `optimizer=auto -> MuSGD(lr=0.01, momentum=0.9)`。
2. CCLKD 消融命令显式使用 `optimizer=SGD`，实际为 `SGD(lr=0.01, momentum=0.937)`。
3. online CCLKD trainer 原先对 student + teacher 参数做一次联合 `clip_grad_norm_`。旧 run 中 teacher detection loss 后期明显恶化，可能同时污染 KD target 并改变 student 的有效更新。

因此，旧 CCLKD 消融已归档为 diagnostic-only，不进入正式结果表。已重启一批修正版实验：`optimizer=auto`，并将 student/teacher 梯度裁剪拆开。

## 2. 协议核查

已确认旧 run 与新 run 都是 YOLO11n，不是 YOLO11s：

| 项 | 旧 CCLKD / 新 CCLKD | LADD 400ep baseline |
|---|---:|---:|
| model | `yolo11n.pt` | `yolo11n.pt` / `yolo11s.pt` |
| data | SAR train, RGB paired teacher train | SAR train |
| classes | 3 | 3 |
| train images | 14664 SAR + 14664 RGB paired | 14664 SAR |
| missing RGB pairs | 0 | N/A |
| imgsz | 256 | 256 |
| epochs | 400 | 400 |
| batch | 64 | 64 |
| mosaic / mixup / cutmix | 0 / 0 / 0 | 0 / 0 / 0 |
| hsv / erasing | 0 / 0 | 0 / 0 |
| deterministic | true | true |
| optimizer, old | `SGD(lr=0.01, momentum=0.937)` | `auto -> MuSGD(lr=0.01, momentum=0.9)` |
| optimizer, restarted | `auto -> MuSGD(lr=0.01, momentum=0.9)` | `auto -> MuSGD(lr=0.01, momentum=0.9)` |

## 3. 代码修正

### Formal online CCLKD launcher

`comparison/code/launch_formal_online_cclkd_job.sh`

Formal LADD protocol 下默认 optimizer 从 `SGD` 改为 `auto`，使其与 baseline 的实际训练协议一致。注意：这只影响 formal online CCLKD comparison launcher，不改 CCLKD paper-protocol reproduction launcher。

### Online trainer gradient clipping

`cclkd_reproduction/code/train_cclkd_online_hbb.py`

`optimizer_step()` 从联合裁剪 student+teacher 参数，改成分别裁剪 student 和 teacher。这样 teacher branch 后期不稳定时，不会通过一次 joint norm 直接压缩 student 的梯度尺度。

## 4. 旧结果归档

远端归档目录：

```text
/root/shared-nvme/LADD_public/archive/invalid_cclkd_laddproto400_sgd_jointclip_20260608_0020
```

归档原因：

```text
formal LADD baseline used optimizer=auto -> MuSGD(lr=0.01,momentum=0.9),
but these CCLKD ablations used explicit SGD(lr=0.01,momentum=0.937)
and joint student+teacher grad clipping. Teacher detection loss later degraded,
so results are diagnostic only.
```

停止前最后指标如下：

| Ablation | Epoch | AP50 | AP50-95 |
|---|---:|---:|---:|
| LLD | 308 | 0.67355 | 0.42475 |
| LLD + FLD | 311 | 0.65129 | 0.40562 |
| LLD + FLD + RLD | 312 | 0.67234 | 0.42715 |
| ATKD | 296 | 0.64178 | 0.39559 |
| CCL only | 308 | 0.66852 | 0.42177 |
| Full | 311 | 0.67437 | 0.42548 |

这些数值明显低于同时运行的 LADD baseline n，并且 full / ccl-only / lld 的排序没有形成可解释的 CCLKD Table 12 式趋势，因此不进入正式对比。

## 5. 新实验状态

新 run 目录：

```text
/root/shared-nvme/LADD_public/runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/online_cclkd/yolo11n/cclkd_auto_sepclip
```

新日志目录：

```text
/root/shared-nvme/LADD_public/logs/formal_nomosaic_20260528/comparisons/online_cclkd_auto_sepclip
```

新实验包含 6 个 YOLO11n seed0 消融：

| Ablation | GPU | Temperature | Loss switches |
|---|---:|---|---|
| LLD | 0 | fixed 1.0 | LLD |
| LLD + FLD | 0 | fixed 1.0 | LLD, FLD |
| LLD + FLD + RLD | 0 | fixed 1.0 | LLD, FLD, RLD |
| ATKD | 1 | 0.5 to 5.0 | LLD, FLD, RLD, PATM |
| CCL only | 1 | fixed 1.0 | CCL |
| Full | 1 | 0.5 to 5.0 | LLD, FLD, RLD, PATM, CCL |

2026-06-08 00:26 左右，新 run 已确认进入训练，并打印：

```text
optimizer=auto -> MuSGD(lr=0.01, momentum=0.9)
```

早期 validation 快照只用于确认训练已启动，不用于方法效果判断：

| Ablation | Epoch | Student box/cls/dfl | Teacher box/cls/dfl | KD loss | AP50 | AP50-95 |
|---|---:|---|---|---:|---:|---:|
| ATKD | 3 | 2.828 / 2.266 / 1.380 | 2.351 / 1.723 / 1.211 | 4.378 | 0.04290 | 0.01218 |
| CCL only | 3 | 2.835 / 2.287 / 1.432 | 2.356 / 1.739 / 1.217 | 0.722 | 0.10201 | 0.03540 |
| Full | 3 | 2.842 / 2.300 / 1.396 | 2.342 / 1.728 / 1.213 | 5.134 | 0.04054 | 0.01194 |
| LLD + FLD | 3 | 2.837 / 2.283 / 1.405 | 2.352 / 1.741 / 1.215 | 2.236 | 0.07181 | 0.02258 |
| LLD + FLD + RLD | 3 | 2.842 / 2.292 / 1.422 | 2.360 / 1.732 / 1.212 | 2.139 | 0.04737 | 0.01596 |

## 6. Baseline 同期状态

2026-06-08 00:26 快照：

| Run | Epoch | Time (s) | AP50 | AP50-95 |
|---|---:|---:|---:|---:|
| YOLO11n SAR baseline | 336 | 29184.8 | 0.75347 | 0.49471 |
| YOLO11s SAR baseline | 333 | 29177.1 | 0.86234 | 0.58841 |

## 7. 判断与下一步

当前最合理的判断是：旧 CCLKD 的异常主要来自 formal LADD 协议没有真正对齐 optimizer，再叠加 online teacher-student joint clipping 导致 teacher branch 不稳定时影响 student 训练。数据增强、类别数、配对文件、图像尺寸都已排除为主因。

新 run 在 100 epoch 左右做第一次正式判断。判断标准：

1. Student AP 是否追近同协议 YOLO11n SAR baseline 的同期趋势。
2. Teacher detection loss 是否继续下降或至少保持稳定，不能再次出现旧 run 那种 ep100 后持续恶化。
3. Ablation 顺序是否出现可解释趋势，尤其是 `full`、`ccl_only`、`lld` 是否不再全部塌在同一低水平。
4. 如果 100 epoch 仍显著落后 baseline，下一步优先检查 online teacher 的训练定义，而不是再调整 256/640 或数据增强。

按 2026-06-08 00:26 的速度估计，新 CCLKD 每个 epoch 约 160-168 秒。100 epoch 预计在 2026-06-08 04:50-05:10 左右完成；400 epoch 保守估计在 2026-06-08 18:30-19:30 左右完成。

## 8. 2026-06-08 二次实现审计：PATM 与 CCL anchor

2026-06-08 进一步检查发现两个实现层问题：

1. PATM temperature 原先对一个 class 内所有 COP token 的 binary entropy 取均值，得到 per-class scalar temperature。这不符合按位置不确定性自适应温度的语义。
2. CCL 原先用 `student_pos` 对齐 `teacher_pos`，同时用 `student_neg` 对齐 `teacher_neg` 作为 negative logit。该写法会给非 k 类 student token 施加“远离非 k 类 teacher token”的梯度，破坏非 COP/异类位置的跨模态一致性。

已修复：

```text
cclkd_reproduction/code/train_cclkd_online_hbb.py
ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py
ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/loss.py
```

修复内容：

1. PATM temperature 改为 per-position vector。DFL KL 使用 `reduction="none"` 得到每个 token 的 LLD，再乘各自的 `T^2` 后平均。
2. CCL 改为以 class-k student positive token 为 anchor，对比 teacher positive token 与 teacher negative token：

```python
s_anchor = normalize(student_feat[pos_k])
t_pos = normalize(teacher_feat[pos_k])
t_neg = normalize(teacher_feat[neg_not_k])
pos_sim = cosine(s_anchor, t_pos)
neg_sim = cosine(s_anchor, t_neg)
```

这样梯度只流入 `s_anchor`：推向同类 teacher，推离异类 teacher；不再把非 k student token 推离非 k teacher token。

远端处理：

```text
/root/shared-nvme/LADD_public/archive/invalid_cclkd_auto_sepclip_patm_ccl_anchor_20260608_0035
```

只归档并停止受影响的三条 run：

| Ablation | 受影响原因 | 停止 epoch | AP50 | AP50-95 |
|---|---|---:|---:|---:|
| ATKD | PATM per-class temperature | 13 | 0.33583 | 0.14753 |
| CCL only | CCL anchor direction | 13 | 0.29767 | 0.12429 |
| Full | PATM + CCL anchor direction | 13 | 0.32416 | 0.14103 |

继续保留并运行的三条：

```text
LLD
LLD + FLD
LLD + FLD + RLD
```

原因：它们使用 fixed temperature (`min=max=1.0`) 且 `ccl_weight=0`，因此不受 per-position PATM 与 CCL anchor 修复的实质影响。

重启的修正版三条位于：

```text
/root/shared-nvme/LADD_public/runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/online_cclkd/yolo11n/cclkd_auto_sepclip_patmpos_cclanchor
```

日志位于：

```text
/root/shared-nvme/LADD_public/logs/formal_nomosaic_20260528/comparisons/online_cclkd_auto_sepclip_patmpos_cclanchor
```

新 PIDs：

| Ablation | PID | GPU |
|---|---:|---:|
| ATKD | 93296 | 1 |
| CCL only | 93297 | 1 |
| Full | 93298 | 1 |
