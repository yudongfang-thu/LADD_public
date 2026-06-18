# OGSOD LADD Probe-A 主线训练规范

最后更新：2026-06-18

本文档记录当前 OGSOD HBB 上 LADD 的正式主线设置。自本版起，主线固定为
`LADD Probe-A / LADD-clean A1B`。旧 no-mosaic A1-A2-B、BN-freeze、A2
稳定学习率等内容保留为历史诊断，不再作为当前主线规范。

## 1. 主线定义

当前正式 LADD 主线是：

```text
latest same-protocol SAR/RGB baseline
+ mosaic100 protocol: mosaic=1.0, close_mosaic=700, 800 epoch
+ A1 teacher decomposition warmup
+ B / Probe-A:
   - detector + z_s -> z_t KD + student reconstruction
   - dynamic teacher decomposition/reach/taskL
   - frozen student reachability probe
   - detached q_s in reach loss
+ cap2 reach-rank: RANK_D_NEG_CAP=2.0
+ no A2
+ historical sep/private/residual/debug losses removed
```

主方法报告中默认写作 `LADD Probe-A` 或 `LADD-clean A1B (Probe-A)`。
对应 launcher mode 为：

```bash
LADD_A1B_MODE=dynamic_probe
```

对应 run tag / project key：

```text
run tag:        clean_a1b_dynprobe_*
project key:    ladd_clean_a1b_dynamic_probe
```

## 2. Baseline 依赖

LADD 必须从同协议 baseline 出发：

- SAR baseline：同容量、同 seed 的 SAR-only YOLO11 HBB `best.pt`。
- RGB teacher：同容量、同 seed 的 RGB-only YOLO11 HBB `best.pt`。
- batch 沿用容量表：`n/s=64`, `m/l=32`, `x=16`。
- baseline、LADD、comparison methods 必须使用同一个增强协议。

配对规则：

```text
SAR(size, seed k, mosaic100) + RGB(size, seed k, mosaic100)
-> LADD Probe-A(size, seed k, mosaic100)
```

旧 formal no-mosaic baseline 可用于鲁棒性/附录实验，但不能混入 mosaic100 主表。

## 3. 阶段设置

当前主线只跑 A1 和 B，不跑 A2：

| 阶段 | epoch | 检测损失 | 主要训练对象 | 作用 |
|---|---:|---:|---|---|
| A1 | `10` | `0.0` | teacher decomposition、teacher decoder、student reachability probe、teacher task heads | 学习 `z_t/u_t` 分解和 reach probe |
| B / Probe-A | `800` | `1.0` | SAR detector、student split、teacher decomposition、teacher decoder、teacher task heads | SAR 检测 + KD，并让 teacher common target 随 B 阶段适配 |

Probe-A 在 B 阶段的关键点：

| 模块/路径 | B 状态 |
|---|---|
| SAR detector backbone/head | train |
| `student_split` | train |
| `teacher_decomposition` | train |
| `teacher_decoder` | train |
| `teacher_task_heads` | train |
| `student_reachability` | frozen/eval |
| reach loss 中的 `q_s` | detach |

这一区别使 Probe-A 既保留 Dynamic 的 `z_t/u_t` 适配能力，又避免 B 阶段 reach
loss 继续拉动学生 reach probe。当前曲线证据显示它比完全 Dynamic 更稳定。

## 4. Loss 口径

A1:

```text
L_A1 = lambda_rec * L_t_rec
     + lambda_reach * (lambda_match_inner * L_reach_match
                     + lambda_rank_inner * L_reach_rank_cap)
     + lambda_taskL * L_task
```

B / Probe-A:

```text
L_B = L_det
    + alpha_kd * L_KD(z_s, stopgrad(z_t))
    + alpha_s_rec * L_s_rec
    + lambda_rec * L_t_rec
    + lambda_reach * (lambda_match_inner * L_reach_match(stopgrad(q_s), z_t, u_t)
                    + lambda_rank_inner * L_reach_rank_cap(stopgrad(q_s), z_t, u_t))
    + lambda_taskL * L_task
```

默认权重：

| 变量 | 值 |
|---|---:|
| `RANK_D_NEG_CAP` | `2.0` |
| `LAMBDA_REC` | `0.1` |
| `LAMBDA_TASKL` | `1.0` |
| `LAMBDA_REACH` | `1.0` |
| `LAMBDA_MATCH_INNER` | `1.0` |
| `LAMBDA_RANK_INNER` | `1.0` |
| `ALPHA_KD` | `1.0` |
| `ALPHA_S_REC` | `0.1` |
| `USE_MASK` | `1` |
| `USE_FG_MASK_FOR_REACH` | `1` |
| `USE_FG_MASK_FOR_REC` | `0` |

已从当前 HBB LADD 代码移除的历史项包括：
`t_sep/s_sep/r_aux/u_aux/mask_reg/recon_task/rs_comp/r_obb/s_repel/path_b/r_sar/dkd/proto_cls`。
若结果 CSV 中出现这些字段，说明不是当前 clean 主线代码。

## 5. 主协议

当前主表协议固定为 mosaic100：

```text
dataset = OGSOD-1.0 HBB
imgsz = 256
epochs_B = 800
mosaic = 1.0
close_mosaic = 700   # 前 100 epoch mosaic on，后 700 epoch mosaic off
mixup = 0.0
cutmix = 0.0
degrees = 0.0
perspective = 0.0
translate = 0.1
scale = 0.5
fliplr = 0.5
flipud = 0.0
hsv_h/s/v = 0.0
erasing = 0.0
cos_lr = true
optimizer = auto
lr0 = 0.01
lrf = 0.01
warmup_epochs = 3.0
warmup_bias_lr = 0.1
deterministic = true
batch = n/s:64, m/l:32, x:16
```

no-mosaic Probe-A 可作为鲁棒性/附录协议：

```text
mosaic = 0.0
close_mosaic = 0
```

但 no-mosaic 结果不进入 mosaic100 主表，也不能与 mosaic100 baseline 直接计算 gap。

## 6. 有效入口

主线启动入口：

```bash
LADD_A1B_MODE=dynamic_probe \
  bash ladd/scripts/launch_ladd_clean_a1b_job.sh <n|s|m|l|x> <seed> <gpu_id>
```

显式传入 baseline 时：

```bash
SAR_BASELINE=/path/to/sar/best.pt \
RGB_TEACHER=/path/to/rgb/best.pt \
LADD_A1B_MODE=dynamic_probe \
  bash ladd/scripts/launch_ladd_clean_a1b_job.sh <n|s|m|l|x> <seed> <gpu_id>
```

Static 和 Dynamic 只作为消融入口：

```bash
# Static ablation
bash ladd/scripts/launch_ladd_clean_a1b_job.sh <n|s|m|l|x> <seed> <gpu_id>

# Dynamic ablation
LADD_A1B_MODE=dynamic \
  bash ladd/scripts/launch_ladd_clean_a1b_job.sh <n|s|m|l|x> <seed> <gpu_id>
```

## 7. 当前实验优先级

| 优先级 | 实验 | 目的 |
|---|---|---|
| P0 | mosaic100 Probe-A `n/s` | 固定主线的最小容量证据 |
| P1 | mosaic100 Probe-A `m` | 容量扩展；依赖 `m` SAR/RGB mosaic100 baseline |
| P2 | no-mosaic Probe-A `n/s` | 检查旧 formal 协议下是否仍稳定 |
| P3 | Static/Dynamic `n/s` | 消融：teacher core 动态适配与 reach probe 冻结的贡献 |

## 8. 主表准入

进入 LADD 主表的结果必须同时满足：

1. run tag 包含 `clean_a1b_dynprobe`；
2. `LADD_A1B_MODE=dynamic_probe`；
3. phase chain 是 `A1 -> B`，没有 A2；
4. 使用当前 cleaned HBB LADD 代码；
5. baseline、LADD、comparison methods 使用同容量、同 seed、同 mosaic100 协议；
6. 结果包不包含 checkpoint 权重或其他大文件。

以下结果只能作为历史/附录/消融：

| 结果类型 | 处理 |
|---|---|
| 旧 A1-A2-B full chain | 历史诊断 |
| old formal no-mosaic LADD | 鲁棒性/附录 |
| Static `clean_a1b` | 消融 |
| Dynamic `clean_a1b_dyn` | 消融/诊断 |
| 旧 close@100 / 400ep / BN-freeze runs | 历史诊断 |
| 含 sep/aux/debug loss 的 run | 非当前主线 |
