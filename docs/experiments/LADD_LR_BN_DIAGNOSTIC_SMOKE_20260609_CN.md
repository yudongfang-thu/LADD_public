# LADD LR / BN / Schedule 诊断 Smoke 记录

最后更新：2026-06-09

本文档记录 2026-06-09 在双卡 4090 服务器上对 LR / BN / schedule 诊断工具做的连通与最小验证。该记录只说明代码和 launcher 可用性，不包含真实训练结果，也不选择 LADD 主线结论。

## 1. 服务器与连通方式

测试节点：

```text
双卡 4090 / Paratera 节点
远端部署目录：/root/shared-nvme/LADD_public
```

在 TUN 模式下，默认 `ssh ladd4090` 不稳定。本次可用方式是通过 Paratera SSH 网关连接，并从本地私有连接文件读取口令；口令不写入仓库、不写入本文档。

```bash
ssh -p 2233 -l 'root@ackcs-00gjh35x' 219.146.211.42
```

本次远端连接返回：

```text
hostname: p-fc024d48c0fa-ackcs-00gjh35x
date: Tue Jun 9 2026 CST
```

## 2. 当前 GPU 状态

smoke 测试结束时两张卡都在跑 CCLKD 实验，显存与利用率接近满载：

```text
GPU 0: 12578 / 24564 MiB, util 98%
GPU 1: 12574 / 24564 MiB, util 97%
```

因此本次没有启动真实 LADD 训练，只做不占显存的语法检查、Python 编译、launcher dry run 和汇总脚本 smoke，避免干扰正在运行的 CCLKD 消融。

## 3. 已完成的远端 Smoke

在 `/root/shared-nvme/LADD_public` 下完成以下检查：

```bash
bash -n ladd/scripts/launch_formal_ladd_job.sh
bash -n ladd/scripts/launch_ladd_lr_bn_diag_matrix.sh
bash -n ladd/code_versions/current_hbb/scripts/ogsod_public/run_hbb_ladd_converged_chain.sh
bash -n ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh
```

结果：

```text
PASS
```

Python 编译检查：

```bash
python3 -m py_compile \
  ladd/tools/summarize_ladd_lr_bn_diagnostics.py \
  ladd/code/train_ladd_hbb.py \
  ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py \
  ladd/code_versions/current_hbb/tools/train_ladd_hbb.py \
  ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/trainer.py
```

结果：

```text
COMPILE_OK
```

诊断矩阵 dry run：

```bash
DRY_RUN=1 LAUNCH=0 \
  ladd/scripts/launch_ladd_lr_bn_diag_matrix.sh n 0 0 smoke
```

确认包含 8 个 smoke tag：

```text
_diag_current_stable
_diag_lr2e3_freeze
_diag_lr3e3_freeze
_diag_tail_lr1e3_lrf0p1
_diag_nofreeze_lr1e3
_diag_delayed_bn200_lr1e3
_diag_delayed_bn200_lr2e3
_diag_b400_lr1e3_freeze
```

结果：

```text
MATRIX_DRY_RUN_OK
```

formal launcher 单条 dry run：

```bash
DRY_RUN=1 RUN_TAG_SUFFIX=_diag_lr2e3_freeze \
  B_OPTIMIZER=MuSGD B_LR0=0.002 B_LRF=0.01 \
  B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.002 \
  B_FREEZE_BN_STATS=1 \
  ladd/scripts/launch_formal_ladd_job.sh cap2 n 0 0
```

确认命令中包含：

```text
B_LR0=0.002
B_LRF=0.01
B_FREEZE_BN_STATS=1
_diag_lr2e3_freeze
```

结果：

```text
FORMAL_LR2E3_DRY_RUN_OK
```

B=400 诊断 dry run：

```bash
DRY_RUN=1 RUN_TAG_SUFFIX=_diag_b400_lr1e3_freeze \
  EPOCHS_B=400 PATIENCE_B=400 \
  B_OPTIMIZER=MuSGD B_LR0=0.001 B_LRF=0.01 \
  B_WARMUP_EPOCHS=0 B_WARMUP_BIAS_LR=0.001 \
  B_FREEZE_BN_STATS=1 \
  ladd/scripts/launch_formal_ladd_job.sh cap2 n 0 0
```

确认命令中包含：

```text
EPOCHS_B=400
PATIENCE_B=400
B_CLOSE_AT_EPOCH=400
```

结果：

```text
FORMAL_B400_DRY_RUN_OK
```

汇总脚本 smoke：

```bash
python3 ladd/tools/summarize_ladd_lr_bn_diagnostics.py \
  runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd \
  --output-csv /tmp/ladd_diag_summary.csv \
  --output-md /tmp/ladd_diag_summary.md
```

结果：

```text
SUMMARY_OK root=runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd rows=8
```

## 4. 结论

本次 smoke 说明：

- 双卡 4090 在 TUN 模式下仍可通过 Paratera SSH 网关访问，但握手延迟较高，长连接可能中断；
- LR / BN / schedule 诊断代码已经同步到远端部署目录；
- 本地与远端的语法检查、Python 编译、诊断矩阵 dry run、formal launcher dry run、汇总脚本 smoke 均通过；
- 因两张 GPU 正在高负载运行 CCLKD，本次没有启动真实 LADD 训练；
- 后续可在 CCLKD 释放显存后，用 `LAUNCH=1` 或手动分配 GPU 启动 YOLO11n seed0 / seed123 的 smoke tier。

