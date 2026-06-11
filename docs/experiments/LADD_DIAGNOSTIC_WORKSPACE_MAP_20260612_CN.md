# LADD 诊断实验工作区与证据地图（2026-06-12）

## 1. 当前在诊断什么

本轮诊断的核心问题是：

**LADD 在 formal no-mosaic 收敛协议下，为什么从 A2 或 B 阶段开始出现 detector AP 退化。**

目前已经拆成几条互相区分的诊断线：

1. **Protocol late-regression**
   - 先确认 800 epoch formal no-mosaic 协议本身是否存在 best-final gap。
   - 结论：90 服务器 YOLO11 baseline 证明协议本身确实存在 late-regression，所以主表应使用 best AP；但 LADD alpha_kd=0.5/0.25 的 final 低于 SAR baseline final，仍有额外退化。

2. **Capacity-aware KD / B 阶段退化**
   - 检查 `alpha_kd=0.5`、`alpha_kd=0.25` 是否缓解 B 阶段后期 collapse。
   - 检查 `B det-only` 是否说明 B 阶段 KD 是主因。
   - 当前判断：只调 B 阶段 KD 不够，因为正常 A2 checkpoint 本身已经低于 baseline。

3. **A2 damage localization**
   - 检查 A2 阶段 detector 是否已经受损。
   - 通过 `A2 det-only` 区分损伤是否来自 LADD aux/reach/rec/sep/task/private aux。
   - 通过 `A2 lr3e-4` 区分损伤是否来自 A2 lr=1e-3 的优化冲击。
   - 当前判断：s 模型可以在 A2 中冲高但后期不稳定；m 模型 A2 损伤更重，不能简单归因于 B 阶段。

4. **4090D vs 90 服务器差异**
   - 用 90 服务器历史 formal baseline 和 mosaic mainline 证据，排除“YOLO11 baseline 或 formal protocol 本身不可收敛”的解释。
   - 90 的 baseline/mosaic 证据主要作为外部参照，不是本轮新训练的主目录。

## 2. 服务器与本地工作目录

### 双卡 4090 服务器：本轮诊断主工作区

```text
/root/shared-nvme/LADD_public_p1
```

这是本轮 A2 damage、capacity KD、B det-only 等诊断实验实际运行的主目录。

主要结果目录：

```text
/root/shared-nvme/LADD_public_p1/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/
```

主要日志目录：

```text
/root/shared-nvme/LADD_public_p1/logs/formal_nomosaic_20260528/ladd/
```

说明：诊断实验没有单独的顶层 `diagnostics/` 子目录，而是挂在 formal run tree 下，通过 run tag 区分。

### 本地仓库：文档与轻量证据整理

```text
/Users/yudongfang/Desktop/光sar/LADD_public
```

本地仓库用于：

- 整理诊断结论文档。
- 存放 compact results、summary CSV、manifest、log extract。
- 同步 GitHub `main`。

### 90 服务器：历史 formal baseline 参照

```text
/mnt/dataY/ydf/projects/LADD_og
```

90 服务器不是本轮 A2 damage 诊断的主训练目录；它用于补充历史 formal baseline 证据，证明 YOLO11 n/s/m/l/x 的 SAR/RGB baseline 在 formal no-mosaic 800ep 协议下可以完整收敛。

## 3. 诊断实验的 run tag 与远端位置

### A2 damage localization

#### P1-1：YOLO11s A2 det-only

目的：判断 s 的 A2 下滑是否主要来自 A2 的 LADD auxiliary/reach/rec/sep/task/private losses。

Run tag：

```text
formal_nomosaic_yolo11s_cap2_s0_diag_a2damage_s_s0_a2detonly_b1_retry1
```

远端结果：

```text
/root/shared-nvme/LADD_public_p1/runs_public/ogsod/hbb/formal_nomosaic_20260528/ladd/yolo11s/cap2/
```

对应 phase run：

```text
ladd_hbb_ogsod11s_formal_nomosaic_yolo11s_cap2_s0_diag_a2damage_s_s0_a2detonly_b1_retry1_a1_e10_b64_s0_gpu0
ladd_hbb_ogsod11s_formal_nomosaic_yolo11s_cap2_s0_diag_a2damage_s_s0_a2detonly_b1_retry1_a2_e50_b64_s0_gpu0
ladd_hbb_ogsod11s_formal_nomosaic_yolo11s_cap2_s0_diag_a2damage_s_s0_a2detonly_b1_retry1_b_e1_b64_s0_gpu0
```

日志：

```text
/root/shared-nvme/LADD_public_p1/logs/formal_nomosaic_20260528/ladd/formal_nomosaic_yolo11s_cap2_s0_diag_a2damage_s_s0_a2detonly_b1_retry1_gpu0/
```

#### P1-2：YOLO11m A2 det-only

目的：判断 m 的 A2 损伤是否来自 A2 auxiliary objectives。

Run tag：

```text
formal_nomosaic_yolo11m_cap2_s0_diag_a2damage_m_s0_a2detonly_b1_retry1
```

对应 phase run：

```text
ladd_hbb_ogsod11m_formal_nomosaic_yolo11m_cap2_s0_diag_a2damage_m_s0_a2detonly_b1_retry1_a1_e10_b32_s0_gpu1
ladd_hbb_ogsod11m_formal_nomosaic_yolo11m_cap2_s0_diag_a2damage_m_s0_a2detonly_b1_retry1_a2_e50_b32_s0_gpu1
ladd_hbb_ogsod11m_formal_nomosaic_yolo11m_cap2_s0_diag_a2damage_m_s0_a2detonly_b1_retry1_b_e1_b32_s0_gpu1
```

日志：

```text
/root/shared-nvme/LADD_public_p1/logs/formal_nomosaic_20260528/ladd/formal_nomosaic_yolo11m_cap2_s0_diag_a2damage_m_s0_a2detonly_b1_retry1_gpu1/
```

#### P1-3：YOLO11s A2 lr3e-4

目的：判断 s 的 A2 损伤是否来自 A2 `lr0=1e-3` 优化冲击。

Run tag：

```text
formal_nomosaic_yolo11s_cap2_s0_diag_a2damage_s_s0_a2lr3e4_b1
```

对应 phase run：

```text
ladd_hbb_ogsod11s_formal_nomosaic_yolo11s_cap2_s0_diag_a2damage_s_s0_a2lr3e4_b1_a1_e10_b64_s0_gpu1
ladd_hbb_ogsod11s_formal_nomosaic_yolo11s_cap2_s0_diag_a2damage_s_s0_a2lr3e4_b1_a2_e50_b64_s0_gpu1
ladd_hbb_ogsod11s_formal_nomosaic_yolo11s_cap2_s0_diag_a2damage_s_s0_a2lr3e4_b1_b_e1_b64_s0_gpu1
```

日志：

```text
/root/shared-nvme/LADD_public_p1/logs/formal_nomosaic_20260528/ladd/formal_nomosaic_yolo11s_cap2_s0_diag_a2damage_s_s0_a2lr3e4_b1_gpu1/
```

#### 尚未作为本轮完成证据的 A2 damage run

根据原计划还包括：

```text
formal_nomosaic_yolo11m_cap2_s0_diag_a2damage_m_s0_a2lr3e4_b1
```

这条用于判断 m 是否对 A2 lr=1e-3 更敏感。若尚未启动或尚未完成，不应写入完成结论。

### Capacity KD / B-stage diagnostics

#### s alpha_kd=0.5 B400

目的：判断降低 B 阶段 KD 强度是否缓解 late collapse。

GitHub 轻量证据：

```text
ladd/results/capacity_kd_20260611/alpha0p5_b400/
```

#### s alpha_kd=0.25 B400

目的：进一步降低 B 阶段 KD 强度，观察是否恢复到 SAR baseline 附近。

GitHub 轻量证据：

```text
ladd/results/capacity_kd_20260611/alpha0p25_b400/
```

#### s B det-only r2

目的：B 阶段关闭 KD，只保留 detection loss，判断 B 阶段 KD 是否是主要损伤来源。

Run tag：

```text
formal_nomosaic_yolo11s_cap2_s0_diag_capkd_s_s0_bdetonly_b400_r2
```

GitHub 轻量证据：

```text
ladd/results/capacity_kd_20260611/bdetonly_b400_r2/
```

注意：这条 B det-only 继承的是正常 A2 checkpoint，而该 A2 已经低于 baseline。因此即使 B det-only 后续恢复，也不能把 A2 损伤归因排除。

#### m A2 probe

目的：在不启动 m full B 的情况下检查 m 的 A2 是否已经损伤。

GitHub 轻量证据：

```text
ladd/results/capacity_kd_20260611/m_a2_probe/
```

## 4. 已同步到 GitHub 的关键文档与数据

### Capacity KD 与 A2/B 退化

```text
docs/experiments/LADD_CAPACITY_KD_DIAG_RESULTS_20260611_CN.md
docs/experiments/ladd_capacity_kd_results_20260611_snapshot.csv
ladd/results/capacity_kd_20260611/
```

内容：

- `s alpha_kd=0.5 B400`
- `s alpha_kd=0.25 B400`
- `s B det-only r2`
- `m A2 probe`
- compact `results.csv`
- `args.yaml`
- `manifest.txt`
- key log extracts

### 90 formal baseline 证据

```text
docs/experiments/LADD_90_FORMAL_BASELINE_EVIDENCE_20260612_CN.md
ladd/results/ladd90_formal_baselines_20260612/
```

内容：

- YOLO11 n/s/m/l/x 的 SAR/RGB formal baseline summary。
- 18 条 baseline 的 `args.yaml`。
- 18 条 baseline 的 `results.csv`。
- 从原始 log 抽取的关键 log extract。
- 完整原始 log 的 90 服务器路径和大小。

### Formal protocol late-regression 分析

```text
docs/experiments/FORMAL_PROTOCOL_LATE_REGRESSION_ANALYSIS_20260612_CN.md
docs/experiments/formal_protocol_late_regression_summary_20260612.csv
```

内容：

- baseline best/final/drop。
- method final - SAR baseline best。
- method final - SAR baseline final。
- excess drop。
- last20/last50/last100 slope。
- best-window vs final-window train/val loss delta。

## 5. 相关代码在哪里

### LADD HBB 当前训练代码

```text
ladd/code/src/teacher_student_decomposition_kd_hbb/
```

核心文件：

```text
ladd/code/src/teacher_student_decomposition_kd_hbb/model.py
ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py
ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py
ladd/code/src/teacher_student_decomposition_kd_hbb/base_hbb.py
```

### 当前 formal chain 使用的版本化代码

```text
ladd/code_versions/current_hbb/
```

关键入口：

```text
ladd/code_versions/current_hbb/tools/train_ladd_hbb.py
ladd/code_versions/current_hbb/scripts/ogsod_public/run_hbb_ladd_converged_chain.sh
ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh
```

### formal LADD launcher

```text
ladd/scripts/launch_formal_ladd_job.sh
```

本轮诊断实验主要通过该 launcher 启动，并通过环境变量控制：

```text
LADD_A2_DET_ONLY
LADD_B_DET_ONLY
A2_OPTIMIZER
A2_LR0
B_OPTIMIZER
B_LR0
B_FREEZE_BN_STATS
LADD_DIAG_LOG_BN
LADD_DIAG_LOG_GRAD
LADD_GRAD_CLIP_NORM
LADD_ASSERT_PHASE_FREEZE
RUN_TAG_SUFFIX
SERVER_TAG
GIT_COMMIT
```

### baseline 与 comparison 代码

Baseline：

```text
baseline/code/train_ogsod_baseline.py
baseline/scripts/run_formal_baseline.sh
```

Comparison：

```text
comparison/code/launch_formal_transfer_kd_job.sh
comparison/code/launch_formal_from_yolo_kd_job.sh
comparison/code/launch_formal_online_cclkd_job.sh
```

## 6. 诊断数据应该看哪些文件

每个远端 run 目录下最重要的是：

```text
results.csv
args.yaml
ladd_diagnostics.csv
weights/best.pt
weights/last.pt
```

GitHub 只同步轻量证据：

- `results.csv`
- `args.yaml`
- `ladd_diagnostics.csv` 的必要摘录或 summary
- `manifest.txt`
- key log extract

不进入 GitHub：

- `weights/*.pt`
- `.pth`
- TensorBoard event
- wandb
- 完整 run 目录
- 完整大日志

## 7. 当前判断线索

1. 90 formal baseline 证明：YOLO11 detector 与 formal no-mosaic 800ep protocol 本身可以完整收敛，但存在自然 best-final gap。
2. 因此主表应该用 best AP；final AP 用于分析 late-regression 稳定性。
3. LADD `alpha_kd=0.5/0.25` 的 final 低于 YOLO11s SAR baseline final，说明不只是 protocol late-regression，还有 LADD 额外退化。
4. `s B det-only r2` 说明 B 阶段关闭 KD 后可能恢复，但它继承了已退化 A2，因此 A2 是必须单独定位的阶段。
5. `m A2 probe` 显示 m 在 A2 已经低于 baseline best/final，暂时不应继续启动 m full B 或大矩阵。
6. A2 damage 实验的核心解释应同时比较 baseline best、baseline final 和 baseline drop，而不是只用单一 safe threshold。

## 8. 后续维护建议

后续新增诊断 run 时，建议继续使用 `diag_*` run tag，并在 GitHub 中同步：

```text
docs/experiments/<analysis_doc>.md
docs/experiments/<summary>.csv
ladd/results/<compact_artifact_dir>/
```

每个 compact artifact dir 至少包含：

```text
README_CN.md 或 manifest.txt
args.yaml
results.csv
ladd_diagnostics.csv 或关键 diagnostics extract
key log extract
```

同时明确：

- 使用的 git commit。
- 服务器和 GPU。
- run tag。
- actual run dir。
- 是否有 OOM / auto batch reduction / NaN / phase-freeze assert failure。
- 是否排除权重和大日志。
