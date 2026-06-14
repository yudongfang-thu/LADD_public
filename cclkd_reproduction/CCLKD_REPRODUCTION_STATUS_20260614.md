# CCLKD 复现消融综合状态

更新时间：2026-06-14 00:15 CST

## 概述

当前 CCLKD 复现工作分为两个独立分支：

1. **YOLO11n 分支**：面向 LADD formal baseline 协议的受控消融，用于论文主表对比
2. **YOLOv5x 分支**：面向 CCLKD 原文协议的严格复现，用于验证方法实现正确性

两个分支使用相同的 `formulation=paper` 核心实现，但训练协议不同，结果不能混用。

---

## 一、YOLO11n 分支（受控消融）

### 1.1 目标

在 LADD formal no-mosaic baseline 协议下评估 CCLKD 各组件增益，对标原文 Table 12。

### 1.2 协议

| 项 | 设置 |
|---|---|
| 数据集 | OGSOD-1.0 HBB, nc=3 |
| 模型 | YOLO11n |
| 输入尺寸 | 256 |
| Epoch | 400（原文）/ 800（LADD 主协议） |
| Batch | 64（YOLO11n）|
| 学习率 | lr0=0.01, lrf=0.01, cos_lr=True |
| 增强 | mosaic=0.0, mixup=0.0, close_mosaic=0, default Albumentations |
| 训练方式 | Online teacher-student joint training |
| Formulation | `--cclkd-formulation paper` |

### 1.3 消融矩阵

按原文 Table 12 映射：

| 消融名称 | LLD | FLD | RLD | PATM | CCL | 温度设置 |
|---|:---:|:---:|:---:|:---:|:---:|---|
| `lld` | ✓ |  |  |  |  | T=1.0 |
| `lld_fld` | ✓ | ✓ |  |  |  | T=1.0 |
| `lld_fld_rld` | ✓ | ✓ | ✓ |  |  | T=1.0 |
| `atkd` | ✓ | ✓ | ✓ | ✓ |  | PATM [0.5, 5.0] |
| `ccl_only` |  |  |  |  | ✓ | - |
| `full` | ✓ | ✓ | ✓ | ✓ | ✓ | PATM [0.5, 5.0] |

**重要修正**：
- 2026-06-08 之前的结果使用错误 CCL formulation（使用 DFL 回归特征），已作废
- 2026-06-08 15:18 之前的 `lld_fld_rld`/`atkd`/`full` 使用错误 RLD 量级（mean MSE 而非 Frobenius-squared），已作废
- 原文 Table 4 最佳配置为 `lambda_kd=1.0, lambda_cc=1.0`；此前代码默认 `--ccl-weight 0.5` 的结果只能作为诊断

### 1.4 启动入口

**正确入口**（LADD formal 协议）：
```bash
cd /path/to/LADD_public
EPOCHS=400 BATCH_SIZE=64 \
  comparison/code/launch_formal_online_cclkd_ablation_job.sh n <ablation> <seed> <gpu_id>
```

支持的 `<ablation>`：`lld`, `lld_fld`, `lld_fld_rld`, `atkd`, `ccl_only`, `full`

**错误入口**（不要使用）：
- `cclkd_reproduction/code/launch_cclkd_paper_repro_job.sh`（原文协议，不符合 LADD 受控对比）
- `cclkd_reproduction/code/launch_cclkd_n_ablation_job.sh`（旧诊断入口）

### 1.5 输出路径

```
runs_public/ogsod/hbb/formal_nomosaic_20260528/comparisons/online_cclkd/yolo11n/paper_ablation/
logs/formal_nomosaic_20260528/comparisons/online_cclkd_paper_ablation/
```

### 1.6 当前状态

**状态**：实现已修正，等待启动或正在运行

**启动前必查项**：
```bash
DRY_RUN=1 EPOCHS=400 BATCH_SIZE=64 \
  comparison/code/launch_formal_online_cclkd_ablation_job.sh n full 0 0
```

输出中必须同时出现：
- `--cclkd-formulation paper`
- `--mosaic 0.0`
- `--mixup 0.0`
- `--close-mosaic 0`
- `--batch 64`
- `--cos-lr`
- `--deterministic`

### 1.7 作废结果规则

以下结果**不能**进入论文主表：

1. Frozen-teacher CCLKD（不符合原文 online teacher-student 定义）
2. `--cclkd-formulation adapted`（只是 YOLO11 token-level 诊断）
3. 2026-06-08 之前使用错误 CCL 的结果
4. 2026-06-08 15:18 之前使用错误 RLD 量级的结果
5. `--ccl-weight 0.5` 的结果（原文 Table 4 使用 1.0）
6. 使用 CCLKD 原文 paper protocol 但误当作 LADD baseline 协议的结果
7. 只跑 full、没有完整 Table 12 结构消融的结果

---

## 二、YOLOv5x 分支（原文协议复现）

### 2.1 目标

严格按 CCLKD 原文 Table 2/5/12 协议复现，验证方法实现正确性。这是 **CCLKD reproduction gate**，不是 LADD 受控对比。

### 2.2 协议

| 项 | 设置 |
|---|---|
| 数据集 | OGSOD-1.0 HBB, nc=3 |
| 模型 | YOLOv5x (86.7M params) |
| 输入尺寸 | 256 |
| Epoch | 400 |
| Batch | 32（原文 Table 2）|
| 优化器 | SGD |
| 学习率 | 0.01 |
| Momentum | 0.937 |
| Weight Decay | 0.0005 |
| 增强 | YOLOv5 defaults + Mosaic 1.0 + MixUp 0.1 |
| 训练方式 | Online teacher-student joint training |
| Formulation | `--cclkd-formulation paper` |

### 2.3 原文 Table 5 目标

| Metric | Target | Loose Pass Threshold |
|---|---:|---:|
| Params | ~86M | - |
| AP50 | 80.9 | ≥ 78 |
| AP | 46.3 | ≥ 44 |

Per-class AP50 参考：Oil Tank 57.7, Bridge 87.2, Harbor 97.9

### 2.4 消融矩阵（原文 Table 12）

| 配置 | mAP50 | mAP |
|---|---:|---:|
| Baseline (Det-only) | 80.9 | 46.3 |
| LLD | 83.4 | 48.5 |
| LLD + FLD | 84.2 | 49.3 |
| LLD + FLD + RLD | 84.9 | 50.1 |
| LLD + FLD + RLD + PATM | 87.0 | 55.1 |
| CCL only | 85.9 | 54.4 |
| Full CCLKD | 88.7 | 57.3 |

### 2.5 当前实验状态（2026-06-14 00:13）

90 服务器 4 个 YOLOv5x b32/s0/400ep scaling-fix 主实验：

| 实验 | GPU | 进度 | Best AP50 | Best AP | ΔAP vs det-only | KD/det ratio | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| **Full CCLKD** | 0 | 190/399 | 0.60260 | 0.32503 | +0.00470 | 0.44153 | running |
| **ATKD-only** | 1 | 154/399 | 0.57474 | 0.30333 | +0.00723 | 0.06036 | running |
| **CCL-only** | 3 | 242/399 | 0.63011 | 0.35620 | +0.00492 | 0.40563 | running |
| **Det-only baseline** | 5 | 273/399 | 0.64690 | 0.36961 | baseline | 0.00000 | running |

**诊断诊断表**（最新 epoch）：

| 实验 | Student det loss | Teacher det loss | KD total | LLD | FLD | RLD | CCL | COP+ | Temp | Feature OK | NaN/Inf |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 0.05615 | 0.04675 | 0.79898 | 0.01054 | 0.03292 | 0.06164 | 0.69388 | 0.98487 | 2.85794 | ✓ | ✗ |
| ATKD-only | 0.06120 | 0.05009 | 0.11693 | 0.01117 | 0.03840 | 0.06736 | 0.00000 | 0.98302 | 2.87086 | ✓ | ✗ |
| CCL-only | 0.05461 | 0.04535 | 0.69395 | 0.00000 | 0.00000 | 0.00000 | 0.69395 | 0.98741 | 2.84218 | ✓ | ✗ |
| Det-only | 0.05216 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | - | - | - | ✗ |

### 2.6 当前观察

**训练稳定性**：✅ 所有实验无 NaN/Inf，feature capture 正常，COP 正样本率 >98%

**增益方向**：
- ✅ 所有 KD 方法在同 epoch 都显示 positive ΔAP
- ⚠️ 增益幅度较小：Full ΔAP=0.00470, ATKD ΔAP=0.00723, CCL ΔAP=0.00492
- ⚠️ Full 未明显优于 ATKD-only（epoch 150 时 full_minus_atkd_ap = -0.00027）
- ⚠️ CCL KD/det ratio 高（0.40563）但增益有限，可能效率偏低

**效率对比**：
- ATKD-only 增益/开销比最优：gain=0.00723, ratio=0.06036
- CCL-only 开销高但增益有限：gain=0.00492, ratio=0.40563
- Full 开销更高：ratio=0.44153

**Baseline gate 状态**：
- ⚠️ Det-only 当前 AP50=0.64690, AP=0.36961（273 epoch）
- ⚠️ 显著低于原文 Table 5 目标（AP50=80.9, AP=46.3）
- ⚠️ 即使 loose threshold（AP50≥78, AP≥44）也未达到

### 2.7 下一步行动

**P1 - 继续运行到对齐节点**：
- 等待 Full 和 ATKD-only 都到 epoch 200
- 生成 200/250/300/350/399 fixed epoch 对齐表
- 使用 `milestone_component_comparison.csv` 严格对齐评估

**P2 - Baseline gate 诊断**（如果 Det-only 400ep 仍低于 threshold）：
1. 检查数据 split、类别映射、YOLOv5 版本、评估脚本
2. 检查 hyp.yaml 是否完全对齐原文 Table 2
3. 比较 YOLOv5 v6.2 vs v7.0
4. 如需要，测试 batch64 作为 sanity check

**P3 - 可能的调优候选**（仅在同时满足以下条件时考虑）：
- ATKD-only 明显高于 det-only
- Full ≤ ATKD-only 或基本持平
- CCL-only weak gain
- Full weighted KD/det ratio 明显高于 ATKD-only

调优候选：CCL weight 0.25、CCL weight 0.5、KD warmup 10

**禁止行动**：
- ❌ 在 200/250 对齐前修改 loss
- ❌ 在 baseline gate 未明确前启动大规模 sweep
- ❌ 继续修改 CCLKD YOLO11 公式
- ❌ 实现 CoLD/CMDistill-style baseline
- ❌ 混用 YOLO11 结果和 YOLOv5 结果

### 2.8 证据归档

已归档路径：`cclkd_reproduction/yolov5_sanity/results/scalingfix_paper_components_400ep_20260613/`

每个 run 包含：
- `results.csv`
- `cclkd_yolov5_diagnostics.csv`
- `nohup_tail_300.log`
- `nohup_error_grep_tail.log`
- `run_meta.txt`
- `command.sh`
- `opt.yaml`
- `hyp.yaml`

未上传：checkpoint 权重、TensorBoard event、完整 nohup 日志

---

## 三、核心实现边界

### 3.1 CCLKD 方法定义（原文）

**训练模式**：Dual-branch online teacher-student joint training
- Teacher network：easy-to-detect modality (RGB)，有独立 detection loss
- Student network：hard-to-detect modality (SAR)，有独立 detection loss
- Inference：只使用 student/SAR

**核心模块**：

| 模块 | 含义 |
|---|---|
| **COP** | Category-Oriented Partition。基于 teacher 预测类别和 GT，将候选框划分为 target/non-target，提供 category mask |
| **PATM** | Prediction-guided Adaptive Temperature Mechanism。根据 teacher 对某类正样本的预测熵映射到类别温度 T_j，范围 [0.5, 5.0]，熵尺度初始化 5.0 |
| **LLD** | Logit-Level Distillation。对候选框定位分布做 category-aware KL，包括 target 与 non-target |
| **FLD** | Feature-Level Distillation。对候选框区域 feature map 做 box-aligned bilinear sampling，得到区域特征后蒸馏 |
| **RLD** | Relationship-Level Distillation。对同类候选框特征构造自相关矩阵，约束 student 保留 teacher 类内结构 |
| **CCL** | Category-Constrained Contrastive Learning。按类别构造正/负候选框对，class-balanced InfoNCE |

**Loss 结构**：
```
Total = Teacher_Det_Loss + Student_Det_Loss + ATKD_Loss + CCL_Loss
ATKD_Loss = LLD + FLD + RLD (with PATM temperature)
```

### 3.2 当前代码实现（`formulation=paper`）

| 原文要求 | 当前代码 |
|---|---|
| Online teacher-student | `CCLKDOnlineHBBTrainer`，teacher 参数 `requires_grad=True` |
| Teacher RGB detection loss | `teacher_det_loss` 独立计算并进入 total loss |
| Student SAR detection loss | `student_det_loss` 独立计算并进入 total loss |
| Teacher-side COP | `assign_source = teacher_main` |
| PATM | `_adaptive_temperature_class(...)`，按 class 正样本熵得到类别温度 |
| Target/non-target LLD | 正样本和采样负样本都进入 DFL KL |
| Box-aligned FLD/RLD/CCL | `_sample_box_features(...)` 使用 `grid_sample` 从 neck feature map 按 box 采样 |
| Class-balanced CCL | 按当前 mini-batch COP 类别频次做 `1/n_j` 归一权重 |
| FLD/RLD temperature | 默认 `--cclkd-fld-temperature-mode patm`，FLD 复用 class-wise PATM；RLD 按原文乘 T_j^2 |

### 3.3 YOLO11 适配说明

当前实现是在 YOLO11 HBB 检测头上对原文方法的近似：

1. **候选框映射**：原文基于 candidate-box level `p_box`/`p_cls`；YOLO11 使用 anchor/token assignment + DFL logits
2. **FLD 投影**：原文文字描述"区域特征经 1x1 conv 降维"；当前实现直接对 box-sampled neck feature 做 KL，未引入额外可学习投影
3. **CCL 对构造**：原文 Algorithm 2 以 target/non-target candidate-box pair 为单位；当前实现使用正候选框 teacher-student 相似度和负候选框相似度构造二分类 InfoNCE
4. **Determinism**：`grid_sample` CUDA backward 非完全 deterministic，PyTorch 给 warn-only 提示，不视为训练失败
5. **默认 formulation**：`train_cclkd_online_hbb.py` 默认仍是 `adapted`（兼容旧诊断），正式实验必须由 launcher 显式传入 `paper`

---

## 四、关键边界与禁止混用

### 4.1 两个分支的定位

| 项 | YOLO11n 分支 | YOLOv5x 分支 |
|---|---|---|
| **目的** | 受控对比：在统一协议下比较 CCLKD vs LADD/FGD/LD/HalluciDet | 方法复现：验证 CCLKD 原文实现正确性 |
| **协议** | LADD formal no-mosaic baseline | CCLKD 原文 Table 2 |
| **模型** | YOLO11n | YOLOv5x |
| **Batch** | 64 | 32 |
| **Epoch** | 400/800 | 400 |
| **增强** | mosaic=0.0, mixup=0.0 | mosaic=1.0, mixup=0.1 |
| **对标** | 论文主表（vs LADD/FGD/LD/HalluciDet）| 原文 Table 5/12 |
| **结果目录** | `comparison/` | `cclkd_reproduction/` |

### 4.2 禁止写法

**❌ 禁止**：
- "YOLO11 ablation reproduces CCLKD Table 12."
- "CCLKD Table 12 failed on YOLO11."
- "YOLOv5x6 is the paper backbone"（除非 x6 sanity 提供强证据）
- "This stage reproduces CoLD or CMDistill."
- 混用两个分支的结果数值

**✅ 推荐**：
- "We first verify the YOLOv5-X SAR baseline used by CCLKD."
- "YOLOv5-X SAR baseline is the CCLKD reproduction gate."
- "YOLO11 experiments are treated as adaptation diagnostics or extension comparisons."
- "Under LADD formal baseline protocol, we evaluate CCLKD components using YOLO11n."

### 4.3 结果使用规则

| 场景 | 使用分支 | 要求 |
|---|---|---|
| 论文 main table 对比 LADD/FGD/LD/HalluciDet | YOLO11n | formulation=paper, LADD formal 协议, 完整消融 |
| 验证 CCLKD 方法实现正确性 | YOLOv5x | formulation=paper, 原文协议, baseline gate pass |
| 补充材料：YOLO11 adaptation | YOLO11n | 明确标注为 extension comparison |
| 补充材料：原文协议复现 | YOLOv5x | baseline gate 通过后，完整 Table 12 消融 |

---

## 五、快速启动清单

### 5.1 YOLO11n 受控消融（用于论文主表）

```bash
cd /path/to/LADD_public

# 启动前 dry-run 检查
DRY_RUN=1 EPOCHS=400 BATCH_SIZE=64 \
  comparison/code/launch_formal_online_cclkd_ablation_job.sh n full 0 0

# 启动完整消融（seed 0, GPU 0-5）
for ablation in lld lld_fld lld_fld_rld atkd ccl_only full; do
  EPOCHS=400 BATCH_SIZE=64 \
    comparison/code/launch_formal_online_cclkd_ablation_job.sh n "$ablation" 0 <gpu_id>
done
```

### 5.2 YOLOv5x 原文复现（用于验证实现）

```bash
cd /path/to/LADD_public/cclkd_reproduction/yolov5_sanity

# 准备 YOLOv5 repo
bash scripts/prepare_yolov5_repo.sh

# 数据集检查
python tools/check_yolov5_ogsod_dataset.py \
  --data ../../shared/configs/datasets_public/ogsod1_sar_detect.yaml \
  --teacher-data ../../shared/configs/datasets_public/ogsod1_rgb_detect.yaml

# Gate 矩阵 dry-run
DRY_RUN=1 LAUNCH=0 bash scripts/launch_yolov5_sanity_matrix.sh gate

# 当前主实验已在 90 服务器运行，等待 200/250/300/350/399 对齐节点
```

---

## 六、文档索引

| 文档 | 路径 | 用途 |
|---|---|---|
| 原文 PDF | `/Users/yudongfang/Desktop/光sar/CCLKD__2026_GIS__Cross_Modal_Contrastive_Learning_Incomplete_Modalities.pdf` | CCLKD 方法原文 |
| 方法审计 | `CCLKD_PAPER_AUDIT_CN.md` | 原文方法定义、代码映射、作废规则 |
| 消融计划 | `ABLATION_PLAN_CN.md` | YOLO11n Table 12 消融映射 |
| 实现审计 | `IMPLEMENTATION_REVIEW.md` / `IMPLEMENTATION_REVIEW_CORRECTED.md` | 实现细节审计 |
| YOLOv5x README | `yolov5_sanity/README_CN.md` | YOLOv5x baseline gate 说明 |
| YOLOv5x 状态 | `yolov5_sanity/results/CCLKD_YOLOV5X_400EP_RUNNING_STATUS_20260613.md` | 最新运行快照 |
| 协议差距 | `diagnostics/20260607_protocol_gap/README_CN.md` | 原文协议与实际复现的差距分析 |

---

## 七、联系与协作

- **YOLO11n 分支负责人**：等待分配或记录
- **YOLOv5x 分支负责人**：等待分配或记录
- **代码审计**：已完成（2026-06-08）
- **下次评审节点**：YOLOv5x 200/250 epoch 对齐后

---

**最后更新**：2026-06-14 00:15 CST  
**状态摘要**：YOLO11n 分支等待启动，YOLOv5x 分支正在运行（190-273/399 epoch），baseline gate 待验证
