# LADD 与对比方法定义/实现对照

最后更新：2026-06-18

本文档作为当前 public 包的方法口径入口，回答三个问题：方法名应该怎么写、代码从哪里启动、当前实现与原论文或原方法定义有什么边界。实验结果、曲线和 run 状态不在这里展开，只引用本定义。

## 0. 使用规则

- 写论文主表或报告时，先按本文档确定方法名和有效入口。
- 论文实验应使用 `scripts/paper/*`，不要直接调用 raw formal launcher；raw launcher 仅作为底层兼容入口。
- Paper main table 必须通过 `paper_results/main_table_schema.csv` 和 `tools/paper_validate_main_table.py`。
- `official reproduction`、`paper-aligned reimplementation`、`style/adaptation` 不混用。
- 历史 run 只有在满足本文档入口和协议时才可进入当前主表；旧诊断结果必须标注为 archived diagnostic。
- CCLKD 原文复现和 LADD 统一协议受控对比是两条线，不能混用结果。

## 1. Formal OGSOD HBB 协议

当前论文主表固定使用 formal no-mosaic 协议：

```text
dataset = OGSOD-1.0 HBB
imgsz = 256
epochs = 800
cos_lr = true
mosaic = 0.0
close_mosaic = 0
default Albumentations
deterministic = true
batch = n/s:64, m/l:32, x:16
```

Baseline、LADD 和对比方法必须使用同容量、同 seed、同增强协议。历史 mosaic100、close@100、400ep 结果只能作为历史或附录，不能混入 LADD Probe-A nomosaic 主表。

Paper-facing 启动入口为 `scripts/paper/run_paper_baseline.sh`、`scripts/paper/run_paper_ladd_probea.sh`、`scripts/paper/run_paper_comparison_kd.sh`。底层 `baseline/`、`ladd/`、`comparison/code/` launcher 保留 no-mosaic 兼容模式，但 `PAPER_RUN=1` 会强制 nomosaic。

## 2. Baseline

Baseline 是单模态 YOLO11 HBB 检测训练：

| 项 | 定义 |
|---|---|
| SAR baseline | SAR-only YOLO11 detector，作为 LADD 与对比方法的学生/推理参照 |
| RGB baseline | RGB-only YOLO11 detector，作为 frozen teacher 或 teacher 初始化来源 |
| 入口 | `baseline/code/train_ogsod_baseline.py` |
| 标准脚本 | `baseline/scripts/run_formal_baseline.sh` |
| 结果口径 | `baseline/results/BASELINE_RESULTS_CN.md` |

## 3. LADD

LADD 的方法定义是 RGB-guided SAR object detection distillation：训练时使用 paired RGB/SAR，推理时只使用 SAR。核心思想是把 RGB teacher 特征拆成 SAR 可学习部分和 RGB 私有部分，再只蒸馏可学习部分。

每个 FPN/neck 特征层上的当前实现：

| 符号/模块 | 代码 | 含义 |
|---|---|---|
| `f_t -> z_t, u_t` | `TeacherPrivateAwareDecompositionBlock` 或 `TeacherResidualDecompositionBlock` | RGB teacher 特征分解为 transferable/common 与 private/unlearnable |
| `f_s -> z_s, r_s` | `student_split` / `StudentResidualProjBlock` | SAR student 特征分解为蒸馏空间与 SAR residual |
| reach / NRRL | `normalized_reachability_loss()` | 约束 student reachability 更接近 `z_t` 而不是 `u_t` |
| cap2 | `rank_d_neg_cap=2.0` | 在 rank loss 中截断负距离，避免继续奖励反平行坍缩 |
| KD | `_compute_kd_loss()` | 默认让 `z_s` 对齐 `z_t`，支持 MSE/contrastive/hybrid |
| historical aux losses | 已从当前 HBB LADD 代码删除 | sep/residual/private/mask-reg/path/proto/debug loss 不再作为可配置 surface |

代码入口：

| 文件 | 责任 |
|---|---|
| `ladd/code/train_ladd_hbb.py` | HBB 单阶段训练入口，支持 `a1/a2/b/c/b1/b2` |
| `ladd/code/src/teacher_student_decomposition_kd_hbb/model.py` | HBB LADD 模型与分解模块 |
| `ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py` | LADD loss、cap2、KD profile |
| `ladd/code/src/teacher_student_decomposition_kd_hbb/trainer.py` | phase freeze、paired RGB/SAR batch、BN-freeze、diagnostics |
| `ladd/scripts/launch_ladd_clean_a1b_job.sh` | clean A1B 主线 launcher |
| `ladd/scripts/launch_formal_ladd_job.sh` | 历史 formal no-mosaic A1-A2-B launcher |
| `ladd/code_versions/current_hbb/` | 当前 HBB 代码快照，正式部署前应与 `ladd/code/` 保持同步 |

当前正式 LADD-clean 主线固定为 Probe-A：

```text
latest same-protocol SAR/RGB baseline
+ A1 -> B only
+ RANK_D_NEG_CAP=2.0
+ sep/private/residual/debug auxiliary losses removed from current implementation
+ B detector + z_s -> z_t KD
+ B dynamic teacher decomposition/reach/taskL
+ B frozen student reachability probe + detached q_s reach path
```

阶段口径：

| 阶段 | 作用 | 当前主线 |
|---|---|---|
| A1 | 训练 teacher decomposition、reach adapter、teacher task head | 检测损失关闭；保留 reconstruction/reach/taskL |
| A2 | 历史诊断/消融 | 不属于 clean 主线 |
| B / Probe-A | 正式蒸馏与检测训练 | dynamic teacher core；frozen reach probe；保留 detector/KD/student reconstruction/teacher rec/reach/taskL |
| C | 可选后续 fine-tuning/诊断 | 不属于当前主线必要阶段 |

Probe-A / clean A1B 的完整定义、loss 开关和推荐实验协议见 `docs/ladd_clean_a1b_method_definition.md`。

推理边界：正式报告中 LADD 是 SAR-only inference；当前 `student_detect_mode=raw` 表示检测头读取 raw SAR 检测特征，不读取 RGB teacher。当前模型类仍保留 student split/decomposition 相关模块以支持训练诊断，部署/导出时是否剥离需要单独说明。

## 4. 受控对比方法

### 4.1 当前有效方法表

| 方法名应写作 | 有效入口 | 当前实现定义 | 必须注明的边界 |
|---|---|---|---|
| FGD-style / FGD-YOLO adaptation | `comparison/code/launch_formal_from_yolo_kd_job.sh fgd ...` 或 `launch_formal_transfer_kd_job.sh fgd ...` | `locked_fgd_yolo_gtbox_attention_20260618`：GT-box fg/bg feature loss + spatial/channel attention mask；无 legacy relation/normalization sweep | 非 MMDetection 官方逐行复现；official trainable global relation 未实现 |
| LD | `comparison/code/launch_formal_from_yolo_kd_job.sh ld ...` 或 `launch_formal_transfer_kd_job.sh ld ...` | `locked_ld_yolo_dfl_vlr_20260618`：raw YOLO DFL logits 的 foreground/main LD + teacher-quality VLR-style candidate LD | YOLO11/DFL 适配；shape 异常 hard fail |
| CMDistill-style / paper-aligned adaptation | `comparison/code/launch_formal_from_yolo_kd_job.sh cmdistill ...` 或 `launch_formal_transfer_kd_job.sh cmdistill ...` | `--comparison-kd-profile cmdistill`，PCCFD shallow/deep + deepest SLRD + full-output IBCLD | 非官方复现；未找到官方代码；正式 run 需 `KD_CALIBRATION_MODE=affine` |
| HalluciDet-YOLO adaptation | `python comparison/hallucidet/train_hallucidet.py ...` | `locked_hallucidet_yolo_official_unet_b64_20260618`：SAR -> replicate3 -> official-style U-Net(resnet34 ImageNet) -> frozen RGB YOLO detector -> detection loss | standalone 方法；不是 `--comparison-kd-profile`；不是 official Faster R-CNN/FCOS/RetinaNet 复现 |
| CCLKD online comparison | `comparison/code/launch_formal_online_cclkd_job.sh <n|s> <seed> <gpu>` | online teacher-student：RGB teacher 与 SAR student 同步训练，student det + teacher det + CCLKD loss | 受控对比入口，不是原文 400ep 协议；只能与 `cclkd_reproduction/` 的原文复现结果分开使用 |
| CCLKD paper-protocol reproduction | `cclkd_reproduction/code/launch_cclkd_paper_repro_job.sh` 和 `cclkd_reproduction/yolov5_sanity/` | 尽量对齐 CCLKD 原文协议和 online 方法定义 | 只用于回答“能否复现 CCLKD 原文”，不直接替代 LADD formal 主表 |

### 4.2 Frozen-teacher KD profile 的共同设置

FGD、LD、CMDistill 共享 LADD HBB trainer，但 launcher 会把 LADD 主 loss 关掉，使其成为普通 frozen RGB teacher -> SAR student KD：

```text
COMPARISON_KD_PROFILE=<fgd|ld|cmdistill>
PROFILE_KD_REPLACE_BASE=1
STUDENT_BRANCH_MODE=raw
TEACHER_FEATURE_MODE=raw
LAMBDA_REACH=0
LAMBDA_REC=0
LAMBDA_TASKL=0
ALPHA_S_REC=0
```

sep/aux/debug loss 参数已从当前 HBB trainer 删除，不再需要也不能通过配置打开。

因此这些方法不应被描述为“LADD 分解方法的变体”，而应描述为同协议下的 detector KD baselines。

FGD/LD/CMDistill-style 的 raw launcher 支持 nomosaic 和 mosaic100 兼容分支；论文主表只能使用 paper wrapper 强制的 nomosaic。`run_paper_comparison_kd.sh` 默认使用 same-seed SAR baseline 初始化的 transferred KD 设置；from-YOLO pretrain 结果必须单独标注 `init_type=from_yolo_pretrain`，不能和 transferred KD 混进同一 comparison 表。

### 4.3 CCLKD 的两个入口不能混用

`--comparison-kd-profile cclkd` 仍存在于 `train_ladd_hbb.py`，但它只是 loss 级/兼容性部件，不是当前有效的 CCLKD 正式入口。CCLKD 若进入主表，应使用 online teacher-student 入口：

- 原文协议复现：`cclkd_reproduction/`
- LADD formal 受控对比：`comparison/code/launch_formal_online_cclkd_job.sh`

## 5. 已作废或需降级的口径

| 旧口径 | 当前处理 |
|---|---|
| `--comparison-kd-profile hallucidet_style` | 已移除；旧结果只能作为历史 diagnostic |
| `hallucidet`/`HalluciDet-style` feature/response/margin KD | 不再作为当前 HalluciDet 方法发布 |
| HalluciDet custom U-Net standalone | 已归档；当前 active 代码只保留 official-style U-Net |
| FGD normalization/assigner-mask/batch-relation sweep | 已删除 active 可执行入口；当前 FGD 实现锁定 |
| LD foreground-only/topk/VLR-weight sweep | 已从 active CLI/env/loss 配置面移除；当前 LD 实现锁定 |
| 2026-06-10 前 FGD/LD 结果 | 修复前语义不同，不能代表当前实现 |
| frozen-teacher CCLKD formal run | 不符合 CCLKD 原文 online 定义，不能写作 CCLKD official/paper reproduction |
| 双卡 4090 上 `nc=5` yaml 的 CCLKD/HalluciDet 等结果 | HBB OGSOD 应为 `nc=3`，相关 run 作废 |
| close@100 / 400ep / 旧高学习率 LADD | 可作历史诊断，不替代当前 nomosaic Probe-A 主线 |

## 6. 文档责任分工

| 文档 | 责任 |
|---|---|
| 本文档 | 方法定义、实现入口、命名边界 |
| `docs/method/METHOD_OVERVIEW_CN.md` | LADD 方法叙事与机制理解 |
| `docs/experiments/LADD_MAINLINE_STANDARD_CN.md` | LADD formal 主线训练规范 |
| `comparison/METHOD_CODE_MAP_CN.md` | 对比方法代码映射速查 |
| `comparison/IMPLEMENTATION_REVIEW_CN.md` | 对比方法实现审查与历史修复记录 |
| `comparison/*/README.md` | 单个对比方法说明 |
| `cclkd_reproduction/README.md` | CCLKD 原文复现协议 |
| `baseline/results/`、`ladd/results/`、`docs/experiments/*` | 结果、曲线、实验状态 |

后续整理时，结果文档只应引用本文档的方法名和入口；如果代码实现变化，先更新本文档和 `comparison/METHOD_CODE_MAP_CN.md`，再更新结果口径。
