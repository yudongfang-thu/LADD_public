# OGSOD 正式对比实验

最后更新：2026-06-03

本文档涵盖对比实验的计划、方法选择、实现细节和已完成结果。

## 1. 协议约束

所有 controlled comparison 统一使用：

```text
OGSOD-1.0 HBB, YOLO11 同容量, SAR-only inference
800ep, cos_lr, full no-mosaic, default Albumentations
同 seed SAR baseline + 同 seed RGB teacher
```

## 2. 主表方法选择

主表保留 5 个对比方法：3 个通用 KD + 2 个跨模态方法。CoLD 单独在 90 服务器尽力对齐原文复现；其他四个方法直接迁移到 117，并按 3 seed 开跑。

| 名额 | 方法 | 状态 |
|---|---|---|
| 通用 KD #1 | **FGD** (CVPR 2022) | YOLO11n seed0 已完成，与 baseline 持平；117 补三 seed |
| 通用 KD #2 | **CrossKD-style** (CVPR 2024) | YOLO11n seed0 已完成，无正向提升；117 补三 seed |
| 通用 KD #3 | **LD** | 经典 logit distillation；HBB profile 已实现，117 可补三 seed |
| 跨模态 KD #1 | **CoLD** | 同任务 anchor；90 服务器尽力对齐原文慢跑 |
| 跨模态 KD #2 | **HalluciDet-style** | 训练期 RGB privileged information，推理期 SAR-only；117 三 seed |

已完成 FGD/CrossKD 结果 (4090D, YOLO11n, 800ep)：

| 方法 | best AP | vs SAR baseline |
|---|---|---|
| SAR baseline | 0.55916 | — |
| FGD | 0.55867 | -0.00049 |
| CrossKD-style | 0.55764 | -0.00152 |

两者均未超越 baseline，但仍作为受控负结果保留。新策略下不再只停留于 seed0：FGD、CrossKD-style、LD、HalluciDet-style 四个非 CoLD 方法统一迁移到 117 跑三 seed。

4090D 已关闭，已完成结果已拉回本地：

- 摘要：`remote_snapshots/4090d_formal_kd_20260602/SUMMARY_CN.md`
- 归档根目录：`remote_snapshots/4090d_formal_kd_20260602/root/autodl-tmp/LADD`
- 范围：YOLO11n/s SAR/RGB baseline、YOLO11n FGD、YOLO11n CrossKD-style 的 `results.csv`、`best.pt`、`last.pt`、日志和环境元数据；未拉取数据集、cache、图片和 optimizer 中间 checkpoint。

## 3. 已实现方法入口

当前已有四个迁移型 KD profile 接入 HBB 对比框架 (`src/teacher_student_decomposition_kd_hbb/loss.py`)：

| Profile | 来源 | 定位 |
|---|---|---|
| `fgd` | FGD, CVPR 2022 | foreground/background feature KD + batch relation KD |
| `mgd` | MGD, ECCV 2022 | random spatial mask feature recovery |
| `c2kd` | C2KD, CVPR 2024 | teacher logits target/non-target class KD |
| `mmanet` | MMANet, CVPR 2023 | foreground token relation + class margin |

仍需新增：

| Profile | 定位 | 实现状态 |
|---|---|---|
| `ld` | 经典检测 logit KD，低计算量 baseline | 已实现 |
| `hallucidet_style` | privileged RGB feature hallucination，SAR-only inference | 待实现 |

训练入口（B-only transferred KD）：

```bash
python3 tools/train_ladd_hbb.py --comparison-kd-profile fgd ...
# 或使用 launcher:
bash scripts/ogsod_public/formal_nomosaic_20260528/launch_formal_transfer_kd_job.sh fgd n 0 0
```

## 4. from-yolo-pretrain 协议

FGD 和 CrossKD-style 使用 `from-yolo-pretrain` 协议：student 初始化为 `yolo11n.pt`（COCO pretrain），teacher 使用同 seed RGB baseline。完整 800 epoch 训练。

## 5. 不进入主表的结果

| 类别 | 定位 |
|---|---|
| CoLD 原文 reported AP=56.7 | external anchor |
| CCLKD/FED/GaLD reported | external reported table |
| 旧 YOLOv5x CoLD 复现 | 复现诊断 |
| 旧 400ep / close@100 实验 | 历史参考 |

## 6. 后续执行策略

服务器分工：

- 117 服务器：LADD 主实验、消融，以及 FGD/CrossKD-style/LD/HalluciDet-style 四个非 CoLD 对比方法。
- 90 服务器：CoLD 复现/迁移，尽力对齐原文，接受慢跑。

117 启动前置：

1. 同步完整 HBB LADD 代码和 `yolo/ultralytics` 本地 patch；
2. 确认 `tools/train_ladd_hbb.py --help` 可运行；
3. 确认 OGSOD YAML 指向 `/home/xmu/djd/ladd/data/OGSOD-1.0/`；
4. 确认 baseline 权重路径完整。当前 117 已有 YOLO11n/s 三 seed SAR/RGB best.pt 且 md5 匹配源端记录；YOLO11m seed0 SAR/RGB best.pt 已在 117；YOLO11l seed0 尚未迁移到 117。

### 6.1 当前条件：可做与需补

| 容量 | 117 baseline 条件 | 非 CoLD 对比可做项 | 需补 |
|---|---|---|---|
| YOLO11n | SAR/RGB 0/42/123 已齐且 md5 匹配 | FGD/CrossKD-style/LD 三 seed；HalluciDet-style 待实现 | 同步 HBB 代码；实现 HalluciDet-style |
| YOLO11s | SAR/RGB 0/42/123 已齐；重拷贝后 md5 已与源端记录一致 | FGD/CrossKD-style/LD 三 seed；HalluciDet-style 待实现 | 同步 HBB 代码；实现 HalluciDet-style |
| YOLO11m | seed0 SAR/RGB baseline 已在 117 | seed0 FGD/CrossKD-style/LD；HalluciDet-style 待实现 | 补 seed42/123 SAR/RGB；实现 HalluciDet-style |
| YOLO11l | seed0 SAR/RGB baseline 已有，但未迁移到 117 | 暂不能启动 117 l 对比 | 迁移 seed0；补 seed42/123 SAR/RGB |

主容量轴为 YOLO11n/s/m/l；YOLO11x 仅作为附录容量趋势参考。

第二个跨模态方法当前推荐：

| 方法 | 判断 |
|---|---|
| HalluciDet-style | 首选；开源、检测任务、训练期 privileged RGB 与 SAR-only 推理逻辑吻合 |
| PFGF-style / TIRDet-style | 备选；更偏 pseudo-visible / translation，工程更重 |
| CCLKD-lite | 暂不作为主候选；可作为附录或实现灵感 |

HalluciDet-style 迁移边界：

- 不声称官方严格复现，命名为 `HalluciDet-style privileged modality hallucination`；
- 训练期使用 paired RGB/SAR，RGB teacher 或 RGB feature extractor 只提供 privileged supervision；
- 推理期只输入 SAR，RGB 分支和 teacher 全部移除；
- 优先做 YOLO11n 三 seed controlled comparison；稳定后随 baseline 补齐扩展到 s/m/l。
