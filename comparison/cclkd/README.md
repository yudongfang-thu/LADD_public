# CCLKD Paper-Structured Reimplementation

论文：*Cross-modal contrastive learning-based object detection under incomplete modalities*，
Geo-spatial Information Science，2026，DOI
[`10.1080/10095020.2026.2633014`](https://doi.org/10.1080/10095020.2026.2633014)。

原文 PDF 和论文协议复现清单已移到独立目录：

```text
../../cclkd_reproduction/
```

该 DOI 的 version of record license 为 `CC BY 4.0`；见
[`../../cclkd_reproduction/paper/README_CN.md`](../../cclkd_reproduction/paper/README_CN.md)。

## 当前实现

使用统一 HBB profile：

```text
../ladd/code/train_ladd_hbb.py --comparison-kd-profile cclkd
../ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py::_cclkd_style_loss
```

2026-06-05 后的 paper-structured implementation 包含：

- **COP**：基于 RGB teacher 分类 logits 的 dominant class，与 GT assigned label 一致时构成类别正样本 mask；
- **ATKD / LLD**：按类别正样本 teacher entropy 映射自适应温度，只对 YOLO11 DFL spatial-distribution 做 localization KD，不做分类 logit KL；
- **ATKD / FLD**：在类别正样本 token 上执行 teacher-student feature MSE；
- **ATKD / RLD**：在同类 token 特征上构造 `R^T R / n` 的 feature-dimension correlation matrix 并对齐；
- **CCL**：按类别频次反比加权，对 target / non-target spatial distributions 做 teacher-student contrastive alignment；
- 最多 512 token 的类别内随机采样显存保护。

`--cclkd-base-temperature` 作为旧 launcher 兼容参数保留；paper-structured 路径实际使用
`--cclkd-temperature-min / --cclkd-temperature-max / --cclkd-entropy-scale` 控制论文式
entropy temperature mapping。

论文没有公开可运行代码。当前实现仍只是 loss 级 YOLO11 适配版：论文中的 YOLOv5
candidate box / objectness / regional feature extraction 没有一一同构公开实现；本仓库用
YOLO11 DFL raw logits 表示 spatial distribution，用 per-level dense token feature
近似 candidate region feature。更重要的是，当前 trainer 仍是 frozen RGB teacher，
不是论文定义的 “joint teacher-student online distillation”。因此当前代码不能作为
CCLKD 官方条件复现实验，也不应进入正式对比主表。

## 状态

2026-06-05 发现双卡 4090 部署时 active dataset yaml 误为 `nc=5`，而正式 OGSOD
HBB 协议应为 `nc=3`。因此此前双卡 4090 上的 CCLKD smoke 和 formal partial run
全部作废；原始问题证据包已从精简 public 分支移除，仅在历史提交中可追溯。

当前 loss 级实现已通过本地 `py_compile` 和 shell `bash -n` 静态检查。当前本地
`/opt/homebrew/bin/python3` 未安装 `torch`，因此 `--help` 导入和 CPU 合成张量
loss 检查需要在服务器环境或装有 torch 的本地环境中补做。尚未重新做 GPU smoke，
也未启动新正式实验。

正式 CCLKD 路线分两步：

1. 先在 [`../../cclkd_reproduction/`](../../cclkd_reproduction/) 中按原文协议复现：
   YOLO11s / YOLO11n、400 epoch、paper-matched augmentation、online
   teacher-student joint training。
2. 复现确认后，再回到 `comparison/` 中按 LADD 统一受控协议运行 CCLKD 对比。

完成 online trainer 前，当前 frozen-teacher loss 组件只保留为实现部件，不能单独作为
CCLKD 官方条件复现。

## 后续要求

CCLKD 若进入论文主表，必须先补 online teacher-student trainer，并按原文条件完成
YOLO11s / YOLO11n、400 epoch 复现实验。当前 frozen-teacher loss 组件不能单独作为
CCLKD 官方条件复现。
