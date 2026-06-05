# CCLKD Paper-Structured Reimplementation

论文：*Cross-modal contrastive learning-based object detection under incomplete modalities*，
Geo-spatial Information Science，2026，DOI
[`10.1080/10095020.2026.2633014`](https://doi.org/10.1080/10095020.2026.2633014)。

原文 PDF 已归档在：

```text
paper/CCLKD__2026_GIS__Cross_Modal_Contrastive_Learning_Incomplete_Modalities.pdf
```

该 DOI 的 version of record license 为 `CC BY 4.0`；见 `paper/README_CN.md`。

## 当前实现

使用统一 HBB profile：

```text
../ladd/code/train_ladd_hbb.py --comparison-kd-profile cclkd
../ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py::_cclkd_style_loss
```

2026-06-05 后的 paper-structured implementation 包含：

- **COP**：基于 RGB teacher 分类 logits 的 dominant class，与 GT assigned label 一致时构成类别正样本 mask；
- **ATKD / LLD**：按类别正样本 teacher entropy 映射自适应温度，执行分类 logit KD 与 YOLO11 DFL spatial-distribution KD；
- **ATKD / FLD**：在类别正样本 token 上执行 teacher-student feature distribution KD；
- **ATKD / RLD**：在同类 token 特征上构造 self-correlation matrix 并对齐；
- **CCL**：按类别频次反比加权，对 target / non-target spatial distributions 做 teacher-student contrastive alignment；
- 最多 512 token 的类别内随机采样显存保护。

`--cclkd-base-temperature` 作为旧 launcher 兼容参数保留；paper-structured 路径实际使用
`--cclkd-temperature-min / --cclkd-temperature-max / --cclkd-entropy-scale` 控制论文式
entropy temperature mapping。

论文没有公开可运行代码。当前实现仍是 YOLO11 适配版：论文中的 YOLOv5 candidate
box / objectness / regional feature extraction没有一一同构公开实现；本仓库用 YOLO11
DFL raw logits 表示 spatial distribution，用 per-level dense token feature 近似
candidate region feature。teacher 分支仍来自给定 RGB teacher 权重，不是完全复刻原文
“joint teacher-student online distillation”训练器。因此写作时只能称为
`CCLKD paper-structured reimplementation` 或 `CCLKD-style reimplementation`，
不能声称官方严格复现。

## 状态

2026-06-05 发现双卡 4090 部署时 active dataset yaml 误为 `nc=5`，而正式 OGSOD
HBB 协议应为 `nc=3`。因此此前双卡 4090 上的 CCLKD smoke 和 formal partial run
全部作废；相关材料只保留在 `reproduction_issue_20260605/` 作为问题证据。

修正后的实现已通过本地 `py_compile` 和 shell `bash -n` 静态检查。当前本地
`/opt/homebrew/bin/python3` 未安装 `torch`，因此 `--help` 导入和 CPU 合成张量
loss 检查需要在服务器环境或装有 torch 的本地环境中补做。尚未重新做 GPU smoke，
也未启动新正式实验。

## 复现实验问题记录

2026-06-05 新增 CCLKD-style 复现实验问题证据包：

```text
reproduction_issue_20260605/
```

当前观察到：90 服务器上尽量贴近原文的 YOLO11s / 400ep CCLKD-style run 已完整跑完，
但 mAP50-95 为 `0.48567`，低于已有 SAR YOLO11s 400ep baseline 的 `0.53255`。
为了排除 baseline protocol 不完全一致的问题，已在 90 上启动完全同协议 SAR-only baseline
并持续记录 partial 结果。双卡 4090 的 formal comparison CCLKD partial 也一并收录。
