# CCLKD-style

论文：*Cross-modal contrastive learning-based object detection under incomplete modalities*，
Geo-spatial Information Science，2026，DOI
[`10.1080/10095020.2026.2633014`](https://doi.org/10.1080/10095020.2026.2633014)。

## 当前实现

使用统一 HBB profile：

```text
../ladd/code/train_ladd_hbb.py --comparison-kd-profile cclkd
../ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py::_cclkd_style_loss
```

当前 portable implementation 包含：

- teacher-confidence adaptive feature/logit distillation；
- GT-assigned foreground anchor 上的 category-constrained cross-modal contrastive loss；
- feature/logit 独立权重；
- 最多 512 foreground token 的类别分层随机采样显存保护。

论文没有公开可运行代码。当前缺少完整 relationship-level distillation，并把
candidate-box CCL 近似为 assigned anchor-token CCL，因此必须写作 `CCLKD-style`，
不能声称严格复现。

## 状态

最终实现已通过真实 GPU smoke，并已启动正式实验。

## 复现实验问题记录

2026-06-05 新增 CCLKD-style 复现实验问题证据包：

```text
reproduction_issue_20260605/
```

当前观察到：90 服务器上尽量贴近原文的 YOLO11s / 400ep CCLKD-style run 已完整跑完，
但 mAP50-95 为 `0.48567`，低于已有 SAR YOLO11s 400ep baseline 的 `0.53255`。
为了排除 baseline protocol 不完全一致的问题，已在 90 上启动完全同协议 SAR-only baseline
并持续记录 partial 结果。双卡 4090 的 formal comparison CCLKD partial 也一并收录。
