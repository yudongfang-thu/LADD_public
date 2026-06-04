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

代码已接入，尚需短 smoke 后再启动正式实验。
