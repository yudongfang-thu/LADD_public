# 对比方法复核意见响应

最后更新：2026-06-04

本文说明 2026-06-04 第二轮外部代码复核意见中哪些建议被采纳、哪些未被采纳，
以及当前实现仍然保留的便携适配边界。结论基于当前 public 代码、仓库内
Ultralytics Detect head 的实际返回语义，以及 FGD/LD 官方代码。

## 1. 处理结论

| 方法 | 复核意见 | 处置 |
|---|---|---|
| FGD | channel attention 应由 softmax 改为 sigmoid，并移除通道缩放 | 不采纳；该判断与 FGD 官方实现不符 |
| FGD | 当前 relation loss 不是官方 Global KD | 采纳说明；继续标记 `FGD-style`，不在便携 profile 中强行新增可训练 context 模块 |
| LD | teacher 为 eval 时可能只返回解码框，导致 LD 静默为零 | 风险判断不适用于当前 Ultralytics，但采纳 fail-fast 防御 |
| LD | 应使用独立温度参数 | 采纳；新增 `ld_temperature=10.0`，对齐 LD 官方配置 |
| CCLKD | feature/logit 项需要独立权重 | 采纳；新增 `cclkd_logit_weight=1.0` |
| CCLKD | 不应复用 MMANet token 上限 | 采纳；新增 `cclkd_max_tokens=512` |
| CCLKD | top-K 高置信采样存在偏差 | 采纳并改进；改为类别分层随机采样，而非无类别约束的全局随机采样 |
| HalluciDet-style | 当前实现可接受 | 保持不变 |

## 2. 上轮评价中的事实性错误

### 2.1 FGD 官方通道注意力不是 sigmoid

FGD 官方实现的 `get_attention()` 对空间与通道注意力均使用 softmax：

```text
S_attention = H * W * softmax(spatial_map / temp)
C_attention = C     * softmax(channel_map / temp)
```

因此以下建议没有应用：

- `channel_att` 从 softmax 改为 sigmoid；
- 移除 `* channels`；
- 移除 `* (height * width)`。

这些缩放用于保持注意力均值约为 1；尤其移除 `* channels` 会直接改变 feature
loss 量级。当前修正保留官方 softmax 与缩放，并补充官方默认
`fgd_temperature=0.5`。

FGD 官方代码：
[`yzd-v/FGD/mmdet/distillation/losses/fgd.py`](https://github.com/yzd-v/FGD/blob/master/mmdet/distillation/losses/fgd.py)

### 2.2 Teacher eval 不会让当前 LD 获得解码框

当前 teacher 确实处于 `eval()`，但本仓库 Ultralytics Detect head 在非 export
的 eval forward 中返回：

```text
(decoded_predictions, raw_predictions_dict)
```

`raw_predictions_dict["boxes"]` 仍是 `[B, 4*reg_max, N]` 的原始 DFL logits。
`_unwrap_teacher_preds()` 明确取 tuple 第二项，因此 LD 当前不会因为 teacher
处于 eval 模式而静默失效。

尽管该问题目前不存在，原实现遇到 shape 不匹配时静默置空仍不够安全。本次已改为：

- `ld` profile 无法获得匹配 DFL logits 时立即抛出 `RuntimeError`；
- `_ld_style_loss()` 收到缺失、错形或非 DFL tensor 时立即失败；
- 不再复用 `crosskd_temperature`，新增 `ld_temperature=10.0`。

LD 官方配置使用 `KnowledgeDistillationKLDivLoss(..., T=10)`，因此“T=1 才与
原文一致”的说法也不准确。

LD 官方配置：
[`HikariTJU/LD/configs/ld/ld_r50_gflv1_r101_fpn_coco_1x.py`](https://github.com/HikariTJU/LD/blob/master/configs/ld/ld_r50_gflv1_r101_fpn_coco_1x.py)

## 3. 为什么没有完全实现官方 FGD

当前 relation 项仍是 batch-level cosine relation matrix MSE，不是 FGD 官方的
Global KD。官方 Global KD 使用可训练的 spatial context pooling 和 channel-add
模块，也不是复核意见所描述的 feature Gram matrix。

完整迁移需要新增、注册和优化每个尺度的可训练 context 模块，并处理 checkpoint
兼容性。这会把便携 comparison profile 变成独立模型结构，明显扩大工程边界。
因此当前选择是：

- 保留 teacher spatial/channel attention、GT fg/bg 权重和便携 relation 近似；
- 明确写作 `FGD-style (teacher-attention weighted)`；
- 不声称严格复现官方 FGD。

## 4. CCLKD 修正

本次新增：

```text
cclkd_logit_weight = 1.0
cclkd_max_tokens   = 512
```

feature MSE 与 logit KL 现在可以独立调权。对比 token 超过上限时，先按类别分层
随机抽取，再用剩余额度随机补齐。这样既避免旧 top-K 只保留高置信区域，也避免
全局随机采样被多数类别完全占据。

CCLKD 没有公开可运行官方代码，当前仍缺完整 relationship-level distillation，
并将 candidate-box CCL 适配为 assigned anchor-token CCL。因此继续写作
`CCLKD-style portable implementation`。

## 5. 当前默认参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `fgd_temperature` | 0.5 | 对齐 FGD 官方 attention temperature |
| `ld_temperature` | 10.0 | 对齐 LD 官方 KL temperature |
| `cclkd_logit_weight` | 1.0 | 独立控制 logit KL |
| `cclkd_max_tokens` | 512 | CCLKD 专属类别分层采样上限 |

## 6. 正式运行要求

1. FGD、LD、CCLKD 修正前结果不得进入当前主表。
2. LD smoke 必须确认 teacher/student DFL logits 都为 `[B, N, 4*reg_max]`；
   当前代码会在不匹配时直接终止。
3. CCLKD smoke 需要记录 feature、logit、contrastive 三项量级，确认默认权重合理。
4. FGD 和 CCLKD 在论文中必须保留 `-style` 标记。
5. 本次修改首先落在 public 审计包；启动正式实验前必须同步到私有工作区和服务器，
   并再次执行 hash/`--help`/短 smoke 检查。
