# 对比方法复核意见响应

最后更新：2026-06-04

本文说明 2026-06-04 第二、三轮外部代码复核意见中哪些建议被采纳、哪些未被采纳，
以及当前实现仍然保留的便携适配边界。结论基于当前 public 代码、仓库内
Ultralytics Detect head 的实际返回语义、实际 smoke 输出，以及 FGD/LD 官方论文和代码。

## 1. 处理结论

| 方法 | 复核意见 | 处置 |
|---|---|---|
| FGD | channel attention 应由 softmax 改为 sigmoid，并移除通道缩放 | 不采纳；该判断与 FGD 官方实现不符 |
| FGD | 当前 relation loss 不是官方 Global KD | 采纳说明；继续标记 `FGD-style`，不在便携 profile 中强行新增可训练 context 模块 |
| LD | teacher 为 eval 时可能只返回解码框，导致 LD 静默为零 | 风险判断不适用于当前 Ultralytics，但采纳 fail-fast 防御 |
| LD | 应使用独立温度参数 | 采纳；新增 `ld_temperature=10.0`，对齐 LD 官方配置 |
| CCLKD | feature/logit 项需要独立权重 | 采纳；新增 `cclkd_logit_weight=1.0` |
| CCLKD | 不应复用旧 relation-KD token 上限 | 采纳；新增 `cclkd_max_tokens=512` |
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
- 不再复用旧通用 KD 温度参数，新增 `ld_temperature=10.0`。

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

## 7. 第三轮复核意见响应

第三轮意见再次建议将 FGD 通道注意力由 softmax 改为 sigmoid，并将 FGD 官方
Global KD 描述为图内 spatial Gram/self-similarity；同时认为 teacher 在 eval
模式下无法向 LD 提供原始 DFL logits。对官方论文、官方代码和当前运行时进行
交叉核对后，这三项判断均不成立，因此没有据此继续修改代码。

### 7.1 逐项结论

| 第三轮意见 | 结论 | 当前处置 |
|---|---|---|
| FGD 通道注意力应使用 sigmoid | 错误；官方代码使用带温度的 softmax 并乘以 `C` | 保持当前 softmax |
| FGD 空间注意力乘以 `H*W` 只是量纲适配 | 错误；该缩放是官方实现的一部分 | 保持当前缩放 |
| FGD Global KD 是单图内 spatial Gram matrix | 错误；官方使用带可训练变换的 GcBlock-style context pooling | 当前 relation 仍标记为近似的 `FGD-style` |
| teacher eval 时 LD 只能获得 decoded boxes | 不适用于当前 Ultralytics；eval 返回 decoded 结果和 raw dict | 保持 teacher eval，并保留 fail-fast |
| LD 首次正式运行前应 smoke | 合理；用于发现远程服务器代码或运行时不一致 | 每台服务器首次运行前执行 |
| CCLKD 的 `remaining < 0` 是死代码 | 对当前 OGSOD 默认配置几乎不可达，但对通用多类别配置并非死代码 | 保留通用保护分支 |

### 7.2 FGD 官方注意力使用 softmax，而不是 sigmoid

FGD 官方实现 `get_attention()` 的关键代码为：

```text
S_attention = H * W * F.softmax((fea_map / temp).view(N, -1), dim=1)
C_attention = C     * F.softmax(channel_map / temp, dim=1)
```

因此，当前实现中的空间 softmax、通道 softmax、`H*W` 缩放和 `C` 缩放都有
官方代码依据。将通道注意力改为 sigmoid 或移除缩放，反而会主动偏离官方实现，
并改变 feature loss 的量级。第三轮意见将 sigmoid 归因于 FGD 原文，没有得到
官方论文与官方代码支持。

FGD 官方论文：
[`Focal and Global Knowledge Distillation for Detectors`](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.pdf)

FGD 官方代码：
[`yzd-v/FGD/mmdet/distillation/losses/fgd.py`](https://github.com/yzd-v/FGD/blob/master/mmdet/distillation/losses/fgd.py)

### 7.3 FGD 官方 Global KD 不是 Gram matrix

FGD 官方 Global KD 使用 GcBlock-style 模块：先通过可训练的 mask convolution
和 spatial softmax 聚合单图全局上下文，再经过可训练的 channel-add 变换回注特征。
它不是第三轮意见描述的“完整 feature map spatial self-similarity/Gram matrix”。

当前 public 实现使用 batch-level cosine relation matrix MSE，确实不是官方
Global KD；这一差异已经明确记录，因此仍写作 `FGD-style`。没有实现官方
GcBlock 的原因不是将 Gram matrix 漏写，而是完整实现需要新增可训练模块、参数注册、
优化器接入和 checkpoint 兼容逻辑，超出了当前便携 comparison profile 的边界。

### 7.4 LD eval 数据流已经实际验证

当前 Ultralytics Detect head 在非 export 的 eval forward 中返回：

```text
(decoded_predictions, raw_predictions_dict)
```

`_unwrap_teacher_preds()` 取 tuple 的第二项，因而 `raw_predictions_dict["boxes"]`
仍是原始 DFL logits。实际 smoke 输出为：

```text
teacher_training False
decoded    (1, 7, 1344)
raw_boxes  (1, 64, 1344)
raw_scores (1, 3, 1344)
```

这证明 teacher 保持 eval 时，当前代码能够获得原始 DFL logits。不能把“切换
teacher 到 train 模式”作为 LD 报错后的通用修复；这样可能更新 teacher 的 BN
统计并改变教师行为。若远程 smoke 触发 fail-fast，应优先检查：

1. 私有工作区、public 包与服务器代码是否同步；
2. 服务器 Ultralytics head 返回协议是否不同；
3. teacher 是否处于 export 路径或使用了不同检测头；
4. teacher/student 的 `reg_max` 与输出形状是否匹配。

短 smoke 仍然必要，但其作用是发现服务器运行时不一致，而不是证明 eval 模式有错。

### 7.5 CCLKD 保护分支并非通用死代码

在 OGSOD 当前类别数远小于 `cclkd_max_tokens=512` 的条件下，类别分层采样后的
`remaining < 0` 分支几乎不会触发。但若将代码用于类别数大于 token 上限的数据集，
`per_class` 至少为 1，首次按类保留的 token 数可能超过上限，因此该分支仍有意义。
保留它不会改变当前实验行为，也能避免通用配置下突破矩阵规模上限。

## 8. 第三轮复核后的最终决定

1. 不将 FGD 通道注意力改为 sigmoid，不移除官方 attention 缩放。
2. 不把当前 FGD relation 项描述为官方 Global KD，继续使用 `FGD-style`。
3. 不把 teacher 切换到 train 模式；保留 eval、raw-output unwrap 与 fail-fast。
4. 每台远程服务器首次运行 LD 前执行短 smoke，检查原始 DFL shape 和非零 loss。
5. 保留 CCLKD 的 `remaining < 0` 通用保护分支。

第三轮意见没有产生需要修改实验代码的新证据，因此当前版本保持不变。
