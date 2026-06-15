# CMDistill 第四轮实现审查与 GPU Smoke 前建议 v4

日期：2026-06-15
审查分支：`audit/cmdistill_v2_smoke_ready`
最新相关提交：`50fb23b38cab775e7a3ede3253096699905fa3df`

## 0. 总体 Verdict

**Verdict：`ready for 1-epoch GPU smoke`**

第四轮审查没有发现会阻止 1-epoch GPU smoke 的 P0/P1 级别问题。当前
CMDistill-style 实现已经满足短程真实数据流 smoke 的基本条件：

- PCCFD / SLRD / IBCLD 三项结构已经按 CMDistill paper 的组件语义完成
  paper-aligned adaptation。
- SLRD 已经是 per-image relation，不再混合 batch 内不同图像 token。
- IBCLD 已从 per-FPN feature loop 中移出，在 full concatenated detector
  outputs 上只计算一次。
- CMDistill loss 显式组合为
  `feature_weight * mean(PCCFD_shallow, PCCFD_deep) +
  relation_weight * SLRD_deep + logit_weight * IBCLD_full_outputs`。
- `ladd_diagnostics.csv` 和 epoch-1 `cmdistill_smoke_stats` 已包含 smoke
  必需的 CMDistill stats。
- from-YOLO / transfer launcher 的 CMDistill run tag 已更新为
  `v3_smoke_ready_20260615`。
- 文档已经明确这是 non-official CMDistill-style paper-aligned OGSOD/YOLO11
  adaptation，不是 official CMDistill reproduction。

结论：**允许启动 1-epoch GPU smoke**。仍然不允许直接启动 800 epoch formal run。

## 1. P0 / P1 / P2 / P3

### P0

无。没有发现 shape 明显不匹配、loss double-counting、IBCLD per-level 重复、
SLRD batch-mixing、launcher 未开启 affine、或 current_hbb 未同步等阻断问题。

### P1

无强制 P1。当前 diagnostics 足够支持 1-epoch smoke 的解释。

### P2

#### P2-1. 缺少完整 `_compute_decomposition_losses()` synthetic integration test

当前 `comparison/code/smoke_check_comparison_losses.py` 覆盖 helper-level 行为，
但还没有完整构造 `_compute_decomposition_losses()` 的真实控制流。正式 800
epoch 前建议补一个更接近 training path 的 synthetic guard。

#### P2-2. CMDistill prerequisite early-return 建议正式长跑前 hard-fail

`_compute_decomposition_losses()` 对缺少 teacher / paired features 的历史行为是
返回 zero losses。对 `COMPARISON_KD_PROFILE=cmdistill` 来说，缺少
`teacher_model`、`teacher_img`、student feature lists、teacher raw boxes/scores
等应被视为配置错误。建议在正式长跑前只对 CMDistill profile 加 hard-fail，
保留非 CMDistill LADD 的 legacy zero-return 行为。

#### P2-3. `CMDISTILL_MIN_CONFIDENCE=0.05` 必须由真实 smoke 判断

`fg OR teacher_conf >= min_confidence` 是合理的 YOLO dense-output adaptation，
但阈值不是 CMDistill 官方代码细节。1-epoch smoke 后必须检查：

- `cmdistill_ibcld_candidate_ratio`
- `cmdistill_ibcld_teacher_conf_added_count`
- `cmdistill_ibcld_fg_count`
- `cmdistill_ibcld_cls_loss`
- `cmdistill_ibcld_box_loss`

若 candidate ratio 接近 1.0，停止长跑，先做阈值 ablation。

#### P2-4. `CMDISTILL_TEMPERATURE` 仍是 unused compatibility 参数

该参数由 CLI 接受但 strict BCE-based IBCLD 不使用。报告中必须明确说明。

#### P2-5. `cmdistill_smoke_stats` 是 epoch-level，不是 batch-level

这对 1-epoch smoke 足够；若训练在第一个 epoch 内崩溃，该行不会出现，此时
以异常日志为准。不建议增加 batch-level 打印以免污染日志。

### P3

- 继续统一使用 CMDistill-style 命名。
- `REVIEW_ROUND3.md` 中部分历史 P2 项已由 `50fb23b` 修复，阅读时注意它是
  第三轮审查记录，不是当前未修问题列表。

## 2. 三项 Loss 的 Paper-Alignment 判断

### PCCFD

当前实现为 shallowest + deepest feature，student adaptive 1x1 layer 由
`KD_CALIBRATION_MODE=affine` 启用，使用 PKD-style channel-wise Pearson/PCC
normalization 和 MSE/2。该实现对于 controlled comparison smoke 足够
paper-aligned。

### SLRD

当前实现使用 deepest feature only，保留 `[B, HW, C]`，在每张图内部采样
spatial tokens 并用 `torch.bmm()` 计算 `[B, K, K]` affinity matrix，使用 L1
relation loss。`CMDISTILL_MAX_TOKENS` 是可接受的 OGSOD/YOLO11 memory-control
adaptation，必须在报告中说明。

### IBCLD

当前实现使用 decoded box IoU loss 和 BCE student logits vs teacher sigmoid
probability，teacher tensors detach，且只在 full concatenated detector outputs
上计算一次。candidate rule 为 `fg OR teacher_conf >= min_confidence`，属于合理
的 YOLO dense-output adaptation，必须在报告中说明。

## 3. Explicit Normalization / Double Counting

当前 `_cmdistill_combine_components()` 中：

- `CMDISTILL_FEATURE_WEIGHT` 乘一次；
- `CMDISTILL_RELATION_WEIGHT` 乘一次；
- `CMDISTILL_LOGIT_WEIGHT` 乘一次；
- `profile_kd_weight` 在加入 `kd_loss` 时乘一次；
- `alpha_kd` 是 LADD comparison 框架统一的总 KD 外层权重，返回 extra loss 时
  乘一次。

未发现 double counting。FGD / LD / CCLKD 仍走 generic profile path，未被
CMDistill special path 误伤。

## 4. Diagnostics 和 Launcher 审查

当前 CMDistill stats 包括：

- `cmdistill_pcc_levels`
- `cmdistill_slrd_tokens`
- `cmdistill_ibcld_candidate_ratio`
- `cmdistill_ibcld_fg_count`
- `cmdistill_ibcld_teacher_conf_added_count`
- `cmdistill_ibcld_cls_loss`
- `cmdistill_ibcld_box_loss`
- `cmdistill_pcc_loss`
- `cmdistill_relation_loss`
- `cmdistill_ibcld_loss`
- `cmdistill_total_loss`

trainer 已将 nonfinite flags 拆成：

- `nonfinite_metrics_or_cmdistill`
- `nonfinite_bn_stats`
- `nan_or_inf_detected`

两个 CMDistill launcher 默认 tag 均为 `v3_smoke_ready_20260615`，并默认设置
`KD_CALIBRATION_MODE=affine`。

## 5. 推荐 1-Epoch GPU Smoke

先做 from-YOLO smoke：

```bash
EPOCHS_B=1 \
PATIENCE_B=1 \
PROFILE_KD_WEIGHT=1.0 \
CMDISTILL_FEATURE_WEIGHT=1.0 \
CMDISTILL_RELATION_WEIGHT=1.0 \
CMDISTILL_LOGIT_WEIGHT=1.0 \
CMDISTILL_MIN_CONFIDENCE=0.05 \
LADD_DIAG_LOG_EVERY=1 \
EXIST_OK=1 \
bash comparison/code/launch_formal_from_yolo_kd_job.sh cmdistill n 0 0
```

from-YOLO smoke 通过后，再做 transfer smoke：

```bash
EPOCHS_B=1 \
PATIENCE_B=1 \
PROFILE_KD_WEIGHT=1.0 \
CMDISTILL_FEATURE_WEIGHT=1.0 \
CMDISTILL_RELATION_WEIGHT=1.0 \
CMDISTILL_LOGIT_WEIGHT=1.0 \
CMDISTILL_MIN_CONFIDENCE=0.05 \
LADD_DIAG_LOG_EVERY=1 \
EXIST_OK=1 \
bash comparison/code/launch_formal_transfer_kd_job.sh cmdistill n 0 0
```

## 6. Smoke 后必须检查

检查 outer log 的 `cmdistill_smoke_stats` 和 run dir 下的
`ladd_diagnostics.csv`。必须确认：

- `cmdistill_pcc_levels == 2`
- `cmdistill_slrd_tokens > 0`
- `cmdistill_ibcld_candidate_ratio > 0` 且不接近 1.0
- `cmdistill_ibcld_cls_loss` / `cmdistill_ibcld_box_loss` finite
- `cmdistill_pcc_loss` / `cmdistill_relation_loss` / `cmdistill_ibcld_loss`
  / `cmdistill_total_loss` finite
- `kd_loss` 和 detector losses finite
- `nonfinite_metrics_or_cmdistill == 0`

若 `nonfinite_bn_stats == 1`，需要看 BN 具体字段；它不一定表示 CMDistill loss
有问题。

## 7. 必须停止的情况

出现以下任一情况，不要启动正式长跑：

- RuntimeError / shape mismatch。
- outer log 没有 `cmdistill_smoke_stats` 且没有明确异常定位。
- `cmdistill_pcc_levels != 2`。
- `cmdistill_slrd_tokens == 0`。
- `cmdistill_total_loss == 0`。
- 任一 CMDistill component loss 为 NaN/Inf。
- `nonfinite_metrics_or_cmdistill == 1`。
- `cmdistill_ibcld_candidate_ratio == 0` 或接近 1.0。
- `cmdistill_ibcld_teacher_conf_added_count` 极大，说明几乎所有 background
  都被加入。
- `kd_loss` 异常大，detector loss 不下降或爆炸。
- validation metrics / results.csv / diagnostics 没有正常写出。

## 8. 报告措辞

推荐使用：

- CMDistill-style paper-aligned adaptation
- CMDistill-style OGSOD/YOLO11 adaptation
- paper-aligned CMDistill-style comparison baseline

避免使用：

- official CMDistill
- CMDistill official implementation
- CMDistill reproduction
- reproduced CMDistill results

表格脚注建议写：

```text
CMDistill is implemented as a non-official, paper-aligned CMDistill-style
OGSOD/YOLO11 adaptation. No official CMDistill code was available. The original
CMDistill setting is IR teacher -> RGB student, YOLOv5s, 640 input; our controlled
comparison uses RGB teacher -> SAR student, SAR-only inference, YOLO11, 256 input,
no mosaic. CMDISTILL_MAX_TOKENS and CMDISTILL_MIN_CONFIDENCE are adaptation knobs.
CMDISTILL_TEMPERATURE is accepted by CLI for compatibility but unused by the strict
BCE-based IBCLD implementation.
```

## 9. 下一步

当前最有价值的下一步是运行 1-epoch GPU smoke。正式 800 epoch 前再补：

1. CMDistill prerequisite hard-fail guard；
2. 更完整的 `_compute_decomposition_losses()` synthetic integration guard；
3. `CMDISTILL_MIN_CONFIDENCE` 阈值 sanity/ablation。
