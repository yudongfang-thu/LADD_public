# CMDistill 第三轮实现审查与修改意见 v3

日期：2026-06-15
仓库：`yudongfang-thu/LADD_public`
审查分支：`audit/cmdistill_v2_smoke_ready`

## 0. 总体 Verdict

**Verdict：`ready for 1-epoch GPU smoke`**

第三轮审查没有发现必须在 GPU smoke 前修复的 P0/P1 级别实现错误。当前实现可以作为
CMDistill-style paper-aligned OGSOD/YOLO11 adaptation 启动 1-epoch GPU smoke，
但不能直接启动 800 epoch formal long run。

## 1. 已确认正确的第二轮改动

- SLRD batch-mixing 已修复：`_cmdistill_relation_loss()` 保留 `[B, HW, C]`，
  在每张图内部用 `torch.bmm()` 构建 `[B, K, K]` relation matrix。
- IBCLD per-level 重复调用已修复：`_cmdistill_style_loss()` 只负责 PCCFD 和
  SLRD，IBCLD 在 `_compute_decomposition_losses()` 中对 full concatenated
  detector outputs 只计算一次。
- explicit CMDistill normalization 正确：
  `feature_weight * mean(PCCFD_shallow, PCCFD_deep) +
  relation_weight * SLRD_deep + logit_weight * IBCLD_full_outputs`。
- diagnostics 可用于 1-epoch smoke：`_cmdistill_last_stats` 会进入
  `ladd_diagnostics.csv`，包括 candidate ratio、fg count、teacher-conf added
  count、cls/box loss、PCCFD/SLRD/IBCLD/total loss。
- `ladd/code/...` 和 `ladd/code_versions/current_hbb/...` 的 loss/trainer 同步。

## 2. Severity-ordered Issues

### P0

无。当前代码可以启动 1-epoch GPU smoke。

### P1

无强制 P1。当前 diagnostics 足够解释 1-epoch smoke 的关键风险。

### P2

#### P2-1. 缺少真实 `_compute_decomposition_losses()` synthetic integration test

当前 `comparison/code/smoke_check_comparison_losses.py` 覆盖 helper-level 行为，
但没有完整构造真实 `_compute_decomposition_losses()` 控制流。该项不阻塞
1-epoch GPU smoke，但正式长跑前建议补充更强 integration-style guard。

#### P2-2. Run tag 仍是 `v2_strict_20260615`

建议将 CMDistill launcher 默认版本更新为 `v3_smoke_ready_20260615`，否则后续
run dir 难以区分第二轮前后的实现。

#### P2-3. `CMDISTILL_MIN_CONFIDENCE=0.05` 必须通过真实 smoke 判断

该阈值是 YOLO dense-output adaptation，不是 CMDistill 论文官方细节。1-epoch
smoke 后必须检查：

- `0 < cmdistill_ibcld_candidate_ratio < 1`
- candidate ratio 不接近 1.0
- `cmdistill_ibcld_cls_loss` 和 `cmdistill_ibcld_box_loss` finite

如果 candidate ratio 接近 1.0，不要开 800 epoch，先做阈值 ablation。

#### P2-4. `nan_or_inf_detected` 混合 BN stats 与 CMDistill stats

建议新增拆分字段：

- `nonfinite_metrics_or_cmdistill`
- `nonfinite_bn_stats`

保留 `nan_or_inf_detected` 兼容旧日志。

#### P2-5. `CMDISTILL_TEMPERATURE` 仍是 unused compatibility 参数

可以保留，但报告中必须写明该参数被 CLI 接受但 strict BCE-based IBCLD 不使用。

### P3

- README 标题和 scope 建议统一为 CMDistill-style，避免误报为 official
  reproduction。
- `YOLOv11` 建议统一为项目内常用的 `YOLO11`。
- 建议在 epoch 1 输出一行 `cmdistill_smoke_stats`，便于从 outer log 直接判断
  candidate ratio 和 component losses。

## 3. GPU Smoke 建议

允许启动 1-epoch smoke。推荐先 from-yolo，再 transfer。

```bash
COMPARISON_IMPL_VERSION=v3_smoke_ready_20260615 \
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

smoke 后必须检查 `ladd_diagnostics.csv`：

- `cmdistill_pcc_levels == 2`
- `cmdistill_slrd_tokens > 0`
- `cmdistill_ibcld_candidate_ratio > 0` 且不接近 1
- `cmdistill_ibcld_cls_loss` / `cmdistill_ibcld_box_loss` finite
- `cmdistill_pcc_loss` / `cmdistill_relation_loss` / `cmdistill_ibcld_loss`
  / `cmdistill_total_loss` finite
- `kd_loss` 和 detector losses finite

## 4. 报告措辞

推荐使用：

- CMDistill-style paper-aligned adaptation
- CMDistill-style OGSOD/YOLO11 adaptation
- paper-aligned CMDistill-style comparison baseline

避免使用：

- official CMDistill
- CMDistill official implementation
- CMDistill reproduction
- reproduced CMDistill results

建议表格脚注：

```text
CMDistill is implemented as a non-official, paper-aligned CMDistill-style
OGSOD/YOLO11 adaptation. No official CMDistill code was available. The original
CMDistill setting is IR teacher -> RGB student, YOLOv5s, 640 input; our controlled
comparison uses RGB teacher -> SAR student, SAR-only inference, YOLO11, 256 input,
no mosaic. CMDISTILL_MAX_TOKENS and CMDISTILL_MIN_CONFIDENCE are adaptation knobs.
```

## 5. 最终行动清单

现在可以：

- 做第三轮小清理 commit；
- 启动 1-epoch GPU smoke。

暂时不要：

- 直接启动 800 epoch formal run；
- 把结果写成 official CMDistill reproduction；
- 在没有 candidate ratio 证据的情况下固定 `CMDISTILL_MIN_CONFIDENCE=0.05`
  进行大规模长跑。
