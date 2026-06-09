# LADD H1 代码修复记录

日期：2026-06-09

## 1. 本次修复的问题

本次只修复 H1 主线诊断的训练/诊断实现污染，不改变 LADD 方法主体。

- BN-freeze 不再改变任何参数的 `requires_grad`。
- `LADD_DIAG_LOG_GRAD=1` 不再隐式执行梯度裁剪。
- 新增显式 `LADD_GRAD_CLIP_NORM` 参数，默认 `0.0`，表示关闭 LADD trainer 额外裁剪。
- 新增 `LADD_ASSERT_PHASE_FREEZE`，用于在 B phase 检查 frozen module 是否被意外打开。
- launcher、chain 和 phase manifest 记录 H1 相关诊断变量，方便日志审计。

## 2. H1 与 H0 差异

```text
H0: cap2 + A2/B MuSGD lr0=1e-3 no warmup + B_FREEZE_BN_STATS=1
H1: H0 + clean BN-freeze + pure grad log + phase-freeze assert
```

H1 的目标是排除实现污染，不把 H1 宣传为最终主线。

## 3. 不改变的协议

- OGSOD HBB。
- `imgsz=256`。
- no-mosaic formal protocol。
- SAR-only student/inference。
- RGB teacher frozen。
- same-size SAR/RGB baseline pairing。
- A1=10、A2=50、B 默认 800。

## 4. 自检记录

本次本地自检结果：

- `py_compile`：通过。
- `bash -n`：formal launcher、LR/BN matrix、chain 和 phase 脚本均通过。
- static check：通过，两个 HBB trainer 不含 hard-coded `max_norm=10.0`，并包含 `ladd_grad_clip_norm` 与 `ladd_assert_phase_freeze`。
- pytest/smoke：`pytest -q tests/test_ladd_h1_diagnostics.py` 通过。
- LR/BN matrix dry run：通过，printed command 包含 `LADD_GRAD_CLIP_NORM=0.0` 与 `LADD_ASSERT_PHASE_FREEZE=1`。
- formal launcher deep dry run：本地缺少正式 baseline/teacher checkpoint，停在前置 checkpoint 检查；需在服务器权重齐全环境复跑。

## 5. 注意事项

- 本次不启动真实训练。
- 真实训练等代码审核通过后再开。
- 不提交 checkpoint、runs、wandb、raw logs 或私有路径。
