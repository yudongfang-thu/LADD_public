# Codex 指令：LADD H1 主线诊断修改与实验编排

> 目标：让 Codex 在 `yudongfang-thu/LADD_public` 仓库中完成下一轮最小代码修正、诊断脚本、结果汇总脚本和文档更新，然后只启动 P1 短链诊断。
> 注意：本文件是可以整体交给 Codex 的任务说明。不要把 checkpoint、runs、raw server logs、数据集、私钥、密码或服务器连接信息提交到 GitHub。

---

## 0. 背景与当前判断

你现在在仓库：

```bash
https://github.com/yudongfang-thu/LADD_public
```

当前任务不是重写 LADD 方法，而是为“锁定稳定主线方法”做下一轮最小诊断修改和实验编排。

当前主线候选记为 **H0**：

```text
cap2 + A2/B MuSGD lr0=1e-3 no warmup + B FREEZE_BN_STATS=1
```

当前观察：

- YOLO11n 三个 seed 已经比较稳定正向。
- YOLO11s 出现 best 正向但 last 退化的问题。
- YOLO11m 在 A2 或 B 入口已经出现明显退化，因此不能继续盲目跑 full B。
- 下一轮真正要验证的主线候选记为 **H1**：

```text
H1 = H0
   + 修正 BN-freeze 的 requires_grad 污染
   + 取消 ladd_diag_log_grad 的隐式梯度裁剪副作用
```

H1 不应该改变 LADD 方法本身，只消除训练/诊断实现污染。

---

## 1. 创建工作分支

请从当前默认分支创建新分支：

```bash
git status
git pull
git checkout -b diag/ladd-h1-bn-freeze-gradlog
```

如果分支已存在，则切换到该分支并 rebase/merge 最新默认分支，确保改动基于最新代码。

---

## 2. 先定位相关代码

请先搜索这些关键词，确认实际文件路径和函数名：

```bash
grep -R "def _set_bn_stats_eval\|_set_bn_stats_eval\|freeze_bn_stats\|freeze_bn_after_epoch" -n .
grep -R "ladd_diag_log_grad\|clip_grad_norm\|optimizer_step" -n .
grep -R "alpha_kd\|lambda_reach\|lambda_match_inner\|lambda_rank_inner" -n .
grep -R "teacher_decomposition\|student_reachability\|teacher_task_heads" -n .
grep -R "launch_formal_ladd_job\|train_ladd_hbb" -n scripts tools ladd shared . 2>/dev/null
```

优先修改当前正式 LADD HBB trainer 和正式 launcher。不要修改 deprecated/archive/old/debug-only 文件，除非当前正式入口实际 import 了它们。

---

## 3. 修正 BN-freeze：只冻结 BN running stats，不要改变 requires_grad

### 3.1 目标

`FREEZE_BN_STATS=1` 时，BN 层应该进入 eval 模式，以冻结 running mean/var。

但是这个函数绝对不能修改任何参数的 `requires_grad`。

特别是 B 阶段已经冻结的以下模块，不能因为 BN-freeze 又把 BN affine 参数重新打开：

- `teacher_decomposition`
- `student_reachability`
- `teacher_task_heads`

### 3.2 需要修改的逻辑

请找到类似 `_set_bn_stats_eval(model)` 的函数。如果现在里面有类似逻辑：

```python
for parameter in module.parameters():
    parameter.requires_grad_(True)
```

或者任何 `requires_grad_(True/False)`，请移除。

目标实现应接近：

```python
@staticmethod
def _set_bn_stats_eval(model):
    """Freeze BatchNorm running statistics only.

    Important:
    - This function must only set BN modules to eval mode.
    - It must NOT change parameter requires_grad flags.
    - Trainability is controlled exclusively by phase-freeze logic.
    """
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
```

如果仓库里已有 `import torch`，可以不要重复 import。

### 3.3 注意事项

- 不要改变 BN 的 affine 参数是否可训练。
- 不要改 `track_running_stats`。
- 不要改 BN momentum。
- 如果 trainer 每个 epoch 或某些 callback 里会重新 `model.train()`，请确保 BN-freeze 的调用仍然发生在 `model.train()` 之后。沿用现有调用点即可；如果现有调用点不稳定，请把 BN-freeze override 放在每个 epoch train-mode 设置之后。
- 如果存在 `freeze_bn_after_epoch`，请保持原语义：只有 epoch 达到阈值后才冻结 BN stats。

---

## 4. 增加 phase-freeze 诊断，防止 B 阶段 frozen module 被意外打开

请新增一个轻量诊断开关：

```bash
--ladd-assert-phase-freeze
```

默认值为 False。

当 `--ladd-assert-phase-freeze` 为 True，并且当前处于 B phase 时，请检查以下模块是否仍然全参数 frozen：

- `teacher_decomposition`
- `student_reachability`
- `teacher_task_heads`

如果实际代码里的模块命名略有不同，请以当前正式 trainer 里的真实模块名为准，但语义必须对应：

- 教师解耦网络
- 学生可达性模块
- 教师任务头/教师辅助头

检查逻辑：

- 如果模块存在，则所有参数都应该 `requires_grad=False`。
- 如果发现某个 frozen module 里有参数 `requires_grad=True`，请 raise 明确错误，错误信息包含 phase、模块名、参数名。
- 如果模块不存在，不要直接崩溃；先打印 warning，说明该模块名未找到，并继续检查其他模块。
- 为避免日志过多，该检查只需要在 B phase 启动时或第一个 epoch 开始时打印一次。

同时请增加一个简洁日志函数，在 A2/B phase 开始时打印：

- 当前 phase
- trainable parameter count / total parameter count
- BN module 总数
- BN eval 数量
- 是否启用 `freeze_bn_stats`
- 是否启用 `freeze_bn_after_epoch`
- 是否启用 `ladd-assert-phase-freeze`
- 是否启用 grad clipping 以及 grad clip norm

不要让这个诊断改变训练行为。

---

## 5. 修正 ladd_diag_log_grad：它必须是纯日志开关，不能隐式裁剪梯度

当前代码中如果存在类似逻辑：

```python
if self.args.ladd_diag_log_grad:
    torch.nn.utils.clip_grad_norm_(...)
```

请修改。

要求：

- `ladd_diag_log_grad=1` 只允许记录梯度范数或梯度状态，不允许改变训练。
- 不允许因为打开日志而额外裁剪梯度、改变 optimizer step、改变 scaler、改变 zero_grad 时机、改变 requires_grad。
- 新增单独显式参数：

```bash
--ladd-grad-clip-norm
```

默认值为 `0.0` 或 `None`。建议用 `0.0` 表示关闭。

只有当 `--ladd-grad-clip-norm > 0` 时，才允许执行：

```python
torch.nn.utils.clip_grad_norm_(...)
```

日志中必须明确打印：

- `ladd_diag_log_grad=True/False`
- `ladd_grad_clip_norm=0.0` 时显示 grad clipping OFF
- `ladd_grad_clip_norm>0` 时显示 grad clipping ON，并打印 norm 数值

重要：

- 如果现有 trainer override 了 `optimizer_step()`，请小心不要破坏 Ultralytics 原本 AMP/scaler/optimizer step 逻辑。
- 如果你无法在不改变原有 step 语义的情况下安全记录梯度范数，那就先移除隐式裁剪，只保留简单日志或 TODO 注释；不能为了日志引入训练行为改变。
- H1 主线实验必须使用 `ladd_grad_clip_norm=0.0`。

---

## 6. 扩展正式 launcher 的可控参数

请修改当前正式 LADD launcher，例如：

```bash
scripts/launch_formal_ladd_job.sh
```

具体文件名以仓库实际正式入口为准。

要求 launcher 支持以下环境变量，并在非空时转成 Python 参数传入 trainer：

```bash
ALPHA_KD
LAMBDA_REACH
LAMBDA_MATCH_INNER
LAMBDA_RANK_INNER
FREEZE_BN_STATS
FREEZE_BN_AFTER_EPOCH
LADD_DIAG_LOG_GRAD
LADD_GRAD_CLIP_NORM
LADD_ASSERT_PHASE_FREEZE
EPOCHS
EXP_SUFFIX
A2_LR0
B_LR0
```

如果已有同义变量，请保持兼容，不要破坏旧脚本。

示例转参逻辑可以类似：

```bash
EXTRA_ARGS=()

if [[ -n "${ALPHA_KD:-}" ]]; then
  EXTRA_ARGS+=(--alpha-kd "$ALPHA_KD")
fi

if [[ -n "${LAMBDA_REACH:-}" ]]; then
  EXTRA_ARGS+=(--lambda-reach "$LAMBDA_REACH")
fi

if [[ -n "${LAMBDA_MATCH_INNER:-}" ]]; then
  EXTRA_ARGS+=(--lambda-match-inner "$LAMBDA_MATCH_INNER")
fi

if [[ -n "${LAMBDA_RANK_INNER:-}" ]]; then
  EXTRA_ARGS+=(--lambda-rank-inner "$LAMBDA_RANK_INNER")
fi

if [[ -n "${FREEZE_BN_STATS:-}" ]]; then
  EXTRA_ARGS+=(--freeze-bn-stats "$FREEZE_BN_STATS")
fi

if [[ -n "${FREEZE_BN_AFTER_EPOCH:-}" ]]; then
  EXTRA_ARGS+=(--freeze-bn-after-epoch "$FREEZE_BN_AFTER_EPOCH")
fi

if [[ -n "${LADD_DIAG_LOG_GRAD:-}" ]]; then
  EXTRA_ARGS+=(--ladd-diag-log-grad "$LADD_DIAG_LOG_GRAD")
fi

if [[ -n "${LADD_GRAD_CLIP_NORM:-}" ]]; then
  EXTRA_ARGS+=(--ladd-grad-clip-norm "$LADD_GRAD_CLIP_NORM")
fi

if [[ -n "${LADD_ASSERT_PHASE_FREEZE:-}" ]]; then
  EXTRA_ARGS+=(--ladd-assert-phase-freeze "$LADD_ASSERT_PHASE_FREEZE")
fi

if [[ -n "${EPOCHS:-}" ]]; then
  EXTRA_ARGS+=(--epochs "$EPOCHS")
fi

if [[ -n "${A2_LR0:-}" ]]; then
  EXTRA_ARGS+=(--a2-lr0 "$A2_LR0")
fi

if [[ -n "${B_LR0:-}" ]]; then
  EXTRA_ARGS+=(--b-lr0 "$B_LR0")
fi
```

实际参数名请和 trainer 当前 argparse 保持一致。如果 argparse 里还没有这些参数，请补上。

---

## 7. 新增诊断实验队列脚本

请新增脚本：

```bash
scripts/launch_ladd_mainline_diag_20260609.sh
```

要求：

- 默认 `DRY_RUN=1`，只打印命令，不真正启动训练。
- 只有当 `DRY_RUN=0` 时才执行训练命令。
- 不要默认开 full 800 大规模实验。
- 优先启动短链诊断。
- 每个实验名必须包含：`diag_h1`、model size、seed、phase/epoch、关键变量。
- 所有实验都必须记录当前 git commit SHA。

建议脚本如下，请根据当前仓库 launcher 的真实变量名修正，但必须保留实验语义和命名：

```bash
#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
RUN_SET="${RUN_SET:-p1}"
SERVER_TAG="${SERVER_TAG:-unknown_server}"

FORMAL_LAUNCHER="${FORMAL_LAUNCHER:-scripts/launch_formal_ladd_job.sh}"

print_header() {
  local name="$1"
  shift

  echo "================================================================"
  echo "[LADD-DIAG] name=${name}"
  echo "[LADD-DIAG] run_set=${RUN_SET}"
  echo "[LADD-DIAG] git_commit=$(git rev-parse HEAD)"
  echo "[LADD-DIAG] server=${SERVER_TAG}"
  echo "[LADD-DIAG] dry_run=${DRY_RUN}"
  echo "[LADD-DIAG] command:"
  printf ' %q' "$@"
  echo
  echo "================================================================"
}

run_cmd() {
  local name="$1"
  shift

  print_header "$name" "$@"

  if [[ "$DRY_RUN" == "0" ]]; then
    "$@"
  fi
}

if [[ "$RUN_SET" == "p1" ]]; then
  # P1-1: n seed0 H1 B-100 smoke，用于确认 BN/gradlog 修正没有破坏 n 的已知健康行为。
  run_cmd "diag_h1_n_seed0_b100_smoke" \
    env MODEL=n SEED=0 EPOCHS=100 EXP_SUFFIX=diag_h1_n_s0_b100_smoke \
    FREEZE_BN_STATS=1 FREEZE_BN_AFTER_EPOCH=0 \
    LADD_ASSERT_PHASE_FREEZE=1 LADD_DIAG_LOG_GRAD=1 LADD_GRAD_CLIP_NORM=0.0 \
    bash "$FORMAL_LAUNCHER"

  # P1-2: s seed0 H1 B-400，用于判断 s 的 best 正向但 last 退化是否被缓解。
  run_cmd "diag_h1_s_seed0_b400" \
    env MODEL=s SEED=0 EPOCHS=400 EXP_SUFFIX=diag_h1_s_s0_b400 \
    FREEZE_BN_STATS=1 FREEZE_BN_AFTER_EPOCH=0 \
    LADD_ASSERT_PHASE_FREEZE=1 LADD_DIAG_LOG_GRAD=1 LADD_GRAD_CLIP_NORM=0.0 \
    bash "$FORMAL_LAUNCHER"

  # P1-3: m seed0 H1 A2-only，用于判断 m 的问题是否仍然在 A2 阶段发生。
  # 如果正式 launcher 支持只跑 A2，请使用只跑 A2。
  # 如果不支持，请新增/复用现有 A2-only 入口，但不要启动 B full run。
  run_cmd "diag_h1_m_seed0_a2only" \
    env MODEL=m SEED=0 EXP_SUFFIX=diag_h1_m_s0_a2only \
    FREEZE_BN_STATS=0 \
    LADD_ASSERT_PHASE_FREEZE=1 LADD_DIAG_LOG_GRAD=1 LADD_GRAD_CLIP_NORM=0.0 \
    bash "$FORMAL_LAUNCHER"
fi

if [[ "$RUN_SET" == "p2_s" ]]; then
  # s 的 B 阶段 KD 强度诊断：只在 P1 结果看完后再跑。
  run_cmd "diag_h1_s_seed0_alpha_kd_0p5_b400" \
    env MODEL=s SEED=0 EPOCHS=400 EXP_SUFFIX=diag_h1_s_s0_alphaKD0p5_b400 \
    ALPHA_KD=0.5 \
    FREEZE_BN_STATS=1 FREEZE_BN_AFTER_EPOCH=0 \
    LADD_ASSERT_PHASE_FREEZE=1 LADD_DIAG_LOG_GRAD=1 LADD_GRAD_CLIP_NORM=0.0 \
    bash "$FORMAL_LAUNCHER"

  run_cmd "diag_h1_s_seed0_alpha_kd_0p25_b400" \
    env MODEL=s SEED=0 EPOCHS=400 EXP_SUFFIX=diag_h1_s_s0_alphaKD0p25_b400 \
    ALPHA_KD=0.25 \
    FREEZE_BN_STATS=1 FREEZE_BN_AFTER_EPOCH=0 \
    LADD_ASSERT_PHASE_FREEZE=1 LADD_DIAG_LOG_GRAD=1 LADD_GRAD_CLIP_NORM=0.0 \
    bash "$FORMAL_LAUNCHER"

  # det-only 诊断：如果当前代码已有关闭 KD/aux 的参数，请使用现有参数；
  # 如果没有，请新增显式参数，不要通过改代码常量实现。
  run_cmd "diag_h1_s_seed0_detonly_b400" \
    env MODEL=s SEED=0 EPOCHS=400 EXP_SUFFIX=diag_h1_s_s0_detonly_b400 \
    ALPHA_KD=0.0 \
    FREEZE_BN_STATS=1 FREEZE_BN_AFTER_EPOCH=0 \
    LADD_ASSERT_PHASE_FREEZE=1 LADD_DIAG_LOG_GRAD=1 LADD_GRAD_CLIP_NORM=0.0 \
    bash "$FORMAL_LAUNCHER"
fi

if [[ "$RUN_SET" == "p2_m" ]]; then
  # m 的 A2 阶段诊断：只跑 A2，不跑 B full。
  run_cmd "diag_h1_m_seed0_a2_lr3e4" \
    env MODEL=m SEED=0 EXP_SUFFIX=diag_h1_m_s0_a2_lr3e4 \
    A2_LR0=0.0003 \
    LADD_ASSERT_PHASE_FREEZE=1 LADD_DIAG_LOG_GRAD=1 LADD_GRAD_CLIP_NORM=0.0 \
    bash "$FORMAL_LAUNCHER"

  run_cmd "diag_h1_m_seed0_a2_short25" \
    env MODEL=m SEED=0 EPOCHS=25 EXP_SUFFIX=diag_h1_m_s0_a2_short25 \
    LADD_ASSERT_PHASE_FREEZE=1 LADD_DIAG_LOG_GRAD=1 LADD_GRAD_CLIP_NORM=0.0 \
    bash "$FORMAL_LAUNCHER"

  run_cmd "diag_h1_m_seed0_a2_lambda05" \
    env MODEL=m SEED=0 EXP_SUFFIX=diag_h1_m_s0_a2_lambda05 \
    LAMBDA_REACH=0.5 LAMBDA_MATCH_INNER=0.5 LAMBDA_RANK_INNER=0.5 \
    LADD_ASSERT_PHASE_FREEZE=1 LADD_DIAG_LOG_GRAD=1 LADD_GRAD_CLIP_NORM=0.0 \
    bash "$FORMAL_LAUNCHER"
fi

if [[ "$RUN_SET" != "p1" && "$RUN_SET" != "p2_s" && "$RUN_SET" != "p2_m" ]]; then
  echo "Unknown RUN_SET=${RUN_SET}. Valid: p1, p2_s, p2_m" >&2
  exit 2
fi
```

---

## 8. 新增诊断结果汇总脚本

请新增：

```bash
tools/summarize_ladd_diag_runs.py
```

功能：

- 输入一个或多个 run 目录。
- 自动读取每个 run 的 `results.csv` 和 `args.yaml`。
- 输出一个 CSV 和一个 Markdown 汇总表。
- 不提交完整 run 目录，只提交轻量 summary。

推荐接口：

```bash
python tools/summarize_ladd_diag_runs.py \
  --runs runs/path/to/run1 runs/path/to/run2 \
  --out-csv docs/experiments/ladd_mainline_diag_20260609_summary.csv \
  --out-md docs/experiments/LADD_MAINLINE_DIAG_RESULTS_20260609_CN.md
```

需要汇总字段：

- `run_name`
- `git_commit`
- `server_tag`，如果 args/log 中有
- `model_size`
- `seed`
- `phase` 或 `run_type`
- `epochs_finished`
- `best_epoch`
- `best_mAP50_95`
- `last_mAP50_95`
- `best_mAP50`
- `last_mAP50`
- `baseline_mAP50_95`
- `best_gain_vs_baseline`
- `last_gain_vs_baseline`
- `status`
- `notes`

baseline 初始值可先从 `docs/experiments/BASELINE_LADD_STATUS_CN.md` 读取；如果自动解析困难，可以先用一个显式字典，但必须在代码注释里写明“需要和 BASELINE_LADD_STATUS_CN.md 核对”。

已知需要核对的 baseline 至少包括：

```text
YOLO11n seed0 SAR baseline ≈ 0.55654
YOLO11n seed42 SAR baseline ≈ 0.55794
YOLO11n seed123 SAR baseline ≈ 0.56128
YOLO11s seed0 SAR baseline ≈ 0.62897
YOLO11m seed0 SAR baseline ≈ 0.65580
```

注意：

- 如果文档里的 baseline 和这些数值不一致，以仓库文档为准，并在结果表 notes 里说明。
- `results.csv` 里的 mAP 列名可能有不同格式，例如：
  - `metrics/mAP50-95(B)`
  - `metrics/mAP50-95`
  - `mAP50-95`
- 请做 robust column matching。
- status 规则：
  - `PASS`: `best_gain_vs_baseline > 0` 且 `last_gain_vs_baseline >= -0.002`，并且没有 NaN/Inf/collapse。
  - `WEAK`: `best_gain_vs_baseline > 0` 但 `last_gain_vs_baseline < -0.002`。
  - `FAIL`: `best_gain_vs_baseline <= 0` 或出现崩溃。
  - `RUNNING/INCOMPLETE`: 没跑完或 results.csv 不完整。

---

## 9. 新增诊断计划文档

请新增或更新：

```bash
docs/experiments/LADD_MAINLINE_DIAG_20260609_CN.md
```

内容包含以下部分。

### 9.1 当前问题

- n 三 seed 已有稳定正向证据。
- s 存在 B 后期退化。
- m 在 A2 或 B 入口存在退化。
- 因此当前 H0 不能直接冻结为跨 seed、跨容量主线。

### 9.2 H1 修改定义

```text
H1 = H0 + BN-freeze 不改 requires_grad + ladd_diag_log_grad 不再隐式裁剪梯度
```

H1 不改变 LADD 方法本身，只消除训练/诊断实现污染。

### 9.3 合格判据

- 单 run：
  - 无 NaN/Inf。
  - 无 B stage collapse。
  - best mAP50-95 > SAR baseline。
  - last mAP50-95 >= SAR baseline - 0.002。
- n/s：
  - 至少 seed0/42/123 三 seed 全部不负。
  - mean gain > 0。
- m：
  - seed0 的 A2 不能低于 baseline。
  - 否则不进入 B full。
- 只有 n/s 三 seed 和 m seed0 通过，才能考虑把 H1 作为正式主线。
- 如果 H1 仍不能稳住 s/m，再进入 H2：capacity-adaptive KD，例如 s/m 使用 `alpha_kd=0.5` 或 `0.25`。

### 9.4 实验队列

P1：

- `diag_h1_n_seed0_b100_smoke`
- `diag_h1_s_seed0_b400`
- `diag_h1_m_seed0_a2only`

P2_s：

- `diag_h1_s_seed0_alpha_kd_0p5_b400`
- `diag_h1_s_seed0_alpha_kd_0p25_b400`
- `diag_h1_s_seed0_detonly_b400`

P2_m：

- `diag_h1_m_seed0_a2_lr3e4`
- `diag_h1_m_seed0_a2_short25`
- `diag_h1_m_seed0_a2_lambda05`

### 9.5 注意事项

- 不混用 4090D 的 invalid/疑似协议错误结果作为主线证据。
- 不改 dataset yaml、imgsz、augmentation、YOLO 版本、teacher/student 配对协议。
- 不提交 raw logs/checkpoints/runs。
- 所有结果必须记录 git commit SHA 和 server/GPU 信息。

---

## 10. 最小自检

完成代码修改后，请至少执行：

```bash
python -m compileall tools scripts ladd shared
DRY_RUN=1 bash scripts/launch_ladd_mainline_diag_20260609.sh
RUN_SET=p2_s DRY_RUN=1 bash scripts/launch_ladd_mainline_diag_20260609.sh
RUN_SET=p2_m DRY_RUN=1 bash scripts/launch_ladd_mainline_diag_20260609.sh
```

如果有测试框架，新增一个最小测试，验证 `_set_bn_stats_eval` 不会改变 `requires_grad`。

测试语义：

1. 构造一个含 Conv + BatchNorm 的 toy model。
2. 手动把 BN weight/bias 的 `requires_grad` 设成 False。
3. 调用 `_set_bn_stats_eval(model)`。
4. 断言 BN module `training == False`。
5. 断言 BN weight/bias 的 `requires_grad` 仍然是 False。
6. 再手动设成 True，调用一次，断言仍然是 True。
7. 即证明 BN-freeze 只改变 BN train/eval 状态，不改变 trainability。

如果因为 trainer import 复杂，不方便直接 import `_set_bn_stats_eval`，请至少写一个内部 helper 的测试，或者在文档里说明未能加自动测试的原因。

---

## 11. 启动实验的要求

如果代码自检通过，可以启动 P1，但不要启动 P2，也不要启动 full 800 多 seed。

启动顺序：

1. 先 push 代码，记录 commit SHA。
2. 只启动 P1：
   - `diag_h1_n_seed0_b100_smoke`
   - `diag_h1_s_seed0_b400`
   - `diag_h1_m_seed0_a2only`
3. 每个 run 的日志里必须能看到：
   - git commit SHA
   - H1 config
   - `FREEZE_BN_STATS=1`，除了 m A2-only 如协议不需要可为 0
   - `LADD_GRAD_CLIP_NORM=0.0`
   - `ladd_diag_log_grad` 不改变训练
   - B phase frozen modules 的 requires_grad 检查结果
4. P1 结果出来后，用 `tools/summarize_ladd_diag_runs.py` 生成：
   - `docs/experiments/ladd_mainline_diag_20260609_summary.csv`
   - `docs/experiments/LADD_MAINLINE_DIAG_RESULTS_20260609_CN.md`
5. 不要根据 P1 自动启动 P2，除非用户明确要求。

---

## 12. 提交要求

请提交以下类型文件：

- trainer/launcher 的代码修正
- 新增诊断脚本
- 新增 summarizer
- 新增/更新诊断计划文档
- 如果有，新增最小测试

不要提交：

- checkpoint
- `runs/`
- `wandb/`
- tensorboard event
- raw server logs
- dataset yaml 中的私有绝对路径修改
- 任何私钥/密码/服务器连接信息

提交信息建议：

```bash
git add .
git status
git commit -m "fix(ladd): clean BN-freeze and add mainline diagnostics"
git push origin diag/ladd-h1-bn-freeze-gradlog
```

---

## 13. Codex 最终回复格式

完成后请在回复里给出：

1. 修改了哪些文件。
2. 关键代码改动摘要。
3. 是否通过 compileall/dry-run/test。
4. P1 是否已经启动；如果启动，列出 run name、命令、commit SHA、日志位置。
5. 如果没有启动 P1，说明阻塞原因。
6. 是否存在不确定点或需要人工确认的参数名/入口脚本。

---

## 14. 禁止事项

不要做以下事情：

- 不要把 H1 直接宣传为最终主线。
- 不要启动 full 800 的 n/s 三 seed 或 m/l 多 seed。
- 不要自动启动 P2。
- 不要改变数据集协议、输入分辨率、augmentation、YOLO 版本、teacher/student 配对。
- 不要为了让实验好看而改 loss 默认权重。
- 不要让日志开关改变训练行为。
- 不要提交任何大文件或敏感信息。
