# LADD project status and direct-400 handoff

日期: 2026-07-09 CST

本文是 `LADD_public` 的当前公开状态入口, 用于帮助新读者理解项目现在为什么还处于 debug / evidence audit 阶段。它只汇总 compact 记录, 不包含 checkpoint 权重、私有服务器连接信息、完整训练日志或大体积运行目录。

## 当前目标

当前项目目标已经从早期 paper-facing `mosaic100` 主线, 转入更严格的 `pure direct-400` 协议下的 LADD rescue / comparison audit:

```text
在 pure direct-400 协议下, 通过结果审计、失败定位、实现级 rescue 和 comparison 方法融合,
判断 LADD 是否能稳定超过 strong comparison methods。
```

当前结论边界:

- 只把 `pure direct-400` 作为当前正式证据口径。
- `row400-from-800`, `800`, `1600`, `reload`, external reported numbers 不能混入 direct-400 matched gain。
- running / progress rows 不是 final result。
- 不跨 3090 / 90 / 4090 混算 gain。
- LADD plain / singleproj / fusion 只能按 same-machine / same-seed / same-protocol matched controls 判断。
- 当前不升级 claim, 不写论文结论。

## 当前 direct-400 摘要

截至 2026-07-09 本地 PM 记录:

| 项目 | 当前值 |
|---|---|
| active goal | `NOT_COMPLETE` |
| claim ready | `no` |
| 当前 top route | `DISPATCH_FAILURE_LOCALIZATION_AND_SCOPED_90_ARTIFACT_AUDIT` |
| best LADD AP50-95 | `0.50350` |
| best LADD row | `3090_ladd_fusionv1_singleproj_plus_ld_replace_base_s0` |
| best LADD variant | `singleproj_plus_ld_replace_base` |
| FGD seed0 floor AP50-95 | `0.55147` |
| best LADD gap to FGD floor | `-0.04797` |
| 90 comparison monitor AP50-95 range | `0.56183` - `0.56550`, not final-audited |

解释:

- 已完成并通过 final fact audit 的 3090 LADD seed0 rescue rows 仍明显低于 FGD floor。
- CMDistill / LD 的 seed42 / seed123 comparison rows 看起来稳定且强, 但最新审计被 90 SSH/TUN 访问阻断, 仍不能进入 accepted final facts。
- 当前证据不支持扩展 LADD seed42 / seed123, 也不支持 claim LADD 超过 comparison methods。

## 当前 direct-400 审计状态

最新可公开审计状态分成两类: 5 条 3090 LADD seed0 rows 已成为 `FINAL_FACT_READY`, 4 条 90 comparison monitor rows 因远端访问阻断仍是 `BLOCKED`。

### 3090 LADD rows: final facts for failure localization

| group | server | method / variant | seed | final AP50 / AP50-95 | route hint |
|---|---|---:|---:|---:|---|
| LADD reset-v2 | 3090 | `singleproj_kd_only_no_decomp_aux` | 0 | 0.75473 / 0.49190 | below-floor -> failure localization |
| LADD reset-v2 | 3090 | `singleproj_plus_ld_profile` | 0 | 0.75973 / 0.49831 | below-floor -> failure localization |
| LADD reset-v2 | 3090 | `singleproj_plus_cmdistill_profile` | 0 | 0.76372 / 0.49159 | below-floor -> failure localization |
| LADD reset-v2 | 3090 | `plain_plus_cmdistill_profile` | 0 | 0.75749 / 0.49056 | below-floor -> failure localization |
| LADD fusion-v1 | 3090 | `singleproj_plus_ld_replace_base` | 0 | 0.77631 / 0.50350 | below-floor -> failure localization |

### 90 comparison rows: strong monitor values, not final facts yet

| group | server | method / variant | seed | monitor AP50 / AP50-95 | latest audit status |
|---|---|---:|---:|---:|---|
| comparison | 90 | CMDistill | 42 | 0.83796 / 0.56520 | `BLOCKED_BY_90_SSH_ACCESS` |
| comparison | 90 | CMDistill | 123 | 0.83720 / 0.56550 | `BLOCKED_BY_90_SSH_ACCESS` |
| comparison | 90 | LD | 42 | 0.83501 / 0.56463 | `BLOCKED_BY_90_SSH_ACCESS` |
| comparison | 90 | LD | 123 | 0.82935 / 0.56183 | `BLOCKED_BY_90_SSH_ACCESS` |

The four 90 rows above are currently `BLOCKED_BY_90_SSH_ACCESS`: monitor values are useful signals, but not accepted final facts until artifact paths, args, logs, row count, weights listing, protocol and health scan are verified.

Explicitly excluded from this final fact audit payload:

- 3090 `singleproj_plus_fgd_profile` seed0: running / progress-only in the accepted monitor snapshot.
- 90 FGD seed42 / seed123: running / progress-only in the accepted monitor snapshot.
- 90 det-only seed42 / seed123 controls: already final fact ready as controls, not LADD matched gain.

## Why the project is still not claim-ready

The current direct-400 data creates a negative evidence gate:

1. The best current LADD rescue row is far below the FGD seed0 floor.
2. LADD rows and comparison rows come from different subsets and must be audited before table construction.
3. Progress-only FGD rows must not be used as final comparison stability evidence.
4. A launcher / CLI parity blocker exists for future LADD / LADD-fusion launches: `run_ladd_phase.sh` may pass fused-shared arguments that `train_ladd_hbb.py` does not accept. Any patch / preflight / launch still needs explicit review and user approval.

Therefore the correct next research move is not seed expansion or claim writing. It is:

```text
final fact audit -> failure localization -> implementation rescue decision
```

Comparison rows should separately enter:

```text
final fact audit -> comparison stability table
```

## Useful project entry points

Public handoff docs:

- `docs/PROJECT_ONBOARDING_20260709_CN.md`
- `docs/experiments/DIRECT400_EXPERIMENT_LEDGER_20260709_CN.md`
- `docs/CODE_AND_DOC_CHANGE_MAP_20260709_CN.md`

Current code and method:

- `ladd/code/train_ladd_hbb.py`
- `ladd/code/src/teacher_student_decomposition_kd_hbb/model.py`
- `ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py`
- `docs/ladd_method_definition.md`
- `docs/method/METHOD_DEFINITIONS_AND_IMPLEMENTATION_CN.md`
- `docs/method/METHOD_OVERVIEW_CN.md`

Experiment records:

- `docs/experiments/EXPERIMENT_INDEX_CN.md`
- `docs/experiments/DIRECT400_EXPERIMENT_LEDGER_20260709_CN.md`

Deep PM / Coordinator records remain useful locally, but many contain remote
paths or operational details. They should be sanitized before public GitHub
publication.

## Publication / repository hygiene

Recommended GitHub update scope:

- Include compact Markdown / CSV status records needed to understand current evidence.
- Include source code only when a reviewed code change is intentionally part of the commit.
- Exclude `.pt`, `.pth`, raw run directories, private logs, SSH material, remote backups, large temporary folders, and unreviewed server snapshots.

The repository is therefore best updated through small, scoped documentation commits rather than by committing the entire local working tree.
