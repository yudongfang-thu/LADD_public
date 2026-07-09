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
| 当前 top route | `DISPATCH_FINAL_FACT_AUDIT_FOR_MISSING_ROWS` |
| best LADD AP50-95 | `0.50350` |
| best LADD row | `3090_ladd_fusionv1_singleproj_plus_ld_replace_base_s0` |
| best LADD variant | `singleproj_plus_ld_replace_base` |
| FGD seed0 floor AP50-95 | `0.55147` |
| best LADD gap to FGD floor | `-0.04797` |
| comparison final-pending AP50-95 range | `0.56183` - `0.56550` |

解释:

- 已完成 / final-pending 的 LADD seed0 rescue rows 仍明显低于 FGD floor。
- CMDistill / LD 的 seed42 / seed123 comparison rows 看起来稳定且强, 但仍需 final fact audit 后才能进入 comparison stability 表。
- 当前证据不支持扩展 LADD seed42 / seed123, 也不支持 claim LADD 超过 comparison methods。

## 当前 9-row final fact audit payload

当前待审计的 final-pending payload 有 9 条:

| group | server | method / variant | seed | monitor AP50 / AP50-95 | route hint |
|---|---|---:|---:|---:|---|
| LADD reset-v2 | 3090 | `singleproj_kd_only_no_decomp_aux` | 0 | 0.75473 / 0.49190 | below-floor -> failure localization |
| LADD reset-v2 | 3090 | `singleproj_plus_ld_profile` | 0 | 0.75973 / 0.49831 | below-floor -> failure localization |
| LADD reset-v2 | 3090 | `singleproj_plus_cmdistill_profile` | 0 | 0.76372 / 0.49159 | below-floor -> failure localization |
| LADD reset-v2 | 3090 | `plain_plus_cmdistill_profile` | 0 | 0.75749 / 0.49056 | below-floor -> failure localization |
| LADD fusion-v1 | 3090 | `singleproj_plus_ld_replace_base` | 0 | 0.77631 / 0.50350 | below-floor -> failure localization |
| comparison | 90 | CMDistill | 42 | 0.83796 / 0.56520 | comparison stability |
| comparison | 90 | CMDistill | 123 | 0.83720 / 0.56550 | comparison stability |
| comparison | 90 | LD | 42 | 0.83501 / 0.56463 | comparison stability |
| comparison | 90 | LD | 123 | 0.82935 / 0.56183 | comparison stability |

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

Current code and method:

- `ladd/code/train_ladd_hbb.py`
- `ladd/code/src/teacher_student_decomposition_kd_hbb/model.py`
- `ladd/code/src/teacher_student_decomposition_kd_hbb/loss.py`
- `docs/ladd_method_definition.md`
- `docs/method/METHOD_DEFINITIONS_AND_IMPLEMENTATION_CN.md`
- `docs/method/METHOD_OVERVIEW_CN.md`

Experiment and PM records:

- `docs/experiments/EXPERIMENT_INDEX_CN.md`
- `docs/experiments/LADD_PROJECT_HANDOFF_20260630_CN.md`
- `docs/experiments/DIRECT400_PM_STATUS_SNAPSHOT_20260708_CN.md`
- `docs/experiments/DIRECT400_FINAL_FACT_AUDIT_STAGE_GOAL_20260707_CN.md`
- `docs/experiments/DIRECT400_FINAL_FACT_AUDIT_PAYLOAD_20260708_CN.md`
- `docs/experiments/DIRECT400_LADD_OVER_COMPARISON_EVIDENCE_GATE_20260707_CN.md`
- `docs/experiments/DIRECT400_LADD_FAILURE_HYPOTHESIS_BACKLOG_20260707_CN.md`
- `docs/experiments/DIRECT400_LADD_RESCUE_IMPLEMENTATION_DECISION_TREE_20260707_CN.md`
- `docs/experiments/DIRECT400_LADD_LAUNCHER_CLI_PARITY_AUDIT_20260707_CN.md`

Team / project-management records:

- `docs/goals/LADD_CURRENT_STAGE_TARGET_CN.md`
- `docs/goals/LADD_PM_ACTIVE_GOAL_CN.md`
- `docs/goals/LADD_PROJECT_TEAM_ROSTER_CN.md`
- `docs/goals/LADD_TEAM_TASK_BOARD_CN.md`

## Publication / repository hygiene

Recommended GitHub update scope:

- Include compact Markdown / CSV status records needed to understand current evidence.
- Include source code only when a reviewed code change is intentionally part of the commit.
- Exclude `.pt`, `.pth`, raw run directories, private logs, SSH material, remote backups, large temporary folders, and unreviewed server snapshots.

The repository is therefore best updated through small, scoped documentation commits rather than by committing the entire local working tree.
