# Code and documentation change map

日期: 2026-07-09 CST

本文说明当前 GitHub 同步策略: 哪些内容适合公开提交, 哪些源码改动存在已知风险, 哪些本地记录需要先清洗再发布。

## 1. 已提交 / 准备提交的公开文档

| file | purpose | public-safety status |
|---|---|---|
| `README.md` | 增加 2026-07-09 current debug status 入口 | sanitized |
| `docs/PROJECT_STATUS_20260709_CN.md` | compact 当前状态、best LADD、FGD floor、9-row payload 和 claim boundary | sanitized |
| `docs/PROJECT_ONBOARDING_20260709_CN.md` | 新成员阅读路径、项目结构、证据规则、接手 checklist | sanitized |
| `docs/experiments/DIRECT400_EXPERIMENT_LEDGER_20260709_CN.md` | direct-400 public experiment ledger: rows, results, status, blockers | sanitized |
| `docs/CODE_AND_DOC_CHANGE_MAP_20260709_CN.md` | 当前源码 / 文档改动状态和发布策略 | sanitized |

These files intentionally avoid private server paths, SSH commands, raw log paths, checkpoint hashes, and full run directories.

## 2. 当前源码 diff 概览

The local working tree contains real source/script changes. They are not all published as runnable code yet because one known launcher / CLI parity blocker remains.

| area | files | observed change | current publish status |
|---|---|---|---|
| baseline strict batch | `baseline/code/train_ogsod_baseline.py`, `shared/yolo/ultralytics/engine/trainer.py` | add `--strict-batch-size` / strict OOM behavior to avoid silent batch-size fallback | needs focused test before source commit |
| batch policy by protocol | `baseline/scripts/run_formal_baseline.sh`, comparison launchers | m-size batch differs for mosaic100 vs no-mosaic | likely useful, but should be committed with protocol validation |
| LADD reach / feature modes | `ladd/code/train_ladd_hbb.py`, `ladd/code/src/.../loss.py`, `ladd/code/src/.../model.py` | add `reach_input_mode=student_z`; allow `teacher_feature_mode=raw_weak_reach` | needs CLI / loss-surface review |
| LADD launcher surface | `ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh` | default `IMGSZ=256`, add OGSOD protocol guard, expose KD / DSN / fused-shared / reach-probe flags | not safe to publish as runnable until CLI parity is fixed |
| offline mechanism audit | `ladd/code/tools/audit_ladd_learnability_hbb.py` | extend gradient audit: det-loss vs KD-z / KD-u gradients with scope and token controls | useful diagnostic tool, but should be validated separately |
| shared trainer behavior | `shared/train_cli_overrides.py`, `shared/yolo/ultralytics/engine/trainer.py` | add strict batch / override support | needs integration smoke |

## 3. Known blocker: launcher / CLI parity

Current PM records identify this blocker:

```text
run_ladd_phase.sh may pass fused-shared arguments that train_ladd_hbb.py does not accept.
```

Implication:

- Do not present the current launcher diff as a clean training entry point.
- Do not launch new LADD / LADD-fusion / flag-only rescue rows from this surface until patch review and no-launch preflight pass.
- A future source-code PR should either:
  - add trainer argparse support for the fused-shared flags, or
  - guard the launcher so it does not pass unsupported flags.

## 4. Local PM / experiment docs not directly published

The local worktree contains many detailed PM and Coordinator records under `docs/experiments/` and `docs/goals/`. Some include:

- remote absolute run paths,
- log path aliases,
- checkpoint filenames,
- SSH / routing notes,
- very large task-board history,
- raw unfiltered operational details.

These files are useful internally, but should not be pushed wholesale to the public repository without sanitization. The current public ledger extracts the relevant facts while omitting private operational details.

## 5. Recommended GitHub publication stages

### Stage A: sanitized handoff docs

Publish:

- `README.md`
- `docs/PROJECT_STATUS_20260709_CN.md`
- `docs/PROJECT_ONBOARDING_20260709_CN.md`
- `docs/experiments/DIRECT400_EXPERIMENT_LEDGER_20260709_CN.md`
- `docs/CODE_AND_DOC_CHANGE_MAP_20260709_CN.md`

Purpose: make the current state, experiment matrix, and evidence boundary visible to new members.

### Stage B: compact experiment artifacts

After a privacy scan, optionally publish compact tables:

- final fact audit row tables,
- comparison stability tables,
- failure localization summaries,
- selected plots with sanitized captions.

Do not include raw remote logs or checkpoint files.

### Stage C: source-code patch PRs

Split source changes into reviewable commits:

1. strict batch / protocol batch policy,
2. LADD loss-surface mode additions,
3. launcher / CLI parity patch,
4. gradient audit tool extension.

Each should include a small validation command and explicit statement about whether it was used in any reported experiment.

## 6. Current validation for the sanitized docs

Checks used before committing the sanitized documentation package:

```bash
git diff --check -- README.md docs/PROJECT_STATUS_20260709_CN.md \
  docs/PROJECT_ONBOARDING_20260709_CN.md \
  docs/experiments/DIRECT400_EXPERIMENT_LEDGER_20260709_CN.md \
  docs/CODE_AND_DOC_CHANGE_MAP_20260709_CN.md

rg -n "<private-token-patterns>|<remote-path-patterns>|<large-artifact-patterns>" \
  <sanitized-doc-set>
```

The intent is to publish enough for project understanding while keeping private operations and large artifacts out of GitHub.
