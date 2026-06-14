# 本地实验数据整理方案（2026-06-14）

## 目标

四台服务器的结果已经拉回本地。本次整理采用“原始证据不动、索引统一、文档归档”的方式，避免把同一 run 的多个副本误当成独立结果，也避免清理文档时丢失可追溯性。

## 服务器来源

| 代号 | 说明 | 本地主要位置 |
|---|---|---|
| `ladd90` | 90 服务器，主要 baseline / LADD 主参考 | `ladd/results/ladd90_formal_baselines_20260612/`, `ladd/results/converged_mainline_ladd_20260613/` |
| `ladd4090` | 双卡 4090，当前大量 shutdown snapshot 和 comparison/LADD 证据 | `ladd/results/ladd4090_shutdown_sync_20260614/`, `comparison/results_shutdown_sync_20260614/evidence_raw/ladd4090/` |
| `autodl` | AutoDL / 4090D，关机前 critical backup | `remote_backups/autodl_20260614_critical/`, `ladd/results/autodl_shutdown_sync_20260614/` |
| `server117` | RTX 5880 Ada，旧 smoke / 环境证据 | 主要散落在旧 comparison 文档和少量 registry 条目 |

## 当前索引

统一 registry 已生成在：

- `docs/experiments/registry/experiment_registry_20260614.csv`
- `docs/experiments/registry/duplicate_results_20260614.csv`
- `docs/experiments/registry/experiment_registry_summary_20260614.json`
- `docs/experiments/registry/local_large_artifacts_20260614.csv`

生成命令：

```bash
python3 tools/build_experiment_registry.py --root . --out-dir docs/experiments/registry
```

当前扫描结果：

| 项 | 数量 |
|---|---:|
| `results.csv` 总数 | 458 |
| `ladd4090` 来源 | 314 |
| `ladd90` 来源 | 54 |
| `autodl` 来源 | 40 |
| `dual4090_old` invalid/diagnostic 来源 | 20 |
| `server117` 来源 | 4 |
| 内容重复 hash 组 | 115 |
| 位于重复组内的文件数 | 314 |

## 关机同步大文件处理

双卡 4090 关机前生成的轻量证据压缩包已经移出 Git 工作区，避免误提交：

```text
/Users/yudongfang/Desktop/光sar/LADD_public_local_archives/ladd4090_shutdown_sync_20260614/
```

仓库内保留解压后的轻量 evidence 和指针文件：

```text
ladd/results/ladd4090_shutdown_sync_20260614/ARCHIVE_POINTER_20260614.txt
ladd/results/ladd4090_shutdown_sync_20260614/evidence_raw/ladd4090/repo_root_snapshot/
```

当前本地仍有两类被 ignore 的 checkpoint 备份：

| 位置 | 数量 | 大小 | 说明 |
|---|---:|---:|---|
| `remote_backups/` | 20 | 约 1.78 GiB | AutoDL 关机迁移 checkpoint 备份 |
| `weights/` | 6 | 约 0.11 GiB | 少量主动保留 baseline checkpoint |

这些文件不进入 GitHub；如后续磁盘紧张，可将 `remote_backups/` 移到仓库外，只保留 manifest。

## 整理原则

1. `remote_backups/`、`ladd/results/*shutdown_sync*`、`comparison/results_shutdown_sync_20260614/` 作为 raw provenance 层，暂不移动、不删除。
2. 后续分析优先读 registry 的 `canonical_path`，不要直接遍历所有 `results.csv`，否则会重复计数。
3. `.pt` 权重单独保管。AutoDL raw backup 中的权重先留在 raw backup；主工作区 `weights/` 只保留明确需要复用的 baseline/关键 checkpoint。
4. 过时文档移出 Git 工作区到本地 archive，不直接删除。
5. 论文表格或汇报数字应从 registry + curated summary CSV 生成，并显式标注 `validity`。

## 推荐后续结构

```text
docs/experiments/registry/        # 全局 run 索引和重复映射
local archive outside repo         # 过时报告、旧指令、紧急临时分析
ladd/results/                     # LADD curated evidence / raw shutdown evidence
comparison/results_shutdown_sync_20260614/
remote_backups/                   # 服务器关机前原始同步包
weights/                          # 少量主动保留 checkpoint
```

过时文档本地 archive 当前路径：

```text
/Users/yudongfang/Desktop/光sar/LADD_public_local_archives/docs_obsolete_20260614/
/Users/yudongfang/Desktop/光sar/LADD_public_local_archives/docs_experiments_top_obsolete_20260614/
```

其中 `docs_experiments_top_obsolete_20260614/` 收纳了被新项目地图和三条线状态页覆盖的旧计划/旧状态页，例如旧 experiment plan、旧 running status、旧 diagnostic workspace map。

## 使用规则

- 查某个 run 是否重复：先用 `results_hash` 在 `duplicate_results_20260614.csv` 中查 alias。
- 查可用于主表的候选：筛 `family in {baseline,ladd,comparison,cclkd_yolov5x}` 且 `validity=candidate_or_unknown`，再人工复核协议。
- 查作废或历史诊断：筛 `validity in {diagnostic,invalid_or_diagnostic}`。
- 新同步回来的服务器数据先放 raw 层，再重新运行 registry 脚本。
