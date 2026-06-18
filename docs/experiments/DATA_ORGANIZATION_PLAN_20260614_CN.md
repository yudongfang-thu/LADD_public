# 本地实验数据整理方案（2026-06-18）

## 目标

四台服务器的结果已经拉回本地。本次整理采用“原始证据不动、索引统一、文档归档”的方式，避免把同一 run 的多个副本误当成独立结果，也避免清理文档时丢失可追溯性。

## 服务器来源

| 代号 | 说明 | 本地主要位置 |
|---|---|---|
| `ladd90` | 90 服务器，baseline / 旧 LADD / clean evidence 来源 | 旧 LADD 证据已进入 `ladd/results/archive_legacy_ladd_20260618/`；当前 clean 入口见 `docs/experiments/ladd_mosaic100_mainline_curves_20260618/` |
| `ladd4090` | 双卡 4090，当前大量 shutdown snapshot 和 comparison/LADD 证据 | 旧 LADD 证据已进入 `ladd/results/archive_legacy_ladd_20260618/`；comparison evidence 保持原目录 |
| `autodl` | AutoDL / 4090D，关机前 critical backup | `remote_backups/autodl_20260614_critical/`；旧 LADD shutdown 记录已进入 `ladd/results/archive_legacy_ladd_20260618/` |
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
| `results.csv` 总数 | 675 |
| `ladd4090` 来源 | 484 |
| `ladd90` 来源 | 67 |
| `autodl` 来源 | 55 |
| `dual4090_old` invalid/diagnostic 来源 | 20 |
| `server117` 来源 | 4 |
| 内容重复 hash 组 | 181 |
| 位于重复组内的文件数 | 547 |

## 关机同步大文件处理

双卡 4090 关机前生成的轻量证据压缩包不进入 GitHub。旧 LADD 相关 evidence 已统一进入仓内 legacy archive：

```text
ladd/results/archive_legacy_ladd_20260618/
docs/experiments/archive_legacy_ladd_20260618/
```

当前本地仍有两类被 ignore 的 checkpoint 备份：

| 位置 | 数量 | 大小 | 说明 |
|---|---:|---:|---|
| `remote_backups/` | 20 | 约 1.78 GiB | AutoDL 关机迁移 checkpoint 备份 |
| `weights/` | 6 | 约 0.11 GiB | 少量主动保留 baseline checkpoint |

这些文件不进入 GitHub；如后续磁盘紧张，可将 `remote_backups/` 移到仓库外，只保留 manifest。

## 整理原则

1. `remote_backups/` 和 `comparison/results_shutdown_sync_20260614/` 作为 raw provenance 层，暂不移动、不删除；旧 LADD 证据统一保留在 `archive_legacy_ladd_20260618/`。
2. 后续分析优先读 registry 的 `canonical_path`，不要直接遍历所有 `results.csv`，否则会重复计数。
3. `.pt` 权重单独保管。AutoDL raw backup 中的权重先留在 raw backup；主工作区 `weights/` 只保留明确需要复用的 baseline/关键 checkpoint。
4. 旧 LADD A1-A2-B、旧 mosaic、旧 repair/diagnostic 文档进入仓内 legacy archive，不直接删除。
5. 论文表格或汇报数字应从 registry + curated summary CSV 生成，并显式标注 `validity`。

## 推荐后续结构

```text
docs/experiments/registry/        # 全局 run 索引和重复映射
docs/experiments/archive_legacy_ladd_20260618/
ladd/results/archive_legacy_ladd_20260618/
comparison/results_shutdown_sync_20260614/
remote_backups/                   # 服务器关机前原始同步包
weights/                          # 少量主动保留 checkpoint
```

旧 LADD 归档入口见 [archive_legacy_ladd_20260618/README_CN.md](archive_legacy_ladd_20260618/README_CN.md)。当前 LADD 主线入口见 [LADD_MAINLINE_STANDARD_CN.md](LADD_MAINLINE_STANDARD_CN.md)。

## 使用规则

- 查某个 run 是否重复：先用 `results_hash` 在 `duplicate_results_20260614.csv` 中查 alias。
- 查可用于 LADD 主表的候选：筛 `experiment_line=ladd_clean_a1b_mainline` 且 `role=mainline_candidate`，再人工复核 baseline/comparison 是否同协议。
- 查作废或历史诊断：筛 `validity in {diagnostic,invalid_or_diagnostic}`。
- 新同步回来的服务器数据先放 raw 层，再重新运行 registry 脚本。
