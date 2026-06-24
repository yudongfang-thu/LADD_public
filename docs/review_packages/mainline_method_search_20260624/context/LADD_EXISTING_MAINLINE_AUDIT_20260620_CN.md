# LADD 已有主实验审计（2026-06-20）

本审计只回答 OGSOD HBB 主实验是否已经完成，重点区分：

- **当前 LADD 主线**：mosaic100，A -> B，无 A2，`clean_a1b_dynprobe` / `dynamic_probe`，B=800。
- **可参考 proxy / 消融**：`clean_a1b_dyn` dynamic、`clean_a1b` static、旧 `A1B_skipA2`。
- **不可直接混入主表**：A1-A2-B、no-mosaic、short/partial/snapshot、旧 debug loss 链路。

## 搜索范围

已检查：

- 本地 registry / paper_results / docs snapshots。
- 本地 4090 备份：`_remote_backups/4090_20260618/`。
- 90 服务器：`/mnt/dataY/ydf/projects/LADD_public/runs_public/`。
- AutoDL2：`/root/autodl-tmp/LADD_public/runs_public/`。
- 单卡 AutoDL 4090D 无卡模式：`/root/autodl-tmp/LADD_public/runs_public/`。

单卡 AutoDL 说明：本机 TUN 模式下 `connect.westb.seetacloud.com` 会解析到 `198.18.x.x` fake IP，常规别名连接会被关闭；本次通过旧真实 IP `36.103.198.205:39401` 成功只读检查。该目录不是 git 仓库，但保留了完整 `runs_public` 结果。

## 严格主线结论：mosaic100 clean dynamic_probe

| Model | Seed | 状态 | Rows | Best AP50-95 | 判定 | 证据 |
|---|---:|---|---:|---:|---|---|
| YOLO11n | 0 | AutoDL 单卡未找到 mosaic100 dynprobe 完整结果；AutoDL2 正在重跑 | A 阶段进行中 | - | 需要等当前 run 完成 | `runs_public/paper/ogsod_hbb_mosaic100/ladd/yolo11n/seed0/..._r2_polarsfix_*` |
| YOLO11s | 0 | 单卡 AutoDL 已完成 dynprobe B=800 | 800 | 0.63487 @708 | 可作为当前主线 seed0 结果；需补进 paper_results/provenance | `runs_public/ogsod/hbb/ladd_clean_a1b_dynamic_probe/mosaic_first100_close700/yolo11s/cap2/ladd_clean_a1b_dynprobe_ogsod11s_clean_a1b_dynprobe_yolo11s_cap2_s0_mosaic_first100_close700_a_probe_autodl_r1_20260617_1159_b_e800_b64_s0_gpu0/results.csv` |
| YOLO11m | 0 | 90 上 dynprobe partial | 493 | 0.65620 | 不完整，必须重跑/续跑 | `runs_public/ogsod/hbb/ladd_clean_a1b_dynamic_probe/mosaic_first100_close700/yolo11m/cap2/..._b_e800_b64_s0_gpu0/results.csv` |

严格按当前最终主线口径，**S/seed0 已经有已完成的 mosaic100 dynamic_probe 800-row 结果**。N/seed0 正在 AutoDL2 重跑；M/seed0 明确需要补或续跑。下一步不应重复 S 主线，除非 provenance validator 发现它和当前主线实现不一致。

## 可参考 proxy / 历史候选

这些结果可用于判断趋势或写消融/历史说明，但进入主表前需要明确口径。

| Model | Variant | Rows | Best AP50-95 | Last AP50-95 | 可用性 | 证据 |
|---|---|---:|---:|---:|---|---|
| YOLO11n | Static `clean_a1b` | 800 | 0.57113 @758 | 0.56836 | 消融可用；优先复用 | 单卡 AutoDL `runs_public/ogsod/hbb/ladd_clean_a1b/mosaic_first100_close700/yolo11n/cap2/..._b_e800_b64_s0_gpu0/results.csv`；本地 `docs/experiments/ladd_mosaic100_mainline_curves_20260618/data/n_static_results.csv` |
| YOLO11n | Dynamic proxy `clean_a1b_dyn` | 800 | 0.57544 @749 | 0.57030 | 可用于趋势；不是最终 dynprobe 主线 | `paper_results/ogsod_protocol_compare_seed0/raw/mosaic_n_dynamic_proxy_seed0.csv` |
| YOLO11n | Dynprobe snapshot | 347 | 0.49778 @347 | 0.49778 | partial，不可主表 | `_remote_backups/4090_20260618/.../ladd_clean_a1b_dynamic_probe/..._b_e800_b64_s0_gpu0/results.csv` |
| YOLO11s | Dynprobe 主线 | 800 | 0.63487 @708 | 0.62764 | 当前主线 seed0 可用；需补 paper_results/provenance | 单卡 AutoDL `runs_public/ogsod/hbb/ladd_clean_a1b_dynamic_probe/mosaic_first100_close700/yolo11s/cap2/..._b_e800_b64_s0_gpu0/results.csv` |
| YOLO11s | Dynamic proxy `clean_a1b_dyn` | 712 | 0.63647 @656 | 0.60079 | partial，不可主表 | `paper_results/ogsod_protocol_compare_seed0/raw/mosaic_s_dynamic_proxy_seed0.csv` |
| YOLO11s | Static `clean_a1b` | 800 | 0.62716 @739 | 0.62425 | 消融可用 | `_remote_backups/4090_20260618/.../ladd_clean_a1b/..._b_e800_b64_s0_gpu0/results.csv` |
| YOLO11s | 90 legacy A1B skipA2 | 800 | 0.64409 @703 | 0.63619 | 旧实现/旧 tag，不能直接当当前主线 | `runs_public/ogsod/hbb/ladd_mosaic_s_20260616/...A1B_skipA2.../results.csv` |
| YOLO11m | Dynprobe partial | 493 | 0.65620 @493 | 0.65620 | partial，必须补 | `paper_results/ogsod_protocol_compare_seed0/raw/mosaic_m_ladd_seed0.csv` |

## 对当前排队策略的影响

1. AutoDL2 当前跑 `YOLO11n seed0` 主线是合理的，因为单卡 AutoDL 未找到 N 的 mosaic100 dynprobe 完整结果。
2. `YOLO11s seed0` 主线不应重复重跑；应优先把单卡 AutoDL 的 `results.csv`、`args.yaml`、必要 run metadata 迁入当前 paper_results/provenance。
3. `YOLO11m seed0` 主线需要补跑，优先级高，因为 M 只有 493-row partial。
4. Static/Dynamic 消融不应再作为当前 P0 抢算力任务：N static/dynamic 已有完整记录，S static 已完整，S dynamic 是 712-row partial、可先作趋势证据。
5. 当前更值得优先保留/启动的是 `LADD n/m 主线`、`w/o reach`、`w/o taskL`、`w/o s_rec`、`w/o cap2` 等直接支撑方法设计的消融。
