# 主线方法搜索审阅包（2026-06-24）

本目录用于把近期 LADD 主线修改、YOLO-init/reload 混淆排查、DroneVehicle 小风洞负例、OGSOD YOLO-init 主线搜索，以及当前 dynamic 系列扫参证据集中在一起，方便交给更强模型或人工审阅。

## 审阅目标

当前核心问题不是写论文结果，而是判断：我们应如何修改 LADD 主线，才能在跨模态蒸馏、单模态 SAR 推理、YOLO-init 训练设定下获得稳定正增益。

重点请看：

- OGSOD YOLO-init：主线证据，只比较 same-pipeline det-only，不再把 reload 作为主线证据。
- Dynamic/LADD-like 变体：当前最值得继续挖的方向。
- ProbeA/old-commit/AutoDL 证据：用于解释为什么某些环境出现较大早期增益，但不直接等价于当前主线。
- DroneVehicle 小风洞：作为负例和协议诊断，不再作为当前主筛选场。

## 目录结构

- `context/`：背景文档、主线标准、reload 混淆、实验 registry。
- `code_refs/`：当前 LADD HBB 关键源码快照，包括 `model.py`、`loss.py`、`trainer.py`、`base_hbb.py`、`train_ladd_hbb.py`。
- `evidence/local/`：本地已整理的曲线 CSV、图、DroneVehicle/OGSOD plot data、AutoDL 拉回证据、历史 LADD 报告。
- `evidence/remote_3090/tar_snapshot/`：从 `ladd3090-zw1` 拉回的当前 OGSOD YOLO-init 主线/扫参原始轻量证据，包含 `results.csv`、`ladd_diagnostics.csv`、`args.yaml`、启动命令和外层日志。
- `evidence/remote_4090/`：预留给 `ladd4090-zw1` 原始证据。本次 4090 SSH 在采集时不稳定，未成功拉回最新远端目录；已在本地证据中包含 4090 plot data 和图。
- `tables/`：机器可读总表。
- `runtime_logs/`：采集时的服务器状态快照和同步缺口说明。
- `scripts/`：监控脚本快照。

## 关键表

- `tables/all_results_summary.csv`：包内所有可解析结果 CSV 的统一摘要，字段包括 rows、latest、best、late5/10/20/50。
- `tables/ogsod_3090_candidate_deltas.csv`：3090 当前候选相对同机 same-pipeline det-only control 的同 epoch delta。
- `tables/manifest_files.csv`：包内文件清单和大小。

当前 3090 delta 表中较值得优先看的早期现象：

- `dynamic_reach_rawinput_yoloinit`：pre100，但 latest delta 和 late20 delta 当前最高。
- `dynamic_plain_yoloinit`：接近 100 epoch，持续小正。
- `dynamic_singleproj_yoloinit`、`dynamic_wo_s_rec_yoloinit`：超过 200 epoch，稳定小正但不到 +1 AP50-95 point。
- `dynamic_wo_reach_yoloinit`：已是 LOW_PRIORITY 负例，说明 reach 分量可能不是纯负担。

## 证据使用规则

- 不要混用 reload 和 YOLO-init 得出主线结论；reload 只作为混淆/诊断背景。
- 4090 候选只和 4090 same-pipeline det-only 比；3090 候选只和 3090 same-pipeline detonly_control 比。
- 100 epoch 只是早筛；最终主线 claim 仍需 e800 完整曲线、late-window 和 final/best 支撑。
- 不要使用 batch fallback、错误落卡、OOM 后重试变 batch 的结果作为正式证据。
- 包内不包含 checkpoint 权重、TensorBoard event、wandb 或大体积训练产物。

## 已知缺口

- `ladd4090-zw1` 在本次打包时 SSH 连接不稳定，未能拉回最新远端原始目录；包中现有 4090 证据来自本地已同步 plot data、图和主记录文档。
- 当前 3090 新 sweep 仍有多条未到 100 epoch，属于 early screening，不应过度解释。
- 当前包主要服务方法诊断，不是论文最终证据包。

## 建议审阅入口

1. 先读 `context/OGSOD_YOLOINIT_MAINLINE_SEARCH_20260624_CN.md`，了解目标、调度和实验线索。
2. 看 `tables/ogsod_3090_candidate_deltas.csv` 和 `tables/all_results_summary.csv`，快速定位正/负方向。
3. 查对应 `results.csv`、`args.yaml`、`.cmd.sh`，确认协议和参数。
4. 对照 `code_refs/`，判断当前 dynamic/LADD-like 设计中哪些损失或结构最可能导致只小幅正增益。
5. 再看 `evidence/local/dronevehicle_method_search_20260623/`，理解为什么 DroneVehicle 小风洞没有给出正结果。
