# LADD capR/gatedKD 夜间执行清单

日期：2026-06-25 03:16 CST  
Goal 简述：执行本清单，完成 capR/gatedKD 代码同步、smoke、已有 run audit、第一批 OGSOD YOLO-init early-screen 实验调度与监控。  
主分支：`codex/ladd-capr-audit-and-gated-kd-v1`  
主代码：`ladd/code/src/teacher_student_decomposition_kd_hbb/`  
训练入口：`ladd/code/train_ladd_hbb.py`，同步快照 `ladd/code_versions/current_hbb/tools/train_ladd_hbb.py`

## 0. 硬性原则

- 主线证据只看 YOLO-init，不使用 reload 作为正结果。
- 不改变最终 SAR YOLO student inference/export graph。
- `dynamic` 线必须继续跑完；不要为了新实验杀掉它。
- 可以停止明显低优先级、负向、重复、错误配置、batch fallback、OOM fallback、错误落卡、目录混淆的任务给本清单让路。
- 每次启动前必须确认 GPU、日志目录、run name、teacher/baseline 权重、dataset yaml、strict batch、结果目录唯一。
- batch fallback/OOM fallback 后的 run 只能诊断，不能作为正式证据。
- 4090 候选只和 4090 same-pipeline control 比；3090 候选只和 3090 same-pipeline control 比。

## 1. 服务器资源策略

可用服务器：

- `ladd4090-zw1`：双 RTX 4090。费用敏感，若已关机需先重新连接/确认在线；若在线可优先用于 smoke 与关键 capR/gatedKD。
- `ladd3090-zw1`：双 RTX 3090。可用于分担 audit、负控制、低风险补实验。

资源目标：

- 每张 GPU 尽量并行 3-5 条轻量 OGSOD YOLO11n img256 batch64 任务，但以不 OOM、不 fallback、不明显 I/O 拥塞为准。
- 显存利用不是强行塞满；若新任务会超过安全范围或导致慢到不可用，就保持现状。
- 低优先级可让路：已明确负向、重复、非当前 capR/gatedKD 目标、或无 same-pipeline control 的探索。
- 不停止：当前 best/关键 `dynamic`，same-pipeline det-only control，正在接近 100/800 决策点的关键候选。

## 2. 阶段 A：代码与文档同步

待办：

- [ ] 确认本地分支 `codex/ladd-capr-audit-and-gated-kd-v1` 上代码完整。
- [ ] 将本分支 commit/push 到 GitHub，commit body 说明无 checkpoint/大文件。
- [ ] 同步代码到 `ladd4090-zw1:/root/shared-nvme/LADD_public`。
- [ ] 同步代码到 `ladd3090-zw1:/root/shared-nvme/LADD_public`。
- [ ] 在两台服务器分别运行 `python -m py_compile` 检查关键文件。

完成条件：

- 两台服务器都能看到 `docs/experiments/LADD_CAPR_AUDIT_AND_GATEDKD_20260624_CN.md`。
- 两台服务器训练入口 `--help` 能显示 `cap_reachability_gap`、`--kd-target-branch`、`--shuffle-teacher-pairs`。

## 3. 阶段 B：服务器状态与任务清理

待办：

- [ ] 检查两台服务器 GPU 显存/利用率。
- [ ] 列出当前训练进程、PID、run name、GPU、rows、latest/best/late20。
- [ ] 确认 `dynamic` 相关任务是否还在跑；若因关机中断，定位最后 checkpoint/last.pt，按原协议恢复。
- [ ] 标记可停止任务：LOW_PRIORITY、错误落卡、fallback、重复旧探索。
- [ ] 停止低优先级任务前记录 PID、run_dir、原因到本文末尾日志。

完成条件：

- 明确每张 GPU 当前跑什么、为什么保留或停止。
- `dynamic` 线处于 running/resumed/done 三者之一，不丢失。

## 4. 阶段 C：训练 smoke

先在服务器上跑 1 epoch / 2 batch smoke，不直接开 e800。

必跑：

- [ ] det-only smoke：`--ladd-b-det-only --epochs 1 --max-train-batches 2`
- [ ] capR-gated KD smoke：`--rank-d-neg-cap 2.0 --kd-weight-mode cap_reachability_gap --kd-reach-tau 0.2`
- [ ] shuffled teacher smoke：同上 + `--shuffle-teacher-pairs`
- [ ] KD-to-u smoke：同上 + `--kd-target-branch u`

完成条件：

- `results.csv` 生成。
- `ladd_diagnostics.csv` 生成并包含 `cap_saturation_ratio`、`rank_active_ratio`、`kd_reach_active_ratio` 等字段。
- `args.yaml` 正确记录新增参数。
- 无 Traceback / CUDA OOM / NaN / batch fallback。

## 5. 阶段 D：已有 run audit

先 audit 现有 run，避免盲目开更多训练。

待审计：

- [ ] `dynamic_plain_yoloinit`
- [ ] `dynamic_wo_s_rec_yoloinit`
- [ ] `dynamic_singleproj_yoloinit`
- [ ] `dynamic_wo_reach_yoloinit`
- [ ] 4090 `dynamic`
- [ ] 4090 `ProbeA`
- [ ] 4090 `oldcommit_ProbeA`
- [ ] same-machine det-only controls

工具：

```bash
python ladd/code/tools/inspect_ladd_run_args.py --run-dir <run_dir>
```

输出汇总：

- `docs/review_packages/mainline_method_search_20260624/tables/capr_existing_run_audit.csv`
- 或 `runs_public/ogsod/hbb/audits/capr_existing_run_audit_20260625/`

必须回答：

- 当前主实验 `rank_d_neg_cap` 到底是多少？
- `capR_effectively_enabled` 是否为真？
- `cap_saturation_ratio` 是否接近 0？
- `rank_active_ratio` 与 `cap_blocked_active_ratio` 暗示瓶颈是 u_t 不够远，还是 q_s/z_t 不够近？

## 6. 阶段 E：learnability audit

若已有 checkpoint 可用，对关键 run 跑 paired/shuffled audit：

- [ ] dynamic_plain paired
- [ ] dynamic_plain shuffled
- [ ] dynamic_wo_s_rec paired
- [ ] dynamic_wo_s_rec shuffled
- [ ] dynamic_singleproj paired
- [ ] dynamic_wo_reach paired

工具：

```bash
python ladd/code/tools/audit_ladd_learnability_hbb.py \
  --weights <best_or_last.pt> \
  --data shared/configs/datasets_public/ogsod1_sar_detect.yaml \
  --teacher-data shared/configs/datasets_public/ogsod1_rgb_detect.yaml \
  --teacher-weights <rgb_teacher.pt> \
  --split val --imgsz 256 --batch 16 --device <gpu> \
  --max-batches 20 --max-tokens-per-level 4096 \
  --output-dir runs_public/ogsod/hbb/audits/<run_name>
```

关键指标：

- `learnability_gap_direct_mean`
- `learnability_positive_ratio`
- `r2_probe_z`
- `r2_probe_u`
- `learnability_gap_probe`
- paired vs shuffled 差异

解释：

- `z > u`：支持当前分解与 SAR learnability 一致。
- `u >= z`：说明 z/u claim 可能不成立，应回到 A-stage decomposition。
- shuffled 接近 paired：说明方法可能不是利用 paired RGB-SAR。

## 7. 阶段 F：第一批 OGSOD YOLO-init early-screen

前置：阶段 C smoke 必须通过。

每个 run 目标长度 e800，但 100 epoch 做 early screen。

优先启动：

- [ ] `dynamic_capR2_yoloinit`
- [ ] `dynamic_capR4_yoloinit`
- [ ] `dynamic_capR2_gatedKD_yoloinit`
- [ ] `dynamic_capR2_gatedKD_wo_srec_yoloinit`
- [ ] `dynamic_capR2_gatedKD_shuffledT_yoloinit`
- [ ] `dynamic_capR2_gatedKD_toU_yoloinit`

早筛标准：

- `PROMISING_EARLY`：matched >= 100，`late20_delta >= +0.010` 且 `latest_delta > 0`。
- `STRONG_EARLY`：matched >= 100，`late20_delta >= +0.020` 且 `latest_delta > 0`。
- `LOW_PRIORITY`：matched >= 120 且 `late20_delta <= 0`，或明显不稳定且无持续正增益。

机制判据：

- capR-gated KD 必须 `capR_effectively_enabled=True`。
- `cap_saturation_ratio` 不能接近 0。
- `cap_blocked_active_ratio` 不能长期过高。
- `kd_reach_active_ratio` 不应全开或全关。
- paired 应优于 shuffled。
- KD-to-z 应优于 KD-to-u。

监控工具：

```bash
python docs/experiments/monitor_ladd_capr_gatedkd_20260624.py \
  --baseline <same_machine_detonly_run_dir> \
  --run name=<candidate_run_dir>
```

## 8. 明早应交付

- [ ] 本 MD 更新了夜间执行日志。
- [ ] GitHub 上有本分支 commit/push。
- [ ] 两台服务器状态表：GPU、PID、run、rows、latest/best/late20、预计完成时间。
- [ ] smoke 结果路径与结论。
- [ ] 已有 run capR audit 表。
- [ ] learnability audit 初步结果。
- [ ] 第一批 capR/gatedKD early-screen 运行状态。
- [ ] 若任何步骤阻塞，写明阻塞原因、已尝试操作、下一步建议。

## 9. 夜间执行日志

### 2026-06-25 03:16 CST

- 创建本 runbook。
- 创建 Goal：执行本清单到明早。
- 创建当前线程 heartbeat 自动化：`ladd-capr-gatedkd-overnight-runner`，30 分钟一次。
- 下一步：连接两台服务器做状态盘点与同步。

### 2026-06-25 03:20 CST

- 确认 active goal 存在：执行本 runbook 的记录、同步、smoke、audit、恢复/调度与夜间监控。
- 确认 heartbeat 自动化存在：`ladd-capr-gatedkd-overnight-runner`。
- 本地验证：
  - `python -m py_compile` 通过：`loss.py`、`trainer.py`、`train_ladd_hbb.py`、`audit_ladd_learnability_hbb.py`、`inspect_ladd_run_args.py`、`monitor_ladd_capr_gatedkd_20260624.py`。
  - `python ladd/code/train_ladd_hbb.py --help` 已显示新增参数：`cap_reachability_gap`、`--kd-target-branch`、`--kd-reach-*`、`--shuffle-teacher-pairs`。
- 4090 服务器快照：
  - GPU0: 16268/24564 MiB, util 99%；GPU1: 12216/24564 MiB, util 99%。
  - 正在跑：det-only resume、ProbeA resume、dynamic resume、dynamic_kd0p5、dynamic_reach0p5、dynamic_srec0p05、dynamic_teacher_projectedraw。
  - `dynamic` 仍在 GPU1 running，按用户要求保护，不停止。
  - 4090 当前监控结果：det-only rows=526 best=0.50665@485；ProbeA rows=423 status=LOW_PRIORITY；dynamic rows=367 latest AP50-95 0.00016、best=0.47031@327，当前 resume 后指标异常下降但仍保留；dynamic_teacher_projectedraw rows=112 且明显负向；kd0p5/reach0p5/srec0p05 接近或未过 100，继续观察到 100 后再让路。
  - 4090 logscan 仍有旧 `dynamic_reach0p5_yoloinit` Traceback 记录；当前 resume-fixed 进程未因此停止。
- 3090 服务器快照：
  - GPU0: 16037/24576 MiB, util 100%；GPU1: 16729/24576 MiB, util 99%。
  - 正在跑：detonly_control、dynamic_singleproj、dynamic_wo_s_rec、dynamic_plain、dynamic_kd2p0、dynamic_corewarm60、dynamic_kd0p25、dynamic_reach_rawinput。
  - 3090 当前监控结果：detonly rows=370 best=0.47086；singleproj rows=295 late20_delta=+0.01383 status=PROMISING_EARLY；reach_rawinput rows=153 late20_delta=+0.01143 status=PROMISING_EARLY；wo_s_rec/dynamic_plain/kd/corewarm 为 WATCH；wo_reach rows=122 late20_delta=-0.00360 status=LOW_PRIORITY。
  - 当前最适合优先让路的是 3090 `wo_reach`；4090 可在记录后停止 `dynamic_teacher_projectedraw`，以及过 100 后仍明显负向的 kd/reach/srec 小变体。
- 下一步：
  - 只 stage 本轮 capR/gatedKD 相关文件，commit/push 到 GitHub。
  - 同步到两台服务器后运行远端 py_compile/help/smoke。

### 2026-06-25 03:24 CST

- 已 commit/push 到 GitHub：
  - commit: `5128923 Add capR audit and gated KD controls`
  - branch: `codex/ladd-capr-audit-and-gated-kd-v1`
  - 未包含 checkpoint/大文件。
- 已将 11 个本轮相关文件精确同步到两台服务器 `/root/shared-nvme/LADD_public`。
  - 第一次 tar 同步受到 macOS `._*` 元数据和远端权限恢复影响返回失败；第二次使用 `COPYFILE_DISABLE=1` 与 `tar --no-same-owner --no-same-permissions` 成功。
- 远端验证：
  - 4090 与 3090 均通过 `py_compile`。
  - 两台服务器 `train_ladd_hbb.py --help` 均可见 `cap_reachability_gap`、`--kd-target-branch`、`--shuffle-teacher-pairs`。
- 已停止 4090 两条低优先级任务并记录到远端 `docs/experiments/capr_gatedkd_stopped_tasks_20260625.log`：
  - `ogsod_yoloinit_probeA_resume_fixed_bestep383_e800_b64_img256_s0_20260625_021121_gpu0`，PGID 13312，原因：resume 后 latest AP 为 0，LOW_PRIORITY，非 protected dynamic。
  - `ogsod_yoloinit_dynamic_teacher_projectedraw_resume_fixed_bestep64_e800_b64_img256_s0_20260625_021121_gpu1`，PGID 13375，原因：projected-raw 侧线明显负向，用于释放 GPU1，非 protected dynamic。
- 停止后 4090 显存：
  - GPU0: 12058/24564 MiB；GPU1: 8527/24564 MiB。
  - `dynamic` 本体仍在 GPU1 running，未停止。
- 已在 3090 GPU0 启动四个顺序 smoke wrapper：
  - PID: 39656
  - script: `logs/capr_gatedkd_smoke_20260625/run_smokes_gpu0_20260625.sh`
  - log: `logs/capr_gatedkd_smoke_20260625/run_smokes_gpu0_20260625.log`
  - run dir: `runs_public/ogsod/hbb/capr_gatedkd_smoke_20260625/`
  - 使用 batch=16、fraction=0.02、epochs=1；训练入口没有 `--max-train-batches`，因此采用 1 epoch 小 fraction smoke。
- 发现并修复一个 audit 工具问题：
  - 旧 run 的 `args.yaml` 不保存 LADD 自定义 CLI 参数，导致 `inspect_ladd_run_args.py` 对 `rank_d_neg_cap` 等字段输出 null。
  - 已增强工具从 `logs/**/<run_name>.cmd.sh` 解析 CLI 参数；待补充 commit/push/sync。
