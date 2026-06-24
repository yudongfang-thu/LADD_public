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

### 2026-06-25 03:42 CST

- 已补充 commit/push：
  - `c5cecd4 Improve capR run inspection logging`
  - `d12fbef Fix capR inspector default normalize flag`
  - 远端 3090/4090 已同步更新后的 `inspect_ladd_run_args.py` 与本 runbook。
- 既有 run 的 capR 配置审计已生成并拉回本地：
  - `docs/review_packages/mainline_method_search_20260624/tables/capr_existing_run_audit_3090_20260625.csv`
  - `docs/review_packages/mainline_method_search_20260624/tables/capr_existing_run_audit_4090_20260625.csv`
  - 3090 主线/扫参 run 的 `rank_d_neg_cap=2.0`、`normalize_reach=True`、`capR_effectively_enabled_computed=True` 已能从 `.cmd.sh` 恢复；4090 resume-fixed 线多为恢复命令，不能可靠反推原始 LADD 自定义参数。
- 3090 四个 smoke 已全部完成，均生成 `results.csv` 与 `ladd_diagnostics.csv`：
  - `smoke_detonly_20260625_0328_gpu0`
  - `smoke_capR_gatedKD_20260625_0328_gpu0`
  - `smoke_capR_gatedKD_shuffledT_20260625_0328_gpu0`
  - `smoke_capR_gatedKD_toU_20260625_0328_gpu0`
  - smoke 日志：`logs/capr_gatedkd_smoke_20260625/run_smokes_gpu0_20260625.log`
- 3090 当前保留的旧实验：
  - GPU0: `detonly_control_yoloinit`、`dynamic_singleproj_yoloinit`
  - GPU1: `dynamic_wo_s_rec_yoloinit`、`dynamic_plain_yoloinit`、`dynamic_reach_rawinput_yoloinit`
  - `dynamic_plain` 与 4090 `dynamic` 均按用户要求保留继续跑完。
- 3090 第一批 capR/gatedKD early-screen 已启动，均为 YOLO-init e800，使用 3090 same-pipeline `detonly_control_yoloinit` 作对照：
  - GPU0 PID 41799：`dynamic_capR2_yoloinit`
    - run: `runs_public/ogsod/hbb/capr_gatedkd_early_20260625/dynamic_capR2_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_capR2_yoloinit_yolo11n_e800_b64_img256_s0_20260625_034019_gpu0`
    - args: `--rank-d-neg-cap 2.0 --kd-weight-mode none`
  - GPU0 PID 41805：`dynamic_capR4_yoloinit`
    - run: `runs_public/ogsod/hbb/capr_gatedkd_early_20260625/dynamic_capR4_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_capR4_yoloinit_yolo11n_e800_b64_img256_s0_20260625_034019_gpu0`
    - args: `--rank-d-neg-cap 4.0 --kd-weight-mode none`
  - GPU0 PID 41811：`dynamic_capR2_gatedKD_yoloinit`
    - run: `runs_public/ogsod/hbb/capr_gatedkd_early_20260625/dynamic_capR2_gatedKD_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_capR2_gatedKD_yoloinit_yolo11n_e800_b64_img256_s0_20260625_034019_gpu0`
    - args: `--rank-d-neg-cap 2.0 --kd-weight-mode cap_reachability_gap --kd-reach-tau 0.2`
  - GPU1 PID 41817：`dynamic_capR2_gatedKD_wo_srec_yoloinit`
    - run: `runs_public/ogsod/hbb/capr_gatedkd_early_20260625/dynamic_capR2_gatedKD_wo_srec_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_capR2_gatedKD_wo_srec_yoloinit_yolo11n_e800_b64_img256_s0_20260625_034019_gpu1`
    - args: gatedKD + `--alpha-s-rec 0.0`
  - GPU1 PID 41823：`dynamic_capR2_gatedKD_shuffledT_yoloinit`
    - run: `runs_public/ogsod/hbb/capr_gatedkd_early_20260625/dynamic_capR2_gatedKD_shuffledT_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_capR2_gatedKD_shuffledT_yoloinit_yolo11n_e800_b64_img256_s0_20260625_034019_gpu1`
    - args: gatedKD + `--shuffle-teacher-pairs`
- 启动健康检查：
  - `.cmd.sh` 已确认包含对应新增参数。
  - 03:41 CST 进程均在运行；3090 GPU0 有 5 个训练主进程、GPU1 有 5 个训练主进程。
  - 03:41 CST 显存：GPU0 7623/24576 MiB，GPU1 13310/24576 MiB；新 run 仍处于初始化/AMP 检查阶段，后续需继续检查 `results.csv` 与 `ladd_diagnostics.csv`。
- 注意：
  - 03:39 CST 曾有一轮 malformed launch，额外参数被换行写到命令外；该轮未留下训练进程，已记录并弃用，正式有效 run 为 03:40:19 CST 启动的 5 条。
  - `dynamic_capR2_gatedKD_toU_yoloinit` 暂未启动，等待安全余量或第一批达到早筛点后再作为负控制补上。

### 2026-06-25 03:52 CST

- 发现并处理 3090 第一批 early-screen 的 cache race：
  - 03:40:19 CST 启动的 5 条中，`dynamic_capR2_yoloinit` 正常进入训练。
  - 其余 4 条在 dataloader 阶段同时刷新 `/root/shared-nvme/data/OGSOD-1.0/sar/labels/train.cache`，出现 `AssertionError` 后并发 `unlink`，最终 `FileNotFoundError: train.cache`。这不是方法 loss/OOM 问题，但这些 4 条 03:40:19 run 视为 INVALID，不用于结果。
  - 确认 SAR/RGB cache 文件存在后，基于原 `.cmd.sh` 复制参数并替换唯一 timestamp，于 03:44:47 CST 重新启动 4 条 retry-cache run：
    - GPU0 PID 42821/42826：`dynamic_capR4_yoloinit`，run name timestamp `20260625_034447`
    - GPU0 PID 42836/42841：`dynamic_capR2_gatedKD_yoloinit`，run name timestamp `20260625_034447`
    - GPU1 PID 42851/42856：`dynamic_capR2_gatedKD_wo_srec_yoloinit`，run name timestamp `20260625_034447`
    - GPU1 PID 42866/42868：`dynamic_capR2_gatedKD_shuffledT_yoloinit`，run name timestamp `20260625_034447`
  - retry-cache 4 条均确认包含正确参数，且截至 03:51 CST 无 Traceback / OOM / NaN / batch fallback。
- 3090 当前 GPU 负载：
  - 03:51 CST：GPU0 20188/24576 MiB、util 99%；GPU1 20876/24576 MiB、util 100%。
  - 已达到较充分利用且低于 22G 危险线，因此暂不启动 `dynamic_capR2_gatedKD_toU_yoloinit`，避免把显存推到危险区。
- 3090 当前早筛表，均与同机 `detonly_control_yoloinit` 做 epoch-matched 对比：
  - `detonly_control`: rows=384, latest/best/late20 AP50-95 = 0.47530 / 0.47530@384 / 0.47212
  - `dynamic_singleproj`: rows=306, latest=0.46718, late20=0.46252, latest_delta=+0.01569, late20_delta=+0.01424
  - `dynamic_wo_s_rec`: rows=321, latest=0.46668, late20=0.46265, latest_delta=+0.01029, late20_delta=+0.00941
  - `dynamic_plain`: rows=193, latest=0.40494, late20=0.39920, latest_delta=+0.00594, late20_delta=+0.00615
  - `dynamic_reach_rawinput`: rows=165, latest=0.39522, late20=0.38901, latest_delta=+0.01269, late20_delta=+0.01147
  - `dynamic_capR2_yoloinit`: rows=2, latest=0.02820, late20_delta=-0.00773，仅为极早期，不能判断。
  - `dynamic_capR4_yoloinit` retry: rows=1, latest=0.03219，仅为极早期。
  - `dynamic_capR2_gatedKD_yoloinit` retry: rows=1, latest=0.05082，仅为极早期。
  - `dynamic_capR2_gatedKD_wo_srec_yoloinit` retry: rows=1, latest=0.02377，仅为极早期。
  - `dynamic_capR2_gatedKD_shuffledT_yoloinit` retry: rows=1, latest=0.03268，仅为极早期。
- 修复 learnability audit 工具并完成 mini audit：
  - 原 bug：全局 summary 直接拼接 P3/P4/P5 的 `q/z/u` 特征，通道数 64/128/256 不一致导致 `RuntimeError: Sizes of tensors must match`。
  - 修复：probe 只在 per-level 内计算，summary 的 probe 指标改为 token-weighted per-level 汇总；全局只拼 scalar gap/fg，不改变训练或推理路径。
  - 已同步修复后的 `ladd/code/tools/audit_ladd_learnability_hbb.py` 到 3090/4090。
  - 已在 3090 CPU 上对 `dynamic_singleproj` 做 paired/shuffled mini audit（batch=2, max_batches=2, max_tokens_per_level=512），结果拉回：
    - `docs/review_packages/mainline_method_search_20260624/tables/learnability_dynamic_singleproj_paired_cpu_mini_v2_summary_20260625.csv`
    - `docs/review_packages/mainline_method_search_20260624/tables/learnability_dynamic_singleproj_paired_cpu_mini_v2_per_level_20260625.csv`
    - `docs/review_packages/mainline_method_search_20260624/tables/learnability_dynamic_singleproj_shuffled_cpu_mini_v2_summary_20260625.csv`
    - `docs/review_packages/mainline_method_search_20260624/tables/learnability_dynamic_singleproj_shuffled_cpu_mini_v2_per_level_20260625.csv`
  - paired mini summary：tokens=1280, `learnability_gap_direct_mean=2.29088`, `learnability_positive_ratio=1.0`, `r2_probe_z=0.96599`, `r2_probe_u=0.73459`, `learnability_gap_probe=+0.23140`, `cos_probe_z=0.99523`, `cos_probe_u=0.86681`。
  - shuffled mini summary：tokens=1280, `learnability_gap_direct_mean=2.29184`, `learnability_positive_ratio=1.0`, `r2_probe_z=0.96777`, `r2_probe_u=0.73125`, `learnability_gap_probe=+0.23652`, `cos_probe_z=0.99519`, `cos_probe_u=0.86395`。
  - 初步解释：mini audit 支持 z 比 u 更容易被当前 SAR-side feature 线性预测；但 paired 与 shuffled 几乎一致，说明这个 mini 设置主要反映 z/u 几何与分支性质，还不能证明 paired RGB-SAR 信息被利用。后续需要更完整 max-batches、更多 run，并把 paired-vs-shuffled 设计得更敏感。
- 下一步：
  - 继续监控 retry-cache 4 条到 >=10/20 rows，确认 diagnostics 字段存在且无错误。
  - 若 GPU 仍维持 20-21G，不新增 KD-to-u；等某条低优先级结束/停止后再补负控制。
  - 将本次 runbook、audit 工具修复和 mini audit CSV commit/push。

### 2026-06-25 03:59 CST

- 目标与自动化状态：
  - Codex goal 已处于 active 状态，目标为执行本 runbook 的 capR/gatedKD 夜间清单。
  - app 侧 heartbeat automation `ladd-capr-gatedkd-overnight-runner` 可查看，继续负责周期性唤醒巡检。
- 3090 服务器状态：
  - GPU0: 20244/24576 MiB, util 99%；GPU1: 20932/24576 MiB, util 100%。
  - 当前已经达到每卡约 5 条训练的高并行度，且仍低于 22G 危险线。
  - 当前不新增 `KD-to-u`，原因是再加一条可能把显存推近或超过危险区；等待某条低优先级任务结束/停止后再补负控制。
  - retry-cache 有效日志 `034447/034019` 未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
- 3090 最新 early-screen 表（same-machine det-only 对照）：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 308 | 0.46810 | 0.46810 | 0.46348 | +0.01659 | +0.01446 | - | - | PROMISING_EARLY |
| dynamic_wo_s_rec | 323 | 0.46765 | 0.46765 | 0.46350 | +0.01054 | +0.00961 | - | - | WATCH |
| dynamic_plain | 196 | 0.40607 | 0.40607 | 0.40089 | +0.00493 | +0.00608 | - | - | WATCH |
| dynamic_reach_rawinput | 168 | 0.39685 | 0.39685 | 0.39079 | +0.01243 | +0.01167 | - | - | PROMISING_EARLY |
| dynamic_capR2_yoloinit | 4 | 0.04365 | 0.05137 | 0.03199 | +0.01718 | +0.00063 | True | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 3 | 0.00436 | 0.03219 | 0.01724 | +0.00041 | -0.01575 | False | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 3 | 0.00592 | 0.05082 | 0.02138 | +0.00197 | -0.01161 | True | 1.00000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 3 | 0.00564 | 0.03251 | 0.02064 | +0.00169 | -0.01235 | True | 0.99992 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 2 | 0.04208 | 0.04208 | 0.03738 | -0.00395 | -0.01014 | True | 1.00000 | pre100 |

- 3090 诊断观察：
  - capR2/gatedKD run 已写出 `ladd_diagnostics.csv`，`capR_effectively_enabled=True`；capR4 retry 为 `False`，符合 capR4 近似 disabled 对照预期。
  - capR2/gatedKD 的 `cap_saturation_ratio` 当前接近 1，说明 capR 在极早期已经大量截断 d_neg。
  - gatedKD 的 `kd_reach_active_ratio` 极早期接近 1，若到 10/20 epoch 仍然全开，需要优先调 `kd_reach_tau/margin/min_weight/conf_power`，否则 gate 不够选择性。
- 4090 服务器状态：
  - GPU0: 12060/24564 MiB, util 98%；GPU1: 8531/24564 MiB, util 90%。
  - 4090 已停止低优先级 ProbeA/projectedraw，保留 dynamic 本体继续跑；当前 4090 更像 resume/context 线，不作为 capR 新候选 formal same-pipeline 证据。
  - 当前有效 `021121` 日志未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
- 4090 最新表（仅作 resume context）：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_resume | 51 | 0.48511 | 0.48511 | 0.48248 | -0.03395 | -0.03443 | pre100 |
| dynamic_kd0p5 | 50 | 0.35787 | 0.35787 | 0.35246 | -0.16100 | -0.16420 | pre100 |
| dynamic_reach0p5 | 50 | 0.35975 | 0.35975 | 0.35463 | -0.15912 | -0.16203 | pre100 |
| dynamic_srec0p05 | 51 | 0.35401 | 0.35401 | 0.34925 | -0.16505 | -0.16767 | pre100 |

- 下一步：
  - 周期性巡检新 capR/gatedKD retry runs 到 >=10/20 rows，重点看 `kd_reach_active_ratio` 是否仍然全开、是否有 batch fallback/OOM。
  - 若 gatedKD 到 20 epoch 仍全开，优先准备更尖锐 gate 的后续小变体；若某条旧 WATCH/LOW_PRIORITY 线持续无希望，再释放空间补 `dynamic_capR2_gatedKD_toU_yoloinit` 负控制。
  - 本节同步到两台服务器并 commit/push，保持明早审阅材料完整。

### 2026-06-25 04:02 CST

- 自动化续跑 quick poll：
  - 3090 GPU0/GPU1: 20244/24576 MiB、20932/24576 MiB，util 99%/99%；继续处于高并行、低于 22G 危险线。
  - 4090 GPU0/GPU1: 12060/24564 MiB、8531/24564 MiB，util 99%/89%；protected `dynamic_resume` 仍在跑。
  - 3090/4090 当前有效日志均未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
- 3090 最新 rows/status：
  - `dynamic_singleproj`: rows=309, latest_delta=+0.01656, late20_delta=+0.01459, PROMISING_EARLY。
  - `dynamic_wo_s_rec`: rows=324, latest_delta=+0.01075, late20_delta=+0.00973, WATCH，接近 PROMISING 阈值。
  - `dynamic_plain`: rows=196, latest_delta=+0.00493, late20_delta=+0.00608, WATCH。
  - `dynamic_reach_rawinput`: rows=168, latest_delta=+0.01243, late20_delta=+0.01167, PROMISING_EARLY。
  - 新 capR/gatedKD 仍在 pre100 且未到 10/20 epoch 动作点：capR2 rows=5，capR4_retry rows=4，gatedKD rows=4，gatedKD_wo_srec rows=3，shuffledT rows=3。
  - gatedKD 的 `kd_reach_active_ratio` 仍接近 1，但 rows 只有 3-4，继续观察到 >=10/20 后再决定是否调 gate。
- 4090 最新 rows/status：
  - `dynamic_resume`: rows=52, latest_delta=-0.03449, late20_delta=-0.03443。
  - `dynamic_kd0p5`: rows=52, late20_delta=-0.16367；`dynamic_reach0p5`: rows=52, late20_delta=-0.16146；`dynamic_srec0p05`: rows=53, late20_delta=-0.16717。
  - 这些 4090 resume 小变体仍只作 context，不作为 capR formal 证据；暂不停止 protected `dynamic_resume`。
- 调度决定：
  - 不新增 KD-to-u，也不启动更尖锐 gate 变体；原因是 capR/gatedKD 尚未到 10/20 rows，且 3090 显存已接近 20-21G。
  - 下一次有意义动作点：capR/gatedKD retry runs 到 >=10 rows 做健康检查；到 >=20 rows 判断 `kd_reach_active_ratio` 是否全开并决定是否启更尖锐 gate 变体。
