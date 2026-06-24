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

### 2026-06-25 04:13 CST

- 自动化续跑巡检：
  - 3090 GPU0/GPU1: 20244/24576 MiB、20932/24576 MiB，util 99%/100%；仍处于 20-21G 高并行区间，暂不追加 `KD-to-u`。
  - 4090 GPU0/GPU1: 12060/24564 MiB、8531/24564 MiB，util 93%/96%；protected `dynamic_resume` 仍在跑。
  - 3090 capR retry-cache 有效日志 `034019/034447` 未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
  - 4090 当前有效 `021121` 日志未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
- 3090 路径修正：
  - `dynamic_wo_s_rec` 实际目录为 `runs_public/ogsod/hbb/yoloinit_mainline_search_20260624/dynamic_wo_s_rec_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_wo_s_rec_yoloinit_yolo11n_e800_b64_img256_s0_20260624_1608_gpu1`，不是早前误写的 `160706`。
  - `dynamic_reach_rawinput` 实际目录为 `runs_public/ogsod/hbb/yoloinit_dynamic_sweep_20260624/dynamic_reach_rawinput_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_reach_rawinput_yoloinit_yolo11n_e800_b64_img256_s0_20260624_210845_gpu1`，不在 `yoloinit_mainline_search_20260624` 目录。
- 3090 旧 dynamic 线最新 early-screen 表（修正路径后）：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 312 | 0.46933 | 0.46933 | 0.46532 | +0.01599 | +0.01495 | PROMISING_EARLY |
| dynamic_wo_s_rec | 328 | 0.46944 | 0.46944 | 0.46562 | +0.01111 | +0.01014 | PROMISING_EARLY |
| dynamic_plain | 200 | 0.40830 | 0.40830 | 0.40313 | +0.00489 | +0.00574 | WATCH |
| dynamic_reach_rawinput | 172 | 0.39895 | 0.39895 | 0.39318 | +0.01269 | +0.01193 | PROMISING_EARLY |

- 3090 新 capR/gatedKD retry 组最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 7 | 0.07796 | 0.07796 | 0.04382 | -0.00913 | -0.00198 | True | 0.999906 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 6 | 0.06647 | 0.06647 | 0.03011 | +0.00840 | -0.00882 | False | 0.000000 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 6 | 0.04737 | 0.05082 | 0.02640 | -0.01070 | -0.01252 | True | 0.999939 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 6 | 0.06106 | 0.06106 | 0.03154 | +0.00299 | -0.00738 | True | 0.999885 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 6 | 0.05425 | 0.05425 | 0.03850 | -0.00382 | -0.00042 | True | 0.999986 | 1.000000 | pre100 |

- 4090 resume/context 最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | note |
|---|---:|---:|---:|---:|---|
| det-only resume | 72 | 0.52377 | 0.52384 | 0.52199 | same-pipeline context |
| dynamic_resume | 57 | 0.48631 | 0.48674 | 0.48423 | protected dynamic, running |
| dynamic_kd0p5 | 57 | 0.36118 | 0.36118 | 0.35609 | low context |
| dynamic_reach0p5 | 57 | 0.36253 | 0.36253 | 0.35816 | low context |
| dynamic_srec0p05 | 58 | 0.35813 | 0.35813 | 0.35289 | low context |

- 调度决定：
  - 本轮不停止、不新增；3090 已接近安全显存上限，新 capR/gatedKD 组尚未到 10/20 rows 动作点。
  - `dynamic_wo_s_rec` 已从 WATCH 升为 `PROMISING_EARLY`；`dynamic_singleproj` 与 `dynamic_reach_rawinput` 继续保持 `PROMISING_EARLY`。
  - gatedKD 的 `kd_reach_active_ratio` 目前仍近似全开，但只有 6 rows；继续观察到 >=10 rows 做健康检查，>=20 rows 再决定是否开更尖锐 gate 或释放空间补 `KD-to-u` 负控制。

### 2026-06-25 04:18 CST

- 自动化续跑 10-row 健康检查：
  - 3090 GPU0/GPU1: 20244/24576 MiB、20932/24576 MiB，util 99%/100%；仍处于安全高并行上限附近，不追加新任务。
  - 4090 GPU0/GPU1: 12060/24564 MiB、8531/24564 MiB，util 98%/84%；protected `dynamic_resume` 仍在跑。
  - 3090 capR retry-cache 有效日志 `034019/034447` 仍未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
  - 4090 当前有效 `021121` 日志仍未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
- 3090 旧 dynamic 线最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 315 | 0.47073 | 0.47073 | 0.46669 | +0.01623 | +0.01531 | PROMISING_EARLY |
| dynamic_wo_s_rec | 330 | 0.47081 | 0.47081 | 0.46648 | +0.01158 | +0.01032 | PROMISING_EARLY |
| dynamic_plain | 203 | 0.41020 | 0.41020 | 0.40483 | +0.00482 | +0.00554 | WATCH |
| dynamic_reach_rawinput | 175 | 0.40002 | 0.40002 | 0.39495 | +0.01124 | +0.01201 | PROMISING_EARLY |

- 3090 新 capR/gatedKD retry 组最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 10 | 0.11132 | 0.11132 | 0.06057 | +0.00721 | -0.00010 | True | 0.999889 | 0.000000 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 9 | 0.08965 | 0.10012 | 0.04773 | -0.00047 | -0.00812 | False | 0.000000 | 0.000000 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 8 | 0.08097 | 0.08097 | 0.03735 | -0.01096 | -0.01422 | True | 0.999938 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 8 | 0.07321 | 0.07551 | 0.04224 | -0.01872 | -0.00932 | True | 0.999641 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 8 | 0.08210 | 0.09311 | 0.05078 | -0.00983 | -0.00079 | True | 0.999982 | 0.000000 | 1.000000 | pre100 |

- 10-row 健康检查结论：
  - `dynamic_capR2_yoloinit` 已到 10 rows，capR 真实启用，且 `cap_saturation_ratio` 接近 1，说明 capR 在极早期确实大量截断 d_neg；`rank_active_ratio=0`，说明当前 rank loss 基本不活跃。
  - `dynamic_capR4_yoloinit` 仍为 capR disabled 对照，符合 `rank_d_neg_cap=4.0` 近似禁用 capR 的预期。
  - 三条 gatedKD retry 仍未到 10 rows；当前 `kd_active_ratio≈1.0`，但暂不据此调参，继续等 gatedKD 自身 >=10 做健康检查，>=20 做 gate 选择性判断。
- 4090 resume/context 最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | note |
|---|---:|---:|---:|---:|---|
| det-only resume | 76 | 0.52489 | 0.52489 | 0.52279 | same-pipeline context |
| dynamic_resume | 61 | 0.48706 | 0.48706 | 0.48519 | protected dynamic, running |
| dynamic_kd0p5 | 60 | 0.36251 | 0.36251 | 0.35763 | low context |
| dynamic_reach0p5 | 60 | 0.36449 | 0.36449 | 0.35966 | low context |
| dynamic_srec0p05 | 61 | 0.35974 | 0.35974 | 0.35454 | low context |

- 调度决定：
  - 不停止、不新增；3090 显存没有足够安全余量，且 gatedKD 组尚未到自身 10/20 rows 动作点。
  - 下一动作点：gatedKD 系列到 >=10 rows 后记录健康检查；>=20 rows 后若 `kd_reach_active_ratio` 仍近似全开，优先考虑更尖锐 gate 变体或释放低优先级空间补 `dynamic_capR2_gatedKD_toU_yoloinit` 负控制。

### 2026-06-25 04:21 CST

- 自动化短 poll：
  - 3090 GPU0/GPU1: 20244/24576 MiB、20932/24576 MiB，util 100%/99%；继续不追加新任务。
  - 3090 capR retry-cache 日志仍未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
- 新 capR/gatedKD retry 组状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 11 | 0.11319 | 0.11319 | 0.06536 | +0.00111 | +0.00001 | True | 0.999899 | 0.000000 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 10 | 0.10973 | 0.10973 | 0.05393 | +0.00562 | -0.00675 | False | 0.000000 | 0.000000 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 9 | 0.09225 | 0.09225 | 0.04345 | +0.00213 | -0.01240 | True | 0.999911 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 9 | 0.09812 | 0.09812 | 0.04845 | +0.00800 | -0.00740 | True | 0.999724 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 9 | 0.09594 | 0.09594 | 0.05579 | +0.00582 | -0.00006 | True | 0.999973 | 0.000000 | 1.000000 | pre100 |

- 10-row 对照检查结论：
  - `dynamic_capR4_yoloinit` 到 10 rows，`capR_enabled=False`、`cap_saturation_ratio=0`，符合 capR disabled 对照预期。
  - `dynamic_capR2_yoloinit` 到 11 rows，`cap_saturation_ratio` 仍接近 1、`rank_active_ratio=0`，说明 capR 继续大量截断 d_neg 且 rank loss 基本不活跃；late20_delta 目前从轻微负值回到约 0。
  - 三条 gatedKD 仍是 9 rows；`kd_active_ratio` 仍近似 1，但继续等自身 >=10 rows 后再记录健康检查，>=20 rows 后再判断是否调 gate。
- 调度决定：
  - 不停止、不新增；等待 gatedKD 三条到 >=10/20 rows。

### 2026-06-25 04:23 CST

- 自动化续跑 gatedKD 10-row 健康检查：
  - 3090 GPU0/GPU1: 20244/24576 MiB、20932/24576 MiB，util 99%/100%；继续无安全余量追加新任务。
  - 4090 GPU0/GPU1: 12060/24564 MiB、8531/24564 MiB，util 99%/91%；protected `dynamic_resume` 继续运行。
  - 3090 capR retry-cache 日志仍未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
  - 4090 当前有效 `021121` 日志仍未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
- 3090 旧 dynamic 线最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 316 | 0.47057 | 0.47073 | 0.46710 | +0.01578 | +0.01538 | PROMISING_EARLY |
| dynamic_wo_s_rec | 332 | 0.47111 | 0.47111 | 0.46730 | +0.01105 | +0.01044 | PROMISING_EARLY |
| dynamic_plain | 204 | 0.41126 | 0.41126 | 0.40542 | +0.00566 | +0.00549 | WATCH |
| dynamic_reach_rawinput | 176 | 0.40013 | 0.40013 | 0.39550 | +0.01114 | +0.01201 | PROMISING_EARLY |

- 新 capR/gatedKD retry 组状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 11 | 0.11319 | 0.11319 | 0.06536 | +0.00111 | +0.00001 | True | 0.999899 | 0.000000 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 10 | 0.10973 | 0.10973 | 0.05393 | +0.00562 | -0.00675 | False | 0.000000 | 0.000000 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 10 | 0.11051 | 0.11051 | 0.05015 | +0.00640 | -0.01052 | True | 0.999884 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 9 | 0.09812 | 0.09812 | 0.04845 | +0.00800 | -0.00740 | True | 0.999724 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 9 | 0.09594 | 0.09594 | 0.05579 | +0.00582 | -0.00006 | True | 0.999973 | 0.000000 | 1.000000 | pre100 |

- gatedKD 10-row 健康检查结论：
  - `dynamic_capR2_gatedKD_yoloinit` 到 10 rows，capR 真实启用，`cap_saturation_ratio=0.999884`，`rank_active_ratio=0`。
  - `kd_reach_active_ratio` 仍约 1.0，说明当前 `cap_reachability_gap` gate 在 10-row 极早期基本全开，还没有形成 token 选择性。
  - 按清单不在 10-row 立即改 gate；继续等 gatedKD 系列 >=20 rows。如果到 >=20 rows 仍全开，优先开更尖锐 gate 变体或释放空间补 `KD-to-u` 负控制。
  - `gatedKD_wo_srec` 与 `shuffledT` 仍是 9 rows，等待自身 >=10 rows 后再记录健康检查。
- 4090 resume/context 最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | note |
|---|---:|---:|---:|---:|---|
| det-only resume | 79 | 0.52500 | 0.52507 | 0.52338 | same-pipeline context |
| dynamic_resume | 63 | 0.48738 | 0.48738 | 0.48561 | protected dynamic, running |
| dynamic_kd0p5 | 63 | 0.36434 | 0.36434 | 0.35925 | low context |
| dynamic_reach0p5 | 63 | 0.36599 | 0.36599 | 0.36117 | low context |
| dynamic_srec0p05 | 64 | 0.36215 | 0.36215 | 0.35626 | low context |

- 调度决定：
  - 不停止、不新增；3090 显存仍接近上限，且关键 gate 调整决策等待 >=20 rows。

### 2026-06-25 04:24 CST

- 自动化续跑 `gatedKD_wo_srec` 10-row 健康检查：
  - 3090 GPU0/GPU1: 20244/24576 MiB、20932/24576 MiB，util 99%/100%；继续无安全余量追加新任务。
  - 4090 GPU0/GPU1: 12060/24564 MiB、8531/24564 MiB，util 99%/91%；protected `dynamic_resume` 继续运行。
  - 3090 capR retry-cache 日志仍未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
  - 4090 当前有效 `021121` 日志仍未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
- 新 capR/gatedKD retry 组状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 11 | 0.11319 | 0.11319 | 0.06536 | +0.00111 | +0.00001 | True | 0.999899 | 0.000000 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 11 | 0.11554 | 0.11554 | 0.05953 | +0.00346 | -0.00582 | False | 0.000000 | 0.000000 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 10 | 0.11051 | 0.11051 | 0.05015 | +0.00640 | -0.01052 | True | 0.999884 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 10 | 0.10490 | 0.10490 | 0.05410 | +0.00079 | -0.00658 | True | 0.999726 | 0.000001 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 9 | 0.09594 | 0.09594 | 0.05579 | +0.00582 | -0.00006 | True | 0.999973 | 0.000000 | 1.000000 | pre100 |

- 10-row 健康检查结论：
  - `dynamic_capR2_gatedKD_wo_srec_yoloinit` 到 10 rows，capR 真实启用，`cap_saturation_ratio=0.999726`，`rank_active_ratio` 约为 0。
  - `kd_reach_active_ratio` 仍约 1.0，和主 gatedKD 一致，说明去掉 student reconstruction 后，当前 gate 在 10-row 极早期仍基本全开。
  - `dynamic_capR2_gatedKD_shuffledT_yoloinit` 仍是 9 rows，等待自身 >=10 rows 后记录负控制健康检查。
  - 继续等 gatedKD 系列 >=20 rows 后再判断是否需要更尖锐 gate；当前不停止、不新增。
- 3090 旧 dynamic 线最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 317 | 0.47016 | 0.47073 | 0.46749 | +0.01497 | +0.01544 | PROMISING_EARLY |
| dynamic_wo_s_rec | 333 | 0.47158 | 0.47158 | 0.46769 | +0.01102 | +0.01050 | PROMISING_EARLY |
| dynamic_plain | 205 | 0.41148 | 0.41148 | 0.40599 | +0.00519 | +0.00544 | WATCH |
| dynamic_reach_rawinput | 177 | 0.40106 | 0.40106 | 0.39608 | +0.01177 | +0.01205 | PROMISING_EARLY |

- 4090 resume/context 最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | note |
|---|---:|---:|---:|---:|---|
| det-only resume | 80 | 0.52530 | 0.52530 | 0.52358 | same-pipeline context |
| dynamic_resume | 64 | 0.48743 | 0.48743 | 0.48581 | protected dynamic, running |
| dynamic_kd0p5 | 64 | 0.36512 | 0.36512 | 0.35979 | low context |
| dynamic_reach0p5 | 64 | 0.36646 | 0.36646 | 0.36166 | low context |
| dynamic_srec0p05 | 65 | 0.36286 | 0.36286 | 0.35684 | low context |

### 2026-06-25 04:26 CST

- 自动化续跑 `shuffledT` 10-row 负控制健康检查：
  - 3090 GPU0/GPU1: 20244/24576 MiB、20932/24576 MiB，util 99%/99%；继续无安全余量追加新任务。
  - 4090 GPU0/GPU1: 12060/24564 MiB、8531/24564 MiB，util 98%/27%；protected `dynamic_resume` 继续运行。
  - 3090 capR retry-cache 日志仍未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
  - 4090 当前有效 `021121` 日志仍未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
- 新 capR/gatedKD retry 组状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 12 | 0.11910 | 0.11910 | 0.06983 | -0.00571 | -0.00047 | True | 0.999910 | 0.000001 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 11 | 0.11554 | 0.11554 | 0.05953 | +0.00346 | -0.00582 | False | 0.000000 | 0.000000 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 11 | 0.08926 | 0.11051 | 0.05371 | -0.02282 | -0.01164 | True | 0.999914 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 10 | 0.10490 | 0.10490 | 0.05410 | +0.00079 | -0.00658 | True | 0.999726 | 0.000001 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 10 | 0.10785 | 0.10785 | 0.06100 | +0.00374 | +0.00032 | True | 0.999978 | 0.000000 | 1.000000 | pre100 |

- `shuffledT` 10-row 负控制观察：
  - `dynamic_capR2_gatedKD_shuffledT_yoloinit` 到 10 rows，capR 真实启用，`cap_saturation_ratio=0.999978`，`rank_active_ratio=0`。
  - `kd_reach_active_ratio` 仍约 1.0，说明 shuffled teacher 负控制中 gate 也基本全开。
  - 早期 `latest_delta=+0.00374`、`late20_delta=+0.00032`，略高于 det-only 的同阶段窗口；这只是 10-row 极早期，不作结论，但需要在 20/100 row 继续观察。若 shuffledT 到 20/100 仍接近 paired gatedKD，说明当前增益可能来自 auxiliary regularization 而非 paired RGB-SAR 信息。
  - paired gatedKD 当前 rows=11，latest_delta 转负但 late20_delta 仍负；三条 gatedKD 仍需等 >=20 rows 做正式 gate/selectivity 判断。
- 3090 旧 dynamic 线最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 317 | 0.47016 | 0.47073 | 0.46749 | +0.01497 | +0.01544 | PROMISING_EARLY |
| dynamic_wo_s_rec | 333 | 0.47158 | 0.47158 | 0.46769 | +0.01102 | +0.01050 | PROMISING_EARLY |
| dynamic_plain | 206 | 0.41192 | 0.41192 | 0.40654 | +0.00508 | +0.00536 | WATCH |
| dynamic_reach_rawinput | 178 | 0.40144 | 0.40144 | 0.39664 | +0.01165 | +0.01204 | PROMISING_EARLY |

- 4090 resume/context 最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | note |
|---|---:|---:|---:|---:|---|
| det-only resume | 81 | 0.52592 | 0.52592 | 0.52379 | same-pipeline context |
| dynamic_resume | 65 | 0.48748 | 0.48748 | 0.48600 | protected dynamic, running |
| dynamic_kd0p5 | 64 | 0.36512 | 0.36512 | 0.35979 | low context |
| dynamic_reach0p5 | 64 | 0.36646 | 0.36646 | 0.36166 | low context |
| dynamic_srec0p05 | 65 | 0.36286 | 0.36286 | 0.35684 | low context |

- 调度决定：
  - 不停止、不新增；等待 gatedKD 系列 >=20 rows 后再判断 gate 全开问题和 shuffledT 负控制风险。

### 2026-06-25 04:28 CST

- 压缩后接管检查：
  - 本地 goal 仍为 active，目标为执行本 runbook 的 LADD capR/gatedKD 夜间清单。
  - heartbeat 自动化 `ladd-capr-gatedkd-overnight-runner` 可 view，继续作为 30 分钟续跑入口。
  - 上一轮提交 `301ef4d Record shuffled teacher ten-row check` 已确认推送到远端分支 `codex/ladd-capr-audit-and-gated-kd-v1`。
- 服务器负载：
  - 3090 GPU0/GPU1: 20244/24576 MiB、20932/24576 MiB，util 99%/100%；两张卡已经在安全高占用区间，不再追加任务。
  - 4090 GPU0/GPU1: 12060/24564 MiB、8531/24564 MiB，util 99%/89%；`dynamic_resume` 继续保护运行，其他低优先级 resume 线可在后续需要 capR 负控制时让路。
  - 3090 capR retry-cache 日志未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
  - 4090 当前有效 `021121` 日志未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
- 新 capR/gatedKD retry 组状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 13 | 0.12257 | 0.12257 | 0.07389 | -0.01185 | -0.00135 | True | 0.999948 | 0.000000 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 12 | 0.11166 | 0.11554 | 0.06388 | -0.01315 | -0.00643 | False | 0.000000 | 0.000000 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 12 | 0.10831 | 0.11051 | 0.05826 | -0.01650 | -0.01205 | True | 0.999952 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 11 | 0.11465 | 0.11465 | 0.05960 | +0.00257 | -0.00575 | True | 0.999750 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 11 | 0.10759 | 0.10785 | 0.06523 | -0.00449 | -0.00012 | True | 0.999971 | 0.000000 | 1.000000 | pre100 |

- 3090 旧 dynamic 线最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 318 | 0.46973 | 0.47073 | 0.46783 | +0.01456 | +0.01549 | PROMISING_EARLY |
| dynamic_wo_s_rec | 334 | 0.47225 | 0.47225 | 0.46810 | +0.01150 | +0.01059 | PROMISING_EARLY |
| dynamic_plain | 207 | 0.41238 | 0.41238 | 0.40712 | +0.00498 | +0.00533 | WATCH |
| dynamic_reach_rawinput | 179 | 0.40197 | 0.40197 | 0.39722 | +0.01179 | +0.01208 | PROMISING_EARLY |

- 4090 resume/context 最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | note |
|---|---:|---:|---:|---:|---|
| det-only resume | 83 | 0.52637 | 0.52637 | 0.52419 | same-pipeline context |
| dynamic_resume | 66 | 0.48764 | 0.48764 | 0.48617 | protected dynamic, running |
| dynamic_kd0p5 | 66 | 0.36631 | 0.36631 | 0.36091 | low context |
| dynamic_reach0p5 | 66 | 0.36751 | 0.36751 | 0.36266 | low context |
| dynamic_srec0p05 | 67 | 0.36443 | 0.36443 | 0.35804 | low context |

- 调度决定：
  - 3090 已经接近满载且无错误，不额外塞任务。
  - 4090 虽有显存余量，但当前 capR/gatedKD 关键组还未到 20-row 机制判断点；先不杀低优先级 resume 线，避免在判断 gate 是否全开前引入新的混杂。
  - 下一触发点：paired/wo_srec/shuffled gatedKD 均达到 >=20 rows。若 `kd_active_ratio` 仍接近 1.0，再记录 gate 全开问题，并考虑在 4090 停低优先级 resume 线后补 `dynamic_capR2_gatedKD_toU_yoloinit` 或更尖锐 gate 变体。

### 2026-06-25 04:30 CST

- 轻量续跑检查：
  - 3090 GPU0/GPU1: 20244/24576 MiB、20932/24576 MiB，util 99%/99%；仍是接近满载的安全高占用。
  - 4090 GPU0/GPU1: 12060/24564 MiB、8531/24564 MiB，util 98%/84%；`dynamic_resume` 继续保护运行。
  - 3090 capR retry-cache 日志仍未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
  - 4090 当前有效 `021121` 日志仍未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
- 新 capR/gatedKD retry 组状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 13 | 0.12257 | 0.12257 | 0.07389 | -0.01185 | -0.00135 | True | 0.999948 | 0.000000 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 13 | 0.12881 | 0.12881 | 0.06887 | -0.00561 | -0.00637 | False | 0.000000 | 0.000001 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 12 | 0.10831 | 0.11051 | 0.05826 | -0.01650 | -0.01205 | True | 0.999952 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 11 | 0.11465 | 0.11465 | 0.05960 | +0.00257 | -0.00575 | True | 0.999750 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 11 | 0.10759 | 0.10785 | 0.06523 | -0.00449 | -0.00012 | True | 0.999971 | 0.000000 | 1.000000 | pre100 |

- 3090 旧 dynamic 线最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 319 | 0.47081 | 0.47081 | 0.46820 | +0.01547 | +0.01557 | PROMISING_EARLY |
| dynamic_wo_s_rec | 335 | 0.47274 | 0.47274 | 0.46852 | +0.01164 | +0.01067 | PROMISING_EARLY |
| dynamic_plain | 207 | 0.41238 | 0.41238 | 0.40712 | +0.00498 | +0.00533 | WATCH |
| dynamic_reach_rawinput | 179 | 0.40197 | 0.40197 | 0.39722 | +0.01179 | +0.01208 | PROMISING_EARLY |

- 4090 resume/context 最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | note |
|---|---:|---:|---:|---:|---|
| det-only resume | 84 | 0.52668 | 0.52668 | 0.52441 | same-pipeline context |
| dynamic_resume | 67 | 0.48802 | 0.48802 | 0.48635 | protected dynamic, running |
| dynamic_kd0p5 | 67 | 0.36675 | 0.36675 | 0.36146 | low context |
| dynamic_reach0p5 | 67 | 0.36845 | 0.36845 | 0.36318 | low context |
| dynamic_srec0p05 | 68 | 0.36502 | 0.36502 | 0.35866 | low context |

- 调度决定：
  - gatedKD 三条线仍只有 11-12 rows，尚未到 >=20 rows 的 gate/selectivity 判断点。
  - `kd_active_ratio` 仍约 1.0，但按清单先等 20-row 触发点；当前不停止、不新增。

### 2026-06-25 04:32 CST

- 轻量续跑检查：
  - 3090 GPU0/GPU1: 20244/24576 MiB、20932/24576 MiB，util 100%/99%；仍不追加任务。
  - 4090 GPU0/GPU1: 12060/24564 MiB、8531/24564 MiB，util 99%/93%；`dynamic_resume` 继续保护运行。
  - 3090/4090 日志扫描未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
- 新 capR/gatedKD retry 组状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 14 | 0.12779 | 0.12779 | 0.07774 | -0.00456 | -0.00158 | True | 0.999989 | 0.000005 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 13 | 0.12881 | 0.12881 | 0.06887 | -0.00561 | -0.00637 | False | 0.000000 | 0.000001 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 13 | 0.11896 | 0.11896 | 0.06293 | -0.01546 | -0.01231 | True | 0.999971 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 12 | 0.10316 | 0.11465 | 0.06323 | -0.02165 | -0.00707 | True | 0.999881 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 12 | 0.11427 | 0.11427 | 0.06932 | -0.01054 | -0.00098 | True | 0.999939 | 0.000000 | 1.000000 | pre100 |

- 3090 旧 dynamic 线最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 320 | 0.47081 | 0.47081 | 0.46853 | +0.01460 | +0.01560 | PROMISING_EARLY |
| dynamic_wo_s_rec | 335 | 0.47274 | 0.47274 | 0.46852 | +0.01164 | +0.01067 | PROMISING_EARLY |
| dynamic_plain | 208 | 0.41330 | 0.41330 | 0.40770 | +0.00542 | +0.00531 | WATCH |
| dynamic_reach_rawinput | 180 | 0.40273 | 0.40273 | 0.39781 | +0.01217 | +0.01214 | PROMISING_EARLY |

- 4090 resume/context 最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | note |
|---|---:|---:|---:|---:|---|
| det-only resume | 86 | 0.52705 | 0.52705 | 0.52484 | same-pipeline context |
| dynamic_resume | 68 | 0.48858 | 0.48858 | 0.48655 | protected dynamic, running |
| dynamic_kd0p5 | 68 | 0.36724 | 0.36724 | 0.36200 | low context |
| dynamic_reach0p5 | 68 | 0.36930 | 0.36930 | 0.36370 | low context |
| dynamic_srec0p05 | 69 | 0.36487 | 0.36502 | 0.35924 | low context |

- 调度决定：
  - capR/gatedKD retry 组仍未到 20-row gate/selectivity 判断点。
  - 当前 paired gatedKD 和 wo_srec 均明显落后 det-only；shuffledT 的 late20 也转为轻微负值，但样本太早，不作降级。
  - 不停止、不新增；继续等待 >=20 rows。

### 2026-06-25 04:34 CST

- 轻量续跑检查：
  - 3090 GPU0/GPU1: 20244/24576 MiB、20932/24576 MiB，util 99%/100%；继续满载安全运行。
  - 4090 GPU0/GPU1: 12060/24564 MiB、8531/24564 MiB，util 99%/89%；`dynamic_resume` 继续保护运行。
  - 3090/4090 日志扫描未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
- 新 capR/gatedKD retry 组状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 15 | 0.13842 | 0.13842 | 0.08179 | -0.00363 | -0.00171 | True | 0.999993 | 0.000000 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 14 | 0.13041 | 0.13041 | 0.07327 | -0.00194 | -0.00605 | False | 0.000000 | 0.000000 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 13 | 0.11896 | 0.11896 | 0.06293 | -0.01546 | -0.01231 | True | 0.999971 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 12 | 0.10316 | 0.11465 | 0.06323 | -0.02165 | -0.00707 | True | 0.999881 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 12 | 0.11427 | 0.11427 | 0.06932 | -0.01054 | -0.00098 | True | 0.999939 | 0.000000 | 1.000000 | pre100 |

- 3090 旧 dynamic 线最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 321 | 0.47126 | 0.47126 | 0.46884 | +0.01487 | +0.01560 | PROMISING_EARLY |
| dynamic_wo_s_rec | 336 | 0.47293 | 0.47293 | 0.46895 | +0.01130 | +0.01076 | PROMISING_EARLY |
| dynamic_plain | 209 | 0.41383 | 0.41383 | 0.40829 | +0.00592 | +0.00532 | WATCH |
| dynamic_reach_rawinput | 181 | 0.40343 | 0.40343 | 0.39837 | +0.01228 | +0.01213 | PROMISING_EARLY |

- 4090 resume/context 最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | note |
|---|---:|---:|---:|---:|---|
| det-only resume | 87 | 0.52737 | 0.52737 | 0.52507 | same-pipeline context |
| dynamic_resume | 69 | 0.48876 | 0.48876 | 0.48677 | protected dynamic, running |
| dynamic_kd0p5 | 69 | 0.36780 | 0.36780 | 0.36254 | low context |
| dynamic_reach0p5 | 69 | 0.36994 | 0.36994 | 0.36426 | low context |
| dynamic_srec0p05 | 70 | 0.36559 | 0.36559 | 0.35985 | low context |

- 调度决定：
  - gatedKD 三条线仍是 13/12/12 rows，未到 >=20 rows；继续等待。
  - capR2/capR4 正常推进；capR2 的 cap saturation 仍接近 1，rank active 接近 0。
  - 不停止、不新增。

### 2026-06-25 04:36 CST

- 轻量续跑检查：
  - 3090 GPU0/GPU1: 20244/24576 MiB、20932/24576 MiB，util 100%/99%；继续满载安全运行。
  - 4090 GPU0/GPU1: 12060/24564 MiB、8531/24564 MiB，util 98%/93%；`dynamic_resume` 继续保护运行。
  - 3090/4090 日志扫描未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
- 新 capR/gatedKD retry 组状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 15 | 0.13842 | 0.13842 | 0.08179 | -0.00363 | -0.00171 | True | 0.999993 | 0.000000 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 14 | 0.13041 | 0.13041 | 0.07327 | -0.00194 | -0.00605 | False | 0.000000 | 0.000000 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 14 | 0.12420 | 0.12420 | 0.06731 | -0.00815 | -0.01201 | True | 0.999976 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 13 | 0.12563 | 0.12563 | 0.06803 | -0.00879 | -0.00721 | True | 0.999952 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 13 | 0.12796 | 0.12796 | 0.07383 | -0.00646 | -0.00141 | True | 0.999910 | 0.000000 | 1.000000 | pre100 |

- 3090 旧 dynamic 线最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 321 | 0.47126 | 0.47126 | 0.46884 | +0.01487 | +0.01560 | PROMISING_EARLY |
| dynamic_wo_s_rec | 337 | 0.47311 | 0.47311 | 0.46937 | +0.01134 | +0.01085 | PROMISING_EARLY |
| dynamic_plain | 210 | 0.41417 | 0.41417 | 0.40885 | +0.00527 | +0.00528 | WATCH |
| dynamic_reach_rawinput | 182 | 0.40350 | 0.40350 | 0.39891 | +0.01135 | +0.01210 | PROMISING_EARLY |

- 4090 resume/context 最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | note |
|---|---:|---:|---:|---:|---|
| det-only resume | 88 | 0.52733 | 0.52737 | 0.52529 | same-pipeline context |
| dynamic_resume | 70 | 0.48903 | 0.48903 | 0.48699 | protected dynamic, running |
| dynamic_kd0p5 | 70 | 0.36845 | 0.36845 | 0.36307 | low context |
| dynamic_reach0p5 | 70 | 0.37070 | 0.37070 | 0.36480 | low context |
| dynamic_srec0p05 | 71 | 0.36585 | 0.36585 | 0.36044 | low context |

- 调度决定：
  - gatedKD 三条线仍是 14/13/13 rows，未到 >=20 rows；继续等待。
  - `kd_active_ratio` 仍约 1.0，说明 gate 早期仍全开，但按 runbook 先等 20-row 触发点再做机制判断。
  - 不停止、不新增。

### 2026-06-25 04:38 CST

- 轻量续跑检查：
  - 3090 GPU0/GPU1: 20244/24576 MiB、20932/24576 MiB，util 100%/99%；继续满载安全运行。
  - 4090 GPU0/GPU1: 12060/24564 MiB、8531/24564 MiB，util 99%/89%；`dynamic_resume` 继续保护运行。
  - 3090/4090 日志扫描未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
- 新 capR/gatedKD retry 组状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 16 | 0.12270 | 0.13842 | 0.08434 | -0.01330 | -0.00244 | True | 0.999998 | 0.000001 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 15 | 0.13609 | 0.13609 | 0.07745 | -0.00596 | -0.00604 | False | 0.000000 | 0.000000 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 14 | 0.12420 | 0.12420 | 0.06731 | -0.00815 | -0.01201 | True | 0.999976 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 13 | 0.12563 | 0.12563 | 0.06803 | -0.00879 | -0.00721 | True | 0.999952 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 13 | 0.12796 | 0.12796 | 0.07383 | -0.00646 | -0.00141 | True | 0.999910 | 0.000000 | 1.000000 | pre100 |

- 3090 旧 dynamic 线最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 322 | 0.47164 | 0.47164 | 0.46915 | +0.01449 | +0.01557 | PROMISING_EARLY |
| dynamic_wo_s_rec | 337 | 0.47311 | 0.47311 | 0.46937 | +0.01134 | +0.01085 | PROMISING_EARLY |
| dynamic_plain | 210 | 0.41417 | 0.41417 | 0.40885 | +0.00527 | +0.00528 | WATCH |
| dynamic_reach_rawinput | 182 | 0.40350 | 0.40350 | 0.39891 | +0.01135 | +0.01210 | PROMISING_EARLY |

- 4090 resume/context 最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | note |
|---|---:|---:|---:|---:|---|
| det-only resume | 89 | 0.52791 | 0.52791 | 0.52552 | same-pipeline context |
| dynamic_resume | 71 | 0.48956 | 0.48956 | 0.48721 | protected dynamic, running |
| dynamic_kd0p5 | 71 | 0.36914 | 0.36914 | 0.36361 | low context |
| dynamic_reach0p5 | 71 | 0.37049 | 0.37070 | 0.36532 | low context |
| dynamic_srec0p05 | 72 | 0.36665 | 0.36665 | 0.36106 | low context |

- 调度决定：
  - gatedKD 三条线仍是 14/13/13 rows，未到 >=20 rows；继续等待。
  - capR2/capR4 正常推进；gatedKD 和负控制目前均未达到机制判断窗口。
  - 不停止、不新增。

### 2026-06-25 04:40 CST

- 轻量续跑检查：
  - 3090 GPU0/GPU1: 20244/24576 MiB、20932/24576 MiB，util 100%/99%；继续满载安全运行。
  - 4090 GPU0/GPU1: 12060/24564 MiB、8531/24564 MiB，util 98%/87%；`dynamic_resume` 继续保护运行。
  - 3090/4090 日志扫描未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
- 新 capR/gatedKD retry 组状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 16 | 0.12270 | 0.13842 | 0.08434 | -0.01330 | -0.00244 | True | 0.999998 | 0.000001 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 15 | 0.13609 | 0.13609 | 0.07745 | -0.00596 | -0.00604 | False | 0.000000 | 0.000000 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 15 | 0.12342 | 0.12420 | 0.07105 | -0.01863 | -0.01245 | True | 0.999980 | 0.000001 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 14 | 0.12307 | 0.12563 | 0.07196 | -0.00928 | -0.00735 | True | 0.999958 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 14 | 0.13392 | 0.13392 | 0.07812 | +0.00157 | -0.00119 | True | 0.999889 | 0.000000 | 1.000000 | pre100 |

- 3090 旧 dynamic 线最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 322 | 0.47164 | 0.47164 | 0.46915 | +0.01449 | +0.01557 | PROMISING_EARLY |
| dynamic_wo_s_rec | 338 | 0.47407 | 0.47407 | 0.46983 | +0.01178 | +0.01096 | PROMISING_EARLY |
| dynamic_plain | 211 | 0.41449 | 0.41449 | 0.40941 | +0.00522 | +0.00525 | WATCH |
| dynamic_reach_rawinput | 183 | 0.40402 | 0.40402 | 0.39943 | +0.01181 | +0.01209 | PROMISING_EARLY |

- 4090 resume/context 最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | note |
|---|---:|---:|---:|---:|---|
| det-only resume | 90 | 0.52815 | 0.52815 | 0.52575 | same-pipeline context |
| dynamic_resume | 72 | 0.48977 | 0.48977 | 0.48744 | protected dynamic, running |
| dynamic_kd0p5 | 72 | 0.36943 | 0.36943 | 0.36415 | low context |
| dynamic_reach0p5 | 72 | 0.37098 | 0.37098 | 0.36584 | low context |
| dynamic_srec0p05 | 73 | 0.36735 | 0.36735 | 0.36168 | low context |

- 调度决定：
  - gatedKD 三条线仍是 15/14/14 rows，未到 >=20 rows；继续等待。
  - `kd_active_ratio` 仍约 1.0；shuffledT latest 短暂转正但 late20 仍轻微负，尚不能判断负控制风险。
  - 不停止、不新增。

### 2026-06-25 04:42 CST

- 轻量续跑检查：
  - 3090 GPU0/GPU1: 20244/24576 MiB、20932/24576 MiB，util 99%/99%；两张 3090 已充分利用。
  - 4090 GPU0/GPU1: 12060/24564 MiB、8531/24564 MiB，util 99%/93%；`dynamic_resume` 继续保护运行，4090 低优先级 resume 线也在推进。
  - 3090/4090 日志扫描未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
- 新 capR/gatedKD retry 组状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 17 | 0.15336 | 0.15336 | 0.08840 | +0.00193 | -0.00218 | True | 0.999997 | 0.000000 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 16 | 0.12460 | 0.13609 | 0.08040 | -0.01140 | -0.00638 | False | 0.000000 | 0.000000 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 16 | 0.13054 | 0.13054 | 0.07476 | -0.00546 | -0.01202 | True | 0.999952 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 15 | 0.11667 | 0.12563 | 0.07494 | -0.02538 | -0.00856 | True | 0.999953 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 14 | 0.13392 | 0.13392 | 0.07812 | +0.00157 | -0.00119 | True | 0.999889 | 0.000000 | 1.000000 | pre100 |

- 3090 旧 dynamic 线最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 323 | 0.47180 | 0.47180 | 0.46944 | +0.01469 | +0.01555 | PROMISING_EARLY |
| dynamic_wo_s_rec | 339 | 0.47431 | 0.47431 | 0.47027 | +0.01194 | +0.01104 | PROMISING_EARLY |
| dynamic_plain | 212 | 0.41510 | 0.41510 | 0.40996 | +0.00532 | +0.00522 | WATCH |
| dynamic_reach_rawinput | 184 | 0.40419 | 0.40419 | 0.39991 | +0.01127 | +0.01203 | PROMISING_EARLY |

- 4090 resume/context 最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | note |
|---|---:|---:|---:|---:|---|
| det-only resume | 92 | 0.52811 | 0.52817 | 0.52618 | same-pipeline context |
| dynamic_resume | 73 | 0.48979 | 0.48979 | 0.48765 | protected dynamic, running |
| dynamic_kd0p5 | 73 | 0.36996 | 0.36996 | 0.36470 | low context |
| dynamic_reach0p5 | 73 | 0.37118 | 0.37118 | 0.36638 | low context |
| dynamic_srec0p05 | 74 | 0.36795 | 0.36795 | 0.36229 | low context |
| dynamic_teacher_projectedraw | 40 | 0.35064 | 0.35064 | 0.34561 | low context |
| ProbeA resume | 33 | 0.49752 | 0.49752 | 0.49465 | context only |

- 调度决定：
  - 新 gatedKD 三条线仍未到 >=20 rows；先不依据 14-16 epoch 的噪声做调度。
  - 目前 `kd_active_ratio` 继续接近 1.0，说明 capR-gated KD 可能实际接近“全开 gate”；到 20/50/100 rows 时需要重点判断 gate selectivity。
  - `dynamic_singleproj`、`dynamic_wo_s_rec`、`dynamic_reach_rawinput` 仍是目前最值得保护的 dynamic 证据线；不停止 dynamic。

### 2026-06-25 04:44 CST

- 轻量续跑检查：
  - 3090 GPU0/GPU1: 20244/24576 MiB、20934/24576 MiB，util 99%/99%；10 条主训练进程均在运行。
  - 4090 GPU0/GPU1: 12060/24564 MiB、8531/24564 MiB，util 98%/91%；真正仍在运行的是 det-only resume、dynamic resume、dynamic_kd0p5、dynamic_reach0p5、dynamic_srec0p05，`ProbeA` 与 `dynamic_teacher_projectedraw` 当前只是保留结果目录，进程不在。
  - 3090/4090 日志扫描未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / AssertionError / batch fallback。
- 新 capR/gatedKD retry 组状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 17 | 0.15336 | 0.15336 | 0.08840 | +0.00193 | -0.00218 | True | 0.999997 | 0.000000 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 17 | 0.14664 | 0.14664 | 0.08430 | -0.00479 | -0.00629 | False | 0.000000 | 0.000001 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 16 | 0.13054 | 0.13054 | 0.07476 | -0.00546 | -0.01202 | True | 0.999952 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 15 | 0.11667 | 0.12563 | 0.07494 | -0.02538 | -0.00856 | True | 0.999953 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 15 | 0.12727 | 0.13392 | 0.08140 | -0.01478 | -0.00210 | True | 0.999862 | 0.000000 | 1.000000 | pre100 |

- 3090 旧 dynamic 线最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 324 | 0.47202 | 0.47202 | 0.46974 | +0.01472 | +0.01555 | PROMISING_EARLY |
| dynamic_wo_s_rec | 340 | 0.47477 | 0.47477 | 0.47071 | +0.01195 | +0.01116 | PROMISING_EARLY |
| dynamic_plain | 213 | 0.41547 | 0.41547 | 0.41049 | +0.00485 | +0.00517 | WATCH |
| dynamic_reach_rawinput | 185 | 0.40455 | 0.40455 | 0.40037 | +0.01074 | +0.01193 | PROMISING_EARLY |

- 4090 resume/context 最新状态：

| run | rows | latest AP50-95 | best AP50-95 | late20 | note |
|---|---:|---:|---:|---:|---|
| det-only resume | 93 | 0.52826 | 0.52826 | 0.52641 | same-pipeline context, running |
| dynamic_resume | 74 | 0.49015 | 0.49015 | 0.48784 | protected dynamic, running |
| dynamic_kd0p5 | 74 | 0.37077 | 0.37077 | 0.36526 | low context, running |
| dynamic_reach0p5 | 74 | 0.37205 | 0.37205 | 0.36694 | low context, running |
| dynamic_srec0p05 | 75 | 0.36875 | 0.36875 | 0.36290 | low context, running |
| dynamic_teacher_projectedraw | 40 | 0.35064 | 0.35064 | 0.34561 | result dir only; no active process |
| ProbeA resume | 33 | 0.49752 | 0.49752 | 0.49465 | result dir only; no active process |

- 调度判断：
  - 新 gatedKD 三条线仍未到 >=20 rows；继续等待机制判断窗口。
  - 3090 已接近 20-21G，暂不补 `dynamic_capR2_gatedKD_toU_yoloinit`，避免把两张卡推向 OOM/fallback 风险。
  - 4090 虽有显存余量，但本机缺少 3090 capR 命令使用的等价 A-stage source：`runs_public/paper/ogsod_hbb_nomosaic/diagnostics/ladd_dynamic/yolo11n/seed0/ladd_clean_a1b_dyn_ogsod11n_diagnostic_nomosaic_dynamic_yolo11n_s0_a1_e10_b64_s0_gpu0/weights/best.pt`。直接在 4090 补 fresh capR/toU 会使 same-pipeline 对照关系变脏；本轮不启动。
  - 保留 4090 当前 `dynamic_resume`；不停止 dynamic。

### 2026-06-25 05:00 CST

- 04:47 到 05:00 期间做了一次更长窗口检查，避免在 16-18 rows 噪声上过度调度。
- 3090 GPU0/GPU1: 20244/24576 MiB、20934/24576 MiB，util 100%/100%；仍接近安全上限，不再加任务。
- 4090 GPU0/GPU1:
  - 04:53: 12060/24564 MiB、8531/24564 MiB，util 98%/24%。
  - 05:00: 新 4090 fresh 组启动后升至 14960/24564 MiB、14384/24564 MiB，util 99%/99%。
- 3090/4090 常规日志扫描未发现当前有效训练的 Traceback / RuntimeError / CUDA OOM / NaN / batch fallback。

#### 3090 capR/gatedKD 20-row 机制窗口

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 22 | 0.17512 | 0.17930 | 0.11278 | +0.00691 | -0.00079 | True | 0.999994 | 0.000002 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 22 | 0.16589 | 0.16589 | 0.10819 | -0.00232 | -0.00539 | False | 0.000000 | 0.000008 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 21 | 0.16420 | 0.16420 | 0.09476 | -0.00882 | -0.01271 | True | 0.999994 | 0.000002 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 20 | 0.15579 | 0.15579 | 0.09291 | -0.01083 | -0.00836 | True | 0.999984 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 19 | 0.15660 | 0.15660 | 0.09574 | -0.00359 | -0.00208 | True | 0.999980 | 0.000000 | 1.000000 | pre100 |

- 20-row 初步机制判断：
  - `cap_saturation_ratio` 基本为 1，说明 capR=2 在几何上确实大量生效。
  - `rank_active_ratio` 接近 0，说明 rank loss 当前几乎没有 active violation；这个阶段不是“u_t 还不够远”的问题。
  - `kd_active_ratio` 持续约 1.0，说明当前 `cap_reachability_gap` gate 基本全开，尚未形成预期的 token selectivity。
  - paired gatedKD 与 shuffledT 尚未拉开，且 paired 的 late20 delta 更负；20 rows 不能下最终结论，但这是需要重点记录的机制风险。

#### 4090 fresh 负控制组

- 为补齐 `KD-to-u` 负控制，同时避免混用 3090 对照，决定在 4090 启动同机 fresh 小组：det-only control、capR2 gatedKD-to-z、capR2 gatedKD-to-u。
- 先将 3090 上 11MB A-stage source 同步到 4090：
  - `runs_public/paper/ogsod_hbb_nomosaic/diagnostics/ladd_dynamic/yolo11n/seed0/ladd_clean_a1b_dyn_ogsod11n_diagnostic_nomosaic_dynamic_yolo11n_s0_a1_e10_b64_s0_gpu0/weights/best.pt`
  - 4090 校验：`sha256sum` 前 16 位 `d240b1a62a3d4677`。
- 04:55 首次 4090 fresh 启动作废：
  - run timestamp: `20260625_045553`
  - 原因：错误使用 `configs/paper/datasets/ogsod_hbb_sar.yaml`，该 yaml 在 4090 仍指向占位路径 `/path/to/OGSOD-1.0/sar/images/test`。
  - 三条均立刻 `FileNotFoundError/RuntimeError` 退出，无 GPU 训练结果；标记 INVALID，不进入结果比较。
- 04:57 使用 4090 已验证的 clean-cache yaml 重启同机 fresh 小组：
  - SAR yaml: `/root/shared-nvme/LADD_public/debug/zw1_nomosaic_clean_cache_20260623/20260623_214553/yamls/ogsod_hbb_sar_nomosaic_zw1.yaml`
  - RGB yaml: `/root/shared-nvme/LADD_public/debug/zw1_nomosaic_clean_cache_20260623/20260623_214553/yamls/ogsod_hbb_rgb_nomosaic_zw1.yaml`
  - RGB teacher: `runs_public/ogsod/hbb/baseline_controls/mosaic_baselines_20260615/rgb_yolo11n_hbb_mosaicE800_closeAt100_s0_imported_cos_closeAt100_20260524/weights/best.pt`

| run | device | PID wrapper/main | run dir | status |
|---|---:|---|---|---|
| 4090zw1cache_detonly_control | GPU1 | 18814/18819 | `runs_public/ogsod/hbb/capr_gatedkd_early_4090_20260625/detonly_control_zw1cache/yolo11n/seed0/ogsod_yoloinit_4090zw1cache_detonly_control_caprgroup_yolo11n_e800_b64_img256_s0_20260625_045721_gpu1` | running, args.yaml generated |
| 4090zw1cache_capR2_gatedKD_z | GPU0 | 18822/18827 | `runs_public/ogsod/hbb/capr_gatedkd_early_4090_20260625/dynamic_capR2_gatedKD_z_zw1cache_yoloinit/yolo11n/seed0/ogsod_yoloinit_4090zw1cache_capR2_gatedKD_z_yolo11n_e800_b64_img256_s0_20260625_045721_gpu0` | running, args.yaml generated |
| 4090zw1cache_capR2_gatedKD_toU | GPU1 | 18830/18835 | `runs_public/ogsod/hbb/capr_gatedkd_early_4090_20260625/dynamic_capR2_gatedKD_toU_zw1cache_yoloinit/yolo11n/seed0/ogsod_yoloinit_4090zw1cache_capR2_gatedKD_toU_yolo11n_e800_b64_img256_s0_20260625_045721_gpu1` | running, args.yaml generated |

- 05:00 健康检查：
  - 三条均有 Python 子进程，GPU 显存上升到 14960/14384 MiB。
  - 日志未见 Traceback / RuntimeError / CUDA OOM / NaN / batch fallback。
  - 目前仅生成 `args.yaml`，尚未完成第 1 个 epoch；下一轮必须确认 `results.csv` 与 `ladd_diagnostics.csv` 出现。

- 调度决定：
  - 3090 继续跑原 capR/gatedKD 组，不新增。
  - 4090 新 `zw1cache` 组作为独立同机负控制证据，只和同组 det-only control 比，不与 3090 formal-yaml 组直接比较。
  - 继续保护所有 dynamic 主线。

### 2026-06-25 05:03 CST

- 3090 GPU0/GPU1: 20244/24576 MiB、20934/24576 MiB，util 99%/99%；继续满载稳定运行。
- 4090 GPU0/GPU1: 15622/24564 MiB、15670/24564 MiB，util 99%/99%；新 `zw1cache` 三条组已真实进入训练。
- 4090 磁盘：`/root/shared-nvme` 约 48G/50G，剩余 2.4G，使用率 96%；因 YOLO11n 权重较小且 `save-period=100`，短期继续跑，但后续 heartbeat 需要持续监控磁盘。
- 3090/4090 当前有效日志扫描仍未发现 Traceback / RuntimeError / CUDA OOM / NaN / batch fallback。

#### 3090 capR/gatedKD 最新状态

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 23 | 0.18132 | 0.18132 | 0.12161 | -0.00187 | -0.00092 | True | 0.999995 | 0.000011 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 22 | 0.16589 | 0.16589 | 0.10819 | -0.00232 | -0.00539 | False | 0.000000 | 0.000008 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 22 | 0.16448 | 0.16448 | 0.10261 | -0.00373 | -0.01096 | True | 0.999991 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 20 | 0.15579 | 0.15579 | 0.09291 | -0.01083 | -0.00836 | True | 0.999984 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 20 | 0.16698 | 0.16698 | 0.09931 | +0.00036 | -0.00196 | True | 0.999980 | 0.000000 | 1.000000 | pre100 |

- 3090 机制观察延续 05:00 判断：
  - `kd_active_ratio` 仍约 1.0，gate 继续近似全开。
  - paired `gatedKD-z` 与 `gatedKD-wo-srec` 没有优于 shuffledT；目前 shuffledT latest 反而略高于 paired，但 rows 仍太少。
  - 不降级/停止，继续等 50/100 rows；如果 50 rows 仍 `kd_active_ratio≈1` 且 paired 不优于 shuffled，应考虑下一轮 sharper gate 或调整 margin/tau，而不是继续只看 AP。

#### 3090 旧 dynamic 线

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 331 | 0.47483 | 0.47483 | 0.47173 | +0.01515 | +0.01521 | PROMISING_EARLY |
| dynamic_wo_s_rec | 347 | 0.47781 | 0.47781 | 0.47362 | +0.01338 | +0.01184 | PROMISING_EARLY |
| dynamic_plain | 219 | 0.41789 | 0.41816 | 0.41375 | +0.00428 | +0.00517 | WATCH |
| dynamic_reach_rawinput | 192 | 0.40836 | 0.40836 | 0.40365 | +0.01031 | +0.01119 | PROMISING_EARLY |

#### 4090 `zw1cache` fresh 负控制组

| run | rows | latest AP50-95 | best AP50-95 | latest AP50 | best AP50 | latest delta vs detonly | note |
|---|---:|---:|---:|---:|---:|---:|---|
| 4090zw1cache_detonly_control | 2 | 0.02917 | 0.05446 | 0.09042 | 0.15147 | n/a | same-group control |
| 4090zw1cache_capR2_gatedKD_z | 1 | 0.04732 | 0.04732 | 0.13132 | 0.13132 | -0.00714 | first epoch only |
| 4090zw1cache_capR2_gatedKD_toU | 1 | 0.03320 | 0.03320 | 0.09509 | 0.09509 | -0.02126 | first epoch only; negative control |

- 4090 `zw1cache` 健康状态：
  - 三条均生成 `args.yaml`、`results.csv`、`ladd_diagnostics.csv`。
  - 三条均有 Python 主进程，当前 GPU 利用率约 99%/99%。
  - `capR2_gatedKD_z/toU` 的 `kd_active_ratio` 第一行同样约 1.0；与 3090 机制风险一致。
  - 现在 only first epoch，不能评价 AP；下一步看 20 rows 时 z 是否优于 toU、且二者是否优于/劣于 same-group det-only。

- 调度决定：
  - 不新增、不停止。
  - 3090 保持满载；4090 `zw1cache` 已把显存利用拉到约 15.6G/15.7G，未超过 22G 危险线。
  - 继续保护 dynamic 主线；继续监控 4090 磁盘。

### 2026-06-25 05:05 CST

- 3090 GPU0/GPU1: 20244/24576 MiB、20934/24576 MiB，util 99%/100%；继续稳定满载。
- 4090 GPU0/GPU1: 15642/24564 MiB、15694/24564 MiB，瞬时 util 36%/38%，但三条 `zw1cache` 训练进程仍在；前一轮为 99%/99%，判断为采样瞬时波动。
- 4090 磁盘仍约 48G/50G，剩余 2.4G，使用率 96%；后续继续监控，暂不删除结果。
- 3090/4090 当前有效日志扫描未发现 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / batch fallback。

#### 3090 capR/gatedKD 最新状态

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 24 | 0.17743 | 0.18132 | 0.12830 | -0.00402 | -0.00198 | True | 0.999998 | 0.000009 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 23 | 0.16942 | 0.16942 | 0.11644 | -0.01377 | -0.00609 | False | 0.000000 | 0.000081 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 22 | 0.16448 | 0.16448 | 0.10261 | -0.00373 | -0.01096 | True | 0.999991 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 21 | 0.16024 | 0.16024 | 0.09973 | -0.01278 | -0.00774 | True | 0.999977 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 21 | 0.17167 | 0.17167 | 0.10625 | -0.00135 | -0.00121 | True | 0.999972 | 0.000000 | 1.000000 | pre100 |

- 机制观察：
  - capR=2 的 `cap_saturation_ratio` 持续接近 1，说明 capR 的截断确实大量发生。
  - `rank_active_ratio` 仍接近 0，说明当前 rank loss 几乎没有 active violation。
  - `kd_active_ratio` 继续接近 1，说明当前 `cap_reachability_gap` gate 仍基本全开，没有形成预期 token 选择性。
  - shuffledT 早期并不弱于 paired，暂时是机制风险信号；但 rows 仍远低于 50/100，不作为最终结论。

#### 3090 旧 dynamic 主线

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 331 | 0.47483 | 0.47483 | 0.47173 | +0.01515 | +0.01521 | PROMISING_EARLY |
| dynamic_wo_s_rec | 348 | 0.47791 | 0.47791 | 0.47404 | +0.01334 | +0.01195 | PROMISING_EARLY |
| dynamic_plain | 220 | 0.41905 | 0.41905 | 0.41428 | +0.00489 | +0.00517 | WATCH |
| dynamic_reach_rawinput | 193 | 0.40901 | 0.40901 | 0.40415 | +0.01001 | +0.01109 | PROMISING_EARLY |

#### 4090 `zw1cache` fresh 负控制组

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta vs detonly | late20 delta vs detonly | capR | cap saturation | rank active | kd active | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4090zw1cache_detonly_control | 2 | 0.02917 | 0.05446 | 0.04182 | n/a | n/a | n/a | n/a | n/a | n/a | same-group control |
| 4090zw1cache_capR2_gatedKD_z | 2 | 0.01498 | 0.04732 | 0.03115 | -0.01419 | -0.01067 | True | 0.995120 | 0.000013 | 0.999983 | health only |
| 4090zw1cache_capR2_gatedKD_toU | 2 | 0.03045 | 0.03320 | 0.03182 | +0.00128 | -0.00999 | True | 0.995109 | 0.000013 | 0.999989 | negative control; health only |

- 4090 判断：
  - 三条 fresh run 均已生成 `args.yaml`、`results.csv`、`ladd_diagnostics.csv`，并保持训练进程。
  - 第 2 行仍只作为健康信号；不能评价 AP 或 KD-to-u 负控制。
  - `kd_active_ratio` 同样接近 1，与 3090 的 gate 全开风险一致。

- 调度决定：
  - 不新增、不停止。
  - 3090 继续等 50/100 rows；4090 fresh 组先等 20 rows 看 z/toU/control 的基本关系。
  - 继续保护所有 dynamic 主线，尤其 `dynamic_singleproj`、`dynamic_wo_s_rec`、`dynamic_reach_rawinput`。

### 2026-06-25 05:09 CST

- 3090 GPU0/GPU1: 20244/24576 MiB、20934/24576 MiB，util 100%/99%。
- 4090 GPU0/GPU1: 15644/24564 MiB、15694/24564 MiB，util 99%/99%。
- 4090 磁盘仍为 48G/50G，剩余 2.4G，使用率 96%；继续监控，不主动删结果。
- 当前有效日志扫描仍无 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / batch fallback。

#### 3090 capR/gatedKD

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 25 | 0.19192 | 0.19192 | 0.13598 | -0.00370 | -0.00158 | True | 0.999999 | 0.000008 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 24 | 0.18177 | 0.18177 | 0.12364 | +0.00032 | -0.00665 | False | 0.000000 | 0.000114 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 23 | 0.17320 | 0.17320 | 0.11098 | -0.00999 | -0.01156 | True | 0.999988 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 22 | 0.16280 | 0.16280 | 0.10624 | -0.00541 | -0.00733 | True | 0.999977 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 22 | 0.17643 | 0.17643 | 0.11297 | +0.00822 | -0.00060 | True | 0.999975 | 0.000005 | 1.000000 | pre100 |

- 机制风险延续：
  - capR 截断稳定生效，但 gatedKD 的 `kd_active_ratio` 仍为约 1.0，gate 仍近似全开。
  - paired gatedKD-z 没有优于 shuffledT；shuffledT latest delta 当前反而为正。rows 仍远低于 50/100，只记录为风险信号。

#### 3090 旧 dynamic 主线

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 333 | 0.47602 | 0.47602 | 0.47234 | +0.01546 | +0.01514 | PROMISING_EARLY |
| dynamic_wo_s_rec | 349 | 0.47869 | 0.47869 | 0.47446 | +0.01406 | +0.01208 | PROMISING_EARLY |
| dynamic_plain | 221 | 0.41990 | 0.41990 | 0.41481 | +0.00525 | +0.00516 | WATCH |
| dynamic_reach_rawinput | 194 | 0.40937 | 0.40937 | 0.40465 | +0.00972 | +0.01102 | PROMISING_EARLY |

#### 4090 `zw1cache` fresh 负控制组

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta vs detonly | late20 delta vs detonly | capR | cap saturation | rank active | kd active | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4090zw1cache_detonly_control | 4 | 0.03536 | 0.05446 | 0.03048 | n/a | n/a | n/a | n/a | n/a | n/a | same-group control |
| 4090zw1cache_capR2_gatedKD_z | 4 | 0.04749 | 0.04749 | 0.03033 | +0.01213 | -0.00015 | True | 0.995131 | 0.000015 | 1.000000 | health only |
| 4090zw1cache_capR2_gatedKD_toU | 4 | 0.04557 | 0.04557 | 0.03121 | +0.01021 | +0.00073 | True | 0.995057 | 0.000034 | 0.999992 | negative control; health only |

- 4090 判断：
  - fresh 组已稳定进入训练，三条均有结果与诊断。
  - 第 4 行仍只看健康；z 与 toU 都短暂高于 det-only latest，但 rows 太少，不能作为机制或 AP 结论。
  - KD-to-u 负控制与 KD-to-z 早期接近，若到 20/50 rows 仍接近，需要作为 decomposition claim 风险重点记录。

- 调度决定：
  - 不新增、不停止。
  - 3090 已接近显存安全上限；4090 主要风险变为磁盘 96%。
  - 等 3090 capR/gatedKD 到 50 rows、4090 fresh 组到 20 rows 后再做下一次决策。

### 2026-06-25 05:11 CST

- 3090 GPU0/GPU1: 20244/24576 MiB、20934/24576 MiB，util 100%/100%；磁盘 40G/50G，剩余 11G，使用率 79%。
- 4090 GPU0/GPU1: 15644/24564 MiB、15694/24564 MiB，util 99%/99%；磁盘 48G/50G，剩余 2.4G，使用率 96%。
- 当前有效日志扫描仍无 Traceback / RuntimeError / CUDA OOM / NaN / FileNotFound / batch fallback / No space left。

#### 3090 capR/gatedKD

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | capR | cap saturation | rank active | kd active | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| dynamic_capR2_yoloinit | 25 | 0.19192 | 0.19192 | 0.13598 | -0.00370 | -0.00158 | True | 0.999999 | 0.000008 | n/a | pre100 |
| dynamic_capR4_yoloinit retry | 25 | 0.18176 | 0.18177 | 0.13150 | -0.01386 | -0.00607 | False | 0.000000 | 0.000106 | n/a | pre100 |
| dynamic_capR2_gatedKD retry | 24 | 0.17378 | 0.17378 | 0.11840 | -0.00767 | -0.01188 | True | 0.999982 | 0.000003 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_wo_srec retry | 22 | 0.16280 | 0.16280 | 0.10624 | -0.00541 | -0.00733 | True | 0.999977 | 0.000000 | 1.000000 | pre100 |
| dynamic_capR2_gatedKD_shuffledT retry | 22 | 0.17643 | 0.17643 | 0.11297 | +0.00822 | -0.00060 | True | 0.999975 | 0.000005 | 1.000000 | pre100 |

#### 3090 旧 dynamic 主线

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta | late20 delta | status |
|---|---:|---:|---:|---:|---:|---:|---|
| dynamic_singleproj | 333 | 0.47602 | 0.47602 | 0.47234 | +0.01546 | +0.01514 | PROMISING_EARLY |
| dynamic_wo_s_rec | 350 | 0.47915 | 0.47915 | 0.47488 | +0.01436 | +0.01222 | PROMISING_EARLY |
| dynamic_plain | 222 | 0.41989 | 0.41990 | 0.41532 | +0.00421 | +0.00510 | WATCH |
| dynamic_reach_rawinput | 195 | 0.41063 | 0.41063 | 0.40518 | +0.01051 | +0.01098 | PROMISING_EARLY |

#### 4090 `zw1cache` fresh 负控制组

| run | rows | latest AP50-95 | best AP50-95 | late20 | latest delta vs detonly | late20 delta vs detonly | capR | cap saturation | rank active | kd active | note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4090zw1cache_detonly_control | 5 | 0.06541 | 0.06541 | 0.03746 | n/a | n/a | n/a | n/a | n/a | n/a | same-group control |
| 4090zw1cache_capR2_gatedKD_z | 4 | 0.04749 | 0.04749 | 0.03033 | +0.01213 | -0.00015 | True | 0.995131 | 0.000015 | 1.000000 | health only |
| 4090zw1cache_capR2_gatedKD_toU | 5 | 0.05784 | 0.05784 | 0.03653 | -0.00757 | -0.00093 | True | 0.994973 | 0.000041 | 0.999981 | negative control; health only |

- 机制/调度判断：
  - 3090 gatedKD 继续呈现 `kd_active_ratio≈1`，gate 全开风险没有缓解。
  - 4090 fresh 组仍只有 4-5 rows，不能解释 AP；到 20 rows 前只看是否稳定、是否有异常和磁盘是否够用。
  - 不新增、不停止；继续等待 4090 fresh 组 20 rows 与 3090 capR/gatedKD 50 rows。
