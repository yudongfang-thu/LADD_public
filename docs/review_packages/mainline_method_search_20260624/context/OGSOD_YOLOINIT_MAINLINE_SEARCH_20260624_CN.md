# OGSOD YOLO-init 主线搜索记录（2026-06-24）

## 2026-06-24 21:10 CST 目标更新：高并行 dynamic 主线搜索

用户明确指出当前并行度过低，两台双卡服务器显存远未充分利用；本阶段目标改为高并行、以 `dynamic / LADD-like` 为中心的 YOLO-init 主线搜索。

新的执行约束：

- 主线证据仍只看 YOLO-init；reload 暂不作为主线证据。
- DroneVehicle 小风洞只作为负例背景，不再作为主筛选场。
- 新增实验优先围绕 `dynamic` 的可解释变体，不再平均投入 ProbeA 或其他低希望线。
- 每张 GPU 默认保持 4-5 条有效训练作为资源利用目标；若某卡低于 4 条且显存/IO/落卡安全，应主动补充 dynamic 变体。
- 不为了凑并行而超过安全显存上限、触发 OOM、batch fallback、错误落卡或结果目录混淆。
- 100 epoch 作为早筛与资源调度点；最终主线结论仍需要 e800 完整曲线、late-window/final/best。

早筛状态标准同步为：

- `pre100`：matched < 100。
- `WATCH`：有正向迹象但未达到稳定 +1 AP50-95 point。
- `PROMISING_EARLY`：matched >= 100，`late20_delta >= +0.010` AP50-95 且 `latest_delta > 0`。
- `STRONG_EARLY`：matched >= 100，`late20_delta >= +0.020` AP50-95 且 `latest_delta > 0`。
- `LOW_PRIORITY`：matched >= 120 且 `late20_delta <= 0`，或 late-window/latest 明显不稳定且没有持续正增益。

21:06 CST 状态采样：

- 4090：GPU0 7853/24564 MiB、GPU1 8597/24564 MiB；每卡仅约 2 条训练，显存利用不足。
- 4090 当前结果：`dynamic` rows=263，AP50-95 latest/best=0.44400/0.44400，matched delta latest +0.00963，late20 +0.00915，状态 `WATCH`；`oldcommit_ProbeA` rows=159，late20 +0.01088，状态 `PROMISING_EARLY`，但它是旧提交/ProbeA 线，暂作线索而非新增火力重点。
- 3090：GPU0 7623/24576 MiB、GPU1 8417/24576 MiB；每卡约 2 条训练，显存利用不足。
- 3090 当前结果：`singleproj` late20 +0.00386，`wo_s_rec` late20 +0.00484，均为 `WATCH`；`wo_reach` late20 -0.00360，已是 `LOW_PRIORITY`；`dynamic_plain` rows=27，仍为 `pre100`。
- 两台服务器日志扫描未见 Traceback、CUDA OOM、NaN 或 batch fallback。

据此启动第一批高并行 dynamic sweep，使每张 GPU 达到约 4 条训练：

### ladd4090-zw1 新增 4 条

结果目录统一在 `runs_public/ogsod/hbb/yoloinit_dynamic_sweep_20260624/`，日志目录统一在 `logs/ogsod_yoloinit_dynamic_sweep_20260624/`。

| 变体 | GPU | run name | 修改点 |
|---|---:|---|---|
| `dynamic_kd0p5_yoloinit` | 0 | `ogsod_yoloinit_dynamic_kd0p5_yoloinit_yolo11n_e800_b64_img256_s0_20260624_210844_gpu0` | `--alpha-kd 0.5` |
| `dynamic_reach0p5_yoloinit` | 0 | `ogsod_yoloinit_dynamic_reach0p5_yoloinit_yolo11n_e800_b64_img256_s0_20260624_210844_gpu0` | `--lambda-reach/match/rank 0.5` |
| `dynamic_srec0p05_yoloinit` | 1 | `ogsod_yoloinit_dynamic_srec0p05_yoloinit_yolo11n_e800_b64_img256_s0_20260624_210844_gpu1` | `--alpha-s-rec 0.05` |
| `dynamic_teacher_projectedraw_yoloinit` | 1 | `ogsod_yoloinit_dynamic_teacher_projectedraw_yoloinit_yolo11n_e800_b64_img256_s0_20260624_210844_gpu1` | `--teacher-feature-mode projected_raw` |

### ladd3090-zw1 新增 4 条

| 变体 | GPU | run name | 修改点 |
|---|---:|---|---|
| `dynamic_kd2p0_yoloinit` | 0 | `ogsod_yoloinit_dynamic_kd2p0_yoloinit_yolo11n_e800_b64_img256_s0_20260624_210845_gpu0` | `--alpha-kd 2.0` |
| `dynamic_corewarm60_yoloinit` | 0 | `ogsod_yoloinit_dynamic_corewarm60_yoloinit_yolo11n_e800_b64_img256_s0_20260624_210845_gpu0` | core non-det loss warmup end 30 -> 60 |
| `dynamic_kd0p25_yoloinit` | 1 | `ogsod_yoloinit_dynamic_kd0p25_yoloinit_yolo11n_e800_b64_img256_s0_20260624_210845_gpu1` | `--alpha-kd 0.25` |
| `dynamic_reach_rawinput_yoloinit` | 1 | `ogsod_yoloinit_dynamic_reach_rawinput_yoloinit_yolo11n_e800_b64_img256_s0_20260624_210845_gpu1` | `--reach-input-mode raw` |

启动后快速验证：

- 3090 新增实验已进入 AMP/第 1 epoch；21:09 CST 显存约 GPU0 14337 MiB、GPU1 12129 MiB，未见错误。
- 4090 新增实验均已生成日志并保持 alive；21:09 CST 仍处于模型构建/AMP 前后，尚需下一次醒来继续确认是否完全进入训练。
- 本地监控脚本 `docs/experiments/monitor_ogsod_yoloinit_status_20260624.py` 已加入 8 条新 run 与 `STRONG_EARLY` 状态。

21:11 CST 复查：

- 4090：GPU0 15008/24564 MiB、GPU1 15202/24564 MiB，双卡 util 99%；每卡 4 条训练均 alive，新增 dynamic sweep 已落到目标卡，日志扫描 `bad=[]`。新 run 尚未产生 `results.csv`，符合刚启动约 2-3 分钟的状态。
- 3090：GPU0 14569/24576 MiB、GPU1 15359/24576 MiB，双卡 util 99-100%；每卡 4 条训练均 alive，新增 dynamic sweep 已落到目标卡，日志扫描 `bad=[]`。新 run 尚未产生 `results.csv`，符合刚启动状态。
- 当前并行度达到新目标的下限：两台服务器共 16 条训练，其中 4090 每卡 4 条、3090 每卡 4 条。暂不继续加到每卡 5 条，先观察是否出现 I/O 抖动、训练速度异常或显存继续上涨。

21:13 CST 复查：

- 4090：GPU0 16420/24564 MiB、GPU1 16570/24564 MiB，双卡 util 99%；每卡 4 条训练，日志扫描 `bad=[]`。
- 4090 新增 sweep 已有第 1 个 `results.csv` 记录：`kd0p5` 0.03969、`reach0p5` 0.05669、`srec0p05` 0.03773、`teacher_projectedraw` 0.03640 AP50-95。第 1 epoch 不作为方法优劣判断，只说明训练已正常进入评估闭环。
- 3090：GPU0 15979/24576 MiB、GPU1 16701/24576 MiB，双卡 util 99%；每卡 4 条训练，日志扫描 `bad=[]`。
- 3090 新增 sweep 已有第 1 个 `results.csv` 记录：`kd2p0` 0.04453、`corewarm60` 0.05843、`kd0p25` 0.02123、`reach_rawinput` 0.02402 AP50-95。第 1 epoch 不作为结论。
- 调度决策：暂不补第 5 条。虽然显存仍低于 22G 风险线，但两台服务器 util 已满、每卡已有 4 条、每条训练各有 25 个进程，继续加第 5 条更可能造成 CPU/I/O 抖动和 epoch 速度下降。下一次醒来若显存/util/速度仍稳定，再考虑每卡第 5 条；否则保持 4 条直到 100 epoch 早筛点。

21:15 CST 复查：

- 4090：GPU0 16476/24564 MiB、GPU1 16596/24564 MiB，双卡 util 99%；每卡 4 条训练仍 alive，日志扫描 `bad=[]`。
- 4090 主要结果：`dynamic` rows=267，latest AP50-95/AP50 0.44521/0.70920，matched latest delta +0.00791、late20 +0.00901，仍为 `WATCH`；`oldcommit_ProbeA` rows=162，late20 +0.01109，仍为 `PROMISING_EARLY` 但不是当前新增火力方向。
- 4090 新 sweep：`kd0p5` rows=2、`reach0p5` rows=2、`srec0p05` rows=1、`teacher_projectedraw` rows=2，均为 `pre100`。第 1-2 epoch 不作结论。
- 3090：GPU0 16037/24576 MiB、GPU1 16729/24576 MiB，双卡 util 99-100%；每卡 4 条训练仍 alive，日志扫描 `bad=[]`。
- 3090 主要结果：`singleproj` rows=153，late20 +0.00423，`wo_s_rec` rows=160，late20 +0.00510，均为 `WATCH`；`wo_reach` 保持 rows=122 的 `LOW_PRIORITY` 停止态，不占进程；`dynamic_plain` rows=31，仍 `pre100`。
- 3090 新 sweep：`kd2p0/corewarm60/kd0p25/reach_rawinput` 均 rows=2，均 `pre100`，第 1-2 epoch 不作结论。
- 调度决策：继续保持每卡 4 条，不补第 5 条，不停止当前有效候选。当前瓶颈更像 GPU util/CPU/I/O，而不是显存；下一次重点看新 sweep 是否稳定推进到 10+ rows，以及是否存在明显异常慢或日志错误。

21:17 CST 复查：

- 4090：GPU0 16476/24564 MiB、GPU1 16598/24564 MiB；瞬时 util 70/74%，但每卡 4 条训练均 alive，日志扫描 `bad=[]`。
- 4090 主要候选：`dynamic` rows=268，latest AP50-95/AP50 0.44589/0.70935，matched latest delta +0.00850，late20 +0.00897，继续 `WATCH`；`oldcommit_ProbeA` rows=163，late20 +0.01114，继续 `PROMISING_EARLY` 但只作旧条件线索。
- 4090 新 sweep：`kd0p5` rows=2、`reach0p5` rows=2、`srec0p05` rows=2、`teacher_projectedraw` rows=3，均为 `pre100`。第 2-3 epoch 不作方法结论。
- 3090：GPU0 16037/24576 MiB、GPU1 16729/24576 MiB，双卡 util 100%；每卡 4 条训练均 alive，日志扫描 `bad=[]`。
- 3090 主要候选：`singleproj` rows=154，late20 +0.00432，`wo_s_rec` rows=161，late20 +0.00517，均为 `WATCH`；`dynamic_plain` rows=32，仍 `pre100`。
- 3090 新 sweep：`kd2p0` rows=3、`corewarm60` rows=3、`kd0p25` rows=2、`reach_rawinput` rows=3，均为 `pre100`。第 2-3 epoch 不作方法结论。
- 调度决策：不新增、不停止。并行数达标、日志干净、无 100/120 epoch 决策点；保持当前 16 条训练继续推进。

21:19 CST 复查：

- 4090：GPU0 16476/24564 MiB、GPU1 16626/24564 MiB，双卡 util 99%；每卡 4 条训练均 alive，日志扫描 `bad=[]`。
- 4090 `dynamic` rows=269，matched latest delta +0.00720，late20 +0.00885，继续 `WATCH`；`oldcommit_ProbeA` rows=163，late20 +0.01114，继续作为旧条件 `PROMISING_EARLY` 线索。
- 4090 新 sweep：`kd0p5` rows=3、`reach0p5` rows=3、`srec0p05` rows=3、`teacher_projectedraw` rows=4，均为 `pre100`。第 3-4 epoch 不作结论。
- 3090：GPU0 16037/24576 MiB、GPU1 16729/24576 MiB，双卡 util 98-99%；每卡 4 条训练均 alive，日志扫描 `bad=[]`。
- 3090 `singleproj` rows=154，late20 +0.00432，`wo_s_rec` rows=162，late20 +0.00525，均为 `WATCH`；`dynamic_plain` rows=33，仍 `pre100`。
- 3090 新 sweep：`kd2p0` rows=3、`corewarm60` rows=3、`kd0p25` rows=3、`reach_rawinput` rows=4，均为 `pre100`。第 3-4 epoch 不作结论。
- 调度决策：继续保持当前 16 条训练，不新增、不停止。下一步等新 sweep 至少到 10+ rows 后再看是否存在明显异常慢/负向形态；正式筛选仍等 matched >=100。

21:21 CST 复查：

- 4090：GPU0 16476/24564 MiB、GPU1 16626/24564 MiB，双卡 util 99%；每卡 4 条训练均 alive，日志扫描 `bad=[]`。
- 4090 主要状态：`dynamic` rows=269，late20 +0.00885，继续 `WATCH`；`oldcommit_ProbeA` rows=164，late20 +0.01119，继续作为旧条件线索。
- 4090 新 sweep rows：`kd0p5` 4、`reach0p5` 4、`srec0p05` 3、`teacher_projectedraw` 4，均 `pre100`。
- 3090：GPU0 16037/24576 MiB、GPU1 16729/24576 MiB，双卡 util 99-100%；每卡 4 条训练均 alive，日志扫描 `bad=[]`。
- 3090 主要状态：`singleproj` rows=155，late20 +0.00444；`wo_s_rec` rows=162，late20 +0.00525；均 `WATCH`。`dynamic_plain` rows=33，仍 `pre100`。
- 3090 新 sweep rows：`kd2p0` 4、`corewarm60` 4、`kd0p25` 4、`reach_rawinput` 4，均 `pre100`。
- 调度决策：不新增、不停止；保持当前 16 条训练。下一次继续关注新 sweep 到 10+ rows 后是否有明显异常慢或稳定负向。

21:23 CST 复查：

- 4090：GPU0 16476/24564 MiB、GPU1 16626/24564 MiB，双卡 util 99%；每卡 4 条训练均 alive，日志扫描 `bad=[]`。
- 4090 主要状态：`dynamic` rows=271，late20 +0.00869，继续 `WATCH`；`oldcommit_ProbeA` rows=165，late20 +0.01124，继续作为旧条件线索。
- 4090 新 sweep 已稳定产出 5 rows 左右：`kd0p5` rows=5、`reach0p5` rows=5、`srec0p05` rows=5、`teacher_projectedraw` rows=6，均 `pre100`。5-6 rows 仍不作为方法结论。
- 3090：GPU0 16037/24576 MiB、GPU1 16729/24576 MiB，双卡 util 99-100%；每卡 4 条训练均 alive，日志扫描 `bad=[]`。
- 3090 主要状态：`singleproj` rows=156，late20 +0.00457；`wo_s_rec` rows=163，late20 +0.00530；均 `WATCH`。`dynamic_plain` rows=34，仍 `pre100`。
- 3090 新 sweep 已稳定产出 5 rows：`kd2p0/corewarm60/kd0p25/reach_rawinput` 均 rows=5，均 `pre100`。5 rows 仍不作为方法结论。
- 调度决策：不新增、不停止；保持当前 16 条训练。下一次继续关注 10+ rows 后是否有明显异常慢或稳定负向；正式筛选仍等 matched >=100。

## 2026-06-24 20:37 CST 新一轮 goal 与自动化任务

当前重新进入主动探索阶段，目标保持为：在 `ladd4090-zw1` 和 `ladd3090-zw1` 上，用 OGSOD YOLO-init full e800 协议寻找一个稳定正增益、可作为主线的跨模态蒸馏方法。主线证据只看 YOLO-init；reload 暂不作为主线证据；DroneVehicle 小风洞作为负例背景，不再作为主筛选场。

本阶段自动化任务不是单纯巡检，而是让多条探索自动运行，并在超过 100 epoch 后做轻量评估和调度决策。

核心执行规则：

- 每次 heartbeat 先检查最新用户消息，再检查 `ladd4090-zw1` 和 `ladd3090-zw1` 的 GPU 显存/利用率、进程、`results.csv`、日志错误和 batch fallback。
- 4090 候选只和 4090 same-pipeline YOLO-init det-only 比；3090 候选只和 3090 same-pipeline detonly control 比；跨机只作趋势参考。
- 候选 `matched >= 100` 后必须输出：`epoch100_delta`、`latest_matched_delta`、`late20_delta`、positive epoch count、`pre100/WATCH/PROMISING_EARLY/LOW_PRIORITY` 状态。
- `PROMISING_EARLY`：`matched >= 100`、`late20_delta >= +0.010` AP50-95 且 `latest_delta > 0`；继续跑满 800。
- `STRONG_EARLY`：在 `PROMISING_EARLY` 基础上，`late20_delta >= +0.020` AP50-95；优先保留资源。
- `LOW_PRIORITY`：`matched >= 120` 且 `late20_delta <= 0`，或 late-window/latest 明显不稳定且没有持续正增益；如需要释放资源，可以停止或暂停并启动下一批。
- 100 epoch 只用于资源调度和方法筛选；最终主线结论仍需要 e800 完整曲线、late-window 和 final/best 支撑。

自动启动新候选的边界：

- 只有在 GPU 显存、I/O 和已有训练都安全时才启动；不能引入 OOM、batch fallback、错误落卡或结果目录混淆。
- 每次最多给一张 GPU 新增一条训练；启动前确认命令参数受当前代码支持、结果目录唯一、strict batch 开启、A1 cache/teacher/baseline 路径正确。
- 不自动杀已有有效任务；只有 `LOW_PRIORITY`、OOM、batch fallback、错误落卡或用户明确要求时才处理。

候选优先级：

1. 继续围绕 LADD / ProbeA / dynamic 的最小可解释改动，不跳到 unrelated KD。
2. 优先验证 AutoDL 早期较大增益是否来自旧条件或实现差异：old-commit/旧 ProbeA 条件、现代码复刻条件、A1 cache/source 差异。
3. 优先解释当前 dynamic 正增益来源：`single_proj`、`wo_s_rec`、`dynamic_plain`、reach/KD/s_rec 权重与 warmup、teacher raw/projected raw。
4. 若某类候选在 100/120 epoch 后明显负或增益很小，自动降低优先级，把资源转给更接近主线且可解释的新变体。

## 2026-06-24 19:58 CST 新阶段目标与自动化规则

本阶段重新收束为一个明确目标：在 `ladd4090-zw1` 和 `ladd3090-zw1` 上，以 OGSOD YOLO-init full e800 协议寻找稳定正增益主线方法；不再把 reload 作为主线证据，不再把小风洞 DroneVehicle 的负结果作为主筛选场。

自动化任务继续使用本线程 heartbeat，频率约每 15 分钟一次。它需要完成三件事：

- 监控当前所有有效 YOLO-init 候选与 same-pipeline det-only control，检查 GPU、进程、日志和 `results.csv`。
- 任一候选超过 100 epoch 后，立即做轻量同协议评估：`epoch100_delta`、`latest_matched_delta`、`late20_delta`、positive epoch count，以及 `pre100` / `WATCH` / `PROMISING_EARLY` / `LOW_PRIORITY` 状态。
- 若有候选达到 120 epoch 后仍明显低价值，且资源可以安全释放或需要启动下一批候选，则按规则降级、暂停或替换；若 GPU 有安全空闲且不会导致 OOM、batch fallback、I/O 竞争或结果混淆，可以自动启动下一批接近 LADD 主线的 YOLO-init 变体。

100 epoch 早筛标准：

- `PROMISING_EARLY`：matched epoch >= 100，`late20_delta >= +0.010` AP50-95 且 `latest_delta > 0`；继续跑满 800。
- 强信号：若 `late20_delta >= +0.020`，额外标记为 strong early signal。
- `LOW_PRIORITY`：matched epoch >= 120 且 `late20_delta <= 0`，或 latest/late-window 明显不稳定并长期没有持续正增益。
- 100 epoch 只用于资源调度和候选筛选；主线结论仍需要 e800 完整曲线、late-window 和 final/best 支撑。

自动探索优先级：

- 优先围绕 LADD / ProbeA / dynamic 的最小可解释改动，而不是跳到 unrelated KD。
- 优先验证能解释 AutoDL 早期较大增益差异的 YOLO-init 条件，和当前 dynamic 正增益来源相关的结构消融。
- 优先候选包括：student-side 降压、`single_proj` / `wo_s_rec` / `wo_reach` 组合、teacher raw/projected raw、KD/reach warmup 或权重调整、ProbeA/dynamic 的更接近旧主线复刻条件。
- 新实验必须满足：结果目录唯一、strict batch 开启、teacher/baseline/A1 cache 路径正确、same-pipeline det-only 对照存在或正在有效运行、启动后日志无错误且落卡正确。

## 目标

当前目标是寻找一个稳定、可作为主线的跨模态蒸馏方法。现阶段暂不把 reload 作为主线证据，优先评估 YOLO-init 设定：学生检测器从 `yolo11n.pt` 初始化，推理时只使用 SAR 单模态；RGB 只作为训练期 teacher。

可接受的主线信号：

- 完整协议仍按 OGSOD HBB formal 设置跑到 800 epoch。
- 早筛可以看约 100 epoch 的同协议排序，因为当前 OGSOD 曲线在 100 epoch 左右已经进入较稳定上升段。
- 不能只看孤立 early best；需要同 epoch、late-window 或最终结果相对 same-pipeline det-only 为正。
- 若 100 epoch 附近已经稳定高于 det-only 约 1 AP50-95 point，可以作为继续投入的候选；最终论文/主结论仍以 800 epoch 完整结果为准。

## 资源与边界

当前可用服务器：

- `ladd4090-zw1` 双 4090：主要继续监控已跑的 OGSOD e800 复刻。
- `ladd3090-zw1` 双 3090：用于新增 YOLO-init 主线候选。

暂不使用 AutoDL/AutoDL2 等其他服务器。当前已重启本线程 heartbeat automation，用于每 15 分钟左右巡检和推进。

资源原则：充分利用两台服务器，但不为了填满显存而引入 OOM、batch fallback、进程落错卡或结果混淆。

## 固定协议

除非另记，实验采用：

- 数据集：OGSOD HBB full train/val。
- 模型：`yolo11n.pt`。
- 输入：`imgsz=256`。
- 训练长度：`epochs=800`。
- batch：`batch=64`，并使用 strict batch，任何 batch fallback 均不作为正式证据。
- 增强：formal no-mosaic 协议，`mosaic=0.0`，`close_mosaic=0`。
- 方向：RGB teacher -> SAR student；推理只保留 SAR student。
- 对照：same-pipeline YOLO-init det-only，而不是 reload det-only。

## 100 Epoch 早筛依据

`ladd4090-zw1` 上当前同协议 OGSOD e800 复刻显示，约 100 epoch 的排序已经有参考价值：

| run | rows | latest AP50-95/AP50 | best AP50-95 | 与 det-only 同 epoch 差值 |
|---|---:|---:|---:|---:|
| det-only | 211 | 0.40755 / 0.65488 | 0.40755 @211 | - |
| ProbeA | 159 | 0.38617 / 0.63662 | 0.38617 @159 | epoch100 +0.00710；latest matched +0.00853；late20 +0.00832 |
| dynamic | 108 | 0.36049 / 0.60476 | 0.36049 @108 | epoch100 +0.01043；latest matched +0.01088；late20 +0.01040 |
| old-commit ProbeA | 14 | 0.13851 / 0.31134 | 0.13851 @14 | 仅早期，暂不判断 |

当前判断：100 epoch 可作为候选筛选点，但不是最终正结果标准。尤其需要避免把 AutoDL2 上的 `ProbeA vs reload_detonly` 直接等同为 `ProbeA vs YOLO-init det-only`。

2026-06-24 16:15 CST 更新：4090 `dynamic` 已过 100 epoch，latest/late20/epoch100 均约为 +1 AP50-95 point，按早筛规则标记为 `PROMISING_EARLY`，继续跑满 800；`ProbeA` 稳定为正但约 +0.8 point，标记为 `WATCH`。

## 已在跑的 4090 实验

`ladd4090-zw1:/root/shared-nvme/LADD_public`

- det-only control：`runs_public/dronevehicle_method_search/ogsod_nomosaic_autodl_condition/yolo_detonly/ogsod_nomix_yolo_detonly_existingcache_yolo11n_e800_b64_img256_s0_20260624_110207_b/`
- ProbeA：`runs_public/dronevehicle_method_search/ogsod_nomosaic_autodl_condition/yolo_probeA/ogsod_nomix_yolo_probeA_existingcache_yolo11n_e800_b64_img256_s0_20260624_113044_b/`
- dynamic：`runs_public/dronevehicle_method_search/ogsod_nomosaic_autodl_condition/yolo_dynamic/ogsod_nomix_yolo_dynamic_existingcache_yolo11n_e800_b64_img256_s0_20260624_125211_b/`
- old-commit ProbeA：`runs_public/dronevehicle_method_search/ogsod_nomosaic_autodl_condition/yolo_probeA_oldcommit6f663c6b/ogsod_nomix_yolo_probeA_oldcommit6f663c6b_existingcache_yolo11n_e700_b64_img256_s0_20260624_154303_b/`

## 新增 3090 YOLO-init 候选

`ladd3090-zw1:/root/shared-nvme/LADD_public`

### dynamic_singleproj_yoloinit

目的：复查旧消融中较强的 `student_single_proj` 思路在 YOLO-init 主线设定下是否仍有稳定正增益。

关键设置：

- `--phase b`
- `--model yolo11n.pt`
- `--b-detector-source yolo11n.pt`
- `--b-decomp-source <A1 cache>`
- `--ladd-b-a2-core`
- `--b-load-student-reachability`
- `--student-branch-mode single_proj`
- `--alpha-s-rec 0.0`

运行状态（2026-06-24 16:xx CST）：

- GPU0，有效运行。
- 结果目录：`runs_public/ogsod/hbb/yoloinit_mainline_search_20260624/dynamic_singleproj_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_singleproj_yoloinit_yolo11n_e800_b64_img256_s0_20260624_1605_gpu0/`
- 当前 rows=2，latest AP50-95/AP50 = 0.03878 / 0.11880。
- 日志未发现 Traceback、CUDA OOM、NaN 或 batch fallback。

### dynamic_wo_s_rec_yoloinit

目的：复查旧消融中较强的 `wo_s_rec` 思路，即保留 split，但移除学生重构约束。

关键设置：

- `--phase b`
- `--model yolo11n.pt`
- `--b-detector-source yolo11n.pt`
- `--b-decomp-source <A1 cache>`
- `--ladd-b-a2-core`
- `--b-load-student-reachability`
- `--student-branch-mode split`
- `--alpha-s-rec 0.0`

运行状态（2026-06-24 16:xx CST）：

- GPU1，有效运行。
- 结果目录：`runs_public/ogsod/hbb/yoloinit_mainline_search_20260624/dynamic_wo_s_rec_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_wo_s_rec_yoloinit_yolo11n_e800_b64_img256_s0_20260624_1608_gpu1/`
- 当前 rows=1，latest AP50-95/AP50 = 0.05901 / 0.15440。
- 日志未发现 Traceback、CUDA OOM、NaN 或 batch fallback。

备注：第一次 `dynamic_wo_s_rec_yoloinit` 启动时进程错误落在 GPU0，已停止并将对应结果目录标记为 `_INVALID_GPU0_COLLISION`，不用于任何结论。

### detonly_control_yoloinit

目的：为 3090 上的新增候选提供同源 YOLO-init det-only control，避免只和 4090 上路径/teacher/A1 cache 不完全相同的 det-only 曲线比较。

关键设置：

- `--phase b`
- `--model yolo11n.pt`
- `--b-detector-source yolo11n.pt`
- `--b-decomp-source <A1 cache>`
- `--ladd-b-det-only`
- `--alpha-s-rec 0.1`
- `--lambda-reach 1.0`

运行状态（2026-06-24 16:18 CST）：

- GPU0，与 `dynamic_singleproj_yoloinit` 并行。
- 结果目录：`runs_public/ogsod/hbb/yoloinit_mainline_search_20260624/detonly_control_yoloinit/yolo11n/seed0/ogsod_yoloinit_detonly_control_yoloinit_yolo11n_e800_b64_img256_s0_20260624_161654_gpu0/`
- 日志：`logs/ogsod_yoloinit_mainline_search_20260624/detonly_control_yoloinit_gpu0_20260624_161654/ogsod_yoloinit_detonly_control_yoloinit_yolo11n_e800_b64_img256_s0_20260624_161654_gpu0.outer.log`
- AMP passed，`strict_batch_size=True`，已进入 800 epoch training；未见 Traceback、CUDA OOM、NaN 或 batch fallback。

### dynamic_wo_reach_yoloinit

目的：测试 YOLO-init 主线下 reach 约束是否压制检测学习；保留 LADD B-stage / decomposition / KD / s_rec，关闭 reach match/rank。

关键设置：

- `--phase b`
- `--model yolo11n.pt`
- `--b-detector-source yolo11n.pt`
- `--b-decomp-source <A1 cache>`
- `--ladd-b-a2-core`
- `--student-branch-mode split`
- `--teacher-feature-mode decomposed`
- `--alpha-s-rec 0.1`
- `--lambda-reach 0.0`
- `--lambda-match-inner 0.0`
- `--lambda-rank-inner 0.0`

运行状态（2026-06-24 16:18 CST）：

- GPU1，与 `dynamic_wo_s_rec_yoloinit` 并行。
- 结果目录：`runs_public/ogsod/hbb/yoloinit_mainline_search_20260624/dynamic_wo_reach_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_wo_reach_yoloinit_yolo11n_e800_b64_img256_s0_20260624_161654_gpu1/`
- 日志：`logs/ogsod_yoloinit_mainline_search_20260624/dynamic_wo_reach_yoloinit_gpu1_20260624_161654/ogsod_yoloinit_dynamic_wo_reach_yoloinit_yolo11n_e800_b64_img256_s0_20260624_161654_gpu1.outer.log`
- AMP passed，`strict_batch_size=True`，已进入 800 epoch training；未见 Traceback、CUDA OOM、NaN 或 batch fallback。

## 下一步判据

1. 继续让 4090 的 det-only / ProbeA / dynamic e800 跑完整，当前只把 100 epoch 信号作为筛选依据。
2. 3090 新增候选 `singleproj / wo_s_rec / wo_reach` 到 100 epoch 后，优先与 3090 同源 `detonly_control_yoloinit` 做同 epoch 比较；4090 det-only 只作为跨机趋势参照。
3. 若某候选在 100 epoch 附近 late20 稳定超过 same-pipeline det-only 约 1 point，则保留为主线候选并跑满 800。
4. 若候选只在个别 epoch 高，但 late-window 不稳定，则不作为主线正结果。
5. 3090 新增同源 det-only control 后，后续优先使用该 control 对 `singleproj / wo_s_rec / wo_reach` 做同服务器、同配置的 100-epoch 评估；在 control 未到 100 前，4090 det-only 只作为趋势参照。

## 自动化任务（已重启）

2026-06-24 已重新开启当前线程 heartbeat automation：

- automation id：`ogsod-yolo-init-mainline-search`
- 频率：约每 15 分钟检查一次。
- 目标：继续当前 active goal，在 `ladd4090-zw1` 和 `ladd3090-zw1` 上寻找 OGSOD YOLO-init 稳定正增益主线。
- 范围：只使用 `ladd4090-zw1` 与 `ladd3090-zw1`；不使用 AutoDL/AutoDL2/90；不把 reload 作为主线证据。
- 每次醒来先检查最新用户消息，再检查 GPU、进程、日志错误和 `results.csv`。
- 必报指标：rows、latest AP50/AP50-95、best AP50-95@epoch、late5/late10/late20/late50。
- 候选超过 100 rows 后进行轻量评估：epoch100 delta、latest matched delta、late20 delta、positive epoch count。
- 早筛规则：100 epoch 只是筛选信号；若 late20 delta 稳定为正且约 `>= +0.010` AP50-95，标记为 promising 并继续跑满 800；若 `>=120` rows 后 late20 仍不正或明显不稳定，标记为 low-priority。
- 自动启动新候选的边界：只在 GPU/IO 安全、不会 OOM、不会 batch fallback、不会混淆目录时启动；启动前必须确认命令参数受当前代码支持、结果目录唯一、strict batch 开启、teacher/baseline/A1 cache 路径正确；启动后检查日志和进程是否落在预期 GPU。
- 候选优先级：继续围绕接近 LADD 主线的 YOLO-init 变体，优先学生侧减压和动态权重变体，例如 `single_proj`、`wo_s_rec`、`wo_reach`、teacher raw/projected raw、KD/reach warmup/weight 调整。
- 不自动杀已有有效任务；只有明确 OOM、batch fallback、错误落卡且会污染结论，或用户明确要求时才处理。

2026-06-24 16:31 CST 再确认一次新目标和自动化约束：

- 当前 Codex goal 保持 active，目标是：在 `ladd4090-zw1` 和 `ladd3090-zw1` 上以 OGSOD YOLO-init full e800 协议寻找稳定正增益主线方法；自动监控多条候选曲线；候选超过 100 epoch 后做轻量评估；再依据 100-epoch/late-window 规则决定继续、降级或启动下一批接近 LADD 主线的变体。
- automation 继续使用同一个 `ogsod-yolo-init-mainline-search`，状态 `ACTIVE`，心跳频率 `30 min`。本轮不是新建重复 automation，而是更新同一个 automation 的 prompt。
- 监控范围明确为：4090 上 det-only / ProbeA / dynamic / old-commit ProbeA；3090 上 `dynamic_singleproj_yoloinit`、`dynamic_wo_s_rec_yoloinit`、`detonly_control_yoloinit`、`dynamic_wo_reach_yoloinit`。
- 100 epoch 轻量评估必须输出：`epoch100 delta`、`latest matched delta`、`late20 delta`、`positive epoch count`，并用 `PROMISING_EARLY / WATCH / LOW_PRIORITY / pre100` 标记。
- 自动推进原则：如果当前候选已经占满安全并行容量，就只监控不新增；如果有候选过 120 epoch 后 late-window 明显不正，且 GPU 资源需要让位，可以把它降级并准备下一批。下一批优先仍围绕接近 LADD 主线的 YOLO-init 变体，而不是跳到 unrelated KD。

2026-06-24 18:40 CST 重新收束 goal 与自动化任务：

- 当前 active goal 保持不变，不强行结束重建：在 `ladd4090-zw1` 和 `ladd3090-zw1` 上以 OGSOD YOLO-init full e800 协议寻找稳定正增益主线方法。
- automation 继续使用同一个 `ogsod-yolo-init-mainline-search`，状态 `ACTIVE`，心跳频率 `15 min`，避免创建重复提醒。
- 每次醒来按固定顺序执行：检查最新用户消息；检查两台服务器 GPU/进程/日志；解析当前 runs 的 `results.csv`；写入本 MD；必要时同步到服务器。
- 100 epoch 是第一触发点：候选 `rows >= 100` 后必须输出 `epoch100 delta`、`latest matched delta`、`late20 delta`、`positive epoch count`，并给出 `PROMISING_EARLY / WATCH / LOW_PRIORITY / pre100`。
- 120 epoch 是降级触发点：候选 `rows >= 120` 且 `late20_delta <= 0` 或 latest/late-window 明显不稳定时，标记为 `LOW_PRIORITY`；只有在资源需要让位、且不会污染结论时才考虑停止或暂停。
- 下一批候选只在资源安全且已有候选达到评估点后启动；优先围绕 LADD 主线附近的 YOLO-init 变体，例如 student-side 降压、动态权重、reach/KD warmup、teacher raw/projected raw、ProbeA/dynamic 的最小可解释改动。

2026-06-24 19:13 CST 重新开始自动化推进规则：

- 当前 Codex goal 已经处于 `active`，目标文字与本轮新目标一致：两台服务器、OGSOD YOLO-init full e800、寻找稳定正增益主线。因此不结束旧 goal、不创建重复 goal，只更新同一个 automation。
- automation `ogsod-yolo-init-mainline-search` 已更新为 `ACTIVE`，心跳频率 `15 min`，目标从单纯巡检扩展为“多条探索自动推进 + 100 epoch 轻量评估 + 安全条件下启动下一批候选”。
- 每次醒来优先使用远端脚本 `docs/experiments/monitor_ogsod_yoloinit_status_20260624.py` 汇总两台服务器状态；必要时再直接读取日志和 `results.csv`。必须检查 Traceback、CUDA OOM、NaN、RuntimeError、batch fallback、错误落卡和目录混淆。
- 评估仍按同服务器 same-pipeline 对照：4090 候选只和 4090 YOLO-init det-only 比，3090 候选只和 3090 detonly_control 比；跨机结果只作趋势参考。
- 候选 `matched >= 100` 后必须输出 `epoch100 delta`、`latest matched delta`、`late20 delta`、`positive epoch count`，并标记 `pre100 / WATCH / PROMISING_EARLY / LOW_PRIORITY`。
- `PROMISING_EARLY` 条件保持为 `matched >= 100`、`late20_delta >= +0.010 AP50-95` 且 `latest_delta > 0`；若 `late20_delta >= +0.020`，额外标注为 strong early signal。所有 early signal 只决定是否继续投入，不作为最终正结果。
- `LOW_PRIORITY` 条件为 `matched >= 120` 且 `late20_delta <= 0`，或 latest/late-window 明显不稳定且没有持续正增益。只有在资源需要让位、且不会污染结论时，才考虑停止或替换低优先级候选。
- 自动启动下一批候选的边界：GPU/IO 安全，不会 OOM，不会 batch fallback，不会结果目录混淆；每张 GPU 每次最多新增一条；启动前确认命令参数、strict batch、teacher/baseline/A1 cache、same-pipeline det-only 对照和结果目录，启动后检查 PID/log/GPU。
- 下一批候选优先围绕 LADD/ProbeA/dynamic 的最小可解释改动，不跳到 unrelated KD。优先方向包括 student-side 降压、`single_proj/wo_s_rec/wo_reach` 的组合或后续解释性变体、teacher raw/projected raw、KD/reach warmup 或权重调整，以及用于解释 AutoDL 早期大增益差异的 YOLO-init 复刻条件。

## 巡检记录

### 2026-06-24 16:21 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 91%`；GPU1 `8597/24564 MiB, util 95%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：GPU0 `7619/24576 MiB, util 89%`；GPU1 `8389/24576 MiB, util 19%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。GPU1 util 偏低视为瞬时采样，不据此追加任务。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 215 | 0.41013 / 0.65689 | 0.41013 @215 | 0.40873 / 0.40727 / 0.40448 / 0.39587 | - |
| ProbeA | 163 | 0.38828 / 0.64018 | 0.38828 @163 | 0.38706 / 0.38562 / 0.38312 / 0.37474 | latest +0.00858；late20 +0.00837；epoch100 +0.00710，`WATCH` |
| dynamic | 111 | 0.36292 / 0.60698 | 0.36292 @111 | 0.36128 / 0.35965 / 0.35619 / 0.34473 | latest +0.01143；late20 +0.01048；epoch100 +0.01043，`PROMISING_EARLY` |
| old-commit ProbeA | 17 | 0.15515 / 0.34593 | 0.15515 @17 | 0.14114 / 0.12447 / 0.09045 / 0.09045 | pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 2 | 0.04603 / 0.12252 | 0.04900 @1 | 0.04752 / 0.04752 / 0.04752 / 0.04752 | - |
| dynamic_singleproj_yoloinit | 7 | 0.05774 / 0.15420 | 0.06522 @6 | 0.04778 / 0.04719 / 0.04719 / 0.04719 | matched=2，latest -0.00725，pre100 |
| dynamic_wo_s_rec_yoloinit | 7 | 0.08577 / 0.23179 | 0.08577 @7 | 0.05434 / 0.05158 / 0.05158 / 0.05158 | matched=2，latest -0.01571，pre100 |
| dynamic_wo_reach_yoloinit | 1 | 0.04183 / 0.11330 | 0.04183 @1 | 0.04183 / 0.04183 / 0.04183 / 0.04183 | matched=1，latest -0.00717，pre100 |

决策：

- 4090 `dynamic` 继续保留为 early promising 候选，跑满 800。
- 4090 `ProbeA` 继续观察，当前稳定为正但不到 +1 point。
- 3090 新增组均未达到 100 rows，不做正负结论，不停跑。
- 当前两台服务器均已有多条并行任务；本轮不再追加新实验，以避免 I/O 竞争、batch fallback 和目录混淆。

### 2026-06-24 16:23 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 94%`；GPU1 `8597/24564 MiB, util 92%`。日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：GPU0 `7619/24576 MiB, util 92%`；GPU1 `8389/24576 MiB, util 99%`。日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 217 | 0.41090 / 0.65710 | 0.41090 @217 | 0.40988 / 0.40846 / 0.40560 / 0.39703 | - |
| ProbeA | 164 | 0.38883 / 0.64074 | 0.38883 @164 | 0.38759 / 0.38621 / 0.38363 / 0.37530 | latest +0.00837；late20 +0.00837；epoch100 +0.00710，`WATCH` |
| dynamic | 112 | 0.36347 / 0.60754 | 0.36347 @112 | 0.36194 / 0.36033 / 0.35687 / 0.34557 | latest +0.01093；late20 +0.01048；epoch100 +0.01043，`PROMISING_EARLY` |
| old-commit ProbeA | 18 | 0.15834 / 0.33860 | 0.15834 @18 | 0.14609 / 0.13230 / 0.09422 / 0.09422 | pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 3 | 0.00395 / 0.01290 | 0.04900 @1 | 0.03299 / 0.03299 / 0.03299 / 0.03299 | - |
| dynamic_singleproj_yoloinit | 8 | 0.07600 / 0.20414 | 0.07600 @8 | 0.05893 / 0.05079 / 0.05079 / 0.05079 | matched=3，latest +0.01634，pre100 |
| dynamic_wo_s_rec_yoloinit | 8 | 0.08419 / 0.21587 | 0.08577 @7 | 0.06768 / 0.05565 / 0.05565 / 0.05565 | matched=3，latest +0.01355，pre100 |
| dynamic_wo_reach_yoloinit | 2 | 0.04923 / 0.13488 | 0.04923 @2 | 0.04553 / 0.04553 / 0.04553 / 0.04553 | matched=2，latest +0.00320，pre100 |

决策：

- 4090 `dynamic` 仍稳定满足 early promising；继续跑满 800。
- 4090 `ProbeA` 继续保持 `WATCH`，当前正增益约 +0.8 point。
- 3090 组仍是极早期，det-only control 第 3 epoch 明显波动，不能据此判断候选优劣。
- 当前两台服务器 GPU 均接近满载；本轮不新增实验，不停止任何有效任务。

### 2026-06-24 16:25 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 93%`；GPU1 `8597/24564 MiB, util 90%`。日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：GPU0 `7619/24576 MiB, util 98%`；GPU1 `8417/24576 MiB, util 91%`。日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 218 | 0.41139 / 0.65688 | 0.41139 @218 | 0.41044 / 0.40901 / 0.40612 / 0.39761 | - |
| ProbeA | 165 | 0.38947 / 0.64161 | 0.38947 @165 | 0.38818 / 0.38682 / 0.38415 / 0.37586 | latest +0.00815；late20 +0.00835；epoch100 +0.00710，`WATCH` |
| dynamic | 113 | 0.36401 / 0.60882 | 0.36401 @113 | 0.36265 / 0.36100 / 0.35753 / 0.34639 | latest +0.01057；late20 +0.01047；epoch100 +0.01043，`PROMISING_EARLY` |
| old-commit ProbeA | 19 | 0.15973 / 0.35516 | 0.15973 @19 | 0.15034 / 0.13804 / 0.09767 / 0.09767 | pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 4 | 0.02647 / 0.08065 | 0.04900 @1 | 0.03136 / 0.03136 / 0.03136 / 0.03136 | - |
| dynamic_singleproj_yoloinit | 9 | 0.10496 / 0.25816 | 0.10496 @9 | 0.07052 / 0.05681 / 0.05681 / 0.05681 | matched=4，latest +0.02053，pre100 |
| dynamic_wo_s_rec_yoloinit | 9 | 0.10043 / 0.24379 | 0.10043 @9 | 0.08064 / 0.06063 / 0.06063 / 0.06063 | matched=4，latest +0.00917，pre100 |
| dynamic_wo_reach_yoloinit | 3 | 0.01063 / 0.03512 | 0.04923 @2 | 0.03390 / 0.03390 / 0.03390 / 0.03390 | matched=3，latest +0.00668，pre100 |

决策：

- 4090 `dynamic` 连续多轮满足 +1 point 左右 early promising，继续跑满 800。
- 4090 `ProbeA` 正增益稳定但仍小于 +1 point，继续 `WATCH`。
- 3090 `singleproj` 和 `wo_s_rec` 早期相对同源 det-only 有正差，但 rows 太少，不能认定为正结果；继续等 100 rows。
- 两台服务器 GPU 利用率已高，本轮不新增实验，不停止任何有效任务。

### 2026-06-24 16:28 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 95%`；GPU1 `8597/24564 MiB, util 91%`。日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：GPU0 `7619/24576 MiB, util 18%`；GPU1 `8417/24576 MiB, util 99%`。日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。GPU0 util 偏低视为瞬时采样或验证阶段，显存/进程仍正常，不据此追加任务。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 220 | 0.41258 / 0.65819 | 0.41258 @220 | 0.41154 / 0.41014 / 0.40722 / 0.39877 | - |
| ProbeA | 166 | 0.38972 / 0.64240 | 0.38972 @166 | 0.38875 / 0.38739 / 0.38467 / 0.37641 | latest +0.00830；late20 +0.00833；epoch100 +0.00710，`WATCH` |
| dynamic | 114 | 0.36489 / 0.60943 | 0.36489 @114 | 0.36344 / 0.36172 / 0.35822 / 0.34720 | latest +0.01124；late20 +0.01050；epoch100 +0.01043，`PROMISING_EARLY` |
| old-commit ProbeA | 20 | 0.16627 / 0.36391 | 0.16627 @20 | 0.15587 / 0.14435 / 0.10110 / 0.10110 | pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 6 | 0.05807 / 0.15903 | 0.05807 @6 | 0.03691 / 0.03892 / 0.03892 / 0.03892 | - |
| dynamic_singleproj_yoloinit | 11 | 0.12071 / 0.28974 | 0.12071 @11 | 0.09412 / 0.06906 / 0.06756 / 0.06756 | matched=6，latest +0.00715，pre100 |
| dynamic_wo_s_rec_yoloinit | 10 | 0.10745 / 0.25882 | 0.10745 @10 | 0.09022 / 0.06531 / 0.06531 / 0.06531 | matched=6，latest +0.01517，pre100 |
| dynamic_wo_reach_yoloinit | 5 | 0.03278 / 0.10409 | 0.04923 @2 | 0.03436 / 0.03436 / 0.03436 / 0.03436 | matched=5，latest -0.01723，pre100 |

决策：

- 4090 `dynamic` 继续稳定满足 early promising，保留并跑满 800。
- 4090 `ProbeA` 继续 `WATCH`。
- 3090 组仍未到 100 rows；`singleproj/wo_s_rec` 早期相对同源 det-only 为正但不可下结论，`wo_reach` 短窗口略负也不可下结论。
- 不新增实验，不停止任何有效任务。

### 2026-06-24 16:37 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 93%`；GPU1 `8597/24564 MiB, util 91%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：GPU0 `7621/24576 MiB, util 95%`；GPU1 `8417/24576 MiB, util 92%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 226 | 0.41629 / 0.66258 | 0.41629 @226 | 0.41494 / 0.41347 / 0.41067 / 0.40225 | - |
| ProbeA | 171 | 0.39201 / 0.64582 | 0.39201 @171 | 0.39116 / 0.38996 / 0.38725 / 0.37912 | latest +0.00762；late20 +0.00825；epoch100 +0.00710；positive 160/171，`WATCH` |
| dynamic | 119 | 0.36797 / 0.61384 | 0.36797 @119 | 0.36665 / 0.36505 / 0.36166 / 0.35102 | latest +0.01231；late20 +0.01088；epoch100 +0.01043；positive 112/119，`PROMISING_EARLY` |
| old-commit ProbeA | 25 | 0.19528 / 0.41865 | 0.19528 @25 | 0.18765 / 0.17176 / 0.14113 / 0.11841 | matched=25，latest +0.00951；late20 +0.00431，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 12 | 0.12481 / 0.29645 | 0.12481 @12 | 0.10461 / 0.07486 / 0.07031 / 0.07031 | - |
| dynamic_singleproj_yoloinit | 15 | 0.12782 / 0.29055 | 0.13450 @13 | 0.12520 / 0.10411 / 0.08323 / 0.08323 | matched=12，latest -0.00778；late20 +0.00138，pre100 |
| dynamic_wo_s_rec_yoloinit | 15 | 0.12473 / 0.28971 | 0.13532 @13 | 0.12671 / 0.10846 / 0.08578 / 0.08578 | matched=12，latest -0.00312；late20 +0.00444，pre100 |
| dynamic_wo_reach_yoloinit | 9 | 0.09085 / 0.22829 | 0.09085 @9 | 0.06601 / 0.05212 / 0.05212 / 0.05212 | matched=9，latest +0.00073；late20 -0.00374，pre100 |

决策：

- 4090 `dynamic` 已接近 120 rows，仍满足 early promising：late20 约 +1.09 AP50-95 point，继续跑满 800。
- 4090 `ProbeA` 继续 `WATCH`：稳定为正，但目前增益约 +0.8 point，不足以单独作为可靠主线。
- 3090 同源组还没有到 100 rows；虽然 `singleproj/wo_s_rec` 的 late-window 早期略正，latest 已被 det-only 追平/反超，当前不能下结论，也不触发停止。
- 两台服务器 GPU 都在高负载，且没有异常；本轮不新增实验、不停止有效任务，等待 3090 组和 old-commit ProbeA 到达 100 epoch 后再判断。

### 2026-06-24 16:40 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 90%`；GPU1 `8597/24564 MiB, util 89%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：GPU0 `7621/24576 MiB, util 92%`；GPU1 `8417/24576 MiB, util 94%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 228 | 0.41704 / 0.66402 | 0.41704 @228 | 0.41607 / 0.41463 / 0.41182 / 0.40340 | - |
| ProbeA | 173 | 0.39318 / 0.64697 | 0.39318 @173 | 0.39219 / 0.39096 / 0.38829 / 0.38021 | latest +0.00793；late20 +0.00824；epoch100 +0.00710；positive 162/173，`WATCH` |
| dynamic | 121 | 0.36893 / 0.61588 | 0.36893 @121 | 0.36786 / 0.36630 / 0.36297 / 0.35246 | latest +0.01177；late20 +0.01104；epoch100 +0.01043；positive 114/121，`PROMISING_EARLY` |
| old-commit ProbeA | 26 | 0.19973 / 0.41686 | 0.19973 @26 | 0.19159 / 0.17775 / 0.14713 / 0.12154 | matched=26，latest +0.01745；late20 +0.00500，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 13 | 0.13442 / 0.30246 | 0.13442 @13 | 0.11311 / 0.08791 / 0.07524 / 0.07524 | - |
| dynamic_singleproj_yoloinit | 16 | 0.14336 / 0.33040 | 0.14336 @16 | 0.12973 / 0.11193 / 0.08699 / 0.08699 | matched=13，latest +0.00008；late20 +0.00128，pre100 |
| dynamic_wo_s_rec_yoloinit | 16 | 0.15042 / 0.34042 | 0.15042 @16 | 0.13238 / 0.11618 / 0.08982 / 0.08982 | matched=13，latest +0.00090；late20 +0.00416，pre100 |
| dynamic_wo_reach_yoloinit | 11 | 0.11935 / 0.28380 | 0.11935 @11 | 0.09088 / 0.06545 / 0.06331 / 0.06331 | matched=11，latest +0.00727；late20 -0.00205，pre100 |

决策：

- 4090 `dynamic` 过 120 rows 后仍稳定满足 early promising，late20 提升约 +1.10 AP50-95 point，继续作为当前最可信 YOLO-init 主线候选跑满 800。
- 4090 `ProbeA` 仍为 `WATCH`，正增益稳定但不足 +1 point。
- 4090 old-commit ProbeA 和 3090 三个候选均未到 100 rows，当前只记录走势，不做正/负结论。
- 当前没有 OOM、fallback 或错误落卡；两台机器负载健康。本轮不新增实验、不停止任务。

### 2026-06-24 16:43 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 92%`；GPU1 `8597/24564 MiB, util 90%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：GPU0 `7621/24576 MiB, util 94%`；GPU1 `8417/24576 MiB, util 96%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 230 | 0.41727 / 0.66422 | 0.41727 @230 | 0.41688 / 0.41558 / 0.41286 / 0.40452 | - |
| ProbeA | 175 | 0.39360 / 0.64827 | 0.39360 @175 | 0.39294 / 0.39182 / 0.38932 / 0.38128 | latest +0.00722；late20 +0.00815；epoch100 +0.00710；positive 164/175，`WATCH` |
| dynamic | 122 | 0.36974 / 0.61626 | 0.36974 @122 | 0.36848 / 0.36693 / 0.36363 / 0.35317 | latest +0.01209；late20 +0.01112；epoch100 +0.01043；positive 115/122，`PROMISING_EARLY` |
| old-commit ProbeA | 28 | 0.21362 / 0.43672 | 0.21468 @27 | 0.20262 / 0.18923 / 0.16077 / 0.12815 | matched=28，latest +0.01705；late20 +0.00763，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 15 | 0.14205 / 0.31695 | 0.14205 @15 | 0.12914 / 0.10770 / 0.08350 / 0.08350 | - |
| dynamic_singleproj_yoloinit | 18 | 0.14545 / 0.31061 | 0.15524 @17 | 0.13956 / 0.12862 / 0.09403 / 0.09403 | matched=15，latest -0.01423；late20 -0.00027，pre100 |
| dynamic_wo_s_rec_yoloinit | 18 | 0.15767 / 0.34382 | 0.15767 @18 | 0.14371 / 0.13056 / 0.09727 / 0.09727 | matched=15，latest -0.01732；late20 +0.00228，pre100 |
| dynamic_wo_reach_yoloinit | 12 | 0.11469 / 0.27603 | 0.11935 @11 | 0.10198 / 0.07200 / 0.06759 / 0.06759 | matched=12，latest -0.01012；late20 -0.00272，pre100 |

决策：

- 4090 `dynamic` 继续保持最强 early signal：过 120 rows 后 late20 delta 约 +1.11 AP50-95 point，仍为 `PROMISING_EARLY`。
- 4090 `ProbeA` 仍为 `WATCH`，稳定正但不到 +1 point。
- 3090 三个候选仍然 pre100；latest 相对同源 det-only 开始转弱，尤其 `singleproj/wo_s_rec` 在 matched=15 时 latest 为负，但当前 rows 太少，只记录趋势，不提前停止。
- 没有日志异常或资源风险，本轮不新增实验、不停止任务。

### 2026-06-24 16:46 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 81%`；GPU1 `8597/24564 MiB, util 80%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：GPU0 `7621/24576 MiB, util 90%`；GPU1 `8417/24576 MiB, util 93%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 232 | 0.41864 / 0.66599 | 0.41864 @232 | 0.41766 / 0.41660 / 0.41392 / 0.40564 | - |
| ProbeA | 176 | 0.39410 / 0.64871 | 0.39410 @176 | 0.39336 / 0.39226 / 0.38982 / 0.38180 | latest +0.00738；late20 +0.00809；epoch100 +0.00710；positive 165/176，`WATCH` |
| dynamic | 124 | 0.37106 / 0.61869 | 0.37106 @124 | 0.36971 / 0.36818 / 0.36495 / 0.35459 | latest +0.01181；late20 +0.01130；epoch100 +0.01043；positive 117/124，`PROMISING_EARLY` |
| old-commit ProbeA | 29 | 0.21582 / 0.43779 | 0.21582 @29 | 0.20783 / 0.19484 / 0.16644 / 0.13118 | matched=29，latest +0.01220；late20 +0.00815，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 17 | 0.15143 / 0.34212 | 0.15143 @17 | 0.13925 / 0.12193 / 0.09058 / 0.09058 | - |
| dynamic_singleproj_yoloinit | 19 | 0.15797 / 0.35642 | 0.15797 @19 | 0.14597 / 0.13392 / 0.09739 / 0.09739 | matched=17，latest +0.00381；late20 +0.00042，pre100 |
| dynamic_wo_s_rec_yoloinit | 19 | 0.16385 / 0.35602 | 0.16385 @19 | 0.15054 / 0.13690 / 0.10077 / 0.10077 | matched=17，latest +0.00459；late20 +0.00313，pre100 |
| dynamic_wo_reach_yoloinit | 14 | 0.12925 / 0.29837 | 0.12925 @14 | 0.11992 / 0.09296 / 0.07633 / 0.07633 | matched=14，latest -0.00310；late20 -0.00299，pre100 |

决策：

- 4090 `dynamic` 继续是当前最可信候选：late20 delta 约 +1.13 AP50-95 point，仍为 `PROMISING_EARLY`，继续跑满 800。
- 4090 `ProbeA` 仍是稳定小正，`WATCH`。
- 3090 `singleproj/wo_s_rec/wo_reach` 仍未到 100 rows；`singleproj/wo_s_rec` 相对同源 det-only 最新点重新转正但幅度很小，`wo_reach` 仍略负。全部继续观察，不提前停止。
- 两台服务器没有异常；本轮不新增实验、不停止任务。

### 2026-06-24 16:50 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 81%`；GPU1 `8597/24564 MiB, util 91%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：GPU0 `7621/24576 MiB, util 96%`；GPU1 `8417/24576 MiB, util 98%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 234 | 0.41974 / 0.66830 | 0.41974 @234 | 0.41861 / 0.41756 / 0.41497 / 0.40673 | - |
| ProbeA | 178 | 0.39550 / 0.64936 | 0.39550 @178 | 0.39427 / 0.39323 / 0.39081 / 0.38288 | latest +0.00722；late20 +0.00794；epoch100 +0.00710；positive 167/178，`WATCH` |
| dynamic | 126 | 0.37243 / 0.62027 | 0.37243 @126 | 0.37108 / 0.36947 / 0.36624 / 0.35598 | latest +0.01244；late20 +0.01144；epoch100 +0.01043；positive 119/126，`PROMISING_EARLY` |
| old-commit ProbeA | 31 | 0.22513 / 0.44964 | 0.22513 @31 | 0.21817 / 0.20488 / 0.17705 / 0.13712 | matched=31，latest +0.00623；late20 +0.00943，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 20 | 0.16662 / 0.36615 | 0.16662 @20 | 0.15456 / 0.14185 / 0.10127 / 0.10127 | - |
| dynamic_singleproj_yoloinit | 21 | 0.17368 / 0.37411 | 0.17368 @21 | 0.15984 / 0.14478 / 0.10692 / 0.10433 | matched=20，latest +0.00023；late20 -0.00040，pre100 |
| dynamic_wo_s_rec_yoloinit | 21 | 0.17548 / 0.37552 | 0.17548 @21 | 0.16376 / 0.14807 / 0.10984 / 0.10742 | matched=20，latest -0.00082；late20 +0.00276，pre100 |
| dynamic_wo_reach_yoloinit | 16 | 0.13236 / 0.30805 | 0.13236 @16 | 0.12714 / 0.10901 / 0.08325 / 0.08325 | matched=16，latest -0.00364；late20 -0.00353，pre100 |

决策：

- 4090 `dynamic` 继续是当前最可靠的 YOLO-init 主线候选：过 126 rows 后 late20 delta 约 +1.14 AP50-95 point，状态保持 `PROMISING_EARLY`。
- 4090 `ProbeA` 仍为 `WATCH`，正增益稳定但低于 +1 point。
- 4090 old-commit ProbeA 仍未到 100 rows，虽然 early latest/late20 为正，但只记录，不做结论。
- 3090 三个候选仍未到 100 rows；`singleproj/wo_s_rec` 基本贴近同源 det-only，`wo_reach` 早期略负，但未触发 120-row 低优先级规则。继续观察。
- 没有日志异常、显存风险或 batch fallback；本轮不新增实验、不停止任务。

### 2026-06-24 16:55 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 91%`；GPU1 `8597/24564 MiB, util 91%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：GPU0 `7621/24576 MiB, util 94%`；GPU1 `8417/24576 MiB, util 95%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 238 | 0.42271 / 0.67128 | 0.42271 @238 | 0.42126 / 0.41966 / 0.41714 / 0.40894 | - |
| ProbeA | 181 | 0.39739 / 0.65066 | 0.39739 @181 | 0.39615 / 0.39476 / 0.39236 / 0.38452 | latest +0.00740；late20 +0.00778；epoch100 +0.00710；positive 170/181，`WATCH` |
| dynamic | 129 | 0.37467 / 0.62164 | 0.37467 @129 | 0.37323 / 0.37147 / 0.36826 / 0.35806 | latest +0.01293；late20 +0.01175；epoch100 +0.01043；positive 122/129，`PROMISING_EARLY` |
| old-commit ProbeA | 34 | 0.23753 / 0.47113 | 0.23753 @34 | 0.22961 / 0.21872 / 0.19241 / 0.14565 | matched=34，latest +0.00231；late20 +0.00931，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 23 | 0.18319 / 0.38836 | 0.18319 @23 | 0.17025 / 0.15716 / 0.12254 / 0.11086 | - |
| dynamic_singleproj_yoloinit | 24 | 0.18440 / 0.37380 | 0.18440 @24 | 0.17702 / 0.16149 / 0.12884 / 0.11398 | matched=23，latest -0.00395；late20 -0.00056，pre100 |
| dynamic_wo_s_rec_yoloinit | 24 | 0.17899 / 0.37275 | 0.19227 @23 | 0.17555 / 0.16304 / 0.13249 / 0.11635 | matched=23，latest +0.00908；late20 +0.00279，pre100 |
| dynamic_wo_reach_yoloinit | 19 | 0.14964 / 0.33957 | 0.14964 @19 | 0.14141 / 0.13066 / 0.09346 / 0.09346 | matched=19，latest -0.01055；late20 -0.00437，pre100 |

决策：

- 4090 `dynamic` 继续保持最强早筛证据：latest delta 约 +1.29 point，late20 delta 约 +1.18 point，状态 `PROMISING_EARLY`，继续跑满 800。
- 4090 `ProbeA` 正增益继续缩在 +0.8 point 内，仍为 `WATCH`。
- 4090 old-commit ProbeA 与 3090 三个候选都未到 100 rows；3090 `wo_reach` 早期偏负，但按规则还不能判为 low-priority。
- 本轮没有异常、没有资源风险，不新增实验、不停止任务。

### 2026-06-24 16:59 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 86%`；GPU1 `8597/24564 MiB, util 92%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：本轮 SSH 直连 `36.103.203.216:2222` 和 SOCKS 通道均超时，未拿到新的 `results.csv` 快照；这只作为连接层不可达记录，不视为训练异常，也不据此做候选判断。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 240 | 0.42349 / 0.67182 | 0.42349 @240 | 0.42256 / 0.42090 / 0.41824 / 0.41003 | - |
| ProbeA | 183 | 0.39846 / 0.65123 | 0.39846 @183 | 0.39738 / 0.39582 / 0.39339 / 0.38561 | latest +0.00669；late20 +0.00761；epoch100 +0.00710；positive 172/183，`WATCH` |
| dynamic | 131 | 0.37596 / 0.62447 | 0.37596 @131 | 0.37454 / 0.37281 / 0.36955 / 0.35945 | latest +0.01277；late20 +0.01191；epoch100 +0.01043；positive 124/131，`PROMISING_EARLY` |
| old-commit ProbeA | 36 | 0.24924 / 0.47936 | 0.24924 @36 | 0.23930 / 0.22873 / 0.20324 / 0.15131 | matched=36，latest +0.00773；late20 +0.01038，pre100，不判断 |

决策：

- 4090 `dynamic` 继续增强，是当前最稳定的 YOLO-init 主线候选：late20 delta 约 +1.19 AP50-95 point，保持 `PROMISING_EARLY`。
- 4090 `ProbeA` 仍是 `WATCH`，正增益约 +0.7 到 +0.8 point。
- 4090 old-commit ProbeA 仍未到 100 rows，虽然 late20 已超过 +1 point，但按规则仍只记录，不作为正结果。
- 3090 本轮无法更新，继续等待下一轮 SSH 恢复后再判断；本轮不新增实验、不停止任务。

### 2026-06-24 17:03 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 95%`；GPU1 `8597/24564 MiB, util 88%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：SSH direct 恢复成功；GPU0 `7621/24576 MiB, util 98%`；GPU1 `8417/24576 MiB, util 98%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 244 | 0.42615 / 0.67417 | 0.42615 @244 | 0.42485 / 0.42340 / 0.42048 / 0.41225 | - |
| ProbeA | 186 | 0.39966 / 0.65369 | 0.39966 @186 | 0.39880 / 0.39748 / 0.39487 / 0.38721 | latest +0.00589；late20 +0.00730；epoch100 +0.00710；positive 175/186，`WATCH` |
| dynamic | 133 | 0.37736 / 0.62614 | 0.37736 @133 | 0.37579 / 0.37415 / 0.37086 / 0.36082 | latest +0.01286；late20 +0.01209；epoch100 +0.01043；positive 126/133，`PROMISING_EARLY` |
| old-commit ProbeA | 38 | 0.25621 / 0.48994 | 0.25621 @38 | 0.24848 / 0.23687 / 0.21305 / 0.15676 | matched=38，latest +0.01129；late20 +0.01068，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 29 | 0.20735 / 0.42298 | 0.20735 @29 | 0.20086 / 0.18768 / 0.16164 / 0.12881 | - |
| dynamic_singleproj_yoloinit | 28 | 0.20306 / 0.41867 | 0.20306 @28 | 0.19377 / 0.18275 / 0.15569 / 0.12571 | matched=28，latest -0.00103；late20 -0.00009，pre100 |
| dynamic_wo_s_rec_yoloinit | 29 | 0.21230 / 0.42882 | 0.21230 @29 | 0.19971 / 0.18763 / 0.16226 / 0.13072 | matched=29，latest +0.00495；late20 +0.00062，pre100 |
| dynamic_wo_reach_yoloinit | 23 | 0.16446 / 0.35790 | 0.16917 @22 | 0.16160 / 0.14947 / 0.11662 / 0.10583 | matched=23，latest -0.01873；late20 -0.00592，pre100 |

决策：

- 4090 `dynamic` 继续是最强 early 主线候选：latest delta 约 +1.29 point，late20 delta 约 +1.21 point，保持 `PROMISING_EARLY`。
- 4090 `ProbeA` 仍是 `WATCH`，正增益收窄到约 +0.7 point。
- 4090 old-commit ProbeA 仍未到 100 rows，即使 latest/late20 为正也不作正结果。
- 3090 三个候选仍未到 100 rows；`wo_reach` 明显弱于同源 det-only，但尚未到 120 rows，不触发 low-priority 规则；`singleproj/wo_s_rec` 基本贴近 det-only。
- 本轮不新增实验、不停止任务，继续等待 3090 组和 old-commit ProbeA 到 100 epoch。

### 2026-06-24 17:07 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 93%`；GPU1 `8597/24564 MiB, util 90%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：SSH direct 正常；GPU0 `7621/24576 MiB, util 98%`；GPU1 `8417/24576 MiB, util 90%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 246 | 0.42724 / 0.67544 | 0.42724 @246 | 0.42602 / 0.42455 / 0.42156 / 0.41333 | - |
| ProbeA | 188 | 0.40114 / 0.65433 | 0.40114 @188 | 0.39980 / 0.39859 / 0.39591 / 0.38826 | latest +0.00626；late20 +0.00710；epoch100 +0.00710；positive 177/188，`WATCH` |
| dynamic | 135 | 0.37856 / 0.62718 | 0.37856 @135 | 0.37713 / 0.37548 / 0.37216 / 0.36217 | latest +0.01318；late20 +0.01227；epoch100 +0.01043；positive 128/135，`PROMISING_EARLY` |
| old-commit ProbeA | 40 | 0.26611 / 0.50698 | 0.26611 @40 | 0.25755 / 0.24601 / 0.22319 / 0.16215 | matched=40，latest +0.01359；late20 +0.01143，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 31 | 0.22036 / 0.44928 | 0.22036 @31 | 0.21077 / 0.19723 / 0.17259 / 0.13454 | - |
| dynamic_singleproj_yoloinit | 30 | 0.21763 / 0.44263 | 0.21763 @30 | 0.20552 / 0.19308 / 0.16628 / 0.13160 | matched=30，latest +0.00288；late20 -0.00089，pre100 |
| dynamic_wo_s_rec_yoloinit | 30 | 0.21567 / 0.43542 | 0.21567 @30 | 0.20335 / 0.19261 / 0.16767 / 0.13355 | matched=30，latest +0.00092；late20 +0.00050，pre100 |
| dynamic_wo_reach_yoloinit | 25 | 0.17661 / 0.38930 | 0.17995 @24 | 0.16992 / 0.15909 / 0.13094 / 0.11162 | matched=25，latest -0.01901；late20 -0.00663，pre100 |

决策：

- 4090 `dynamic` 继续增强：latest delta 约 +1.32 point，late20 delta 约 +1.23 point，保持 `PROMISING_EARLY`。
- 4090 `ProbeA` 正增益进一步收窄，仍为 `WATCH`。
- 4090 old-commit ProbeA 仍 pre100，当前 latest/late20 为正，但不作为正式正结果。
- 3090 `singleproj/wo_s_rec` 仍贴近 det-only；`wo_reach` 早期明显偏负，但 rows=25，尚未到 120-row low-priority 规则。继续观察。
- 本轮不新增实验、不停止任务。

### 2026-06-24 17:10 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 93%`；GPU1 `8597/24564 MiB, util 92%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：SSH direct 正常；GPU0 `7621/24576 MiB, util 97%`；GPU1 `8417/24576 MiB, util 96%`。4 个训练进程在跑，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 248 | 0.42781 / 0.67619 | 0.42781 @248 | 0.42692 / 0.42559 / 0.42262 / 0.41442 | - |
| ProbeA | 190 | 0.40148 / 0.65456 | 0.40148 @190 | 0.40078 / 0.39956 / 0.39689 / 0.38928 | latest +0.00499；late20 +0.00681；epoch100 +0.00710；positive 179/190，`WATCH` |
| dynamic | 137 | 0.38002 / 0.62815 | 0.38002 @137 | 0.37855 / 0.37681 / 0.37348 / 0.36352 | latest +0.01316；late20 +0.01246；epoch100 +0.01043；positive 130/137，`PROMISING_EARLY` |
| old-commit ProbeA | 41 | 0.26503 / 0.49999 | 0.26611 @40 | 0.26071 / 0.25000 / 0.22744 / 0.16465 | matched=41，latest +0.00907；late20 +0.01124，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 33 | 0.22556 / 0.45341 | 0.22743 @32 | 0.21909 / 0.20739 / 0.18228 / 0.14011 | - |
| dynamic_singleproj_yoloinit | 31 | 0.21818 / 0.44794 | 0.21818 @31 | 0.21028 / 0.19753 / 0.17116 / 0.13440 | matched=31，latest -0.00218；late20 -0.00143，pre100 |
| dynamic_wo_s_rec_yoloinit | 32 | 0.22227 / 0.44751 | 0.22444 @31 | 0.21476 / 0.20322 / 0.17782 / 0.13917 | matched=32，latest -0.00516；late20 +0.00010，pre100 |
| dynamic_wo_reach_yoloinit | 27 | 0.18807 / 0.40449 | 0.18807 @27 | 0.17924 / 0.16842 / 0.14323 / 0.11725 | matched=27，latest -0.01925；late20 -0.00694，pre100 |

决策：

- 4090 `dynamic` 仍是当前唯一达到 `PROMISING_EARLY` 的 YOLO-init 主线候选：已 137 rows，epoch100 delta +1.04 point，late20 delta +1.25 point，latest delta +1.32 point；继续跑满 800。
- 4090 `ProbeA` 已 190 rows，但领先幅度只有约 +0.5 到 +0.7 point，继续标记 `WATCH`，不能当作稳定主线正结果。
- 4090 old-commit ProbeA 仍只有 41 rows，虽然 late20 为正，但未到 100-row 早筛点，不做结论。
- 3090 三条变体都还在 30 rows 左右，均未到 100-row 早筛点；当前 `wo_reach` 明显偏弱，`singleproj/wo_s_rec` 近似 det-only，继续观察。
- 自动化状态确认：active goal 正常；`ogsod-yolo-init-mainline-search` heartbeat 为 `ACTIVE`，频率 `FREQ=MINUTELY;INTERVAL=30`，挂在当前线程。本轮不新增实验、不停止任务。

### 2026-06-24 17:15 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 91%`；GPU1 `8597/24564 MiB, util 92%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：SSH direct 正常；GPU0 `7621/24576 MiB, util 93%`；GPU1 `8417/24576 MiB, util 98%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 252 | 0.42965 / 0.67845 | 0.42965 @252 | 0.42900 / 0.42774 / 0.42492 / 0.41663 | - |
| ProbeA | 193 | 0.40321 / 0.65648 | 0.40321 @193 | 0.40208 / 0.40094 / 0.39838 / 0.39079 | latest +0.00579；late20 +0.00644；epoch100 +0.00710；positive 182/193，`WATCH` |
| dynamic | 140 | 0.38205 / 0.63037 | 0.38205 @140 | 0.38059 / 0.37886 / 0.37548 / 0.36553 | latest +0.01370；late20 +0.01267；epoch100 +0.01043；positive 133/140，`PROMISING_EARLY` |
| old-commit ProbeA | 44 | 0.28034 / 0.51960 | 0.28034 @44 | 0.27145 / 0.26248 / 0.24060 / 0.17220 | matched=44，latest +0.01082；late20 +0.01090，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 36 | 0.24277 / 0.47164 | 0.24277 @36 | 0.23223 / 0.22150 / 0.19716 / 0.14810 | - |
| dynamic_singleproj_yoloinit | 34 | 0.23741 / 0.46737 | 0.23741 @34 | 0.22448 / 0.21173 / 0.18661 / 0.14273 | matched=34，latest +0.00783；late20 -0.00053，pre100 |
| dynamic_wo_s_rec_yoloinit | 35 | 0.24096 / 0.46959 | 0.24096 @35 | 0.23073 / 0.21704 / 0.19368 / 0.14744 | matched=35，latest +0.00515；late20 +0.00185，pre100 |
| dynamic_wo_reach_yoloinit | 30 | 0.21296 / 0.43405 | 0.21296 @30 | 0.19947 / 0.18469 / 0.16055 / 0.12626 | matched=30，latest -0.00179；late20 -0.00663，pre100 |

决策：

- 4090 `dynamic` 继续是当前唯一 `PROMISING_EARLY` 候选：已 140 rows，latest delta +1.37 point，late20 delta +1.27 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍然稳定为正但幅度不足：late20 约 +0.64 point，继续 `WATCH`，不作为可靠主线正结果。
- 4090 old-commit ProbeA 仍 pre100；当前 latest/late20 为正且约 +1 point，但未到 100-row 早筛点，不能下结论。
- 3090 `singleproj/wo_s_rec/wo_reach` 仍未到 100 rows；`singleproj` 最新点为正但 late20 略负，`wo_s_rec` 贴近 det-only，`wo_reach` 早期偏负。全部继续观察，不触发停止或 low-priority。
- 当前两台服务器均有 4 个训练进程且 GPU util 高；按记录中的自动推进原则，本轮不新增实验、不停止有效任务，等待 3090 组和 old-commit ProbeA 到 100-row 评估点。

### 2026-06-24 17:18 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 92%`；GPU1 `8597/24564 MiB, util 83%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：SSH direct 正常；GPU0 `7621/24576 MiB, util 93%`；GPU1 `8417/24576 MiB, util 93%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 254 | 0.43060 / 0.67970 | 0.43060 @254 | 0.42987 / 0.42863 / 0.42601 / 0.41772 | - |
| ProbeA | 195 | 0.40392 / 0.65793 | 0.40392 @195 | 0.40302 / 0.40190 / 0.39941 / 0.39179 | latest +0.00447；late20 +0.00619；epoch100 +0.00710；positive 184/195，`WATCH` |
| dynamic | 141 | 0.38285 / 0.63132 | 0.38285 @141 | 0.38133 / 0.37955 / 0.37618 / 0.36621 | latest +0.01431；late20 +0.01280；epoch100 +0.01043；positive 134/141，`PROMISING_EARLY` |
| old-commit ProbeA | 46 | 0.28724 / 0.52649 | 0.28724 @46 | 0.27960 / 0.27016 / 0.24944 / 0.17715 | matched=46，latest +0.01235；late20 +0.01092，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 39 | 0.25020 / 0.48303 | 0.25293 @38 | 0.24553 / 0.23453 / 0.21111 / 0.15592 | - |
| dynamic_singleproj_yoloinit | 36 | 0.24321 / 0.47334 | 0.24321 @36 | 0.23339 / 0.22183 / 0.19707 / 0.14815 | matched=36，latest +0.00044；late20 -0.00009，pre100 |
| dynamic_wo_s_rec_yoloinit | 37 | 0.24436 / 0.47550 | 0.24436 @37 | 0.23825 / 0.22650 / 0.20257 / 0.15255 | matched=37，latest -0.00160；late20 +0.00068，pre100 |
| dynamic_wo_reach_yoloinit | 31 | 0.21439 / 0.43498 | 0.21439 @31 | 0.20492 / 0.19019 / 0.16530 / 0.12911 | matched=31，latest -0.00597；late20 -0.00729，pre100 |

决策：

- 4090 `dynamic` 优势继续略增，是当前最强 YOLO-init 主线候选：141 rows，latest delta +1.43 point，late20 delta +1.28 point，epoch100 delta +1.04 point，继续跑满 800。
- 4090 `ProbeA` 正增益继续收窄，latest 只剩 +0.45 point、late20 +0.62 point；继续 `WATCH`，不作为主线正结果。
- 4090 old-commit ProbeA 仍 pre100；虽然当前 latest/late20 约 +1 point，但按规则不到 100 rows 不下结论。
- 3090 `singleproj/wo_s_rec/wo_reach` 仍未到 100 rows；`singleproj/wo_s_rec` 基本贴近 det-only，`wo_reach` 仍偏负，但未到 120-row low-priority 门槛。
- 本轮不新增实验、不停止任务。下一次关键节点仍是：4090 old-commit ProbeA 到 100 rows；3090 同源组到 100 rows 后做正式早筛。
- 同步状态：本地记录与 `ladd4090-zw1` 已同步；`ladd3090-zw1` 本轮采集成功，但记录同步阶段 direct IP、SOCKS fallback、alias/master 三种通道均失败，作为 SSH 连接层波动记录，下一轮重试同步。

### 2026-06-24 17:23 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 93%`；GPU1 `8597/24564 MiB, util 93%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP 本轮首次连接超时，随后 SOCKS fallback 成功；GPU0 `7621/24576 MiB, util 97%`；GPU1 `8417/24576 MiB, util 98%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 257 | 0.43247 / 0.68233 | 0.43247 @257 | 0.43120 / 0.43010 / 0.42759 / 0.41936 | - |
| ProbeA | 197 | 0.40494 / 0.66039 | 0.40494 @197 | 0.40405 / 0.40286 / 0.40044 / 0.39279 | latest +0.00528；late20 +0.00600；epoch100 +0.00710；positive 186/197，`WATCH` |
| dynamic | 144 | 0.38459 / 0.63380 | 0.38459 @144 | 0.38336 / 0.38163 / 0.37822 / 0.36821 | latest +0.01442；late20 +0.01314；epoch100 +0.01043；positive 137/144，`PROMISING_EARLY` |
| old-commit ProbeA | 48 | 0.29467 / 0.53855 | 0.29467 @48 | 0.28761 / 0.27776 / 0.25732 / 0.18197 | matched=48，latest +0.01269；late20 +0.01042，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 42 | 0.26506 / 0.50174 | 0.26506 @42 | 0.25702 / 0.24648 / 0.22481 / 0.16340 | - |
| dynamic_singleproj_yoloinit | 38 | 0.25012 / 0.48378 | 0.25012 @38 | 0.24262 / 0.23085 / 0.20680 / 0.15338 | matched=38，latest -0.00281；late20 +0.00019，pre100 |
| dynamic_wo_s_rec_yoloinit | 39 | 0.25101 / 0.48050 | 0.25216 @38 | 0.24568 / 0.23568 / 0.21165 / 0.15763 | matched=39，latest +0.00081；late20 +0.00055，pre100 |
| dynamic_wo_reach_yoloinit | 34 | 0.23069 / 0.45931 | 0.23069 @34 | 0.21936 / 0.20578 / 0.18016 / 0.13740 | matched=34，latest +0.00111；late20 -0.00698，pre100 |

决策：

- 4090 `dynamic` 继续增强，仍是唯一明确 `PROMISING_EARLY` 的主线候选：144 rows，latest delta +1.44 point，late20 delta +1.31 point，epoch100 delta +1.04 point，继续跑满 800。
- 4090 `ProbeA` 仍为 `WATCH`：正增益维持在 +0.5 到 +0.7 point，明显弱于 dynamic。
- 4090 old-commit ProbeA 仍 pre100；当前 latest/late20 为正，但不到 100 rows 不作为结论。
- 3090 `singleproj/wo_s_rec/wo_reach` 仍未到 100 rows；三条基本贴近 det-only，`wo_reach` late20 仍偏负，但未到 120-row low-priority 门槛。
- 本轮不新增实验、不停止任务。下一关键节点仍是 old-commit ProbeA 与 3090 同源组达到 100 rows 后的正式早筛。

### 2026-06-24 17:26 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 92%`；GPU1 `8597/24564 MiB, util 88%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：本轮 direct IP 超时、SOCKS fallback 超时、alias/master 返回 password 认证失败；视为 SSH 连接层不可达，不视为训练异常。本轮不更新 3090 曲线，也不基于 3090 做候选判断。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 259 | 0.43278 / 0.68300 | 0.43305 @258 | 0.43223 / 0.43105 / 0.42859 / 0.42042 | - |
| ProbeA | 199 | 0.40602 / 0.66193 | 0.40602 @199 | 0.40504 / 0.40379 / 0.40144 / 0.39380 | latest +0.00461；late20 +0.00575；epoch100 +0.00710；positive 188/199，`WATCH` |
| dynamic | 146 | 0.38587 / 0.63546 | 0.38587 @146 | 0.38463 / 0.38298 / 0.37957 / 0.36956 | latest +0.01519；late20 +0.01342；epoch100 +0.01043；positive 139/146，`PROMISING_EARLY` |
| old-commit ProbeA | 50 | 0.29958 / 0.54690 | 0.29958 @50 | 0.29370 / 0.28443 / 0.26522 / 0.18660 | matched=50，latest +0.01325；late20 +0.01041，pre100，不判断 |

决策：

- 4090 `dynamic` 继续增强，是当前唯一明确 `PROMISING_EARLY` 的主线候选：146 rows，latest delta +1.52 point，late20 delta +1.34 point，epoch100 delta +1.04 point，继续跑满 800。
- 4090 `ProbeA` 正增益进一步收窄，latest +0.46 point、late20 +0.58 point，仍为 `WATCH`。
- 4090 old-commit ProbeA 到 50 rows，当前 latest/late20 约 +1 point，但未到 100-row 早筛点，不作为结论。
- 3090 本轮不可达，沿用上一轮 17:23 记录作为最近有效快照；不新增实验、不停止任务。
- 同步状态：本地记录与 `ladd4090-zw1` 同步；`ladd3090-zw1` 因连接不可达，本轮未同步，下一轮重试。

### 2026-06-24 17:29 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 92%`；GPU1 `8597/24564 MiB, util 91%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP 本轮恢复成功；GPU0 `7623/24576 MiB, util 98%`；GPU1 `8417/24576 MiB, util 97%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 261 | 0.43388 / 0.68453 | 0.43388 @261 | 0.43315 / 0.43189 / 0.42958 / 0.42148 | - |
| ProbeA | 200 | 0.40657 / 0.66252 | 0.40657 @200 | 0.40557 / 0.40430 / 0.40193 / 0.39431 | latest +0.00505；late20 +0.00564；epoch100 +0.00710；positive 189/200，`WATCH` |
| dynamic | 147 | 0.38639 / 0.63622 | 0.38639 @147 | 0.38522 / 0.38361 / 0.38021 / 0.37024 | latest +0.01505；late20 +0.01354；epoch100 +0.01043；positive 140/147，`PROMISING_EARLY` |
| old-commit ProbeA | 51 | 0.30048 / 0.54833 | 0.30048 @51 | 0.29634 / 0.28797 / 0.26899 / 0.19217 | matched=51，latest +0.01070；late20 +0.01063，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 45 | 0.27545 / 0.51372 | 0.27545 @45 | 0.26766 / 0.25860 / 0.23741 / 0.17056 | - |
| dynamic_singleproj_yoloinit | 41 | 0.25950 / 0.49428 | 0.25950 @41 | 0.25211 / 0.24275 / 0.22014 / 0.16082 | matched=41，latest -0.00152；late20 +0.00017，pre100 |
| dynamic_wo_s_rec_yoloinit | 42 | 0.26295 / 0.49350 | 0.26295 @42 | 0.25722 / 0.24773 / 0.22548 / 0.16501 | matched=42，latest -0.00211；late20 +0.00066，pre100 |
| dynamic_wo_reach_yoloinit | 37 | 0.23850 / 0.46487 | 0.24076 @36 | 0.23241 / 0.22164 / 0.19503 / 0.14546 | matched=37，latest -0.00746；late20 -0.00686，pre100 |

决策：

- 4090 `dynamic` 继续保持唯一明确 `PROMISING_EARLY`：147 rows，latest delta +1.51 point，late20 delta +1.35 point，epoch100 delta +1.04 point，继续跑满 800。
- 4090 `ProbeA` 到 200 rows，仍只有约 +0.5 到 +0.7 point，继续 `WATCH`。
- 4090 old-commit ProbeA 到 51 rows，当前 latest/late20 约 +1 point，但未到 100-row 早筛点，不下结论。
- 3090 `singleproj/wo_s_rec/wo_reach` 仍未到 100 rows；`singleproj/wo_s_rec` 近似 det-only，`wo_reach` early late20 仍偏负但未到 120-row low-priority 门槛。
- 本轮不新增实验、不停止任务。下一关键节点仍是 old-commit ProbeA 和 3090 同源组到 100 rows。
- 同步状态：本地记录与 `ladd4090-zw1` 已同步；`ladd3090-zw1` 采集成功，但记录同步阶段 direct IP 与 SOCKS fallback 均断开，下一轮重试同步。

### 2026-06-24 17:32 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 92%`；GPU1 `8597/24564 MiB, util 95%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP 本轮恢复成功；GPU0 `7623/24576 MiB, util 96%`；GPU1 `8417/24576 MiB, util 98%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 263 | 0.43437 / 0.68486 | 0.43437 @263 | 0.43373 / 0.43276 / 0.43047 / 0.42252 | - |
| ProbeA | 202 | 0.40803 / 0.66336 | 0.40803 @202 | 0.40674 / 0.40539 / 0.40293 / 0.39533 | latest +0.00587；late20 +0.00548；epoch100 +0.00710；positive 191/202，`WATCH` |
| dynamic | 149 | 0.38769 / 0.63875 | 0.38769 @149 | 0.38644 / 0.38490 / 0.38152 / 0.37156 | latest +0.01526；late20 +0.01377；epoch100 +0.01043；positive 142/149，`PROMISING_EARLY` |
| old-commit ProbeA | 52 | 0.30230 / 0.54762 | 0.30230 @52 | 0.29859 / 0.29113 / 0.27272 / 0.19790 | matched=52，latest +0.01195；late20 +0.01116，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 47 | 0.27885 / 0.52027 | 0.27885 @47 | 0.27361 / 0.26531 / 0.24534 / 0.17512 | - |
| dynamic_singleproj_yoloinit | 42 | 0.25875 / 0.49457 | 0.25950 @41 | 0.25483 / 0.24644 / 0.22403 / 0.16316 | matched=42，latest -0.00631；late20 -0.00078，pre100 |
| dynamic_wo_s_rec_yoloinit | 44 | 0.27049 / 0.50781 | 0.27049 @44 | 0.26362 / 0.25465 / 0.23367 / 0.16968 | matched=44，latest -0.00006；late20 +0.00025，pre100 |
| dynamic_wo_reach_yoloinit | 38 | 0.24251 / 0.47187 | 0.24251 @38 | 0.23672 / 0.22567 / 0.19993 / 0.14802 | matched=38，latest -0.01042；late20 -0.00668，pre100 |

决策：

- 4090 `dynamic` 继续是唯一明确 `PROMISING_EARLY` 候选：149 rows，latest delta +1.53 point，late20 delta +1.38 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为 `WATCH`：到 202 rows 仍只有约 +0.55 point late20，明显弱于 dynamic。
- 4090 old-commit ProbeA 到 52 rows，当前 latest/late20 仍约 +1 point，但未到 100-row 早筛点，不下结论。
- 3090 三条变体仍未到 100 rows；`singleproj` latest/late20 转弱，`wo_s_rec` 近似 det-only，`wo_reach` early 偏负，但均未到 120-row low-priority 门槛。
- 本轮不新增实验、不停止任务；继续等待 old-commit ProbeA 和 3090 同源组到 100 rows。

### 2026-06-24 17:37 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 96%`；GPU1 `8597/24564 MiB, util 92%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP 本轮成功；GPU0 `7623/24576 MiB, util 91%`；GPU1 `8417/24576 MiB, util 92%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 266 | 0.43654 / 0.68653 | 0.43654 @266 | 0.43532 / 0.43423 / 0.43191 / 0.42408 | - |
| ProbeA | 205 | 0.40944 / 0.66502 | 0.40944 @205 | 0.40840 / 0.40699 / 0.40445 / 0.39689 | latest +0.00586；late20 +0.00536；epoch100 +0.00710；positive 194/205，`WATCH` |
| dynamic | 151 | 0.38904 / 0.63975 | 0.38904 @151 | 0.38770 / 0.38616 / 0.38286 / 0.37289 | latest +0.01547；late20 +0.01407；epoch100 +0.01043；positive 144/151，`PROMISING_EARLY` |
| old-commit ProbeA | 55 | 0.30859 / 0.55388 | 0.30859 @55 | 0.30480 / 0.29925 / 0.28280 / 0.21433 | matched=55，latest +0.01045；late20 +0.01149，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 50 | 0.28806 / 0.52965 | 0.28806 @50 | 0.28283 / 0.27524 / 0.25695 / 0.18178 | - |
| dynamic_singleproj_yoloinit | 45 | 0.27415 / 0.51614 | 0.27415 @45 | 0.26418 / 0.25652 / 0.23673 / 0.17012 | matched=45，latest -0.00130；late20 -0.00068，pre100 |
| dynamic_wo_s_rec_yoloinit | 46 | 0.27759 / 0.51498 | 0.27759 @46 | 0.27024 / 0.26187 / 0.24186 / 0.17432 | matched=46，latest +0.00061；late20 +0.00010，pre100 |
| dynamic_wo_reach_yoloinit | 41 | 0.25295 / 0.48329 | 0.25295 @41 | 0.24584 / 0.23705 / 0.21362 / 0.15544 | matched=41，latest -0.00807；late20 -0.00635，pre100 |

决策：

- 4090 `dynamic` 继续增强，仍是唯一明确 `PROMISING_EARLY` 候选：151 rows，latest delta +1.55 point，late20 delta +1.41 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为 `WATCH`：到 205 rows 只有约 +0.54 point late20，远弱于 dynamic。
- 4090 old-commit ProbeA 到 55 rows，当前 latest/late20 仍约 +1 point，但未到 100-row 早筛点，不下结论。
- 3090 同源组仍未到 100 rows；`singleproj` 轻微转负，`wo_s_rec` 贴近 det-only，`wo_reach` 仍偏负，但均未触发 120-row low-priority。
- 本轮不新增实验、不停止任务；继续等待 100-row 早筛点。

### 2026-06-24 17:39 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 93%`；GPU1 `8597/24564 MiB, util 88%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP 本轮成功；GPU0 `7623/24576 MiB, util 93%`；GPU1 `8417/24576 MiB, util 94%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 268 | 0.43739 / 0.68767 | 0.43739 @268 | 0.43657 / 0.43515 / 0.43289 / 0.42513 | - |
| ProbeA | 206 | 0.40991 / 0.66544 | 0.40991 @206 | 0.40893 / 0.40753 / 0.40496 / 0.39741 | latest +0.00532；late20 +0.00533；epoch100 +0.00710；positive 195/206，`WATCH` |
| dynamic | 153 | 0.39073 / 0.64194 | 0.39073 @153 | 0.38912 / 0.38747 / 0.38420 / 0.37422 | latest +0.01604；late20 +0.01439；epoch100 +0.01043；positive 146/153，`PROMISING_EARLY` |
| old-commit ProbeA | 56 | 0.30999 / 0.55554 | 0.30999 @56 | 0.30670 / 0.30152 / 0.28584 / 0.21893 | matched=56，latest +0.01139；late20 +0.01167，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 52 | 0.29537 / 0.53597 | 0.29537 @52 | 0.28928 / 0.28144 / 0.26396 / 0.19164 | - |
| dynamic_singleproj_yoloinit | 46 | 0.27498 / 0.51643 | 0.27498 @46 | 0.26728 / 0.25970 / 0.24076 / 0.17240 | matched=46，latest -0.00200；late20 -0.00100，pre100 |
| dynamic_wo_s_rec_yoloinit | 47 | 0.28632 / 0.52345 | 0.28632 @47 | 0.27491 / 0.26607 / 0.24628 / 0.17671 | matched=47，latest +0.00747；late20 +0.00094，pre100 |
| dynamic_wo_reach_yoloinit | 42 | 0.25604 / 0.49136 | 0.25604 @42 | 0.24935 / 0.24088 / 0.21797 / 0.15783 | matched=42，latest -0.00902；late20 -0.00685，pre100 |

决策：

- 4090 `dynamic` 继续增强，仍是唯一明确 `PROMISING_EARLY` 候选：153 rows，latest delta +1.60 point，late20 delta +1.44 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为 `WATCH`：到 206 rows 只有约 +0.53 point late20。
- 4090 old-commit ProbeA 到 56 rows，当前 latest/late20 仍约 +1 point，但未到 100-row 早筛点，不下结论。
- 3090 同源组仍未到 100 rows；`wo_s_rec` 最新点短暂较高但 late20 只有 +0.09 point，`singleproj/wo_reach` 偏负，均不触发正式判断。
- 本轮不新增实验、不停止任务；继续等待 100-row 早筛点。

### 2026-06-24 17:42 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 92%`；GPU1 `8597/24564 MiB, util 85%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP 本轮成功；GPU0 `7623/24576 MiB, util 94%`；GPU1 `8417/24576 MiB, util 98%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 270 | 0.43850 / 0.68877 | 0.43850 @270 | 0.43763 / 0.43621 / 0.43384 / 0.42617 | - |
| ProbeA | 208 | 0.41083 / 0.66717 | 0.41083 @208 | 0.40993 / 0.40859 / 0.40596 / 0.39842 | latest +0.00490；late20 +0.00524；epoch100 +0.00710；positive 197/208，`WATCH` |
| dynamic | 154 | 0.39055 / 0.64258 | 0.39073 @153 | 0.38970 / 0.38807 / 0.38485 / 0.37488 | latest +0.01567；late20 +0.01454；epoch100 +0.01043；positive 147/154，`PROMISING_EARLY` |
| old-commit ProbeA | 57 | 0.31187 / 0.55459 | 0.31187 @57 | 0.30862 / 0.30360 / 0.28876 / 0.22366 | matched=57，latest +0.01114；late20 +0.01167，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 54 | 0.29976 / 0.53985 | 0.29976 @54 | 0.29467 / 0.28749 / 0.27106 / 0.20298 | - |
| dynamic_singleproj_yoloinit | 47 | 0.28235 / 0.52455 | 0.28235 @47 | 0.27200 / 0.26341 / 0.24478 / 0.17474 | matched=47，latest +0.00350；late20 -0.00056，pre100 |
| dynamic_wo_s_rec_yoloinit | 49 | 0.28751 / 0.52607 | 0.28751 @49 | 0.28246 / 0.27304 / 0.25436 / 0.18119 | matched=49，latest +0.00039；late20 +0.00108，pre100 |
| dynamic_wo_reach_yoloinit | 44 | 0.26201 / 0.50150 | 0.26201 @44 | 0.25606 / 0.24777 / 0.22677 / 0.16249 | matched=44，latest -0.00854；late20 -0.00665，pre100 |

决策：

- 4090 `dynamic` 继续保持唯一明确 `PROMISING_EARLY`：154 rows，latest delta +1.57 point，late20 delta +1.45 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为 `WATCH`：到 208 rows 只有约 +0.52 point late20，未达到主线候选强度。
- 4090 old-commit ProbeA 到 57 rows，当前 latest/late20 仍约 +1 point，但未到 100-row 早筛点，不下结论。
- 3090 同源组仍未到 100 rows；`wo_s_rec` 贴近 det-only，`singleproj/wo_reach` 短窗口不稳或偏负，不触发正式判断。
- 本轮不新增实验、不停止任务；继续等待 100-row 早筛点。

### 2026-06-24 17:45 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 95%`；GPU1 `8597/24564 MiB, util 91%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP 首次连接超时后 SOCKS fallback 成功；GPU0 `7623/24576 MiB, util 94%`；GPU1 `8417/24576 MiB, util 90%`。4 个训练进程在跑，15 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 272 | 0.44011 / 0.69038 | 0.44011 @272 | 0.43869 / 0.43733 / 0.43483 / 0.42722 | - |
| ProbeA | 210 | 0.41199 / 0.66875 | 0.41199 @210 | 0.41090 / 0.40965 / 0.40697 / 0.39943 | latest +0.00507；late20 +0.00520；epoch100 +0.00710；positive 199/210，`WATCH` |
| dynamic | 156 | 0.39198 / 0.64306 | 0.39198 @156 | 0.39085 / 0.38927 / 0.38613 / 0.37618 | latest +0.01649；late20 +0.01483；epoch100 +0.01043；positive 149/156，`PROMISING_EARLY` |
| old-commit ProbeA | 59 | 0.31633 / 0.55837 | 0.31633 @59 | 0.31191 / 0.30745 / 0.29427 / 0.23259 | matched=59，latest +0.01310；late20 +0.01188，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 56 | 0.30236 / 0.54160 | 0.30236 @56 | 0.29920 / 0.29259 / 0.27731 / 0.21289 | - |
| dynamic_singleproj_yoloinit | 49 | 0.28584 / 0.52443 | 0.28584 @49 | 0.28022 / 0.27041 / 0.25258 / 0.17923 | matched=49，latest -0.00128；late20 -0.00070，pre100 |
| dynamic_wo_s_rec_yoloinit | 50 | 0.29225 / 0.53028 | 0.29225 @50 | 0.28581 / 0.27624 / 0.25819 / 0.18341 | matched=50，latest +0.00419；late20 +0.00124，pre100 |
| dynamic_wo_reach_yoloinit | 45 | 0.26731 / 0.50879 | 0.26731 @45 | 0.25937 / 0.25139 / 0.23131 / 0.16482 | matched=45，latest -0.00814；late20 -0.00610，pre100 |

决策：

- 4090 `dynamic` 继续是唯一明确 `PROMISING_EARLY`：156 rows，latest delta +1.65 point，late20 delta +1.48 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为 `WATCH`：到 210 rows late20 只有约 +0.52 point，未达到强主线候选标准。
- 4090 old-commit ProbeA 到 59 rows，仍未到 100-row 早筛点；当前 early delta 约 +1.2 point，只作为待观察信号。
- 3090 同源组仍未到 100 rows；`wo_s_rec` 当前 latest 为 +0.42 point 但 late20 只有 +0.12 point，`singleproj/wo_reach` 偏负；均未触发正式判断。
- 本轮不新增实验、不停止任务；继续等待 old-commit ProbeA 和 3090 同源组到 100 rows。

### 2026-06-24 17:50 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 93%`；GPU1 `8597/24564 MiB, util 91%`。4 个训练进程在跑，12 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：普通 alias 本轮未复用到 master，使用 direct IP + 临时密码文件 fallback 成功；GPU0 `7623/24576 MiB, util 98%`；GPU1 `8417/24576 MiB, util 96%`。4 个训练进程在跑，5 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 275 | 0.44060 / 0.69150 | 0.44108 @274 | 0.44022 / 0.43893 / 0.43634 / 0.42877 | - |
| ProbeA | 212 | 0.41355 / 0.66986 | 0.41355 @212 | 0.41209 / 0.41076 / 0.40808 / 0.40048 | latest +0.00532；late20 +0.00522；epoch100 +0.00710；positive 201/212，`WATCH` |
| dynamic | 159 | 0.39339 / 0.64422 | 0.39339 @159 | 0.39251 / 0.39111 / 0.38800 / 0.37813 | latest +0.01575；late20 +0.01526；epoch100 +0.01043；positive 152/159，`PROMISING_EARLY` |
| old-commit ProbeA | 62 | 0.32399 / 0.57013 | 0.32399 @62 | 0.31869 / 0.31365 / 0.30239 / 0.24475 | matched=62，latest +0.01440；late20 +0.01220，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 59 | 0.30888 / 0.54574 | 0.30888 @59 | 0.30427 / 0.29947 / 0.28575 / 0.22586 | - |
| dynamic_singleproj_yoloinit | 51 | 0.28964 / 0.52977 | 0.28964 @51 | 0.28555 / 0.27641 / 0.25958 / 0.18611 | matched=51，latest -0.00307；late20 -0.00098，pre100 |
| dynamic_wo_s_rec_yoloinit | 53 | 0.29815 / 0.53715 | 0.29815 @53 | 0.29349 / 0.28627 / 0.26867 / 0.19902 | matched=53，latest +0.00069；late20 +0.00112，pre100 |
| dynamic_wo_reach_yoloinit | 48 | 0.27747 / 0.52290 | 0.27747 @48 | 0.27032 / 0.26144 / 0.24355 / 0.17165 | matched=48，latest -0.00568；late20 -0.00574，pre100 |

决策：

- 4090 `dynamic` 仍是唯一明确 `PROMISING_EARLY`：159 rows，latest delta +1.58 point，late20 delta +1.53 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 到 212 rows 仍只有约 +0.52 point late20，继续标为 `WATCH`，不作为强主线。
- 4090 old-commit ProbeA 到 62 rows，pre100；当前 early delta 约 +1.2 point，但需要等到 100-row 才能和 dynamic/AutoDL 现象对齐判断。
- 3090 三条同源变体仍 pre100；`wo_s_rec` 轻微正但幅度很小，`singleproj/wo_reach` 偏负。还不到停止或追加新变体的规则点。
- 本轮不新增实验、不停止任务；下一轮继续等待 old-commit ProbeA 和 3090 同源组接近 100 rows。

### 2026-06-24 17:53 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 92%`；GPU1 `8597/24564 MiB, util 94%`。4 个训练进程在跑，12 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP + 临时密码文件连接成功；GPU0 `7623/24576 MiB, util 91%`；GPU1 `8417/24576 MiB, util 99%`。4 个训练进程在跑，5 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 277 | 0.44204 / 0.69312 | 0.44204 @277 | 0.44099 / 0.43984 / 0.43728 / 0.42977 | - |
| ProbeA | 214 | 0.41473 / 0.67130 | 0.41473 @214 | 0.41343 / 0.41191 / 0.40917 / 0.40151 | latest +0.00553；late20 +0.00522；epoch100 +0.00710；positive 203/214，`WATCH` |
| dynamic | 160 | 0.39448 / 0.64448 | 0.39448 @160 | 0.39314 / 0.39170 / 0.38862 / 0.37878 | latest +0.01624；late20 +0.01538；epoch100 +0.01043；positive 153/160，`PROMISING_EARLY` |
| old-commit ProbeA | 63 | 0.32561 / 0.57144 | 0.32561 @63 | 0.32126 / 0.31575 / 0.30492 / 0.24859 | matched=63，latest +0.01390；late20 +0.01225，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 61 | 0.31244 / 0.55038 | 0.31244 @61 | 0.30831 / 0.30375 / 0.29108 / 0.23401 | - |
| dynamic_singleproj_yoloinit | 53 | 0.29349 / 0.53369 | 0.29349 @53 | 0.28962 / 0.28290 / 0.26645 / 0.19666 | matched=53，latest -0.00397；late20 -0.00111，pre100 |
| dynamic_wo_s_rec_yoloinit | 54 | 0.29867 / 0.53789 | 0.29867 @54 | 0.29572 / 0.28909 / 0.27187 / 0.20428 | matched=54，latest -0.00109；late20 +0.00081，pre100 |
| dynamic_wo_reach_yoloinit | 49 | 0.27969 / 0.52251 | 0.27969 @49 | 0.27385 / 0.26496 / 0.24719 / 0.17385 | matched=49，latest -0.00743；late20 -0.00609，pre100 |

决策：

- 4090 `dynamic` 继续是当前唯一强候选：160 rows，latest delta +1.62 point，late20 delta +1.54 point，epoch100 delta +1.04 point；继续跑满 800，并作为下一批 LADD-like 变体的主要参考。
- 4090 `ProbeA` 仍为弱正向 `WATCH`：214 rows，late20 约 +0.52 point，没有达到可作为稳定主线的强度。
- 4090 old-commit ProbeA 到 63 rows，pre100；早期仍有约 +1.2 point 的 late20 信号，但不能提前声明正结果。
- 3090 同源组均未到 100 rows；`wo_s_rec` 的 late20 仍微正但 latest 已略负，`singleproj/wo_reach` 偏负。继续观察，不触发停止或新增。
- 本轮不新增实验、不停止任务；下一决策点仍是 old-commit ProbeA 和 3090 同源组到 100 rows。

### 2026-06-24 17:56 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 93%`；GPU1 `8597/24564 MiB, util 89%`。4 个训练进程在跑，12 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP + 临时密码文件连接成功；GPU0 `7623/24576 MiB, util 86%`；GPU1 `8417/24576 MiB, util 94%`。4 个训练进程在跑，5 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 279 | 0.44262 / 0.69375 | 0.44262 @279 | 0.44171 / 0.44076 / 0.43824 / 0.43078 | - |
| ProbeA | 215 | 0.41522 / 0.67129 | 0.41522 @215 | 0.41408 / 0.41249 / 0.40974 / 0.40202 | latest +0.00509；late20 +0.00526；epoch100 +0.00710；positive 204/215，`WATCH` |
| dynamic | 161 | 0.39486 / 0.64502 | 0.39486 @161 | 0.39372 / 0.39228 / 0.38922 / 0.37942 | latest +0.01640；late20 +0.01549；epoch100 +0.01043；positive 154/161，`PROMISING_EARLY` |
| old-commit ProbeA | 64 | 0.32676 / 0.57210 | 0.32676 @64 | 0.32335 / 0.31763 / 0.30724 / 0.25236 | matched=64，latest +0.01422；late20 +0.01242，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 62 | 0.31453 / 0.55109 | 0.31453 @62 | 0.31049 / 0.30567 / 0.29356 / 0.23781 | - |
| dynamic_singleproj_yoloinit | 54 | 0.29896 / 0.53747 | 0.29896 @54 | 0.29225 / 0.28623 / 0.26953 / 0.20169 | matched=54，latest -0.00080；late20 -0.00154，pre100 |
| dynamic_wo_s_rec_yoloinit | 56 | 0.30203 / 0.54284 | 0.30203 @56 | 0.29931 / 0.29416 / 0.27801 / 0.21370 | matched=56，latest -0.00033；late20 +0.00071，pre100 |
| dynamic_wo_reach_yoloinit | 51 | 0.28484 / 0.52673 | 0.28484 @51 | 0.28030 / 0.27141 / 0.25423 / 0.18090 | matched=51，latest -0.00787；late20 -0.00633，pre100 |

决策：

- 4090 `dynamic` 继续保持强候选：161 rows，latest delta +1.64 point，late20 delta +1.55 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为弱正向 `WATCH`：215 rows，late20 约 +0.53 point，未达到主线强度。
- 4090 old-commit ProbeA 到 64 rows，pre100；early delta 仍约 +1.2 point，但继续等待 100-row 判断点。
- 3090 三条同源变体仍未到 100 rows；`wo_s_rec` 仅微正，`singleproj/wo_reach` 偏负。未触发停止或新增。
- 本轮不新增实验、不停止任务；下一轮继续等待 100-row 决策点。

### 2026-06-24 17:58 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 92%`；GPU1 `8597/24564 MiB, util 80%`。4 个训练进程在跑，12 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP 第一次连接被远端关闭，重试成功；GPU0 `7623/24576 MiB, util 92%`；GPU1 `8417/24576 MiB, util 98%`。4 个训练进程在跑，5 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 280 | 0.44310 / 0.69475 | 0.44310 @280 | 0.44221 / 0.44122 / 0.43871 / 0.43130 | - |
| ProbeA | 217 | 0.41648 / 0.67299 | 0.41648 @217 | 0.41522 / 0.41365 / 0.41087 / 0.40307 | latest +0.00558；late20 +0.00527；epoch100 +0.00710；positive 206/217，`WATCH` |
| dynamic | 162 | 0.39559 / 0.64593 | 0.39559 @162 | 0.39431 / 0.39288 / 0.38983 / 0.38006 | latest +0.01641；late20 +0.01559；epoch100 +0.01043；positive 155/162，`PROMISING_EARLY` |
| old-commit ProbeA | 65 | 0.32792 / 0.57525 | 0.32792 @65 | 0.32504 / 0.31956 / 0.30941 / 0.25614 | matched=65，latest +0.01358；late20 +0.01235，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 64 | 0.31786 / 0.55526 | 0.31786 @64 | 0.31468 / 0.30948 / 0.29848 / 0.24518 | - |
| dynamic_singleproj_yoloinit | 55 | 0.30045 / 0.53831 | 0.30045 @55 | 0.29511 / 0.28886 / 0.27269 / 0.20673 | matched=55，latest -0.00059；late20 -0.00163，pre100 |
| dynamic_wo_s_rec_yoloinit | 57 | 0.30580 / 0.54712 | 0.30580 @57 | 0.30127 / 0.29611 / 0.28109 / 0.21810 | matched=57，latest +0.00217；late20 +0.00090，pre100 |
| dynamic_wo_reach_yoloinit | 52 | 0.28881 / 0.53209 | 0.28881 @52 | 0.28285 / 0.27469 / 0.25779 / 0.18569 | matched=52，latest -0.00656；late20 -0.00617，pre100 |

决策：

- 4090 `dynamic` 继续保持强候选：162 rows，latest delta +1.64 point，late20 delta +1.56 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为弱正向 `WATCH`：217 rows，late20 约 +0.53 point，未达到主线强度。
- 4090 old-commit ProbeA 到 65 rows，pre100；early delta 仍约 +1.2 point，继续等待 100-row 判断点。
- 3090 三条同源变体仍未到 100 rows；`wo_s_rec` latest 暂回正但 late20 仍只有 +0.09 point，`singleproj/wo_reach` 偏负。未触发停止或新增。
- 本轮不新增实验、不停止任务；下一轮继续等待 100-row 决策点。

### 2026-06-24 18:02 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 94%`；GPU1 `8597/24564 MiB, util 89%`。4 个训练进程在跑，12 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP 首次连接被远端关闭，重试成功；GPU0 `7623/24576 MiB, util 96%`；GPU1 `8417/24576 MiB, util 97%`。4 个训练进程在跑，5 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 283 | 0.44408 / 0.69613 | 0.44408 @283 | 0.44351 / 0.44246 / 0.44019 / 0.43282 | - |
| ProbeA | 218 | 0.41687 / 0.67287 | 0.41687 @218 | 0.41578 / 0.41426 / 0.41142 / 0.40360 | latest +0.00548；late20 +0.00530；epoch100 +0.00710；positive 207/218，`WATCH` |
| dynamic | 164 | 0.39700 / 0.64726 | 0.39700 @164 | 0.39568 / 0.39410 / 0.39108 / 0.38136 | latest +0.01654；late20 +0.01582；epoch100 +0.01043；positive 157/164，`PROMISING_EARLY` |
| old-commit ProbeA | 67 | 0.33079 / 0.57671 | 0.33079 @67 | 0.32791 / 0.32330 / 0.31345 / 0.26343 | matched=67，latest +0.01287；late20 +0.01239，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 66 | 0.31918 / 0.55760 | 0.31920 @65 | 0.31764 / 0.31297 / 0.30278 / 0.25238 | - |
| dynamic_singleproj_yoloinit | 57 | 0.30312 / 0.54469 | 0.30312 @57 | 0.29964 / 0.29366 / 0.27854 / 0.21638 | matched=57，latest -0.00051；late20 -0.00165，pre100 |
| dynamic_wo_s_rec_yoloinit | 59 | 0.30962 / 0.55287 | 0.30962 @59 | 0.30535 / 0.30054 / 0.28679 / 0.22676 | matched=59，latest +0.00074；late20 +0.00104，pre100 |
| dynamic_wo_reach_yoloinit | 54 | 0.28906 / 0.53346 | 0.28995 @53 | 0.28722 / 0.28054 / 0.26415 / 0.19632 | matched=54，latest -0.01070；late20 -0.00691，pre100 |

决策：

- 4090 `dynamic` 继续保持强候选：164 rows，latest delta +1.65 point，late20 delta +1.58 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为弱正向 `WATCH`：218 rows，late20 约 +0.53 point，未达到主线强度。
- 4090 old-commit ProbeA 到 67 rows，pre100；early delta 仍约 +1.2 point，继续等待 100-row 判断点。
- 3090 三条同源变体仍未到 100 rows；`wo_s_rec` 小幅正向但只有 +0.10 point late20，`singleproj/wo_reach` 偏负。未触发停止或新增。
- 本轮不新增实验、不停止任务；下一轮继续等待 100-row 决策点。

### 2026-06-24 18:05 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 90%`；GPU1 `8597/24564 MiB, util 90%`。4 个训练进程在跑，12 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP + 临时密码文件连接成功；GPU0 `7623/24576 MiB, util 93%`；GPU1 `8417/24576 MiB, util 98%`。4 个训练进程在跑，5 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 285 | 0.44498 / 0.69691 | 0.44498 @285 | 0.44435 / 0.44328 / 0.44110 / 0.43381 | - |
| ProbeA | 220 | 0.41825 / 0.67394 | 0.41825 @220 | 0.41695 / 0.41551 / 0.41258 / 0.40465 | latest +0.00567；late20 +0.00536；epoch100 +0.00710；positive 209/220，`WATCH` |
| dynamic | 166 | 0.39852 / 0.64934 | 0.39852 @166 | 0.39707 / 0.39539 / 0.39233 / 0.38265 | latest +0.01710；late20 +0.01599；epoch100 +0.01043；positive 159/166，`PROMISING_EARLY` |
| old-commit ProbeA | 68 | 0.33177 / 0.57881 | 0.33177 @68 | 0.32914 / 0.32520 / 0.31531 / 0.26690 | matched=68，latest +0.01306；late20 +0.01241，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 68 | 0.32134 / 0.56047 | 0.32134 @68 | 0.31957 / 0.31622 / 0.30676 / 0.25901 | - |
| dynamic_singleproj_yoloinit | 58 | 0.30574 / 0.54810 | 0.30574 @58 | 0.30209 / 0.29586 / 0.28132 / 0.22097 | matched=58，latest +0.00028；late20 -0.00150，pre100 |
| dynamic_wo_s_rec_yoloinit | 60 | 0.31228 / 0.55630 | 0.31228 @60 | 0.30746 / 0.30254 / 0.28939 / 0.23085 | matched=60，latest +0.00114；late20 +0.00088，pre100 |
| dynamic_wo_reach_yoloinit | 55 | 0.29363 / 0.53613 | 0.29363 @55 | 0.28926 / 0.28317 / 0.26728 / 0.20153 | matched=55，latest -0.00741；late20 -0.00705，pre100 |

决策：

- 4090 `dynamic` 继续保持强候选：166 rows，latest delta +1.71 point，late20 delta +1.60 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为弱正向 `WATCH`：220 rows，late20 约 +0.54 point，未达到主线强度。
- 4090 old-commit ProbeA 到 68 rows，pre100；early delta 仍约 +1.2 point，继续等待 100-row 判断点。
- 3090 三条同源变体仍未到 100 rows；`wo_s_rec` 小幅正向但只有 +0.09 point late20，`singleproj` latest 刚转正但 late20 仍负，`wo_reach` 持续偏负。未触发停止或新增。
- 本轮不新增实验、不停止任务；下一轮继续等待 100-row 决策点。

### 2026-06-24 18:08 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 88%`；GPU1 `8597/24564 MiB, util 84%`。4 个训练进程在跑，12 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP + 临时密码文件连接成功；GPU0 `7623/24576 MiB, util 96%`；GPU1 `8417/24576 MiB, util 95%`。4 个训练进程在跑，5 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 287 | 0.44608 / 0.69817 | 0.44608 @287 | 0.44504 / 0.44410 / 0.44197 / 0.43477 | - |
| ProbeA | 222 | 0.41862 / 0.67413 | 0.41862 @222 | 0.41794 / 0.41658 / 0.41367 / 0.40570 | latest +0.00482；late20 +0.00532；epoch100 +0.00710；positive 211/222，`WATCH` |
| dynamic | 168 | 0.39978 / 0.65095 | 0.39978 @168 | 0.39840 / 0.39668 / 0.39361 / 0.38395 | latest +0.01753；late20 +0.01621；epoch100 +0.01043；positive 161/168，`PROMISING_EARLY` |
| old-commit ProbeA | 70 | 0.33395 / 0.58247 | 0.33395 @70 | 0.33147 / 0.32825 / 0.31885 / 0.27370 | matched=70，latest +0.01259；late20 +0.01247，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 70 | 0.32273 / 0.56117 | 0.32273 @70 | 0.32114 / 0.31871 / 0.31025 / 0.26538 | - |
| dynamic_singleproj_yoloinit | 60 | 0.30966 / 0.55081 | 0.30966 @60 | 0.30565 / 0.30038 / 0.28689 / 0.22899 | matched=60，latest -0.00148；late20 -0.00162，pre100 |
| dynamic_wo_s_rec_yoloinit | 62 | 0.31621 / 0.55811 | 0.31621 @62 | 0.31206 / 0.30667 / 0.29480 / 0.23859 | matched=62，latest +0.00168；late20 +0.00124，pre100 |
| dynamic_wo_reach_yoloinit | 57 | 0.29809 / 0.54278 | 0.29809 @57 | 0.29327 / 0.28806 / 0.27300 / 0.21082 | matched=57，latest -0.00554；late20 -0.00719，pre100 |

决策：

- 4090 `dynamic` 继续保持强候选：168 rows，latest delta +1.75 point，late20 delta +1.62 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为弱正向 `WATCH`：222 rows，late20 约 +0.53 point，latest 缩到 +0.48 point，未达到主线强度。
- 4090 old-commit ProbeA 到 70 rows，pre100；early delta 仍约 +1.2 point，继续等待 100-row 判断点。
- 3090 三条同源变体仍未到 100 rows；`wo_s_rec` 小幅正向但只有 +0.12 point late20，`singleproj/wo_reach` 偏负。未触发停止或新增。
- 本轮不新增实验、不停止任务；下一轮继续等待 100-row 决策点。

### 2026-06-24 18:10 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 96%`；GPU1 `8597/24564 MiB, util 93%`。4 个训练进程在跑，12 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP + 临时密码文件连接成功；GPU0 `7623/24576 MiB, util 97%`；GPU1 `8417/24576 MiB, util 95%`。4 个训练进程在跑，5 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 289 | 0.44623 / 0.69940 | 0.44623 @289 | 0.44564 / 0.44481 / 0.44278 / 0.43569 | - |
| ProbeA | 223 | 0.41905 / 0.67446 | 0.41905 @223 | 0.41837 / 0.41708 / 0.41420 / 0.40622 | latest +0.00465；late20 +0.00528；epoch100 +0.00710；positive 212/223，`WATCH` |
| dynamic | 169 | 0.39982 / 0.65179 | 0.39982 @169 | 0.39897 / 0.39732 / 0.39421 / 0.38459 | latest +0.01684；late20 +0.01629；epoch100 +0.01043；positive 162/169，`PROMISING_EARLY` |
| old-commit ProbeA | 71 | 0.33589 / 0.58354 | 0.33589 @71 | 0.33296 / 0.32975 / 0.32062 / 0.27682 | matched=71，latest +0.01261；late20 +0.01257，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 72 | 0.32427 / 0.56406 | 0.32427 @72 | 0.32279 / 0.32079 / 0.31323 / 0.27151 | - |
| dynamic_singleproj_yoloinit | 61 | 0.31318 / 0.55320 | 0.31318 @61 | 0.30784 / 0.30273 / 0.28957 / 0.23284 | matched=61，latest +0.00074；late20 -0.00151，pre100 |
| dynamic_wo_s_rec_yoloinit | 63 | 0.31777 / 0.56045 | 0.31777 @63 | 0.31410 / 0.30863 / 0.29745 / 0.24224 | matched=63，latest +0.00035；late20 +0.00133，pre100 |
| dynamic_wo_reach_yoloinit | 58 | 0.30032 / 0.54489 | 0.30032 @58 | 0.29534 / 0.29034 / 0.27589 / 0.21529 | matched=58，latest -0.00514；late20 -0.00693，pre100 |

决策：

- 4090 `dynamic` 继续保持强候选：169 rows，latest delta +1.68 point，late20 delta +1.63 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为弱正向 `WATCH`：223 rows，late20 约 +0.53 point，latest 只有 +0.46 point，未达到主线强度。
- 4090 old-commit ProbeA 到 71 rows，pre100；early delta 仍约 +1.2 point，继续等待 100-row 判断点。
- 3090 三条同源变体仍未到 100 rows；`wo_s_rec` 小幅正向但只有 +0.13 point late20，`singleproj` latest 略正但 late20 仍负，`wo_reach` 持续偏负。未触发停止或新增。
- 本轮不新增实验、不停止任务；下一轮继续等待 100-row 决策点。

### 2026-06-24 18:13 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 87%`；GPU1 `8597/24564 MiB, util 90%`。4 个训练进程在跑，12 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP + 临时密码文件连接成功；GPU0 `7623/24576 MiB, util 98%`；GPU1 `8417/24576 MiB, util 97%`。4 个训练进程在跑，5 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 290 | 0.44669 / 0.70085 | 0.44669 @290 | 0.44598 / 0.44516 / 0.44319 / 0.43615 | - |
| ProbeA | 225 | 0.41994 / 0.67662 | 0.41994 @225 | 0.41907 / 0.41801 / 0.41525 / 0.40726 | latest +0.00455；late20 +0.00516；epoch100 +0.00710；positive 214/225，`WATCH` |
| dynamic | 170 | 0.39995 / 0.65247 | 0.39995 @170 | 0.39940 / 0.39787 / 0.39479 / 0.38522 | latest +0.01613；late20 +0.01632；epoch100 +0.01043；positive 163/170，`PROMISING_EARLY` |
| old-commit ProbeA | 72 | 0.33737 / 0.58441 | 0.33737 @72 | 0.33427 / 0.33109 / 0.32237 / 0.27991 | matched=72，latest +0.01276；late20 +0.01261，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 73 | 0.32614 / 0.56560 | 0.32614 @73 | 0.32375 / 0.32166 / 0.31466 / 0.27436 | - |
| dynamic_singleproj_yoloinit | 62 | 0.31413 / 0.55372 | 0.31413 @62 | 0.31005 / 0.30484 / 0.29234 / 0.23678 | matched=62，latest -0.00040；late20 -0.00122，pre100 |
| dynamic_wo_s_rec_yoloinit | 64 | 0.31932 / 0.56403 | 0.31932 @64 | 0.31604 / 0.31069 / 0.29989 / 0.24603 | matched=64，latest +0.00146；late20 +0.00141，pre100 |
| dynamic_wo_reach_yoloinit | 59 | 0.30302 / 0.54715 | 0.30302 @59 | 0.29814 / 0.29268 / 0.27882 / 0.21953 | matched=59，latest -0.00586；late20 -0.00693，pre100 |

决策：

- 4090 `dynamic` 继续保持强候选：170 rows，latest delta +1.61 point，late20 delta +1.63 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为弱正向 `WATCH`：225 rows，late20 约 +0.52 point，latest 只有 +0.46 point，未达到主线强度。
- 4090 old-commit ProbeA 到 72 rows，pre100；early delta 仍约 +1.2 point，继续等待 100-row 判断点。
- 3090 三条同源变体仍未到 100 rows；`wo_s_rec` 小幅正向但只有 +0.14 point late20，`singleproj/wo_reach` 偏负。未触发停止或新增。
- 本轮不新增实验、不停止任务；下一轮继续等待 100-row 决策点。

### 2026-06-24 18:15 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 95%`；GPU1 `8597/24564 MiB, util 88%`。4 个训练进程在跑，12 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP + 临时密码文件连接成功；GPU0 `7623/24576 MiB, util 93%`；GPU1 `8417/24576 MiB, util 36%`。4 个训练进程仍在，5 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。GPU1 本轮 util 偏低，先观察是否为短时采样波动。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 292 | 0.44732 / 0.70163 | 0.44732 @292 | 0.44662 / 0.44583 / 0.44395 / 0.43706 | - |
| ProbeA | 226 | 0.42060 / 0.67723 | 0.42060 @226 | 0.41951 / 0.41851 / 0.41578 / 0.40779 | latest +0.00431；late20 +0.00511；epoch100 +0.00710；positive 215/226，`WATCH` |
| dynamic | 171 | 0.40067 / 0.65316 | 0.40067 @171 | 0.39983 / 0.39845 / 0.39537 / 0.38585 | latest +0.01628；late20 +0.01636；epoch100 +0.01043；positive 164/171，`PROMISING_EARLY` |
| old-commit ProbeA | 74 | 0.33969 / 0.58770 | 0.33969 @74 | 0.33712 / 0.33369 / 0.32566 / 0.28587 | matched=74，latest +0.01225；late20 +0.01269，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 75 | 0.32824 / 0.56921 | 0.32824 @75 | 0.32581 / 0.32347 / 0.31738 / 0.27993 | - |
| dynamic_singleproj_yoloinit | 63 | 0.31682 / 0.55648 | 0.31682 @63 | 0.31226 / 0.30718 / 0.29504 / 0.24043 | matched=63，latest -0.00060；late20 -0.00108，pre100 |
| dynamic_wo_s_rec_yoloinit | 66 | 0.32169 / 0.56568 | 0.32169 @66 | 0.31910 / 0.31454 / 0.30435 / 0.25337 | matched=66，latest +0.00251；late20 +0.00157，pre100 |
| dynamic_wo_reach_yoloinit | 61 | 0.30723 / 0.55137 | 0.30723 @61 | 0.30274 / 0.29708 / 0.28425 / 0.22723 | matched=61，latest -0.00521；late20 -0.00684，pre100 |

决策：

- 4090 `dynamic` 继续保持当前最强候选：171 rows，latest delta +1.63 point，late20 delta +1.64 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为弱正向 `WATCH`：226 rows，late20 约 +0.51 point，未达到主线强度。
- 4090 old-commit ProbeA 到 74 rows，pre100；early delta 仍约 +1.2 point，继续等待 100-row 判断点。
- 3090 三条同源变体仍未到 100 rows；`wo_s_rec` 小幅正向但只有 +0.16 point late20，`singleproj/wo_reach` 偏负。未触发停止或新增。
- 本轮不新增实验、不停止任务；下一轮重点看 old-commit ProbeA 和 3090 变体是否跨过 100-row 决策点。

### 2026-06-24 18:21 CST

补充动作：

- 新增并同步轻量解析脚本 `docs/experiments/monitor_ogsod_yoloinit_status_20260624.py`，后续用统一口径解析 rows、latest/best、late windows、同 epoch delta、positive epoch count 和 100/120 epoch 状态。

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 90%`；GPU1 `8597/24564 MiB, util 94%`。4 个训练组仍在跑，12 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP + 临时密码文件连接成功；GPU0 `7623/24576 MiB, util 97%`；GPU1 `8417/24576 MiB, util 94%`。4 个训练组仍在跑，5 个相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 296 | 0.44900 / 0.70333 | 0.44900 @296 | 0.44808 / 0.44723 / 0.44546 / 0.43882 | - |
| ProbeA | 229 | 0.42201 / 0.67874 | 0.42201 @229 | 0.42111 / 0.41992 / 0.41740 / 0.40936 | latest +0.00499；late20 +0.00507；epoch100 +0.00710；positive 218/229，`WATCH` |
| dynamic | 175 | 0.40247 / 0.65522 | 0.40247 @175 | 0.40158 / 0.40049 / 0.39762 / 0.38834 | latest +0.01609；late20 +0.01645；epoch100 +0.01043；positive 168/175，`PROMISING_EARLY` |
| old-commit ProbeA | 77 | 0.34238 / 0.58906 | 0.34238 @77 | 0.34057 / 0.33742 / 0.33036 / 0.29417 | matched=77，latest +0.01339；late20 +0.01303，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 79 | 0.33161 / 0.57165 | 0.33161 @79 | 0.33013 / 0.32742 / 0.32249 / 0.29020 | - |
| dynamic_singleproj_yoloinit | 67 | 0.32228 / 0.56350 | 0.32228 @67 | 0.31906 / 0.31456 / 0.30411 / 0.25495 | matched=67，latest +0.00203；late20 -0.00074，pre100 |
| dynamic_wo_s_rec_yoloinit | 69 | 0.32539 / 0.57136 | 0.32539 @69 | 0.32293 / 0.31948 / 0.31001 / 0.26327 | matched=69，latest +0.00321；late20 +0.00149，pre100 |
| dynamic_wo_reach_yoloinit | 64 | 0.31282 / 0.56128 | 0.31282 @64 | 0.30880 / 0.30347 / 0.29200 / 0.23842 | matched=64，latest -0.00504；late20 -0.00648，pre100 |

决策：

- 4090 `dynamic` 仍是唯一满足早筛强正向的候选：175 rows，latest delta +1.61 point，late20 delta +1.65 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为弱正向 `WATCH`：229 rows，late20 约 +0.51 point，强度不足，不作为可靠主线结论。
- 4090 old-commit ProbeA 到 77 rows，pre100；早期 delta 约 +1.3 point，继续等待 100-row 决策点。
- 3090 三条同源变体仍未到 100 rows；`wo_s_rec` latest/late20 小幅正向，`singleproj` latest 转正但 late20 仍略负，`wo_reach` 持续偏负。暂不停止，等 100/120-row 规则再判断。
- 本轮不新增实验、不停止任务；下一轮继续等 old-commit ProbeA 与 3090 变体接近 100 rows。

### 2026-06-24 18:25 CST

补充动作：

- 更新 `docs/experiments/monitor_ogsod_yoloinit_status_20260624.py`，将进程输出压缩为 PID/状态/运行时/CPU/显存/脚本/设备/run name，避免长命令行淹没结果段；本地 `py_compile` 通过。

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 96%`；GPU1 `8597/24564 MiB, util 95%`。相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct 入口先被关闭，SOCKS fallback 成功；GPU0 `7623/24576 MiB, util 96%`；GPU1 `8417/24576 MiB, util 94%`。相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 298 | 0.45031 / 0.70441 | 0.45031 @298 | 0.44914 / 0.44805 / 0.44624 / 0.43972 | - |
| ProbeA | 231 | 0.42332 / 0.68019 | 0.42332 @231 | 0.42217 / 0.42084 / 0.41845 / 0.41039 | latest +0.00498；late20 +0.00506；epoch100 +0.00710；positive 220/231，`WATCH` |
| dynamic | 177 | 0.40407 / 0.65595 | 0.40407 @177 | 0.40263 / 0.40146 / 0.39874 / 0.38956 | latest +0.01597；late20 +0.01643；epoch100 +0.01043；positive 170/177，`PROMISING_EARLY` |
| old-commit ProbeA | 78 | 0.34323 / 0.58906 | 0.34323 @78 | 0.34148 / 0.33857 / 0.33188 / 0.29676 | matched=78，latest +0.01363；late20 +0.01318，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 82 | 0.33444 / 0.57468 | 0.33444 @82 | 0.33258 / 0.33040 / 0.32559 / 0.29695 | - |
| dynamic_singleproj_yoloinit | 69 | 0.32356 / 0.56443 | 0.32356 @69 | 0.32155 / 0.31791 / 0.30797 / 0.26182 | matched=69，latest +0.00138；late20 -0.00054，pre100 |
| dynamic_wo_s_rec_yoloinit | 71 | 0.32786 / 0.57314 | 0.32786 @71 | 0.32538 / 0.32224 / 0.31344 / 0.26954 | matched=71，latest +0.00444；late20 +0.00166，pre100 |
| dynamic_wo_reach_yoloinit | 66 | 0.31623 / 0.56372 | 0.31623 @66 | 0.31255 / 0.30765 / 0.29675 / 0.24577 | matched=66，latest -0.00295；late20 -0.00603，pre100 |

决策：

- 4090 `dynamic` 继续是唯一强早筛候选：177 rows，latest delta +1.60 point，late20 delta +1.64 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为弱正向 `WATCH`：231 rows，late20 约 +0.51 point，不够主线强度。
- 4090 old-commit ProbeA 到 78 rows，pre100；早期 delta 约 +1.3 point，继续等待 100-row 决策点。
- 3090 三条同源变体仍未到 100 rows；`wo_s_rec` latest 正到 +0.44 point 但 late20 只有 +0.17 point，`singleproj` late20 略负，`wo_reach` 仍偏负。暂不停止，等 100/120-row 规则。
- 本轮不新增实验、不停止任务；下一轮继续等 old-commit ProbeA 与 3090 变体接近 100 rows。

### 2026-06-24 18:28 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 89%`；GPU1 `8597/24564 MiB, util 94%`。相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：SOCKS fallback 连接成功；GPU0 `7623/24576 MiB, util 94%`；GPU1 `8417/24576 MiB, util 88%`。相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 300 | 0.45103 / 0.70522 | 0.45103 @300 | 0.45017 / 0.44893 / 0.44705 / 0.44060 | - |
| ProbeA | 233 | 0.42401 / 0.68123 | 0.42401 @233 | 0.42312 / 0.42184 / 0.41946 / 0.41142 | latest +0.00496；late20 +0.00502；epoch100 +0.00710；positive 222/233，`WATCH` |
| dynamic | 178 | 0.40438 / 0.65700 | 0.40438 @178 | 0.40320 / 0.40193 / 0.39930 / 0.39017 | latest +0.01610；late20 +0.01643；epoch100 +0.01043；positive 171/178，`PROMISING_EARLY` |
| old-commit ProbeA | 80 | 0.34444 / 0.59315 | 0.34444 @80 | 0.34314 / 0.34080 / 0.33452 / 0.30179 | matched=80，latest +0.01290；late20 +0.01313，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 83 | 0.33519 / 0.57566 | 0.33519 @83 | 0.33340 / 0.33130 / 0.32648 / 0.29915 | - |
| dynamic_singleproj_yoloinit | 70 | 0.32429 / 0.56644 | 0.32429 @70 | 0.32268 / 0.31938 / 0.30988 / 0.26497 | matched=70，latest +0.00156；late20 -0.00037，pre100 |
| dynamic_wo_s_rec_yoloinit | 73 | 0.33093 / 0.57733 | 0.33093 @73 | 0.32807 / 0.32489 / 0.31676 / 0.27560 | matched=73，latest +0.00479；late20 +0.00210，pre100 |
| dynamic_wo_reach_yoloinit | 67 | 0.31792 / 0.56617 | 0.31792 @67 | 0.31441 / 0.30963 / 0.29884 / 0.24914 | matched=67，latest -0.00233；late20 -0.00601，pre100 |

决策：

- 4090 `dynamic` 继续保持强早筛：178 rows，latest delta +1.61 point，late20 delta +1.64 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为弱正向 `WATCH`：233 rows，late20 约 +0.50 point，不够主线强度。
- 4090 old-commit ProbeA 到 80 rows，pre100；早期 delta 约 +1.3 point，继续等待 100-row 决策点。
- 3090 三条同源变体仍未到 100 rows；`wo_s_rec` 目前最好但 late20 仅 +0.21 point，`singleproj` 接近 0，`wo_reach` 偏负。暂不停止，等 100/120-row 规则。
- 本轮不新增实验、不停止任务；下一轮继续等 old-commit ProbeA 与 3090 变体接近 100 rows。

### 2026-06-24 18:31 CST

补充动作：

- 更新 `docs/experiments/monitor_ogsod_yoloinit_status_20260624.py`，将 `PROCS` 从 worker 列表改为按 run name 聚合，输出每组 procs、pids、device、cpu_total、mem_total 和状态计数；本地 `py_compile` 通过，远端 4090/3090 均可执行。

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 93%`；GPU1 `8597/24564 MiB, util 89%`。4 个训练组均在：det-only、ProbeA、dynamic、old-commit ProbeA；相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP 连接成功；GPU0 `7623/24576 MiB, util 99%`；GPU1 `8417/24576 MiB, util 47%`。4 个训练组均在：detonly_control、singleproj、wo_s_rec、wo_reach；GPU1 本轮 util 采样偏低但进程组存在且日志干净，继续观察。相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 303 | 0.45194 / 0.70719 | 0.45202 @302 | 0.45141 / 0.45027 / 0.44822 / 0.44192 | - |
| ProbeA | 235 | 0.42467 / 0.68199 | 0.42467 @235 | 0.42400 / 0.42282 / 0.42041 / 0.41244 | latest +0.00425；late20 +0.00493；epoch100 +0.00710；positive 224/235，`WATCH` |
| dynamic | 180 | 0.40561 / 0.65769 | 0.40561 @180 | 0.40448 / 0.40303 / 0.40045 / 0.39140 | latest +0.01606；late20 +0.01645；epoch100 +0.01043；positive 173/180，`PROMISING_EARLY` |
| old-commit ProbeA | 82 | 0.34566 / 0.59345 | 0.34566 @82 | 0.34454 / 0.34255 / 0.33682 / 0.30655 | matched=82，latest +0.01270；late20 +0.01303，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 86 | 0.33661 / 0.57723 | 0.33661 @86 | 0.33565 / 0.33370 / 0.32910 / 0.30516 | - |
| dynamic_singleproj_yoloinit | 72 | 0.32677 / 0.56886 | 0.32677 @72 | 0.32470 / 0.32188 / 0.31336 / 0.27093 | matched=72，latest +0.00250；late20 +0.00014，pre100 |
| dynamic_wo_s_rec_yoloinit | 74 | 0.33243 / 0.57827 | 0.33243 @74 | 0.32947 / 0.32620 / 0.31845 / 0.27867 | matched=74，latest +0.00545；late20 +0.00242，pre100 |
| dynamic_wo_reach_yoloinit | 69 | 0.31960 / 0.56553 | 0.31960 @69 | 0.31749 / 0.31314 / 0.30291 / 0.25602 | matched=69，latest -0.00258；late20 -0.00560，pre100 |

决策：

- 4090 `dynamic` 继续保持强早筛：180 rows，latest delta +1.61 point，late20 delta +1.65 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为弱正向 `WATCH`：235 rows，late20 约 +0.49 point，且 latest delta 缩到 +0.43 point，不够主线强度。
- 4090 old-commit ProbeA 到 82 rows，pre100；早期 delta 约 +1.3 point，继续等待 100-row 决策点。
- 3090 三条同源变体仍未到 100 rows；`wo_s_rec` 目前最好但 late20 仅 +0.24 point，`singleproj` 接近 0，`wo_reach` 偏负。暂不停止，等 100/120-row 规则。
- 本轮不新增实验、不停止任务；下一轮继续等 old-commit ProbeA 与 3090 变体接近 100 rows。

### 2026-06-24 18:34 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 96%`；GPU1 `8597/24564 MiB, util 91%`。4 个训练组均在：det-only、ProbeA、dynamic、old-commit ProbeA；相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP 连接成功；GPU0 `7623/24576 MiB, util 90%`；GPU1 `8417/24576 MiB, util 98%`。4 个训练组均在：detonly_control、singleproj、wo_s_rec、wo_reach；相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 305 | 0.45246 / 0.70740 | 0.45246 @305 | 0.45199 / 0.45108 / 0.44896 / 0.44278 | - |
| ProbeA | 237 | 0.42536 / 0.68299 | 0.42536 @237 | 0.42468 / 0.42365 / 0.42133 / 0.41345 | latest +0.00330；late20 +0.00475；epoch100 +0.00710；positive 226/237，`WATCH` |
| dynamic | 182 | 0.40676 / 0.66019 | 0.40676 @182 | 0.40556 / 0.40409 / 0.40156 / 0.39261 | latest +0.01576；late20 +0.01639；epoch100 +0.01043；positive 175/182，`PROMISING_EARLY` |
| old-commit ProbeA | 83 | 0.34599 / 0.59547 | 0.34599 @83 | 0.34509 / 0.34329 / 0.33784 / 0.30874 | matched=83，latest +0.01273；late20 +0.01297，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 87 | 0.33772 / 0.57860 | 0.33772 @87 | 0.33631 / 0.33445 / 0.32997 / 0.30699 | - |
| dynamic_singleproj_yoloinit | 73 | 0.32732 / 0.57009 | 0.32732 @73 | 0.32551 / 0.32293 / 0.31506 / 0.27389 | matched=73，latest +0.00118；late20 +0.00039，pre100 |
| dynamic_wo_s_rec_yoloinit | 76 | 0.33400 / 0.57982 | 0.33400 @76 | 0.33201 / 0.32870 / 0.32162 / 0.28422 | matched=76，latest +0.00459；late20 +0.00288，pre100 |
| dynamic_wo_reach_yoloinit | 71 | 0.32172 / 0.56813 | 0.32172 @71 | 0.31986 / 0.31620 / 0.30664 / 0.26239 | matched=71，latest -0.00170；late20 -0.00514，pre100 |

决策：

- 4090 `dynamic` 继续保持强早筛：182 rows，latest delta +1.58 point，late20 delta +1.64 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为弱正向 `WATCH`：237 rows，late20 约 +0.48 point，latest delta 缩到 +0.33 point，不够主线强度。
- 4090 old-commit ProbeA 到 83 rows，pre100；早期 delta 仍约 +1.3 point，继续等待 100-row 决策点。
- 3090 三条同源变体仍未到 100 rows；`wo_s_rec` 目前最好但 late20 仅 +0.29 point，`singleproj` 接近 0，`wo_reach` 偏负。暂不停止，等 100/120-row 规则。
- 本轮不新增实验、不停止任务；下一轮继续等 old-commit ProbeA 与 3090 变体接近 100 rows。

### 2026-06-24 18:36 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 91%`；GPU1 `8597/24564 MiB, util 95%`。4 个训练组均在：det-only、ProbeA、dynamic、old-commit ProbeA；相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP 连接成功；GPU0 `7623/24576 MiB, util 88%`；GPU1 `8417/24576 MiB, util 98%`。4 个训练组均在：detonly_control、singleproj、wo_s_rec、wo_reach；相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 306 | 0.45346 / 0.70757 | 0.45346 @306 | 0.45242 / 0.45153 / 0.44938 / 0.44321 | - |
| ProbeA | 238 | 0.42600 / 0.68358 | 0.42600 @238 | 0.42507 / 0.42410 / 0.42178 / 0.41395 | latest +0.00329；late20 +0.00464；epoch100 +0.00710；positive 227/238，`WATCH` |
| dynamic | 183 | 0.40710 / 0.66053 | 0.40710 @183 | 0.40610 / 0.40465 / 0.40209 / 0.39321 | latest +0.01533；late20 +0.01631；epoch100 +0.01043；positive 176/183，`PROMISING_EARLY` |
| old-commit ProbeA | 85 | 0.34743 / 0.59751 | 0.34743 @85 | 0.34623 / 0.34468 / 0.33982 / 0.31296 | matched=85，latest +0.01281；late20 +0.01288，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 89 | 0.33942 / 0.57991 | 0.33942 @89 | 0.33778 / 0.33599 / 0.33170 / 0.31049 | - |
| dynamic_singleproj_yoloinit | 74 | 0.32805 / 0.57144 | 0.32805 @74 | 0.32641 / 0.32398 / 0.31651 / 0.27676 | matched=74，latest +0.00107；late20 +0.00049，pre100 |
| dynamic_wo_s_rec_yoloinit | 77 | 0.33467 / 0.58056 | 0.33467 @77 | 0.33304 / 0.32987 / 0.32306 / 0.28696 | matched=77，latest +0.00439；late20 +0.00299，pre100 |
| dynamic_wo_reach_yoloinit | 72 | 0.32255 / 0.56899 | 0.32255 @72 | 0.32078 / 0.31760 / 0.30833 / 0.26546 | matched=72，latest -0.00172；late20 -0.00490，pre100 |

决策：

- 4090 `dynamic` 继续保持强早筛：183 rows，latest delta +1.53 point，late20 delta +1.63 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 仍为弱正向 `WATCH`：238 rows，late20 约 +0.46 point，latest delta 缩到 +0.33 point，不够主线强度。
- 4090 old-commit ProbeA 到 85 rows，pre100；早期 delta 仍约 +1.3 point，继续等待 100-row 决策点。
- 3090 三条同源变体仍未到 100 rows；`wo_s_rec` 目前最好但 late20 仅 +0.30 point，`singleproj` 接近 0，`wo_reach` 偏负。暂不停止，等 100/120-row 规则。
- 本轮不新增实验、不停止任务；下一轮继续等 old-commit ProbeA 与 3090 变体接近 100 rows。

### 2026-06-24 18:44 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 93%`；GPU1 `8597/24564 MiB, util 88%`。4 个训练组均在：det-only、ProbeA、dynamic、old-commit ProbeA；相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP 首次连接被远端关闭，随后 SOCKS fallback 成功；GPU0 `7623/24576 MiB, util 91%`；GPU1 `8417/24576 MiB, util 93%`。4 个训练组均在：detonly_control、singleproj、wo_s_rec、wo_reach；相关日志文件扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 312 | 0.45594 / 0.71177 | 0.45594 @312 | 0.45500 / 0.45390 / 0.45186 / 0.44579 | - |
| ProbeA | 242 | 0.42859 / 0.68770 | 0.42859 @242 | 0.42722 / 0.42595 / 0.42365 / 0.41601 | latest +0.00369；late20 +0.00430；epoch100 +0.00710；positive 231/242，`WATCH` |
| dynamic | 187 | 0.40883 / 0.66312 | 0.40883 @187 | 0.40798 / 0.40677 / 0.40412 / 0.39555 | latest +0.01450；late20 +0.01594；epoch100 +0.01043；positive 180/187，`PROMISING_EARLY` |
| old-commit ProbeA | 88 | 0.34910 / 0.59707 | 0.34910 @88 | 0.34802 / 0.34656 / 0.34256 / 0.31870 | matched=88，latest +0.01193；late20 +0.01283，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 94 | 0.34307 / 0.58437 | 0.34307 @94 | 0.34184 / 0.33981 / 0.33599 / 0.31830 | - |
| dynamic_singleproj_yoloinit | 78 | 0.33172 / 0.57500 | 0.33172 @78 | 0.32978 / 0.32765 / 0.32198 / 0.28749 | matched=78，latest +0.00060；late20 +0.00063，pre100 |
| dynamic_wo_s_rec_yoloinit | 81 | 0.33802 / 0.58472 | 0.33802 @81 | 0.33615 / 0.33408 / 0.32816 / 0.29685 | matched=81，latest +0.00452；late20 +0.00357，pre100 |
| dynamic_wo_reach_yoloinit | 76 | 0.32699 / 0.57445 | 0.32699 @76 | 0.32510 / 0.32248 / 0.31506 / 0.27735 | matched=76，latest -0.00242；late20 -0.00367，pre100 |

决策：

- 4090 `dynamic` 仍是当前最可信主线候选：187 rows，latest delta +1.45 point，late20 delta +1.59 point，epoch100 delta +1.04 point；继续跑满 800。
- 4090 `ProbeA` 继续弱正向 `WATCH`：242 rows，late20 约 +0.43 point，latest 约 +0.37 point，不够主线强度。
- 4090 old-commit ProbeA 到 88 rows，距离 100-row 触发点较近；早期 latest/late20 仍约 +1.2 到 +1.3 point，但暂不定性。
- 3090 同源组仍未到 100 rows；`wo_s_rec` 最好但只有 +0.36 point late20，`singleproj` 基本贴住 det-only，`wo_reach` 当前为负。继续等 100/120-row 规则，不提前停止。
- 本轮没有 OOM、fallback、NaN、错误落卡或目录混淆；不新增实验、不停止任务。

### 2026-06-24 18:46 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 90%`；GPU1 `8597/24564 MiB, util 80%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：SOCKS fallback 连接成功；GPU0 `7623/24576 MiB, util 85%`；GPU1 `8417/24576 MiB, util 96%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 313 | 0.45602 / 0.71170 | 0.45602 @313 | 0.45536 / 0.45431 / 0.45229 / 0.44623 | - |
| ProbeA | 244 | 0.42908 / 0.68823 | 0.42908 @244 | 0.42824 / 0.42688 / 0.42461 / 0.41702 | latest +0.00293；late20 +0.00413；epoch100 +0.00710；positive 233/244，`WATCH` |
| dynamic | 188 | 0.40929 / 0.66316 | 0.40929 @188 | 0.40842 / 0.40726 / 0.40459 / 0.39612 | latest +0.01441；late20 +0.01579；epoch100 +0.01043；positive 181/188，`PROMISING_EARLY` |
| old-commit ProbeA | 90 | 0.35049 / 0.59834 | 0.35049 @90 | 0.34928 / 0.34776 / 0.34428 / 0.32214 | matched=90，latest +0.01234；late20 +0.01280，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 96 | 0.34474 / 0.58580 | 0.34474 @96 | 0.34324 / 0.34136 / 0.33753 / 0.32102 | - |
| dynamic_singleproj_yoloinit | 79 | 0.33251 / 0.57614 | 0.33251 @79 | 0.33067 / 0.32854 / 0.32323 / 0.28993 | matched=79，latest +0.00090；late20 +0.00074，pre100 |
| dynamic_wo_s_rec_yoloinit | 82 | 0.33943 / 0.58526 | 0.33943 @82 | 0.33710 / 0.33507 / 0.32932 / 0.29919 | matched=82，latest +0.00499；late20 +0.00373，pre100 |
| dynamic_wo_reach_yoloinit | 77 | 0.32878 / 0.57506 | 0.32878 @77 | 0.32635 / 0.32357 / 0.31660 / 0.28017 | matched=77，latest -0.00150；late20 -0.00347，pre100 |

决策：

- 没有新 run 达到 100/120-row 触发点；本轮不启动、不停止。
- 4090 `dynamic` 继续保持 `PROMISING_EARLY`，仍是当前最有主线价值的候选。
- 4090 `ProbeA` 继续弱正向 `WATCH`；old-commit ProbeA 到 90 rows，继续等 100-row 正式评估。
- 3090 三条变体仍是 pre100；`wo_s_rec` 仍最好但增益偏小，`wo_reach` 仍偏负。继续等 100/120-row 规则。

### 2026-06-24 18:49 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 93%`；GPU1 `8597/24564 MiB, util 92%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：SOCKS fallback 连接成功；GPU0 `7623/24576 MiB, util 86%`；GPU1 `8417/24576 MiB, util 94%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 315 | 0.45668 / 0.71259 | 0.45668 @315 | 0.45616 / 0.45517 / 0.45312 / 0.44706 | - |
| ProbeA | 245 | 0.42961 / 0.68781 | 0.42961 @245 | 0.42877 / 0.42737 / 0.42509 / 0.41753 | latest +0.00348；late20 +0.00408；epoch100 +0.00710；positive 234/245，`WATCH` |
| dynamic | 190 | 0.40995 / 0.66442 | 0.40995 @190 | 0.40918 / 0.40812 / 0.40558 / 0.39725 | latest +0.01346；late20 +0.01550；epoch100 +0.01043；positive 183/190，`PROMISING_EARLY` |
| old-commit ProbeA | 91 | 0.35083 / 0.59938 | 0.35083 @91 | 0.34985 / 0.34832 / 0.34502 / 0.32385 | matched=91，latest +0.01225；late20 +0.01278，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 97 | 0.34559 / 0.58590 | 0.34559 @97 | 0.34397 / 0.34215 / 0.33830 / 0.32236 | - |
| dynamic_singleproj_yoloinit | 81 | 0.33429 / 0.57794 | 0.33429 @81 | 0.33248 / 0.33029 / 0.32545 / 0.29456 | matched=81，latest +0.00079；late20 +0.00086，pre100 |
| dynamic_wo_s_rec_yoloinit | 84 | 0.34121 / 0.58623 | 0.34121 @84 | 0.33916 / 0.33687 / 0.33153 / 0.30350 | matched=84，latest +0.00562；late20 +0.00417，pre100 |
| dynamic_wo_reach_yoloinit | 79 | 0.33028 / 0.57633 | 0.33028 @79 | 0.32842 / 0.32568 / 0.31941 / 0.28517 | matched=79，latest -0.00133；late20 -0.00308，pre100 |

决策：

- 没有候选新达到 100/120-row 触发点；本轮不启动下一批，也不停止任务。
- 4090 `dynamic` 继续是唯一明确 `PROMISING_EARLY` 的主线候选；虽然 latest delta 从峰值略回落，但 late20 仍约 +1.55 AP50-95 point。
- 4090 `ProbeA` 继续弱正向 `WATCH`；old-commit ProbeA 到 91 rows，继续等 100-row 正式评估。
- 3090 `detonly_control` 到 97 rows，已接近 100；三条候选仍 pre100。`wo_s_rec` 小正，`singleproj` 近零，`wo_reach` 偏负但未到正式降级点。

### 2026-06-24 18:51 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 34%`；GPU1 `8597/24564 MiB, util 88%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。GPU0 低利用率视为瞬时采样：det-only 与 ProbeA 两组进程仍在，不据此加任务。
- `ladd3090-zw1`：SOCKS fallback 连接成功；GPU0 `7623/24576 MiB, util 95%`；GPU1 `8417/24576 MiB, util 98%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 317 | 0.45717 / 0.71329 | 0.45717 @317 | 0.45668 / 0.45584 / 0.45389 / 0.44786 | - |
| ProbeA | 247 | 0.43037 / 0.68864 | 0.43037 @247 | 0.42952 / 0.42837 / 0.42601 / 0.41855 | latest +0.00309；late20 +0.00393；epoch100 +0.00710；positive 236/247，`WATCH` |
| dynamic | 191 | 0.41047 / 0.66465 | 0.41047 @191 | 0.40962 / 0.40859 / 0.40607 / 0.39781 | latest +0.01347；late20 +0.01536；epoch100 +0.01043；positive 184/191，`PROMISING_EARLY` |
| old-commit ProbeA | 92 | 0.35134 / 0.60001 | 0.35134 @92 | 0.35038 / 0.34889 / 0.34572 / 0.32546 | matched=92，latest +0.01240；late20 +0.01276，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 99 | 0.34666 / 0.58621 | 0.34666 @99 | 0.34541 / 0.34362 / 0.33981 / 0.32481 | - |
| dynamic_singleproj_yoloinit | 82 | 0.33489 / 0.57768 | 0.33489 @82 | 0.33329 / 0.33110 / 0.32649 / 0.29682 | matched=82，latest +0.00045；late20 +0.00090，pre100 |
| dynamic_wo_s_rec_yoloinit | 85 | 0.34188 / 0.58736 | 0.34188 @85 | 0.34013 / 0.33774 / 0.33260 / 0.30552 | matched=85，latest +0.00544；late20 +0.00437，pre100 |
| dynamic_wo_reach_yoloinit | 80 | 0.33076 / 0.57718 | 0.33076 @80 | 0.32922 / 0.32664 / 0.32069 / 0.28753 | matched=80，latest -0.00149；late20 -0.00285，pre100 |

决策：

- 没有候选新达到 100/120-row 触发点；本轮不启动下一批，也不停止任务。
- 4090 `dynamic` 继续是唯一明确 `PROMISING_EARLY` 的主线候选，late20 仍约 +1.54 AP50-95 point。
- 4090 `ProbeA` 弱正继续收窄，保持 `WATCH`；old-commit ProbeA 到 92 rows，继续等 100-row 正式评估。
- 3090 `detonly_control` 到 99 rows，下一轮应可提供 100-row control；三条候选仍 pre100。`wo_s_rec` 小正，`singleproj` 近零，`wo_reach` 偏负但未到正式降级点。

### 2026-06-24 18:54 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 93%`；GPU1 `8597/24564 MiB, util 90%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：SOCKS fallback 连接成功；GPU0 `7623/24576 MiB, util 93%`；GPU1 `8417/24576 MiB, util 99%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 318 | 0.45739 / 0.71428 | 0.45739 @318 | 0.45695 / 0.45616 / 0.45425 / 0.44826 | - |
| ProbeA | 248 | 0.43074 / 0.68914 | 0.43074 @248 | 0.42996 / 0.42884 / 0.42647 / 0.41905 | latest +0.00293；late20 +0.00385；epoch100 +0.00710；positive 237/248，`WATCH` |
| dynamic | 192 | 0.41103 / 0.66547 | 0.41103 @192 | 0.41006 / 0.40902 / 0.40656 / 0.39836 | latest +0.01383；late20 +0.01522；epoch100 +0.01043；positive 185/192，`PROMISING_EARLY` |
| old-commit ProbeA | 93 | 0.35210 / 0.60092 | 0.35210 @93 | 0.35098 / 0.34950 / 0.34639 / 0.32701 | matched=93，latest +0.01223；late20 +0.01272，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 100 | 0.34708 / 0.58703 | 0.34708 @100 | 0.34606 / 0.34431 / 0.34055 / 0.32599 | control 已到 100 |
| dynamic_singleproj_yoloinit | 83 | 0.33597 / 0.57817 | 0.33597 @83 | 0.33414 / 0.33196 / 0.32745 / 0.29899 | matched=83，latest +0.00078；late20 +0.00097，pre100 |
| dynamic_wo_s_rec_yoloinit | 86 | 0.34235 / 0.58887 | 0.34235 @86 | 0.34100 / 0.33857 / 0.33363 / 0.30757 | matched=86，latest +0.00574；late20 +0.00453，pre100 |
| dynamic_wo_reach_yoloinit | 81 | 0.33139 / 0.57882 | 0.33139 @81 | 0.33010 / 0.32760 / 0.32190 / 0.28987 | matched=81，latest -0.00211；late20 -0.00269，pre100 |

决策：

- 3090 `detonly_control` 已达到 100 rows，后续三条 3090 候选到 100 后可以直接用同源 control 做正式轻量评估。
- 当前没有候选新达到 100/120-row 触发点；本轮不启动下一批，也不停止任务。
- 4090 `dynamic` 继续是唯一明确 `PROMISING_EARLY` 的主线候选，late20 仍约 +1.52 AP50-95 point。
- 4090 `ProbeA` 弱正继续收窄，保持 `WATCH`；old-commit ProbeA 到 93 rows，继续等 100-row 正式评估。
- 3090 三条候选仍 pre100：`wo_s_rec` 小正，`singleproj` 近零，`wo_reach` 偏负但未到正式降级点。

### 2026-06-24 18:57 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 92%`；GPU1 `8597/24564 MiB, util 95%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：SOCKS fallback 连接成功；GPU0 `7623/24576 MiB, util 98%`；GPU1 `8417/24576 MiB, util 94%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 320 | 0.45853 / 0.71513 | 0.45853 @320 | 0.45752 / 0.45684 / 0.45496 / 0.44905 | - |
| ProbeA | 249 | 0.43151 / 0.69015 | 0.43151 @249 | 0.43045 / 0.42935 / 0.42694 / 0.41956 | latest +0.00299；late20 +0.00375；epoch100 +0.00710；positive 238/249，`WATCH` |
| dynamic | 194 | 0.41192 / 0.66638 | 0.41192 @194 | 0.41092 / 0.40987 / 0.40754 / 0.39945 | latest +0.01342；late20 +0.01497；epoch100 +0.01043；positive 187/194，`PROMISING_EARLY` |
| old-commit ProbeA | 95 | 0.35352 / 0.60250 | 0.35352 @95 | 0.35211 / 0.35070 / 0.34769 / 0.32983 | matched=95，latest +0.01216；late20 +0.01268，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 102 | 0.34829 / 0.58893 | 0.34829 @102 | 0.34712 / 0.34555 / 0.34193 / 0.32814 | control 已过 100 |
| dynamic_singleproj_yoloinit | 85 | 0.33736 / 0.58015 | 0.33736 @85 | 0.33586 / 0.33371 / 0.32934 / 0.30299 | matched=85，latest +0.00092；late20 +0.00112，pre100 |
| dynamic_wo_s_rec_yoloinit | 88 | 0.34360 / 0.58858 | 0.34360 @88 | 0.34238 / 0.34022 / 0.33560 / 0.31137 | matched=88，latest +0.00487；late20 +0.00476，pre100 |
| dynamic_wo_reach_yoloinit | 83 | 0.33322 / 0.57988 | 0.33322 @83 | 0.33152 / 0.32950 / 0.32422 / 0.29440 | matched=83，latest -0.00197；late20 -0.00226，pre100 |

决策：

- 当前没有候选新达到 100/120-row 触发点；本轮不启动下一批，也不停止任务。
- 4090 `dynamic` 继续是唯一明确 `PROMISING_EARLY` 的主线候选，late20 仍约 +1.50 AP50-95 point。
- 4090 `ProbeA` 继续弱正 `WATCH`，latest/late20 增益仍在收窄；old-commit ProbeA 到 95 rows，继续等 100-row 正式评估。
- 3090 `detonly_control` 已过 100 rows，可作为后续同源参照；三条候选仍 pre100。`wo_s_rec` 小正但不足 +1 point，`singleproj` 近零，`wo_reach` 偏负但未到正式降级点。

### 2026-06-24 19:00 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 91%`；GPU1 `8597/24564 MiB, util 89%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：首次 SOCKS 连接超时，随后 direct IP 连接成功；GPU0 `7623/24576 MiB, util 94%`；GPU1 `8417/24576 MiB, util 96%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 322 | 0.45881 / 0.71502 | 0.45881 @321 | 0.45821 / 0.45745 / 0.45567 / 0.44982 | - |
| ProbeA | 251 | 0.43192 / 0.69171 | 0.43192 @251 | 0.43119 / 0.43018 / 0.42782 / 0.42055 | latest +0.00227；late20 +0.00345；epoch100 +0.00710；positive 240/251，`WATCH` |
| dynamic | 195 | 0.41232 / 0.66780 | 0.41232 @195 | 0.41139 / 0.41029 / 0.40803 / 0.39999 | latest +0.01287；late20 +0.01481；epoch100 +0.01043；positive 188/195，`PROMISING_EARLY` |
| old-commit ProbeA | 96 | 0.35412 / 0.60316 | 0.35412 @96 | 0.35277 / 0.35131 / 0.34832 / 0.33117 | matched=96，latest +0.01192；late20 +0.01260，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 104 | 0.34965 / 0.59035 | 0.34965 @104 | 0.34823 / 0.34682 / 0.34331 / 0.33017 | control 已过 100 |
| dynamic_singleproj_yoloinit | 86 | 0.33826 / 0.58111 | 0.33826 @86 | 0.33665 / 0.33456 / 0.33026 / 0.30489 | matched=86，latest +0.00165；late20 +0.00116，pre100 |
| dynamic_wo_s_rec_yoloinit | 89 | 0.34417 / 0.58965 | 0.34417 @89 | 0.34297 / 0.34107 / 0.33654 / 0.31323 | matched=89，latest +0.00475；late20 +0.00484，pre100 |
| dynamic_wo_reach_yoloinit | 84 | 0.33442 / 0.58080 | 0.33442 @84 | 0.33235 / 0.33039 / 0.32530 / 0.29647 | matched=84，latest -0.00117；late20 -0.00207，pre100 |

决策：

- 当前没有候选新达到 100/120-row 触发点；本轮不启动下一批，也不停止任务。
- 4090 `dynamic` 继续是唯一明确 `PROMISING_EARLY` 的主线候选，late20 仍约 +1.48 AP50-95 point，但正增益幅度相对 180-row 附近略收窄。
- 4090 `ProbeA` 继续弱正 `WATCH`；old-commit ProbeA 到 96 rows，继续等 100-row 正式评估。
- 3090 `detonly_control` 已到 104 rows，可作为后续同源参照；三条候选仍 pre100。`wo_s_rec` 小正但不足 +1 point，`singleproj` 近零，`wo_reach` 偏负但未到正式降级点。

### 2026-06-24 19:03 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 92%`；GPU1 `8597/24564 MiB, util 89%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP 连接成功；GPU0 `7623/24576 MiB, util 91%`；GPU1 `8417/24576 MiB, util 95%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 324 | 0.45989 / 0.71593 | 0.45989 @324 | 0.45908 / 0.45811 / 0.45643 / 0.45057 | - |
| ProbeA | 253 | 0.43280 / 0.69487 | 0.43280 @253 | 0.43200 / 0.43098 / 0.42869 / 0.42153 | latest +0.00272；late20 +0.00322；epoch100 +0.00710；positive 242/253，`WATCH` |
| dynamic | 197 | 0.41355 / 0.66879 | 0.41355 @197 | 0.41241 / 0.41124 / 0.40900 / 0.40108 | latest +0.01389；late20 +0.01456；epoch100 +0.01043；positive 190/197，`PROMISING_EARLY` |
| old-commit ProbeA | 97 | 0.35484 / 0.60312 | 0.35484 @97 | 0.35347 / 0.35192 / 0.34895 / 0.33244 | matched=97，latest +0.01187；late20 +0.01253，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 106 | 0.35078 / 0.59232 | 0.35078 @106 | 0.34957 / 0.34808 / 0.34472 / 0.33212 | control 已过 100 |
| dynamic_singleproj_yoloinit | 88 | 0.33920 / 0.58324 | 0.33920 @88 | 0.33803 / 0.33609 / 0.33187 / 0.30854 | matched=88，latest +0.00047；late20 +0.00103，pre100 |
| dynamic_wo_s_rec_yoloinit | 91 | 0.34529 / 0.59139 | 0.34529 @91 | 0.34411 / 0.34255 / 0.33832 / 0.31663 | matched=91，latest +0.00395；late20 +0.00484，pre100 |
| dynamic_wo_reach_yoloinit | 86 | 0.33501 / 0.58276 | 0.33501 @86 | 0.33383 / 0.33197 / 0.32722 / 0.30043 | matched=86，latest -0.00160；late20 -0.00188，pre100 |

决策：

- 当前没有候选新达到 100/120-row 触发点；本轮不启动下一批，也不停止任务。
- 4090 `dynamic` 继续是唯一明确 `PROMISING_EARLY` 的主线候选，late20 仍约 +1.46 AP50-95 point。
- 4090 `ProbeA` 继续弱正 `WATCH`；old-commit ProbeA 到 97 rows，继续等 100-row 正式评估。
- 3090 三条候选仍 pre100。`wo_s_rec` 小正但不足 +1 point，`singleproj` 近零，`wo_reach` 偏负但未到正式降级点。

### 2026-06-24 19:06 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 84%`；GPU1 `8597/24564 MiB, util 90%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP 连接成功；GPU0 `7623/24576 MiB, util 96%`；GPU1 `8417/24576 MiB, util 52%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。GPU1 低利用率视为瞬时采样，不据此加任务。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 326 | 0.46078 / 0.71653 | 0.46078 @326 | 0.45978 / 0.45883 / 0.45717 / 0.45136 | - |
| ProbeA | 254 | 0.43361 / 0.69385 | 0.43361 @254 | 0.43242 / 0.43144 / 0.42916 / 0.42202 | latest +0.00301；late20 +0.00314；epoch100 +0.00710；positive 243/254，`WATCH` |
| dynamic | 199 | 0.41387 / 0.66906 | 0.41387 @199 | 0.41329 / 0.41210 / 0.40990 / 0.40214 | latest +0.01246；late20 +0.01420；epoch100 +0.01043；positive 192/199，`PROMISING_EARLY` |
| old-commit ProbeA | 99 | 0.35610 / 0.60438 | 0.35610 @99 | 0.35483 / 0.35317 / 0.35016 / 0.33487 | matched=99，latest +0.01215；late20 +0.01238，pre100，不判断 |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 108 | 0.35160 / 0.59232 | 0.35160 @108 | 0.35069 / 0.34916 / 0.34603 / 0.33399 | control 已过 100 |
| dynamic_singleproj_yoloinit | 89 | 0.34008 / 0.58387 | 0.34008 @89 | 0.33870 / 0.33685 / 0.33269 / 0.31035 | matched=89，latest +0.00066；late20 +0.00099，pre100 |
| dynamic_wo_s_rec_yoloinit | 93 | 0.34649 / 0.59269 | 0.34649 @93 | 0.34534 / 0.34386 / 0.33992 / 0.31993 | matched=93，latest +0.00389；late20 +0.00474，pre100 |
| dynamic_wo_reach_yoloinit | 88 | 0.33637 / 0.58644 | 0.33637 @88 | 0.33529 / 0.33341 / 0.32901 / 0.30426 | matched=88，latest -0.00236；late20 -0.00183，pre100 |

决策：

- 当前没有候选新达到 100/120-row 触发点；本轮不启动下一批，也不停止任务。
- 4090 `dynamic` 继续是唯一明确 `PROMISING_EARLY` 的主线候选，late20 仍约 +1.42 AP50-95 point。
- 4090 `ProbeA` 继续弱正 `WATCH`；old-commit ProbeA 到 99 rows，下一轮几乎会触发 100-row 正式评估。
- 3090 三条候选仍 pre100。`wo_s_rec` 小正但不足 +1 point，`singleproj` 近零，`wo_reach` 偏负但未到正式降级点。

### 2026-06-24 19:10 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 88%`；GPU1 `8597/24564 MiB, util 91%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP 连接成功；GPU0 `7623/24576 MiB, util 94%`；GPU1 `8417/24576 MiB, util 91%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 329 | 0.46226 / 0.71866 | 0.46242 @328 | 0.46132 / 0.46020 / 0.45832 / 0.45253 | - |
| ProbeA | 256 | 0.43450 / 0.69572 | 0.43450 @256 | 0.43345 / 0.43232 / 0.43009 / 0.42300 | latest +0.00269；late20 +0.00303；epoch100 +0.00710；positive 245/256，`WATCH` |
| dynamic | 201 | 0.41521 / 0.67018 | 0.41521 @201 | 0.41414 / 0.41302 / 0.41081 / 0.40318 | latest +0.01330；late20 +0.01392；epoch100 +0.01043；positive 194/201，`PROMISING_EARLY` |
| old-commit ProbeA | 101 | 0.35662 / 0.60495 | 0.35662 @101 | 0.35594 / 0.35435 / 0.35134 / 0.33713 | latest +0.01091；late20 +0.01223；epoch100 +0.01159；positive 94/101，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 111 | 0.35291 / 0.59300 | 0.35291 @111 | 0.35191 / 0.35074 / 0.34783 / 0.33648 | control 已过 100 |
| dynamic_singleproj_yoloinit | 91 | 0.34212 / 0.58738 | 0.34212 @91 | 0.34018 / 0.33842 / 0.33435 / 0.31369 | matched=91，latest +0.00078；late20 +0.00088，pre100 |
| dynamic_wo_s_rec_yoloinit | 94 | 0.34683 / 0.59292 | 0.34683 @94 | 0.34587 / 0.34442 / 0.34064 / 0.32145 | matched=94，latest +0.00376；late20 +0.00466，pre100 |
| dynamic_wo_reach_yoloinit | 89 | 0.33754 / 0.58670 | 0.33754 @89 | 0.33592 / 0.33413 / 0.32991 / 0.30612 | matched=89，latest -0.00188；late20 -0.00180，pre100 |

决策：

- 4090 old-commit ProbeA 首次超过 100 rows，正式触发轻量评估：epoch100 delta +0.01159，latest +0.01091，late20 +0.01223，positive 94/101，满足 `PROMISING_EARLY`。
- 当前 4090 有两个 early promising 候选：`dynamic` 与 `old-commit ProbeA`；两者都继续跑，不提前称为最终正结果。
- 4090 `ProbeA` 继续弱正 `WATCH`，增益约 +0.3 AP50-95 point。
- 3090 三条候选仍 pre100。`wo_s_rec` 小正但不足 +1 point，`singleproj` 近零，`wo_reach` 偏负但未到正式降级点。
- 本轮不启动下一批，也不停止任务；优先等 3090 三条候选完成 100-row 判读。

### 2026-06-24 19:16 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 91%`；GPU1 `8597/24564 MiB, util 91%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：direct IP 连接成功；GPU0 `7623/24576 MiB, util 94%`；GPU1 `8417/24576 MiB, util 95%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 333 | 0.46327 / 0.72070 | 0.46327 @333 | 0.46281 / 0.46183 / 0.45980 / 0.45407 | - |
| ProbeA | 260 | 0.43608 / 0.69801 | 0.43608 @260 | 0.43534 / 0.43413 / 0.43196 / 0.42495 | latest +0.00253；late20 +0.00287；epoch100 +0.00710；positive 249/260，`WATCH` |
| dynamic | 204 | 0.41652 / 0.67221 | 0.41652 @204 | 0.41561 / 0.41445 / 0.41216 / 0.40473 | latest +0.01314；late20 +0.01361；epoch100 +0.01043；positive 197/204，`PROMISING_EARLY` |
| old-commit ProbeA | 104 | 0.35816 / 0.60703 | 0.35816 @104 | 0.35725 / 0.35604 / 0.35307 / 0.34029 | latest +0.01057；late20 +0.01192；epoch100 +0.01159；positive 97/104，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 115 | 0.35522 / 0.59512 | 0.35522 @115 | 0.35392 / 0.35270 / 0.35009 / 0.33944 | control 已过 100 |
| dynamic_singleproj_yoloinit | 94 | 0.34361 / 0.58875 | 0.34361 @94 | 0.34243 / 0.34056 / 0.33670 / 0.31853 | matched=94，latest +0.00054；late20 +0.00071，pre100 |
| dynamic_wo_s_rec_yoloinit | 98 | 0.34932 / 0.59581 | 0.34932 @98 | 0.34805 / 0.34670 / 0.34346 / 0.32683 | matched=98，latest +0.00311；late20 +0.00441，pre100 |
| dynamic_wo_reach_yoloinit | 93 | 0.34023 / 0.59113 | 0.34023 @93 | 0.33895 / 0.33712 / 0.33331 / 0.31290 | matched=93，latest -0.00237；late20 -0.00187，pre100 |

决策：

- 4090 `dynamic` 仍是最干净的当前主线 early signal，late20 约 +1.36 AP50-95 point，继续跑满 800。
- 4090 old-commit ProbeA 仍满足 `PROMISING_EARLY`，但它是 old-commit/e700 上下文，暂作为复刻差异线索，不直接替代 full e800 主线证据。
- 4090 当前 ProbeA 继续弱正 `WATCH`，增益只有约 +0.29 AP50-95 point。
- 3090 三条候选仍未到 100；`wo_s_rec` 预计下一轮最先触发 100-row 轻量评估，目前小正但不足 +1 point；`singleproj` 近零；`wo_reach` 仍偏负。
- 本轮不启动下一批、不停止任务。下一步优先等 3090 `wo_s_rec/singleproj/wo_reach` 到 100 rows 后正式判读；如果 `wo_reach` 到 120 后 late20 仍不正，再标记 `LOW_PRIORITY` 并考虑给下一批候选让位。

### 2026-06-24 19:20 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 93%`；GPU1 `8597/24564 MiB, util 89%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：ControlMaster 通道可用；GPU0 `7623/24576 MiB, util 92%`；GPU1 `8417/24576 MiB, util 99%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 336 | 0.46518 / 0.72170 | 0.46518 @336 | 0.46396 / 0.46311 / 0.46097 / 0.45524 | - |
| ProbeA | 262 | 0.43696 / 0.69874 | 0.43696 @262 | 0.43613 / 0.43505 / 0.43280 / 0.42590 | latest +0.00291；late20 +0.00276；epoch100 +0.00710；positive 251/262，`WATCH` |
| dynamic | 206 | 0.41760 / 0.67311 | 0.41760 @206 | 0.41660 / 0.41537 / 0.41307 / 0.40575 | latest +0.01301；late20 +0.01344；epoch100 +0.01043；positive 199/206，`PROMISING_EARLY` |
| old-commit ProbeA | 106 | 0.35946 / 0.60909 | 0.35946 @106 | 0.35835 / 0.35714 / 0.35423 / 0.34229 | latest +0.01062；late20 +0.01174；epoch100 +0.01159；positive 99/106，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 117 | 0.35603 / 0.59657 | 0.35603 @117 | 0.35498 / 0.35368 / 0.35115 / 0.34088 | control 已过 100 |
| dynamic_singleproj_yoloinit | 96 | 0.34582 / 0.59111 | 0.34582 @96 | 0.34384 / 0.34201 / 0.33829 / 0.32135 | matched=96，latest +0.00108；late20 +0.00075，pre100 |
| dynamic_wo_s_rec_yoloinit | 100 | 0.35097 / 0.59776 | 0.35097 @100 | 0.34944 / 0.34794 / 0.34488 / 0.32926 | latest +0.00389；late20 +0.00434；epoch100 +0.00389；positive 78/100，`WATCH` |
| dynamic_wo_reach_yoloinit | 95 | 0.34123 / 0.59133 | 0.34123 @95 | 0.34023 / 0.33844 / 0.33480 / 0.31595 | matched=95，latest -0.00261；late20 -0.00196，pre100 |

决策：

- 3090 `dynamic_wo_s_rec_yoloinit` 首次达到 100 rows，正式轻量评估为 `WATCH`：epoch100/latest +0.00389，late20 +0.00434，positive 78/100。它是小正，但不到 +1 AP50-95 point 的 early promising 线。
- 3090 `singleproj` 仍 pre100 且近零；`wo_reach` 仍 pre100 且偏负，未到 120-row 降级点。
- 4090 `dynamic` 继续是最干净 early promising，latest +0.01301、late20 +0.01344；old-commit ProbeA 继续 positive 但只作复刻差异线索。
- 本轮不启动下一批、不停止任务。下一步等 `singleproj/wo_reach` 到 100 rows；若 `wo_reach` 到 120 后 late20 仍为负，再按规则标记 `LOW_PRIORITY` 并考虑释放或替换。

### 2026-06-24 19:22 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 90%`；GPU1 `8597/24564 MiB, util 95%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：ControlMaster 通道可用；GPU0 `7623/24576 MiB, util 92%`；GPU1 `8417/24576 MiB, util 95%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 337 | 0.46513 / 0.72151 | 0.46518 @336 | 0.46438 / 0.46351 / 0.46137 / 0.45562 | - |
| ProbeA | 263 | 0.43742 / 0.69960 | 0.43742 @263 | 0.43653 / 0.43551 / 0.43325 / 0.42636 | latest +0.00305；late20 +0.00277；epoch100 +0.00710；positive 252/263，`WATCH` |
| dynamic | 207 | 0.41815 / 0.67385 | 0.41815 @207 | 0.41708 / 0.41583 / 0.41353 / 0.40626 | latest +0.01318；late20 +0.01337；epoch100 +0.01043；positive 200/207，`PROMISING_EARLY` |
| old-commit ProbeA | 107 | 0.35991 / 0.60926 | 0.35991 @107 | 0.35889 / 0.35765 / 0.35479 / 0.34325 | latest +0.01062；late20 +0.01166；epoch100 +0.01159；positive 100/107，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 118 | 0.35697 / 0.59697 | 0.35697 @118 | 0.35569 / 0.35422 / 0.35169 / 0.34159 | control 已过 100 |
| dynamic_singleproj_yoloinit | 97 | 0.34621 / 0.59188 | 0.34621 @97 | 0.34458 / 0.34277 / 0.33906 / 0.32263 | matched=97，latest +0.00062；late20 +0.00076，pre100 |
| dynamic_wo_s_rec_yoloinit | 101 | 0.35085 / 0.59710 | 0.35097 @100 | 0.35003 / 0.34850 / 0.34552 / 0.33040 | latest +0.00347；late20 +0.00428；epoch100 +0.00389；positive 79/101，`WATCH` |
| dynamic_wo_reach_yoloinit | 96 | 0.34121 / 0.59128 | 0.34123 @95 | 0.34065 / 0.33906 / 0.33551 / 0.31740 | matched=96，latest -0.00353；late20 -0.00202，pre100 |

决策：

- 本轮没有新增 100/120-row 触发点；不启动下一批，也不停止任务。
- 3090 `wo_s_rec` 过 100 后仍是小正 `WATCH`，没有接近 +1 point；`singleproj` 仍 near-zero pre100；`wo_reach` 仍负向 pre100。
- 4090 `dynamic` 的 early positive 仍稳定在约 +1.3 AP50-95 point；old-commit ProbeA 仍为复刻差异线索。
- 下一步继续等 `singleproj/wo_reach` 到 100 rows；`wo_reach` 若到 120 后 late20 仍负，再降级为 `LOW_PRIORITY`。

### 2026-06-24 19:24 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 92%`；GPU1 `8597/24564 MiB, util 88%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：ControlMaster 通道可用；GPU0 `7623/24576 MiB, util 98%`；GPU1 `8417/24576 MiB, util 64%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。GPU1 util 低视为瞬时采样，不据此追加任务。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 338 | 0.46496 / 0.72194 | 0.46518 @336 | 0.46472 / 0.46376 / 0.46174 / 0.45601 | - |
| ProbeA | 264 | 0.43747 / 0.70024 | 0.43747 @264 | 0.43688 / 0.43590 / 0.43367 / 0.42682 | latest +0.00208；late20 +0.00273；epoch100 +0.00710；positive 253/264，`WATCH` |
| dynamic | 208 | 0.41812 / 0.67400 | 0.41815 @207 | 0.41747 / 0.41628 / 0.41397 / 0.40676 | latest +0.01219；late20 +0.01326；epoch100 +0.01043；positive 201/208，`PROMISING_EARLY` |
| old-commit ProbeA | 108 | 0.35978 / 0.60967 | 0.35991 @107 | 0.35930 / 0.35807 / 0.35532 / 0.34419 | latest +0.01017；late20 +0.01157；epoch100 +0.01159；positive 101/108，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 120 | 0.35760 / 0.59786 | 0.35760 @120 | 0.35671 / 0.35531 / 0.35275 / 0.34299 | control 已过 100 |
| dynamic_singleproj_yoloinit | 98 | 0.34683 / 0.59272 | 0.34683 @98 | 0.34535 / 0.34354 / 0.33981 / 0.32389 | matched=98，latest +0.00062；late20 +0.00076，pre100 |
| dynamic_wo_s_rec_yoloinit | 102 | 0.35164 / 0.59764 | 0.35164 @102 | 0.35062 / 0.34905 / 0.34614 / 0.33152 | latest +0.00335；late20 +0.00420；epoch100 +0.00389；positive 80/102，`WATCH` |
| dynamic_wo_reach_yoloinit | 97 | 0.34202 / 0.59179 | 0.34202 @97 | 0.34113 / 0.33965 / 0.33618 / 0.31872 | matched=97，latest -0.00357；late20 -0.00212，pre100 |

决策：

- 本轮没有新增 100/120-row 触发点；不启动下一批，也不停止任务。
- 3090 `wo_s_rec` 到 102 后仍是小正 `WATCH`，未接近 +1 point；`singleproj` 仍 near-zero pre100；`wo_reach` 仍负向 pre100。
- 4090 `dynamic` 仍保持 early positive，latest +0.01219、late20 +0.01326；但增益相对最早 +1.4~1.5 point 区间略收窄，继续观察到更长窗口。
- 下一步等 3090 `singleproj/wo_reach` 到 100 rows；`wo_reach` 若到 120 后 late20 仍负，再按规则降级。

### 2026-06-24 19:27 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 89%`；GPU1 `8597/24564 MiB, util 88%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：ControlMaster 通道可用；GPU0 `7623/24576 MiB, util 98%`；GPU1 `8417/24576 MiB, util 92%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 340 | 0.46617 / 0.72307 | 0.46617 @340 | 0.46537 / 0.46445 / 0.46252 / 0.45678 | - |
| ProbeA | 266 | 0.43837 / 0.70028 | 0.43837 @266 | 0.43765 / 0.43669 / 0.43450 / 0.42773 | latest +0.00183；late20 +0.00260；epoch100 +0.00710；positive 255/266，`WATCH` |
| dynamic | 210 | 0.41919 / 0.67567 | 0.41919 @210 | 0.41834 / 0.41723 / 0.41489 / 0.40776 | latest +0.01227；late20 +0.01311；epoch100 +0.01043；positive 203/210，`PROMISING_EARLY` |
| old-commit ProbeA | 109 | 0.36020 / 0.61042 | 0.36020 @109 | 0.35971 / 0.35848 / 0.35582 / 0.34507 | latest +0.00939；late20 +0.01143；epoch100 +0.01159；positive 102/109，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 122 | 0.35902 / 0.59923 | 0.35902 @122 | 0.35791 / 0.35645 / 0.35385 / 0.34439 | control 已过 120 |
| dynamic_singleproj_yoloinit | 100 | 0.34749 / 0.59276 | 0.34749 @100 | 0.34669 / 0.34490 / 0.34126 / 0.32634 | latest +0.00041；late20 +0.00072；epoch100 +0.00041；positive 61/100，`WATCH` |
| dynamic_wo_s_rec_yoloinit | 104 | 0.35276 / 0.59804 | 0.35276 @104 | 0.35166 / 0.35020 / 0.34731 / 0.33368 | latest +0.00311；late20 +0.00400；epoch100 +0.00389；positive 82/104，`WATCH` |
| dynamic_wo_reach_yoloinit | 99 | 0.34320 / 0.59414 | 0.34320 @99 | 0.34207 / 0.34085 / 0.33749 / 0.32130 | matched=99，latest -0.00346；late20 -0.00231，pre100 |

决策：

- 3090 `dynamic_singleproj_yoloinit` 首次达到 100 rows，正式轻量评估为 `WATCH`，但增益几乎为零：epoch100/latest +0.00041，late20 +0.00072，positive 61/100，不满足 early promising。
- 3090 `wo_s_rec` 仍为小正 `WATCH`，latest +0.00311、late20 +0.00400，也不到 +1 point。
- 3090 `wo_reach` 到 99 rows，仍未触发 100-row 正式判读，当前 latest/late20 仍为负。
- 4090 `dynamic` 继续是最干净的 early promising，但增益保持在约 +1.2~+1.3 AP50-95 point；继续观察到更长窗口。
- 本轮不启动下一批、不停止任务。下一步等 `wo_reach` 到 100 rows 做正式判读，并继续等 120-row 降级点。

### 2026-06-24 19:32 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 93%`；GPU1 `8597/24564 MiB, util 95%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：重建 ControlMaster 后连接成功；GPU0 `7623/24576 MiB, util 97%`；GPU1 `8417/24576 MiB, util 98%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 343 | 0.46719 / 0.72399 | 0.46719 @343 | 0.46642 / 0.46557 / 0.46370 / 0.45795 | - |
| ProbeA | 268 | 0.44025 / 0.70128 | 0.44025 @268 | 0.43860 / 0.43757 / 0.43541 / 0.42864 | latest +0.00286；late20 +0.00252；epoch100 +0.00710；positive 257/268，`WATCH` |
| dynamic | 212 | 0.42016 / 0.67664 | 0.42016 @212 | 0.41913 / 0.41811 / 0.41580 / 0.40875 | latest +0.01193；late20 +0.01294；epoch100 +0.01043；positive 205/212，`PROMISING_EARLY` |
| old-commit ProbeA | 112 | 0.36208 / 0.61179 | 0.36208 @112 | 0.36097 / 0.35993 / 0.35743 / 0.34748 | latest +0.00954；late20 +0.01105；epoch100 +0.01159；positive 105/112，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 125 | 0.36072 / 0.60021 | 0.36072 @125 | 0.35963 / 0.35817 / 0.35544 / 0.34638 | control 已过 120 |
| dynamic_singleproj_yoloinit | 102 | 0.34905 / 0.59448 | 0.34905 @102 | 0.34775 / 0.34616 / 0.34267 / 0.32863 | latest +0.00076；late20 +0.00074；epoch100 +0.00041；positive 63/102，`WATCH` |
| dynamic_wo_s_rec_yoloinit | 106 | 0.35417 / 0.59894 | 0.35417 @106 | 0.35279 / 0.35141 / 0.34847 / 0.33575 | latest +0.00339；late20 +0.00375；epoch100 +0.00389；positive 84/106，`WATCH` |
| dynamic_wo_reach_yoloinit | 101 | 0.34400 / 0.59512 | 0.34400 @101 | 0.34309 / 0.34187 / 0.33876 / 0.32368 | latest -0.00338；late20 -0.00248；epoch100 -0.00355；positive 9/101，`WATCH` |

决策：

- 3090 `dynamic_wo_reach_yoloinit` 首次达到 100 rows，正式轻量评估为负向：epoch100 -0.00355，latest -0.00338，late20 -0.00248，positive 9/101。它还未到 120-row 降级触发点，因此暂不停止，但已经是最可能进入 `LOW_PRIORITY` 的候选。
- 3090 `singleproj` 继续 near-zero `WATCH`，`wo_s_rec` 继续小正 `WATCH`；两者均不满足 early promising。
- 4090 `dynamic` 仍是当前最干净的主线 early signal，latest +0.01193、late20 +0.01294；old-commit ProbeA 继续作为复刻差异线索。
- 本轮不启动下一批、不停止任务。下一步优先观察 `wo_reach` 到 120 rows 后是否满足 `LOW_PRIORITY`，若是再考虑释放或替换为下一批 LADD-like 变体。

### 2026-06-24 19:37 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 92%`；GPU1 `8597/24564 MiB, util 93%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：ControlMaster 通道可用；GPU0 `7623/24576 MiB, util 92%`；GPU1 `8417/24576 MiB, util 94%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 347 | 0.46786 / 0.72570 | 0.46786 @347 | 0.46743 / 0.46670 / 0.46511 / 0.45944 | - |
| ProbeA | 272 | 0.44163 / 0.70319 | 0.44163 @272 | 0.44071 / 0.43937 / 0.43721 / 0.43045 | latest +0.00152；late20 +0.00238；epoch100 +0.00710；positive 261/272，`WATCH` |
| dynamic | 215 | 0.42124 / 0.67912 | 0.42124 @215 | 0.42032 / 0.41933 / 0.41712 / 0.41016 | latest +0.01111；late20 +0.01264；epoch100 +0.01043；positive 208/215，`PROMISING_EARLY` |
| old-commit ProbeA | 114 | 0.36328 / 0.61322 | 0.36328 @114 | 0.36212 / 0.36091 / 0.35848 / 0.34895 | latest +0.00963；late20 +0.01076；epoch100 +0.01159；positive 107/114，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 128 | 0.36269 / 0.60263 | 0.36269 @128 | 0.36154 / 0.35998 / 0.35710 / 0.34829 | control 已过 120 |
| dynamic_singleproj_yoloinit | 105 | 0.35109 / 0.59637 | 0.35109 @105 | 0.34973 / 0.34821 / 0.34473 / 0.33180 | latest +0.00072；late20 +0.00072；epoch100 +0.00041；positive 66/105，`WATCH` |
| dynamic_wo_s_rec_yoloinit | 109 | 0.35577 / 0.60070 | 0.35577 @109 | 0.35463 / 0.35315 / 0.35023 / 0.33861 | latest +0.00388；late20 +0.00357；epoch100 +0.00389；positive 87/109，`WATCH` |
| dynamic_wo_reach_yoloinit | 104 | 0.34588 / 0.59646 | 0.34588 @104 | 0.34470 / 0.34338 / 0.34058 / 0.32704 | latest -0.00377；late20 -0.00274；epoch100 -0.00355；positive 9/104，`WATCH` |

决策：

- 本轮没有新增 120-row 降级触发点；不启动下一批、不停止任务。
- 3090 `wo_reach` 到 104 rows 后仍稳定负向，latest -0.00377、late20 -0.00274，若到 120 rows 仍为负，应按规则标记 `LOW_PRIORITY`。
- 3090 `singleproj` 基本零增益，`wo_s_rec` 小正但仍不到 +1 point，二者继续 `WATCH`。
- 4090 `dynamic` 仍为最强 early signal，但 latest/late20 增益继续从早期 +1.4~+1.5 point 收窄到约 +1.1~+1.3 point，需要继续观察到更长窗口再决定是否足够稳定作为主线。

### 2026-06-24 19:44 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 91%`；GPU1 `8597/24564 MiB, util 90%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：重建 ControlMaster 后连接成功；GPU0 `7623/24576 MiB, util 93%`；GPU1 `8417/24576 MiB, util 99%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 350 | 0.46890 / 0.72723 | 0.46890 @350 | 0.46820 / 0.46758 / 0.46602 / 0.46051 | - |
| ProbeA | 274 | 0.44222 / 0.70407 | 0.44222 @274 | 0.44141 / 0.44029 / 0.43809 / 0.43137 | latest +0.00114；late20 +0.00223；epoch100 +0.00710；positive 263/274，`WATCH` |
| dynamic | 218 | 0.42324 / 0.68206 | 0.42324 @218 | 0.42177 / 0.42068 / 0.41848 / 0.41156 | latest +0.01185；late20 +0.01235；epoch100 +0.01043；positive 211/218，`PROMISING_EARLY` |
| old-commit ProbeA | 117 | 0.36492 / 0.61623 | 0.36492 @117 | 0.36382 / 0.36239 / 0.36002 / 0.35107 | latest +0.00981；late20 +0.01043；epoch100 +0.01159；positive 110/117，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 133 | 0.36563 / 0.60523 | 0.36563 @133 | 0.36433 / 0.36294 / 0.35999 / 0.35139 | control 已过 120 |
| dynamic_singleproj_yoloinit | 108 | 0.35319 / 0.59775 | 0.35319 @108 | 0.35175 / 0.35005 / 0.34679 / 0.33473 | latest +0.00159；late20 +0.00076；epoch100 +0.00041；positive 69/108，`WATCH` |
| dynamic_wo_s_rec_yoloinit | 112 | 0.35710 / 0.60323 | 0.35710 @112 | 0.35618 / 0.35478 / 0.35192 / 0.34114 | latest +0.00370；late20 +0.00352；epoch100 +0.00389；positive 90/112，`WATCH` |
| dynamic_wo_reach_yoloinit | 107 | 0.34724 / 0.59806 | 0.34724 @107 | 0.34628 / 0.34494 / 0.34230 / 0.33010 | latest -0.00383；late20 -0.00309；epoch100 -0.00355；positive 9/107，`WATCH` |

决策：

- 本轮没有达到 120-row 降级触发点；不启动下一批、不停止任务。
- 3090 `wo_reach` 到 107 rows 后仍持续负向，latest -0.00383、late20 -0.00309，若 120 rows 仍为负，应标记 `LOW_PRIORITY`。
- 3090 `singleproj` 基本零增益；`wo_s_rec` 小正但远低于 +1 point；二者继续 `WATCH`。
- 4090 `dynamic` 继续保持 early positive，latest +0.01185、late20 +0.01235，仍是当前最值得保留到 e800 的候选；current ProbeA 仍弱正。

### 2026-06-24 19:48 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 91%`；GPU1 `8597/24564 MiB, util 89%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：重建 ControlMaster 后连接成功；GPU0 `7623/24576 MiB, util 91%`；GPU1 `8417/24576 MiB, util 97%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 354 | 0.47104 / 0.72867 | 0.47104 @354 | 0.46987 / 0.46886 / 0.46737 / 0.46197 | - |
| ProbeA | 277 | 0.44410 / 0.70582 | 0.44410 @277 | 0.44309 / 0.44190 / 0.43949 / 0.43276 | latest +0.00206；late20 +0.00222；epoch100 +0.00710；positive 266/277，`WATCH` |
| dynamic | 221 | 0.42449 / 0.68421 | 0.42449 @221 | 0.42349 / 0.42212 / 0.41989 / 0.41299 | latest +0.01157；late20 +0.01212；epoch100 +0.01043；positive 214/221，`PROMISING_EARLY` |
| old-commit ProbeA | 119 | 0.36581 / 0.61542 | 0.36581 @119 | 0.36498 / 0.36355 / 0.36101 / 0.35242 | latest +0.01015；late20 +0.01024；epoch100 +0.01159；positive 112/119，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 136 | 0.36713 / 0.60641 | 0.36713 @136 | 0.36608 / 0.36466 / 0.36170 / 0.35321 | control 已过 120 |
| dynamic_singleproj_yoloinit | 110 | 0.35459 / 0.59794 | 0.35459 @110 | 0.35316 / 0.35145 / 0.34817 / 0.33655 | latest +0.00249；late20 +0.00092；epoch100 +0.00041；positive 71/110，`WATCH` |
| dynamic_wo_s_rec_yoloinit | 115 | 0.35850 / 0.60418 | 0.35850 @115 | 0.35760 / 0.35639 / 0.35359 / 0.34347 | latest +0.00328；late20 +0.00350；epoch100 +0.00389；positive 93/115，`WATCH` |
| dynamic_wo_reach_yoloinit | 110 | 0.34836 / 0.59915 | 0.34836 @110 | 0.34764 / 0.34643 / 0.34390 / 0.33282 | latest -0.00374；late20 -0.00335；epoch100 -0.00355；positive 9/110，`WATCH` |

决策：

- 本轮仍未达到 120-row 降级触发点；不启动下一批、不停止任务。
- 3090 `wo_reach` 到 110 rows 后持续负向，latest -0.00374、late20 -0.00335；若到 120 rows 仍为负，应标记 `LOW_PRIORITY`。
- 3090 `singleproj` 近零，`wo_s_rec` 小正但仍远低于 +1 point；两者继续 `WATCH`。
- 4090 `dynamic` 仍是当前最强 YOLO-init 主线候选，latest +0.01157、late20 +0.01212，但相对早期峰值继续略收窄，需跑更长窗口验证稳定性。

### 2026-06-24 19:52 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 89%`；GPU1 `8597/24564 MiB, util 89%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：重建 ControlMaster 后连接成功；GPU0 `7623/24576 MiB, util 95%`；GPU1 `8417/24576 MiB, util 97%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 357 | 0.47206 / 0.73075 | 0.47207 @356 | 0.47123 / 0.47008 / 0.46839 / 0.46307 | - |
| ProbeA | 280 | 0.44523 / 0.70684 | 0.44523 @280 | 0.44454 / 0.44328 / 0.44087 / 0.43413 | latest +0.00213；late20 +0.00216；epoch100 +0.00710；positive 269/280，`WATCH` |
| dynamic | 223 | 0.42521 / 0.68505 | 0.42521 @223 | 0.42440 / 0.42309 / 0.42081 / 0.41394 | latest +0.01081；late20 +0.01188；epoch100 +0.01043；positive 216/223，`PROMISING_EARLY` |
| old-commit ProbeA | 122 | 0.36758 / 0.61788 | 0.36758 @122 | 0.36642 / 0.36512 / 0.36252 / 0.35429 | latest +0.00993；late20 +0.01001；epoch100 +0.01159；positive 115/122，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 138 | 0.36809 / 0.60757 | 0.36809 @138 | 0.36703 / 0.36568 / 0.36283 / 0.35439 | control 已过 120 |
| dynamic_singleproj_yoloinit | 112 | 0.35501 / 0.59988 | 0.35549 @111 | 0.35446 / 0.35276 / 0.34946 / 0.33822 | latest +0.00161；late20 +0.00107；epoch100 +0.00041；positive 73/112，`WATCH` |
| dynamic_wo_s_rec_yoloinit | 117 | 0.35958 / 0.60625 | 0.35958 @117 | 0.35860 / 0.35739 / 0.35470 / 0.34495 | latest +0.00355；late20 +0.00354；epoch100 +0.00389；positive 95/117，`WATCH` |
| dynamic_wo_reach_yoloinit | 112 | 0.34991 / 0.60098 | 0.34991 @112 | 0.34869 / 0.34748 / 0.34493 / 0.33448 | latest -0.00349；late20 -0.00347；epoch100 -0.00355；positive 9/112，`WATCH` |

决策：

- 本轮仍未达到 120-row 降级触发点；不启动下一批、不停止任务。
- 3090 `wo_reach` 到 112 rows 后仍持续负向，latest -0.00349、late20 -0.00347；距离 120-row 降级点还差 8 rows。
- 3090 `singleproj` 继续近零；`wo_s_rec` 小正但不到 +1 point。
- 4090 `dynamic` 仍是最强候选，但 latest/late20 正增益继续收窄到约 +1.08/+1.19 AP50-95 point，后续需要重点看是否跌破 +1 point。

### 2026-06-24 19:55 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 92%`；GPU1 `8597/24564 MiB, util 90%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1`：ControlMaster 通道可用；GPU0 `7623/24576 MiB, util 98%`；GPU1 `8417/24576 MiB, util 86%`。4 个训练组均在，日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 359 | 0.47259 / 0.73126 | 0.47259 @359 | 0.47202 / 0.47094 / 0.46913 / 0.46379 | - |
| ProbeA | 282 | 0.44646 / 0.70780 | 0.44646 @282 | 0.44538 / 0.44424 / 0.44181 / 0.43503 | latest +0.00241；late20 +0.00210；epoch100 +0.00710；positive 271/282，`WATCH` |
| dynamic | 225 | 0.42668 / 0.68636 | 0.42668 @225 | 0.42540 / 0.42417 / 0.42175 / 0.41489 | latest +0.01129；late20 +0.01166；epoch100 +0.01043；positive 218/225，`PROMISING_EARLY` |
| old-commit ProbeA | 123 | 0.36809 / 0.61775 | 0.36809 @123 | 0.36690 / 0.36568 / 0.36304 / 0.35487 | latest +0.01001；late20 +0.00998；epoch100 +0.01159；positive 116/123，`WATCH` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 140 | 0.36871 / 0.60876 | 0.36871 @140 | 0.36793 / 0.36672 / 0.36393 / 0.35553 | control 已过 120 |
| dynamic_singleproj_yoloinit | 114 | 0.35657 / 0.60123 | 0.35657 @114 | 0.35551 / 0.35399 / 0.35076 / 0.33978 | latest +0.00194；late20 +0.00124；epoch100 +0.00041；positive 75/114，`WATCH` |
| dynamic_wo_s_rec_yoloinit | 118 | 0.36065 / 0.60745 | 0.36065 @118 | 0.35919 / 0.35792 / 0.35526 / 0.34569 | latest +0.00368；late20 +0.00357；epoch100 +0.00389；positive 96/118，`WATCH` |
| dynamic_wo_reach_yoloinit | 113 | 0.35067 / 0.60212 | 0.35067 @113 | 0.34928 / 0.34800 / 0.34545 / 0.33529 | latest -0.00277；late20 -0.00349；epoch100 -0.00355；positive 9/113，`WATCH` |

决策：

- 本轮仍未达到 120-row 降级触发点；不启动下一批、不停止任务。
- 3090 `wo_reach` 到 113 rows 后仍持续负向，late20 -0.00349；距离 120-row 降级点还差 7 rows。
- 3090 `singleproj` 近零，`wo_s_rec` 小正但远不到 +1 point。
- 4090 `dynamic` 仍是最强候选，latest +0.01129、late20 +0.01166；old-commit ProbeA 的 late20 微降到 +0.00998，按阈值从 `PROMISING_EARLY` 退回 `WATCH`，继续只作复刻差异线索。

### 2026-06-24 20:13 CST

服务器状态：

- `ladd4090-zw1` 20:02 巡检：GPU0 `7853/24564 MiB, util 92%`；GPU1 `8597/24564 MiB, util 90%`。日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。
- `ladd3090-zw1` 20:08 巡检：GPU0 `7623/24576 MiB, util 94%`；GPU1 `8417/24576 MiB, util 93%`。日志扫描未见 Traceback、CUDA OOM、RuntimeError、NaN 或 batch fallback。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 364 | 0.47353 / 0.73311 | 0.47353 @364 | 0.47311 / 0.47257 / 0.47072 / 0.46553 | - |
| ProbeA | 286 | 0.44817 / 0.70949 | 0.44821 @285 | 0.44736 / 0.44613 / 0.44376 / 0.43688 | latest +0.00304；late20 +0.00223；epoch100 +0.00710；positive 275/286，`WATCH` |
| dynamic | 229 | 0.42823 / 0.68788 | 0.42823 @229 | 0.42752 / 0.42620 / 0.42367 / 0.41678 | latest +0.01121；late20 +0.01133；epoch100 +0.01043；positive 222/229，`PROMISING_EARLY` |
| old-commit ProbeA | 127 | 0.37116 / 0.62215 | 0.37116 @127 | 0.36963 / 0.36802 / 0.36521 / 0.35719 | latest +0.01031；late20 +0.00984；epoch100 +0.01159；positive 120/127，`WATCH` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 149 | 0.37427 / 0.61564 | 0.37427 @149 | 0.37318 / 0.37154 / 0.36887 / 0.36055 | control |
| dynamic_singleproj_yoloinit | 120 | 0.36128 / 0.60811 | 0.36128 @120 | 0.36014 / 0.35812 / 0.35478 / 0.34430 | latest +0.00368；late20 +0.00203；epoch100 +0.00041；positive 81/120，`WATCH` |
| dynamic_wo_s_rec_yoloinit | 126 | 0.36520 / 0.61342 | 0.36520 @126 | 0.36438 / 0.36281 / 0.35985 / 0.35105 | latest +0.00382；late20 +0.00389；epoch100 +0.00389；positive 104/126，`WATCH` |
| dynamic_wo_reach_yoloinit | 121 | 0.35498 / 0.60728 | 0.35498 @121 | 0.35387 / 0.35244 / 0.34970 / 0.34090 | latest -0.00363；late20 -0.00361；epoch100 -0.00355；positive 9/121，`LOW_PRIORITY` |

决策与动作：

- `dynamic_wo_reach_yoloinit` 已达到 120-row 降级线，late20=-0.00361，判为 `LOW_PRIORITY`。已停止该进程，保留其 121-row `results.csv` 作为负向证据。
- 发现一个重要混淆：4090 `dynamic` 使用 AutoDL-condition 相关 YAML / teacher / A1 cache，而 3090 当前 variants 使用 `configs/paper` + `formal_nomosaic` teacher + `ladd_dynamic` A1 cache；因此 4090 dynamic 的 +1 point 不能直接解释 3090 variants。
- 为补齐 3090 same-control 的主候选，已在 GPU1 启动 `dynamic_plain_yoloinit`，由 `wo_s_rec` 同协议命令复制而来，仅把 `alpha_s_rec` 从 `0.0` 改回 `0.1`，并保留 `lambda_reach/lambda_match_inner/lambda_rank_inner=1.0`。
- 新 run：`runs_public/ogsod/hbb/yoloinit_mainline_search_20260624/dynamic_plain_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_plain_yoloinit_yolo11n_e800_b64_img256_s0_20260624_201305_gpu1/`
- 启动健康检查：PID 外层 `31989`、主进程 `31994`；GPU1 初始 `4659 MiB`，日志初检 `CLEAN`。下一步等它产出 `results.csv` 并在 100 epoch 后与 3090 `detonly_control` 做同 epoch 评估。
- 20:16 复查：监控脚本已纳入 `dynamic_plain`，rows=`1`，latest AP50/AP50-95=`0.14672/0.05137`，status=`pre100`；GPU1 `8387/24576 MiB, util 95%`，当前进程为 `dynamic_plain` + `wo_s_rec`，`wo_reach` 已不在进程表中。日志扫描 `bad=[]`。

### 2026-06-24 20:18 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 93%`；GPU1 `8597/24564 MiB, util 90%`。4 个训练组均在，日志扫描 `bad=[]`。
- `ladd3090-zw1`：GPU0 `7623/24576 MiB, util 94%`；GPU1 `8389/24576 MiB, util 98%`。当前有效训练为 detonly_control、singleproj、wo_s_rec、dynamic_plain；`wo_reach` 仅保留结果文件，不在进程表。日志扫描 `bad=[]`。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 375 | 0.47738 / 0.73677 | 0.47738 @375 | 0.47632 / 0.47562 / 0.47425 / 0.46931 | - |
| ProbeA | 295 | 0.45167 / 0.71530 | 0.45167 @295 | 0.45109 / 0.45003 / 0.44786 / 0.44095 | latest +0.00294；late20 +0.00280；epoch100 +0.00710；positive 284/295，`WATCH` |
| dynamic | 237 | 0.43318 / 0.69423 | 0.43318 @237 | 0.43189 / 0.43045 / 0.42784 / 0.42058 | latest +0.01112；late20 +0.01126；epoch100 +0.01043；positive 230/237，`PROMISING_EARLY` |
| old-commit ProbeA | 135 | 0.37547 / 0.62674 | 0.37547 @135 | 0.37469 / 0.37320 / 0.37002 / 0.36174 | latest +0.01009；late20 +0.01013；epoch100 +0.01159；positive 128/135，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 156 | 0.37799 / 0.62035 | 0.37799 @156 | 0.37690 / 0.37550 / 0.37265 / 0.36438 | control |
| dynamic_singleproj_yoloinit | 125 | 0.36467 / 0.61244 | 0.36467 @125 | 0.36339 / 0.36176 / 0.35820 / 0.34791 | latest +0.00395；late20 +0.00276；epoch100 +0.00041；positive 86/125，`WATCH` |
| dynamic_wo_s_rec_yoloinit | 131 | 0.36885 / 0.61656 | 0.36885 @131 | 0.36744 / 0.36591 / 0.36279 / 0.35418 | latest +0.00457；late20 +0.00400；epoch100 +0.00389；positive 109/131，`WATCH` |
| dynamic_wo_reach_yoloinit | 122 | 0.35555 / 0.60837 | 0.35555 @122 | 0.35446 / 0.35300 / 0.35024 / 0.34156 | latest -0.00347；late20 -0.00360；epoch100 -0.00355；positive 9/122，`LOW_PRIORITY`，已停止 |
| dynamic_plain_yoloinit | 2 | 0.02820 / 0.07819 | 0.05137 @1 | 0.03979 / 0.03979 / 0.03979 / 0.03979 | matched=2；status=`pre100`，暂不评价 |

决策：

- 本轮没有新的 100/120 epoch 决策触发；不新增、不停止。
- 4090 `dynamic` 仍是当前最强有效候选，latest/late20 维持约 +1.1 AP50-95 point，但还需要继续跑满 e800 验证后期稳定性。
- 4090 old-commit ProbeA 的 late20 回到 +1 point 附近，但它是 e700/old-commit 复刻线索，不作为正式主线直接证据。
- 3090 `singleproj` 和 `wo_s_rec` 均为小正但远不到 +1 point；继续 `WATCH`。`dynamic_plain` 只有 2 rows，属于 `pre100`。

### 2026-06-24 20:21 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 93%`；GPU1 `8597/24564 MiB, util 93%`。4 个训练组均在，日志扫描 `bad=[]`。
- `ladd3090-zw1`：GPU0 `7623/24576 MiB, util 98%`；GPU1 `8417/24576 MiB, util 94%`。detonly_control、singleproj、wo_s_rec、dynamic_plain 有效运行；日志扫描 `bad=[]`。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 377 | 0.47705 / 0.73773 | 0.47738 @375 | 0.47675 / 0.47611 / 0.47474 / 0.46996 | - |
| ProbeA | 296 | 0.45253 / 0.71578 | 0.45253 @296 | 0.45151 / 0.45046 / 0.44830 / 0.44140 | latest +0.00353；late20 +0.00284；epoch100 +0.00710；positive 285/296，`WATCH` |
| dynamic | 239 | 0.43356 / 0.69506 | 0.43356 @239 | 0.43289 / 0.43150 / 0.42885 / 0.42154 | latest +0.01039；late20 +0.01116；epoch100 +0.01043；positive 232/239，`PROMISING_EARLY` |
| old-commit ProbeA | 136 | 0.37651 / 0.62828 | 0.37651 @136 | 0.37524 / 0.37382 / 0.37061 / 0.36231 | latest +0.01103；late20 +0.01018；epoch100 +0.01159；positive 129/136，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 157 | 0.37862 / 0.62104 | 0.37862 @157 | 0.37743 / 0.37603 / 0.37321 / 0.36493 | control |
| dynamic_singleproj_yoloinit | 127 | 0.36583 / 0.61288 | 0.36583 @127 | 0.36471 / 0.36305 / 0.35955 / 0.34933 | latest +0.00322；late20 +0.00301；epoch100 +0.00041；positive 88/127，`WATCH` |
| dynamic_wo_s_rec_yoloinit | 132 | 0.36916 / 0.61649 | 0.36916 @132 | 0.36809 / 0.36646 / 0.36340 / 0.35477 | latest +0.00398；late20 +0.00401；epoch100 +0.00389；positive 110/132，`WATCH` |
| dynamic_wo_reach_yoloinit | 122 | 0.35555 / 0.60837 | 0.35555 @122 | 0.35446 / 0.35300 / 0.35024 / 0.34156 | latest -0.00347；late20 -0.00360；epoch100 -0.00355；positive 9/122，`LOW_PRIORITY`，已停止 |
| dynamic_plain_yoloinit | 3 | 0.00474 / 0.02023 | 0.05137 @1 | 0.02810 / 0.02810 / 0.02810 / 0.02810 | matched=3；status=`pre100`，早期波动不评价 |

决策：

- 本轮没有新的 100/120 epoch 决策触发；不新增、不停止。
- 4090 `dynamic` 仍保持 `PROMISING_EARLY`，latest 正增益略收窄到 +0.01039，但 late20 仍为 +0.01116。
- 3090 `singleproj` / `wo_s_rec` 继续小正但不足 +1 point；`dynamic_plain` 仍处于 very early rows，等待 100 epoch。

### 2026-06-24 20:24 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 92%`；GPU1 `8597/24564 MiB, util 89%`。4 个训练组均在，日志扫描 `bad=[]`。
- `ladd3090-zw1`：GPU0 `7623/24576 MiB, util 95%`；GPU1 `8417/24576 MiB, util 94%`。detonly_control、singleproj、wo_s_rec、dynamic_plain 有效运行；日志扫描 `bad=[]`。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 379 | 0.47761 / 0.73887 | 0.47761 @379 | 0.47724 / 0.47658 / 0.47523 / 0.47056 | - |
| ProbeA | 298 | 0.45306 / 0.71709 | 0.45306 @298 | 0.45216 / 0.45128 / 0.44913 / 0.44229 | latest +0.00275；late20 +0.00289；epoch100 +0.00710；positive 287/298，`WATCH` |
| dynamic | 240 | 0.43344 / 0.69467 | 0.43356 @239 | 0.43317 / 0.43194 / 0.42932 / 0.42201 | latest +0.00995；late20 +0.01108；epoch100 +0.01043；positive 233/240，`PROMISING_EARLY` |
| old-commit ProbeA | 138 | 0.37745 / 0.63014 | 0.37745 @138 | 0.37619 / 0.37494 / 0.37178 / 0.36344 | latest +0.01002；late20 +0.01015；epoch100 +0.01159；positive 131/138，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 159 | 0.37935 / 0.62234 | 0.37935 @159 | 0.37827 / 0.37700 / 0.37427 / 0.36601 | control |
| dynamic_singleproj_yoloinit | 128 | 0.36630 / 0.61359 | 0.36630 @128 | 0.36523 / 0.36368 / 0.36021 / 0.35002 | latest +0.00361；late20 +0.00311；epoch100 +0.00041；positive 89/128，`WATCH` |
| dynamic_wo_s_rec_yoloinit | 134 | 0.37056 / 0.61772 | 0.37056 @134 | 0.36932 / 0.36764 / 0.36463 / 0.35595 | latest +0.00447；late20 +0.00406；epoch100 +0.00389；positive 112/134，`WATCH` |
| dynamic_wo_reach_yoloinit | 122 | 0.35555 / 0.60837 | 0.35555 @122 | 0.35446 / 0.35300 / 0.35024 / 0.34156 | latest -0.00347；late20 -0.00360；epoch100 -0.00355；positive 9/122，`LOW_PRIORITY`，已停止 |
| dynamic_plain_yoloinit | 5 | 0.03828 / 0.11824 | 0.05137 @1 | 0.03325 / 0.03325 / 0.03325 / 0.03325 | matched=5；status=`pre100`，早期波动不评价 |

决策：

- 本轮没有新的 100/120 epoch 决策触发；不新增、不停止。
- 4090 `dynamic` 的 latest delta 降到 +0.00995，刚低于 +1 point，但 late20 仍为 +0.01108，按规则仍为 `PROMISING_EARLY`，继续观察是否持续跌破 +1 point。
- 3090 `singleproj` / `wo_s_rec` 继续小正不足 +1 point；`dynamic_plain` 仍只有 5 rows，等待 100 epoch。

### 2026-06-24 20:27 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 96%`；GPU1 `8597/24564 MiB, util 93%`。4 个训练组均在，日志扫描 `bad=[]`。
- `ladd3090-zw1`：GPU0 `7623/24576 MiB, util 92%`；GPU1 `8417/24576 MiB, util 96%`。detonly_control、singleproj、wo_s_rec、dynamic_plain 有效运行；日志扫描 `bad=[]`。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 381 | 0.47819 / 0.73950 | 0.47819 @381 | 0.47755 / 0.47706 / 0.47575 / 0.47117 | - |
| ProbeA | 299 | 0.45382 / 0.71738 | 0.45382 @299 | 0.45270 / 0.45172 / 0.44958 / 0.44273 | latest +0.00306；late20 +0.00293；epoch100 +0.00710；positive 288/299，`WATCH` |
| dynamic | 242 | 0.43434 / 0.69634 | 0.43434 @242 | 0.43371 / 0.43280 / 0.43026 / 0.42295 | latest +0.00944；late20 +0.01091；epoch100 +0.01043；positive 235/242，`PROMISING_EARLY` |
| old-commit ProbeA | 139 | 0.37773 / 0.63096 | 0.37773 @139 | 0.37674 / 0.37551 / 0.37237 / 0.36399 | latest +0.00990；late20 +0.01014；epoch100 +0.01159；positive 132/139，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 161 | 0.37988 / 0.62318 | 0.37997 @160 | 0.37919 / 0.37805 / 0.37537 / 0.36711 | control |
| dynamic_singleproj_yoloinit | 129 | 0.36664 / 0.61380 | 0.36664 @129 | 0.36575 / 0.36423 / 0.36084 / 0.35070 | latest +0.00361；late20 +0.00318；epoch100 +0.00041；positive 90/129，`WATCH` |
| dynamic_wo_s_rec_yoloinit | 135 | 0.37102 / 0.61824 | 0.37102 @135 | 0.36987 / 0.36829 / 0.36525 / 0.35653 | latest +0.00464；late20 +0.00413；epoch100 +0.00389；positive 113/135，`WATCH` |
| dynamic_wo_reach_yoloinit | 122 | 0.35555 / 0.60837 | 0.35555 @122 | 0.35446 / 0.35300 / 0.35024 / 0.34156 | latest -0.00347；late20 -0.00360；epoch100 -0.00355；positive 9/122，`LOW_PRIORITY`，已停止 |
| dynamic_plain_yoloinit | 6 | 0.06256 / 0.16799 | 0.06256 @6 | 0.03549 / 0.03813 / 0.03813 / 0.03813 | matched=6；status=`pre100`，早期波动不评价 |

决策：

- 本轮没有新的 100/120 epoch 决策触发；不新增、不停止。
- 4090 `dynamic` 的 latest delta 连续第二次低于 +1 point（+0.00944），但 late20 仍为 +0.01091，按规则仍保持 `PROMISING_EARLY`。下一轮需要重点看 late20 是否也跌破 +0.010。
- 3090 `singleproj` / `wo_s_rec` 仍是小正；`dynamic_plain` 仍处于 6-row early noise，等待 100 epoch。

### 2026-06-24 20:30 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 92%`；GPU1 `8597/24564 MiB, util 90%`。4 个训练组均在，日志扫描 `bad=[]`。
- `ladd3090-zw1`：GPU0 `7623/24576 MiB, util 98%`；GPU1 `8417/24576 MiB, util 95%`。detonly_control、singleproj、wo_s_rec、dynamic_plain 有效运行；日志扫描 `bad=[]`。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 382 | 0.47836 / 0.73969 | 0.47836 @382 | 0.47781 / 0.47728 / 0.47601 / 0.47147 | - |
| ProbeA | 301 | 0.45400 / 0.71908 | 0.45400 @301 | 0.45343 / 0.45247 / 0.45043 / 0.44362 | latest +0.00270；late20 +0.00300；epoch100 +0.00710；positive 290/301，`WATCH` |
| dynamic | 243 | 0.43498 / 0.69690 | 0.43498 @243 | 0.43405 / 0.43323 / 0.43075 / 0.42342 | latest +0.00932；late20 +0.01084；epoch100 +0.01043；positive 236/243，`PROMISING_EARLY` |
| old-commit ProbeA | 140 | 0.37819 / 0.63248 | 0.37819 @140 | 0.37728 / 0.37599 / 0.37297 / 0.36454 | latest +0.00984；late20 +0.01016；epoch100 +0.01159；positive 133/140，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 163 | 0.38161 / 0.62471 | 0.38161 @163 | 0.38029 / 0.37903 / 0.37646 / 0.36822 | control |
| dynamic_singleproj_yoloinit | 131 | 0.36778 / 0.61460 | 0.36778 @131 | 0.36678 / 0.36543 / 0.36209 / 0.35206 | latest +0.00350；late20 +0.00330；epoch100 +0.00041；positive 92/131，`WATCH` |
| dynamic_wo_s_rec_yoloinit | 137 | 0.37258 / 0.61981 | 0.37258 @137 | 0.37121 / 0.36965 / 0.36655 / 0.35772 | latest +0.00510；late20 +0.00428；epoch100 +0.00389；positive 115/137，`WATCH` |
| dynamic_wo_reach_yoloinit | 122 | 0.35555 / 0.60837 | 0.35555 @122 | 0.35446 / 0.35300 / 0.35024 / 0.34156 | latest -0.00347；late20 -0.00360；epoch100 -0.00355；positive 9/122，`LOW_PRIORITY`，已停止 |
| dynamic_plain_yoloinit | 8 | 0.09258 / 0.23727 | 0.09258 @8 | 0.06301 / 0.04992 / 0.04992 / 0.04992 | matched=8；status=`pre100`，早期波动不评价 |

决策：

- 本轮没有新的 100/120 epoch 决策触发；不新增、不停止。
- 4090 `dynamic` 的 latest delta 继续低于 +1 point（+0.00932），late20 仍为 +0.01084，继续 `PROMISING_EARLY`，但安全余量继续缩小。
- 3090 `wo_s_rec` 小正略升到 latest +0.00510 / late20 +0.00428，但仍不到主线候选阈值；`dynamic_plain` 仍未到 100 epoch。

### 2026-06-24 20:34 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 89%`；GPU1 `8597/24564 MiB, util 81%`。4 个训练组均在，日志扫描 `bad=[]`。
- `ladd3090-zw1`：第一次 ControlMaster 断开，重建后巡检成功。GPU0 `7623/24576 MiB, util 98%`；GPU1 `8417/24576 MiB, util 94%`。detonly_control、singleproj、wo_s_rec、dynamic_plain 有效运行；日志扫描 `bad=[]`。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 385 | 0.47955 / 0.74067 | 0.47955 @385 | 0.47872 / 0.47801 / 0.47682 / 0.47238 | - |
| ProbeA | 303 | 0.45525 / 0.72089 | 0.45525 @303 | 0.45430 / 0.45323 / 0.45126 / 0.44452 | latest +0.00331；late20 +0.00305；epoch100 +0.00710；positive 292/303，`WATCH` |
| dynamic | 245 | 0.43550 / 0.69842 | 0.43550 @245 | 0.43480 / 0.43398 / 0.43167 / 0.42435 | latest +0.00937；late20 +0.01066；epoch100 +0.01043；positive 238/245，`PROMISING_EARLY` |
| old-commit ProbeA | 142 | 0.37939 / 0.63309 | 0.37939 @142 | 0.37834 / 0.37699 / 0.37417 / 0.36567 | latest +0.01027；late20 +0.01022；epoch100 +0.01159；positive 135/142，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 166 | 0.38321 / 0.62709 | 0.38321 @166 | 0.38205 / 0.38062 / 0.37806 / 0.36987 | control |
| dynamic_singleproj_yoloinit | 133 | 0.36863 / 0.61644 | 0.36863 @133 | 0.36776 / 0.36650 / 0.36340 / 0.35338 | latest +0.00300；late20 +0.00340；epoch100 +0.00041；positive 94/133，`WATCH` |
| dynamic_wo_s_rec_yoloinit | 139 | 0.37339 / 0.62026 | 0.37339 @139 | 0.37240 / 0.37086 / 0.36778 / 0.35889 | latest +0.00517；late20 +0.00440；epoch100 +0.00389；positive 117/139，`WATCH` |
| dynamic_wo_reach_yoloinit | 122 | 0.35555 / 0.60837 | 0.35555 @122 | 0.35446 / 0.35300 / 0.35024 / 0.34156 | latest -0.00347；late20 -0.00360；epoch100 -0.00355；positive 9/122，`LOW_PRIORITY`，已停止 |
| dynamic_plain_yoloinit | 10 | 0.11132 / 0.27134 | 0.11132 @10 | 0.08790 / 0.06057 / 0.06057 / 0.06057 | matched=10；latest +0.00721；late20 -0.00010；status=`pre100` |

决策：

- 本轮没有新的 100/120 epoch 决策触发；不新增、不停止。
- 4090 `dynamic` late20 仍守住 +0.01066，但 latest 已连续多轮低于 +1 point，继续重点观察 late20 是否跌破 +0.010。
- 3090 `dynamic_plain` 到 10 rows 后 latest 已转为 +0.00721，但仍是 `pre100`，不能作为正结果；继续等 100 epoch 同源评估。

### 2026-06-24 20:40 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB`；GPU1 `8597/24564 MiB`。4 个训练组均在，日志扫描 `bad=[]`。
- `ladd3090-zw1`：GPU0 `7623/24576 MiB, util 93%`；GPU1 `8417/24576 MiB, util 97%`。detonly_control、singleproj、wo_s_rec、dynamic_plain 有效运行；日志扫描 `bad=[]`。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 390 | 0.48146 / 0.74485 | 0.48146 @390 | 0.48068 / 0.47970 / 0.47826 / 0.47391 | - |
| ProbeA | 307 | 0.45761 / 0.72331 | 0.45761 @307 | 0.45633 / 0.45509 / 0.45299 / 0.44630 | latest +0.00369；late20 +0.00323；epoch100 +0.00710；positive 296/307，`WATCH` |
| dynamic | 249 | 0.43805 / 0.70171 | 0.43805 @249 | 0.43658 / 0.43548 / 0.43349 / 0.42622 | latest +0.00953；late20 +0.01029；epoch100 +0.01043；positive 242/249，`PROMISING_EARLY` |
| old-commit ProbeA | 146 | 0.38130 / 0.63494 | 0.38130 @146 | 0.38038 / 0.37908 / 0.37645 / 0.36787 | latest +0.01062；late20 +0.01030；epoch100 +0.01159；positive 139/146，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 170 | 0.38525 / 0.62905 | 0.38525 @170 | 0.38432 / 0.38285 / 0.38021 / 0.37207 | control |
| dynamic_singleproj_yoloinit | 136 | 0.37041 / 0.61800 | 0.37041 @136 | 0.36950 / 0.36814 / 0.36529 / 0.35534 | latest +0.00328；late20 +0.00358；epoch100 +0.00041；positive 97/136，`WATCH` |
| dynamic_wo_s_rec_yoloinit | 142 | 0.37436 / 0.62283 | 0.37436 @142 | 0.37359 / 0.37240 / 0.36943 / 0.36060 | latest +0.00437；late20 +0.00443；epoch100 +0.00389；positive 120/142，`WATCH` |
| dynamic_wo_reach_yoloinit | 122 | 0.35555 / 0.60837 | 0.35555 @122 | 0.35446 / 0.35300 / 0.35024 / 0.34156 | latest -0.00347；late20 -0.00360；epoch100 -0.00355；positive 9/122，`LOW_PRIORITY`，已停止 |
| dynamic_plain_yoloinit | 13 | 0.12257 / 0.28560 | 0.12257 @13 | 0.11225 / 0.08763 / 0.07389 / 0.07389 | matched=13；latest -0.01185；late20 -0.00135；status=`pre100` |

决策：

- 本轮没有新的 100/120 epoch 决策触发；不新增、不停止。
- 4090 `dynamic` 仍满足 `PROMISING_EARLY`，但 late20 已收窄到 +0.01029，距离 +1 point 阈值只剩约 0.00029，需要下一轮重点观察。
- 4090 `old-commit ProbeA` 仍是早期正信号，latest +0.01062 / late20 +0.01030；由于它是 old-commit/e700 线，只作解释 AutoDL 早期差异的线索，不直接作为正式主线证据。
- 3090 `singleproj` 与 `wo_s_rec` 继续小正但没有达到 +1 point 早筛阈值；`dynamic_plain` 仍远未到 100 epoch，不评价。

### 2026-06-24 20:49 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `7853/24564 MiB, util 93%`；GPU1 `8597/24564 MiB, util 89%`。4 个训练组均在，日志扫描 `bad=[]`。
- `ladd3090-zw1`：GPU0 `7623/24576 MiB, util 94%`；GPU1 `8417/24576 MiB, util 98%`。detonly_control、singleproj、wo_s_rec、dynamic_plain 有效运行；`wo_reach` 已停止，仅保留结果证据。日志扫描 `bad=[]`。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 394 | 0.48259 / 0.74505 | 0.48259 @394 | 0.48218 / 0.48124 / 0.47952 / 0.47515 | - |
| ProbeA | 311 | 0.45961 / 0.72453 | 0.45961 @311 | 0.45872 / 0.45722 / 0.45485 / 0.44815 | latest +0.00401；late20 +0.00341；epoch100 +0.00710；positive 300/311，`WATCH` |
| dynamic | 253 | 0.43920 / 0.70309 | 0.43920 @253 | 0.43861 / 0.43731 / 0.43527 / 0.42809 | latest +0.00912；late20 +0.00980；epoch100 +0.01043；positive 246/253，`WATCH` |
| old-commit ProbeA | 149 | 0.38307 / 0.63597 | 0.38307 @149 | 0.38195 / 0.38068 / 0.37809 / 0.36949 | latest +0.01064；late20 +0.01035；epoch100 +0.01159；positive 142/149，`PROMISING_EARLY` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 175 | 0.38878 / 0.63216 | 0.38878 @175 | 0.38725 / 0.38578 / 0.38294 / 0.37484 | control |
| dynamic_singleproj_yoloinit | 140 | 0.37275 / 0.62188 | 0.37275 @140 | 0.37151 / 0.37024 / 0.36754 / 0.35791 | latest +0.00404；late20 +0.00361；epoch100 +0.00041；positive 101/140，`WATCH` |
| dynamic_wo_s_rec_yoloinit | 147 | 0.37774 / 0.62681 | 0.37774 @147 | 0.37681 / 0.37520 / 0.37242 / 0.36354 | latest +0.00437；late20 +0.00465；epoch100 +0.00389；positive 125/147，`WATCH` |
| dynamic_wo_reach_yoloinit | 122 | 0.35555 / 0.60837 | 0.35555 @122 | 0.35446 / 0.35300 / 0.35024 / 0.34156 | latest -0.00347；late20 -0.00360；epoch100 -0.00355；positive 9/122，`LOW_PRIORITY`，已停止 |
| dynamic_plain_yoloinit | 18 | 0.15326 / 0.32898 | 0.15336 @17 | 0.13911 / 0.12568 / 0.09201 / 0.09201 | matched=18；latest -0.00532；late20 -0.00235；status=`pre100` |

决策：

- 4090 `dynamic` 的 late20 已从 +0.01029 降到 +0.00980，按阈值从 `PROMISING_EARLY` 退回 `WATCH`；它仍为持续正增益，但不再满足 +1 point 早筛线。
- 4090 `old-commit ProbeA` 仍满足 `PROMISING_EARLY`，但继续只作为解释 AutoDL 早期差异的线索。
- 3090 `singleproj` / `wo_s_rec` 保持小正，未接近主线阈值；`dynamic_plain` 仍 pre100，不评价。
- 本轮不新增、不停止；继续等待 `dynamic_plain` 到 100 epoch，并观察 4090 `dynamic` 是否继续收窄。

### 2026-06-24 21:27 CST

服务器状态：

- `ladd4090-zw1`：GPU0 `16476/24564 MiB, util 99%`；GPU1 `16626/24564 MiB, util 99%`。当前每卡 4 条训练，日志扫描 `bad=[]`。
- `ladd3090-zw1`：GPU0 `16037/24576 MiB, util 99%`；GPU1 `16729/24576 MiB, util 99%`。当前每卡 4 条训练，日志扫描 `bad=[]`。

4090 同协议结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs det-only |
|---|---:|---:|---:|---:|---:|
| det-only | 418 | 0.48842 / 0.75229 | 0.48860 @417 | 0.48820 / 0.48733 / 0.48590 / 0.48193 | - |
| ProbeA | 331 | 0.46847 / 0.73380 | 0.46847 @331 | 0.46760 / 0.46654 / 0.46433 / 0.45735 | latest +0.00541；late20 +0.00524；epoch100 +0.00710；positive 320/331，`WATCH` |
| dynamic | 272 | 0.44690 / 0.71018 | 0.44690 @272 | 0.44618 / 0.44537 / 0.44341 / 0.43685 | latest +0.00679；late20 +0.00858；epoch100 +0.01043；positive 265/272，`WATCH` |
| old-commit ProbeA | 167 | 0.39376 / 0.65121 | 0.39376 @167 | 0.39253 / 0.39115 / 0.38824 / 0.37970 | latest +0.01155；late20 +0.01135；epoch100 +0.01159；positive 160/167，`PROMISING_EARLY` |
| dynamic_kd0p5 | 7 | 0.06943 / 0.17722 | 0.07043 @6 | 0.05206 / 0.04797 / 0.04797 / 0.04797 | latest -0.01248；late20 -0.00134；status=`pre100` |
| dynamic_reach0p5 | 7 | 0.09412 / 0.24136 | 0.09412 @7 | 0.04746 / 0.04870 / 0.04870 / 0.04870 | latest +0.01221；late20 -0.00060；status=`pre100` |
| dynamic_srec0p05 | 6 | 0.08514 / 0.22137 | 0.08514 @6 | 0.04439 / 0.04328 / 0.04328 / 0.04328 | latest +0.00923；late20 -0.00059；status=`pre100` |
| dynamic_teacher_projectedraw | 8 | 0.08361 / 0.22859 | 0.08361 @8 | 0.06039 / 0.04845 / 0.04845 / 0.04845 | latest -0.00849；late20 -0.00621；status=`pre100` |

3090 同源组结果：

| run | rows | latest AP50-95/AP50 | best AP50-95 | late5/10/20/50 AP50-95 | vs 3090 det-only |
|---|---:|---:|---:|---:|---:|
| detonly_control_yoloinit | 197 | 0.40221 / 0.64946 | 0.40221 @197 | 0.40042 / 0.39868 / 0.39545 / 0.38700 | control |
| dynamic_singleproj_yoloinit | 157 | 0.38464 / 0.63355 | 0.38464 @157 | 0.38325 / 0.38139 / 0.37793 / 0.36872 | latest +0.00602；late20 +0.00472；epoch100 +0.00041；positive 118/157，`WATCH` |
| dynamic_wo_s_rec_yoloinit | 165 | 0.38928 / 0.64329 | 0.38928 @165 | 0.38762 / 0.38618 / 0.38297 / 0.37412 | latest +0.00675；late20 +0.00543；epoch100 +0.00389；positive 143/165，`WATCH` |
| dynamic_wo_reach_yoloinit | 122 | 0.35555 / 0.60837 | 0.35555 @122 | 0.35446 / 0.35300 / 0.35024 / 0.34156 | latest -0.00347；late20 -0.00360；epoch100 -0.00355；positive 9/122，`LOW_PRIORITY`，已停止 |
| dynamic_plain_yoloinit | 36 | 0.24323 / 0.47062 | 0.24323 @36 | 0.23214 / 0.22170 / 0.19722 / 0.14705 | latest +0.00046；late20 +0.00006；status=`pre100` |
| dynamic_kd2p0 | 7 | 0.07726 / 0.21110 | 0.07726 @7 | 0.05233 / 0.04904 / 0.04904 / 0.04904 | latest -0.00983；late20 +0.00323；status=`pre100` |
| dynamic_corewarm60 | 7 | 0.06730 / 0.18218 | 0.07386 @6 | 0.04631 / 0.04488 / 0.04488 / 0.04488 | latest -0.01979；late20 -0.00092；status=`pre100` |
| dynamic_kd0p25 | 6 | 0.07774 / 0.21437 | 0.07774 @6 | 0.04449 / 0.04061 / 0.04061 / 0.04061 | latest +0.01967；late20 +0.00169；status=`pre100` |
| dynamic_reach_rawinput | 7 | 0.07849 / 0.21155 | 0.07888 @6 | 0.04682 / 0.04069 / 0.04069 / 0.04069 | latest -0.00860；late20 -0.00512；status=`pre100` |

决策：

- 当前 4 张 GPU 均已达到每卡 4 条训练，且 GPU util 约 99-100%。虽然显存只有约 16-17G，但算力、CPU worker 与 I/O 压力已经接近满载，本轮不再追加第 5 条训练。
- 4090 `dynamic` 从早期 +1 point 降到 latest +0.00679 / late20 +0.00858，仍是持续小正但不满足 `PROMISING_EARLY`。
- 3090 `singleproj` 与 `wo_s_rec` 仍是小正，`wo_s_rec` 略好；都没有达到 +1 point 早筛线。
- 旧条件 `old-commit ProbeA` 仍满足 `PROMISING_EARLY`，但它只解释 AutoDL 早期差异，不能直接替代当前 dynamic 主线。
- 新增的 8 条 dynamic sweep 只有 6-8 rows，仍是 `pre100`；下一轮继续观察是否有明显异常，正式筛选等 matched >=100。
