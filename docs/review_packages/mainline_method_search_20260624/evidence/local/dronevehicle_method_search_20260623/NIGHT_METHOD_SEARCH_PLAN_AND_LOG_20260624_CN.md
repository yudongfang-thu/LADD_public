# DroneVehicle 夜间方法搜索目标草案与执行日志

创建时间：2026-06-24 02:58 CST

状态：已于 2026-06-24 03:20 CST 重新写入 active goal。用户在 ChatGPT app 中删除了旧 goal 后，本线程已重新创建正式目标；Heartbeat automation `dronevehicle-night-method-search` 已确认存在且为 `ACTIVE`，15 分钟检查一次当前线程。

正式执行目标：

```text
在 DroneVehicle sub2k full-val 小风洞上寻找稳定的 LADD 主线候选：
目标 A 最多验证 CMDistill-from-YOLO-init、LD-from-YOLO-init、D3T-style 三种跨模态单模态推理方法是否能产生正结果；
目标 B 并行探索更接近 LADD 主线的 YOLO-init/reload 候选方案，优先 shared/private、learnability、reachability、fused shared、student shared distill 机制；
所有实验按本文档判据记录 run 路径、曲线、best/final/late-window、positive/negative/inconclusive/invalid 标签，并遵守服务器资源规则。
```

## 1. 当前问题

DroneVehicle sub2k full-val 被用作 LADD 主线重构的小风洞。现在的核心问题不是继续证明 reload 曲线能不能短期擦线，而是：

1. 先在这个小风洞上找到至少一种真正跨模态蒸馏方法，能从 YOLO-init 或合格 reload 设定下产生正结果，证明风洞本身有正向潜力。
2. 同时探索更接近 LADD 主线的方案，为后续主线替换或重构提供候选机制。

### 1.1 夜间总目标

夜间任务的总目标不是尽可能多地堆实验，而是在 DroneVehicle 小风洞里建立一条可信的判断链：

```text
小风洞是否能产生跨模态单模态推理正结果
  -> 如果能，哪个机制最可能迁移回 LADD 主线
  -> 如果不能，是外部方法都失败，还是当前实现 / 协议 / 数据方向有问题
```

更具体地说，夜间应尽量达成以下目标：

1. 在目标 A 中，最多用三种外部或对比式跨模态蒸馏方法验证风洞是否有正向潜力。
2. 在目标 B 中，并行寻找至少一个更接近 LADD 主线的候选机制，优先从 shared/private、learnability、reachability、fusion shared、student shared distill 这些结构出发。
3. 每个实验都必须有同协议解释，不允许把 reload 高起点、batch fallback、不同 schedule、不同输入模态混成正结果。
4. 明早交付的不是零散口头判断，而是一份可读记录：哪些实验跑了、各自结果是多少、是否满足正结果判据、下一步应该继续还是淘汰。

### 1.2 目标完成状态

夜间搜索结束时，可能出现四种状态：

| 状态 | 含义 | 后续策略 |
|---|---|---|
| 强成功 | 找到一个贴近 LADD 主线的目标 B 方法，并且满足 YOLO-init 或 reload 正结果判据 | 优先围绕该方法做 control、seed、shuffled-pair 和更完整曲线 |
| 弱成功 | 目标 A 中至少一种开源/对比跨模态方法满足正结果，但目标 B 暂未满足 | 证明风洞不是死的；继续把有效机制转译到 LADD-like 主线 |
| 目标 A 暂停 | CMDistill、LD、D3T-style 三者都未产生正结果 | 不再横向找第四个外部方法，集中排查目标 B 和协议/实现 |
| 负向诊断 | 所有有效 run 均失败，且没有任何接近正结果的曲线 | 输出失败证据，判断是数据方向、实现、协议还是跨模态设定本身的问题 |

### 1.3 “正结果对象”必须包含什么

一个可以被称为正结果的对象，必须至少包含：

1. `results.csv` 路径、log 路径、启动命令或脚本路径。
2. 明确初始化类型：YOLO-init / reload / 其他，不得混淆。
3. 明确推理输入：目标 A 必须是单模态 student 推理；需要双模态输入的不算目标 A 正结果。
4. 至少一张曲线：method 与 RGB baseline from scratch、必要的 det-only reload control 画在一起。
5. 至少一个表格：rows、best、best epoch、final/latest、late5、late10、late20、与 baseline final/best 的差值。
6. 判定标签：`positive` / `weak-positive` / `negative` / `inconclusive` / `invalid`。
7. 如果是 reload 正结果，必须报告加载点、最低回落点、恢复后的 peak，以及 peak 是否超过加载点和 RGB baseline final。
8. 如果是 YOLO-init 正结果，必须说明 peak 是否来自真实训练上升，而不是预训练/收敛权重继承。

### 1.4 明早应交付的内容

若夜间自动化被启动，明早应在本文件中留下：

1. 一张总表：目标 A/B 每个 run 的状态、best AP50-95、final/latest AP50-95、判定、下一步。
2. 一个结论段：是否已经证明 DroneVehicle 小风洞能产生跨模态单模态推理正结果。
3. 一个主线建议段：下一轮最值得投入的 LADD-like 方案是什么，为什么。
4. 一个失败段：哪些 run 无效、失败或不值得继续，原因是什么。
5. 所有新增 run 的路径，方便复查日志和曲线。

现有 fixed baseline：

| role | best AP50 | best AP50-95 | final AP50 | final AP50-95 | late20 AP50-95 |
|---|---:|---:|---:|---:|---:|
| RGB student baseline from scratch | 0.56886 | 0.36087 | 0.55255 | 0.35385 | 0.35410 |
| IR teacher baseline from scratch | 0.63800 | 0.43299 | 0.62123 | 0.42480 | - |

固定协议优先使用：

```text
DroneVehicle sub2k seed0 full-val
IR teacher -> RGB student
YOLO11n
imgsz=512
batch=64
epochs=200
mosaic=0.0
close_mosaic=0
mixup=0.1
strict batch size
```

## 2. 正结果判据

### 2.1 YOLO-init / from scratch

从 `yolo11n.pt` 或等价 YOLO 初始化开始训练时，正结果必须满足：

1. 曲线中出现的 peak AP50-95 必须超过 RGB baseline final AP50-95 `0.35385`。
2. 更强目标是超过 RGB baseline best AP50-95 `0.36087`。
3. 这个 peak 不能只是加载了已收敛模型后的高起点，也不能是 epoch 1/2 的继承值一路下降。
4. 允许后期退化；只要训练过程中从 YOLO-init 状态真正涨出高于 baseline 收敛值的凸点，就可以作为方法有正信号。

### 2.2 Reload

从已收敛 RGB baseline 权重 reload 后继续训练时，正结果必须满足：

1. 初期因为重置学习率出现短时下降是允许的。
2. 后续必须涨回来，并且 peak 超过刚加载时的性能点。
3. 同时 peak 必须超过 RGB baseline final AP50-95 `0.35385`。
4. 更强目标是超过 RGB baseline best AP50-95 `0.36087` 和同协议 det-only reload control。
5. 只靠加载点附近的高值、随后单调下降，不算正结果。

### 2.3 额外排除项

以下情况不进入主线正结果：

1. 发生 OOM 后 batch fallback，尤其是 batch64 降到 32/16/8。
2. 只比 det-only 擦线 `1e-4` 量级，但曲线解释依赖 reload 高起点。
3. 方法推理时仍需要双模态输入；目标 A 必须是单模态 student 推理。
4. 普通检测蒸馏在跨模态上历史已负增益的方案，比如 FGD，不再作为优先探索项。

## 3. 目标 A：证明风洞有正向跨模态蒸馏方法

目标 A 的目的不是提出新主线，而是建立一个外部参照：如果成熟的跨模态蒸馏思路在 DroneVehicle 小风洞上能从 YOLO-init 跑出正结果，就说明数据集、协议和方向至少有产出正信号的可能；如果它们都失败，后续主线探索需要更谨慎地怀疑数据方向、实现或评价协议。

目标 A 的成功标准：

1. 方法必须是跨模态蒸馏或跨模态 teacher-student，不能只是多模态融合。
2. 推理阶段必须是单模态 student；需要 RGB+IR 同时输入的方法不合格。
3. 首选 YOLO-init；若使用 reload，必须按第 2.2 节的 reload 正结果判据报告。
4. 至少 AP50-95 peak 超过 RGB baseline final `0.35385`；更强目标是超过 RGB baseline best `0.36087`。
5. 需要和同协议 baseline 曲线放在一起画，不能只报单个 best 数字。

目标 A 的方法数量先限制为三类，避免无限发散：

| 顺序 | 方法 | 当前定位 | 是否已启动 | 通过后动作 | 失败后动作 |
|---:|---|---|---|---|---|
| A1 | CMDistill from YOLO-init | 已有实现，先验证风洞是否能从头跑出正信号 | 已启动 | 作为 sanity-positive 方法记录，不自动升级为主线 | 看 A2 |
| A2 | LD from YOLO-init | 旧对比方法中在 OGSOD n 上有过弱正信号，可能适合快速验证 | 已启动 | 作为 cross-modal positive evidence 记录 | 看 A3 |
| A3 | D3T-style output KD | 开源跨模态单模态推理方法的轻量迁移；优先做 output/logit/box/quality KD，不硬搬 cvpods | 未启动 | 若正向，整理为目标 A 结论 | 若失败，暂停目标 A，转向目标 B |

说明：当前已经实际启动的是 CMDistill-from-YOLO-init 与 LD-from-YOLO-init。若后续需要把 A1/A2 中的某个位置换成 CCLKD-from-YOLO-init，需要用户再次明确确认。

目标 A 暂停条件：

```text
CMDistill from YOLO-init 失败
AND LD from YOLO-init 失败
AND D3T-style output KD 失败
```

满足暂停条件后，不再继续横向找第四个外部跨模态方法，转为集中做目标 B。

## 4. 目标 B：探索更接近 LADD 主线的候选方案

目标 B 是真正服务主线的部分。它不要求必须来自外部开源方法，但应尽量保留 LADD 的核心问题意识：哪些 teacher 信息是 student 可学的，哪些是模态私有或不可迁移的，怎样避免把不可学信息强行蒸馏到 student 里。

目标 B 的成功标准：

1. 结构上优先接近 LADD：shared/private 分解、teacher transferable/private、student shared/private、learnability/reachability、融合 shared target。
2. 结果上必须满足第 2 节的 YOLO-init 或 reload 正结果判据。
3. 解释上必须能说明相对 LADD 原主线改变了什么：例如从强 reach loss 改成 reachability weighting，从单 teacher shared 改成 fused shared，从直接 feature match 改成 gated / selective KD。
4. 工程上必须能落到当前 YOLO11n HBB 训练栈中，不依赖难以维护的大规模外部框架。
5. 每个候选必须有自己的子目录；无效 run、诊断 run、正式 run 分开记录。

目标 B 的优先成果形式：

| 成果类型 | 说明 |
|---|---|
| 可继续主线候选 | 满足正结果判据，并且机制与 LADD 主线足够接近 |
| 机制性线索 | 未完全正向，但显示某个机制有稳定改善窗口，例如 late-window 明显好于 control |
| 反证 | 明确显示某类 LADD-like 设计会引入负迁移，例如强 feature KD、强 reach loss、错误 private 分支 |
| 实现诊断 | 发现 bug、batch fallback、reload confound、BN 污染或不一致 schedule |

目标 B 可以与目标 A 并行。显存和 GPU 利用率允许时，两张 4090 可以同时跑目标 A 和目标 B，但必须避免队列 race、OOM、batch fallback。

目标 B 的偏好顺序：

1. 优先找与既有 LADD 主线更接近的方案：shared/private 分解、learnability / reachability、teacher transferable vs private、student shared/private。
2. YOLO-init 与 reload 都可以接受，但必须分别按第 2 节的正结果判据解释。
3. 可以并行试多个变体，但每个变体必须有独立子目录、独立日志、独立 results.csv，不混入同一个方法目录。
4. 每一步都要写入本文件的执行日志，尤其记录为什么启动、结果如何、是否淘汰、下一步是什么。

优先候选：

| 类别 | 候选 | 理由 | 当前动作 |
|---|---|---|---|
| B1 | reachable / fused shared LADD variant | 最接近用户提出的新主线：双模态 shared 融合、CAP2/reachable 拉近 shared 拉远 private、fusion 蒸馏 student shared | 等目标启动后，在空卡 strict batch64 重启有效实验 |
| B2 | DSN shared-private refine | 已有 S1/S2 代码与早期信号，但此前不是合格正结果；可作为结构参考，不直接声称正向 | 只在严格判据下重看 |
| B3 | reachability as KD weight | 把 learnability 从强结构约束改成 token/sample weighting，保留 LADD 思路但降低负迁移 | 可作为低风险并行变体 |
| B4 | old split A2-only controlled | 90 旧方案 A2 曾出现弱正信号，需要同结构 det-only control 验证 | 等空卡和主队列压力降低 |

## 5. 当前已知运行状态

最后已知快照：2026-06-24 02:49 CST 左右。

| run | GPU | rows | best AP50/AP50-95 | latest AP50/AP50-95 | 估计完成 | 备注 |
|---|---:|---:|---:|---:|---|---|
| CMDistill from YOLO-init | 0 | 31 | 0.43014 / 0.25905 | 0.42287 / 0.24550 | 约 03:29 CST | 从 `yolo11n.pt` 开始，仍在早期爬升段 |
| LD from YOLO-init | 1 | 2 | 0.16299 / 0.08223 | 0.16299 / 0.08223 | 约 04:14 CST | 刚启动，不能判断趋势 |

对应远端目录：

```text
runs_public/dronevehicle_method_search/sub2k_seed0_fullval/cmdistill_style/ir_to_rgb_from_yolo/
runs_public/dronevehicle_method_search/sub2k_seed0_fullval/ld_style/ir_to_rgb_from_yolo/
logs/dronevehicle_method_search/sub2k_seed0_fullval/cmdistill_style/ir_to_rgb_from_yolo/
logs/dronevehicle_method_search/sub2k_seed0_fullval/ld_style/ir_to_rgb_from_yolo/
```

## 6. 可用服务器与资源规则

用户授权的资源边界如下：

| 服务器 | 资源 | 可杀进程规则 | 可加进程规则 | 显存目标 |
|---|---|---|---|---|
| LADD4090 双卡服务器 / `ladd4090-zw1` | 双卡 4090 | 自由度最高，上面的所有进程都可以杀 | 可以自由启动和调度 | 有合适任务时尽量充分利用；推荐占用区间约 15G-21G，不强行塞满，超过 22G 视为危险 |
| LADD3090 双卡服务器 | 双卡 3090 | 已有实验优先级不高，但最好不要杀原有进程 | 可以加新进程分散算力 | 有合适任务时利用空余显存，但避免挤爆已有任务 |
| AutoDL 双卡 4090 服务器 | 双卡 4090 | 原有进程优先级较高，不要杀原有进程；只杀/关自己后续新加的进程 | 可以加新进程，但必须避开原有高优先级任务 | 有合适任务时利用空余显存，不为了填满而叠高风险任务 |

调度原则：

1. 先用 LADD4090 双卡服务器承载主要搜索，因为这台机器允许自由杀进程和重排队列。
2. LADD3090 与 AutoDL 只用于分散算力；除非用户再次明确授权，不杀这两台机器上的原有进程。
3. 每次新发任务前先看 `nvidia-smi`、现有 pid、显存和 util；15G-21G 是“有任务可并行时希望充分利用”的推荐区间，不是必须塞满的硬指标。超过 22G 视为危险，不继续叠任务。
4. LADD4090 上若发现 race、OOM、batch fallback 或无效队列，可以主动杀掉相关进程并重启 strict-batch run。
5. 3090 / AutoDL 上只清理由本轮夜间搜索新增的进程，不清理原有高优先级进程。
6. 若继续叠任务会引入 OOM、batch fallback、队列 race、I/O 争用或结果解释混乱，即使显存仍有空余，也应保持空闲。
7. 示例：如果某卡当前任务占用 16G，虽然低于推荐区间上沿，但预计再加一个任务会把总显存推到 23G，则不应加任务，应保持 16G 当前状态。

## 7. 夜间自动化建议，但尚未启动

建议节奏：

1. 前 2 小时使用 15 分钟 heartbeat，主要防止训练报错、OOM、batch fallback、队列卡住。
2. 之后改为 30 分钟 heartbeat，主要解析 results.csv、更新本日志、决定是否启动下一步。
3. 每次 heartbeat 先检查 GPU 显存和进程，再决定是否发新任务。
4. 若 GPU 接近满载、已有 primary/refine 队列可能 race，禁止追加新队列。

每次 heartbeat 必做：

```text
1. nvidia-smi / 进程 / 显存 / util
2. 检查目标 A 当前 run 的 results.csv 行数、best、latest、late5/late10/late20
3. 检查日志是否有 OOM、batch fallback、NaN、FileNotFound、CUDA assert
4. 按正结果判据判断：positive / negative / inconclusive / invalid
5. 更新本 MD 的执行日志
6. 只有在满足触发条件时，才启动下一方法或目标 B 变体
```

## 8. 执行日志格式

每次动作追加一条：

```text
### YYYY-MM-DD HH:MM CST - 动作标题

- 目标：A / B / 监控 / 文档
- 做了什么：
- 使用资源：
- 结果：
- 判定：positive / negative / inconclusive / invalid / waiting
- 下一步：
```

## 9. 执行日志

### 2026-06-24 08:36 CST - Heartbeat 巡检：核查 AutoDL2 线索并启动 P16 标准 LR YOLO-init 对照

- 目标：B / 监控
- 做了什么：响应用户指出“双卡 AutoDL 似乎有正结果”的线索，只读检查 `autodl-nmb1/autodl2` 上 full DroneVehicle batch16 baseline/LADD 结果；同时检查 `ladd4090-zw1` 上 P15 最新状态。发现 P15 YOLO-init 仍沿用 reload 的 `lr1e-3_nowarmup`，这对 from-scratch 训练不公平，因此扩展 `reachable_fused_shared` 单次脚本，新增标准 LR/warmup 变体 `c0_yoloinit_std` 与 `sum_mlp_cap2_ema_yoloinit_std`，并在 GPU0 启动 P16 成对 control/method。
- 使用资源：`autodl-nmb1` 只读检查，未杀进程、未启动新任务；该机 GPU0/GPU1 约 `12621/8904 MiB`，有原有 OGSOD 任务。`ladd4090-zw1` GPU0 在 P15 control 完成后空出，P16 control 启动 pid=`99627`，run tag `reachable_fused_c0_yoloinit_std_stdlr0p01_warmup3_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_083333`；P16 method 启动 pid=`100153`，run tag `reachable_fused_sum_mlp_cap2_ema_yoloinit_std_stdlr0p01_warmup3_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_083448`。P16 双 run 启动后 GPU0 约 `19209/24564 MiB`，GPU1 约 `19867/24564 MiB`，均低于 22G 危险线。
- 结果：AutoDL2 full DroneVehicle baseline 是 batch16/full-data，不是当前 sub2k/b64 风洞协议。该机 RGB student baseline rows=`200`，best AP50/AP50-95=`0.74778/0.51053`，final=`0.74816/0.51007`；IR teacher baseline best/final AP50-95=`0.56964/0.56889`。AutoDL2 LADD dynprobe B rows=`200`，best epoch=`145`，best AP50/AP50-95=`0.75093/0.50992`，final=`0.74396/0.50374`，late20=`0.50511`，低于同机 RGB baseline best `0.51053` 与 final `0.51007`；dynamic B partial rows=`90`，best/latest AP50-95=`0.50545`，也低于 baseline；wo_reach B rows=`16`，best epoch1 AP50-95=`0.48553` 后下降。P15 low-LR control 已完成 rows=`200`，best AP50/AP50-95=`0.52543/0.32541`，final=`0.52518/0.32522`；P15 low-LR method rows=`186`，best AP50/AP50-95=`0.52866/0.33038`，latest=`0.52630/0.32883`，best 比 low-LR control 高 `+0.00382`，但仍远低于 sub2k RGB baseline final `0.35385`。P16 标准 LR control 已写出 rows=`7`，best/latest AP50/AP50-95=`0.26912/0.15205`；P16 标准 LR method 已写出 rows=`2`，best/latest=`0.13347/0.06336`。P16 日志扫描无 OOM、Traceback、batch fallback、NaN、FileNotFound、CUDA assert。
- 判定：AutoDL2 上“疑似正结果”按同机 full-data baseline 复核后不是正结果，只是 LADD AP50 略高但 AP50-95 未超过 baseline；且它是 full-data/b16 协议，不能直接混入当前 sub2k/b64 风洞结论。P15 low-LR method 只能记为 `weak mechanism clue`，因为 schedule 对 YOLO-init 不公平且未达到 baseline final。P16 是新的 `valid-started` 标准 LR/warmup YOLO-init 对照组，后续才是公平判断 fused-shared 是否有正信号。
- 下一步：继续监控 P16 control/method 到 50/100/150/200 epoch，重点看标准 LR 下 control 能否接近 sub2k RGB baseline、method 是否超过 control 以及是否超过 baseline final `0.35385`。P15 low-LR method 跑满后归档为 schedule-confounded weak clue，不作为主线正结果。

### 2026-06-24 08:20 CST - 用户询问进展：P15 method 有微弱 best 优势但仍远低于 baseline

- 目标：A / B / 监控
- 做了什么：响应用户“目前进展”询问，重新只读检查 `ladd4090-zw1` GPU、P15 两条 YOLO-init run、目标 A 已完成 run、代表性 reload run，以及 P15 日志错误标记。
- 使用资源：`ladd4090-zw1`；GPU0 约 `9807/24564 MiB`、util `44%`，运行 P15 control；GPU1 约 `19281/24564 MiB`、util `64%`，运行 P15 method 与旧 OGSOD RGB baseline。活动计数中 control/method worker 各约 `26`，另有 `3` 个旧 OGSOD queue/watcher。未启动新任务。
- 结果：目标 A 中 CMDistill-from-YOLO-init 已完成，rows=`200`，best epoch=`184`，best AP50/AP50-95=`0.56976/0.36215`，final=`0.56725/0.35933`，late20 AP50-95=`0.35987`，满足 YOLO-init 正结果判据。LD-from-YOLO-init 已完成，rows=`200`，best epoch=`141`，best AP50/AP50-95=`0.56220/0.36098`，final=`0.55153/0.35357`，late20=`0.35468`，只算弱/边缘证据。P15 control rows=`184`，best epoch=`146`，best AP50/AP50-95=`0.52543/0.32541`，latest=`0.52603/0.32284`，late5/late10/late20=`0.32353/0.32342/0.32325`。P15 method rows=`135`，best epoch=`130`，best AP50/AP50-95=`0.52138/0.32675`，latest=`0.51850/0.32131`，late5/late10/late20=`0.32337/0.32332/0.32254`。P15 method 相比 control：epoch25 `+0.01776`，epoch50 `-0.00679`，epoch75 `+0.00178`，epoch100 `+0.00285`，epoch125 `+0.00020`，best `+0.00134` AP50-95；但 method best 距 RGB baseline final `0.35385` 仍差 `-0.02710`，距 baseline best `0.36087` 差 `-0.03412`。DSN S2 与 P14 reload EMA 均已完成且按 reload 判据为 negative。P15 日志扫描无 OOM、Traceback、batch fallback、NaN、FileNotFound、CUDA assert。
- 判定：目标 A 是 `weak success`：至少 CMDistill-from-YOLO-init 证明 DroneVehicle 小风洞能跑出跨模态单模态推理正结果，但 CMDistill 本身不作为主线。目标 B 当前仍无正结果；P15 fused-shared YOLO-init 只有微弱 control 优势，不满足 YOLO-init 正结果。整体进度是“风洞可用已证明，主线候选仍未找到”。
- 下一步：继续等待 P15 control/method 跑满 200 epoch 做 final 判定；若 method final/best 仍低于 baseline final，则把 P15 标为 negative/weak mechanism clue，然后在释放的 GPU0 上启动下一条更接近 LADD 的候选。

### 2026-06-24 08:16 CST - Heartbeat 巡检：P15 method 到 100+，只有微弱对照优势

- 目标：A / B / 监控
- 做了什么：按只读方式检查 `ladd4090-zw1` GPU、P15 训练进程、队列进程、P15 两条 YOLO-init run、目标 A 已完成 run，以及 07:18 启动相关日志错误标记。
- 使用资源：`ladd4090-zw1`；GPU0 约 `9807/24564 MiB`、util `74%`，运行 P15 control；GPU1 约 `19251/24564 MiB`、util `91%`，运行 P15 method 与旧 OGSOD RGB baseline。队列计数中 control/method worker 各约 `26` 个进程，另有 `3` 个旧 OGSOD formal baseline queue/watcher；未见新的 DroneVehicle method-search 队列抢跑。
- 结果：P15 control `c0_yoloinit_nofusion` rows=`171`，best epoch=`146`，best AP50/AP50-95=`0.52543/0.32541`，latest epoch=`171`，latest=`0.52529/0.32316`，late5/late10/late20 AP50-95=`0.32300/0.32284/0.32205`，epoch100=`0.50694/0.31334`，epoch125=`0.52227/0.32324`，epoch150=`0.52256/0.32280`。P15 method `sum_mlp_cap2_ema_yoloinit` rows=`126`，best epoch=`117`，best AP50/AP50-95=`0.52306/0.32472`，latest epoch=`126`，latest=`0.51752/0.32114`，late5/late10/late20 AP50-95=`0.32185/0.32183/0.32136`，epoch100=`0.50633/0.31619`，epoch125=`0.52371/0.32344`。epoch-matched：method-control 在 epoch75 为 `+0.00178`，epoch100 为 `+0.00285`，epoch125 为 `+0.00020` AP50-95；但当前 best 仍低于 control best `0.00069`，且相比 RGB baseline final `0.35385` 分别为 control `-0.02844`、method `-0.02913`。目标 A 状态不变：CMDistill-from-YOLO-init 已完成 positive candidate（best `0.36215`，final `0.35933`）；LD-from-YOLO-init 已完成 weak/marginal（best `0.36098`，final `0.35357`）。日志扫描无 OOM、Traceback、batch fallback、NaN、FileNotFound、CUDA assert。
- 判定：P15 为 `valid-running / negative-leaning mechanism clue`。YOLO-init 方法在 100/125 epoch 与 control 有微弱对齐优势，但幅度太小且整体 best 未超过 control，更没有接近 baseline final，不能称正结果。由于 control 尚未完成、method 仍在中后期，继续保留到 final。
- 下一步：等待 control 完成 200 epoch 后释放 GPU0；下一轮若 GPU0 空出且 P15 method 仍明显低于 baseline final，可启动下一条更接近 LADD 的候选，优先选择不与 P15 method 在 GPU1 上抢显存的单 GPU0 任务。

### 2026-06-24 08:01 CST - Heartbeat 巡检：P15 control 过 100，method 尚未到 100

- 目标：A / B / 监控
- 做了什么：按只读方式检查 `ladd4090-zw1` GPU、P15 训练进程、队列进程、P15 两条 YOLO-init run、目标 A 已完成 run，以及 07:18 启动相关日志错误标记。
- 使用资源：`ladd4090-zw1`；GPU0 约 `9807/24564 MiB`、util `4%`，运行 P15 control；GPU1 约 `19047/24564 MiB`、util `17%`，运行 P15 method 与旧 OGSOD RGB baseline。队列计数中 control/method worker 各约 `26` 个进程，另有 `3` 个旧 OGSOD formal baseline queue/watcher；未见新的 DroneVehicle method-search 队列抢跑。
- 结果：P15 control `c0_yoloinit_nofusion` rows=`120`，best epoch=`118`，best AP50/AP50-95=`0.52044/0.32209`，latest epoch=`120`，latest=`0.51655/0.31878`，late5/late10/late20 AP50-95=`0.31961/0.31794/0.31746`，epoch75=`0.49650/0.30323`，epoch100=`0.50694/0.31334`。P15 method `sum_mlp_cap2_ema_yoloinit` rows=`89`，best epoch=`83`，best AP50/AP50-95=`0.50595/0.31163`，latest epoch=`89`，latest=`0.50197/0.31034`，late5/late10/late20 AP50-95=`0.30549/0.30613/0.30396`，epoch75=`0.49660/0.30501`。epoch-matched：method-control 在 epoch25 为 `+0.01776`，epoch50 为 `-0.00679`，epoch75 为 `+0.00178` AP50-95。当前 best 相比 RGB baseline final `0.35385` 分别为 control `-0.03176`、method `-0.04222`。目标 A 状态不变：CMDistill-from-YOLO-init 已完成 positive candidate（best `0.36215`，final `0.35933`）；LD-from-YOLO-init 已完成 weak/marginal（best `0.36098`，final `0.35357`）。日志扫描无 OOM、Traceback、batch fallback、NaN、FileNotFound、CUDA assert。
- 判定：P15 为 `valid-running / inconclusive but negative-leaning`。method 尚未到 100 epoch，不能做 100 epoch 对齐判断；它在 epoch75 略高于 control，但整体 best 仍低于 control，且二者离 baseline final 仍有明显距离。
- 下一步：继续等 method 到 100 epoch，再看 epoch100 matched delta 与是否接近 baseline final；当前不追加任务，避免干扰两条 from-scratch 曲线中段。

### 2026-06-24 07:46 CST - Heartbeat 巡检：P15 到 50+，method 暂时落后 control

- 目标：A / B / 监控
- 做了什么：按只读方式检查 `ladd4090-zw1` GPU、P15 训练进程、队列进程、P15 两条 YOLO-init run、目标 A 已完成 run，以及 07:18 启动相关日志错误标记。
- 使用资源：`ladd4090-zw1`；GPU0 约 `9807/24564 MiB`、util `6%`，运行 P15 control；GPU1 约 `18931/24564 MiB`、util `23%`，运行 P15 method 与旧 OGSOD RGB baseline。队列进程仅见旧 OGSOD formal baseline queue/watcher，未见新的 DroneVehicle method-search 队列抢跑。
- 结果：P15 control `c0_yoloinit_nofusion` rows=`74`，best/latest epoch=`74`，AP50/AP50-95=`0.49797/0.30422`，late5/late10/late20 AP50-95=`0.29571/0.29636/0.29438`，epoch25=`0.37370/0.21986`，epoch50=`0.46919/0.28454`。P15 method `sum_mlp_cap2_ema_yoloinit` rows=`55`，best/latest epoch=`55`，AP50/AP50-95=`0.47790/0.29483`，late5/late10/late20 AP50-95=`0.28991/0.28469/0.27778`，epoch25=`0.40040/0.23762`，epoch50=`0.45533/0.27775`。epoch-matched：method-control 在 epoch25 为 `+0.01776` AP50-95，但 epoch50 为 `-0.00679`；当前 best 相比 RGB baseline final `0.35385` 分别为 control `-0.04963`、method `-0.05902`。目标 A 状态不变：CMDistill-from-YOLO-init 已完成 positive candidate（best `0.36215`，final `0.35933`）；LD-from-YOLO-init 已完成 weak/marginal（best `0.36098`，final `0.35357`）。日志扫描无 OOM、Traceback、batch fallback、NaN、FileNotFound、CUDA assert。
- 判定：P15 为 `valid-running / negative-leaning but inconclusive`。YOLO-init 起步和 batch 均有效，但 50 epoch 位置 method 已低于同组 control，且二者仍低于 baseline final；暂不称负，需看到 100 epoch 后是否继续追上或反超。
- 下一步：不追加新实验，避免在两条 from-scratch 曲线中段制造 I/O/显存/解释混淆；下一轮重点看 control/method 是否到 100 epoch，以及 method 是否能重新超过 control 并接近 baseline final。

### 2026-06-24 07:31 CST - Heartbeat 巡检：P15 早期继续爬升，尚未到判定区间

- 目标：A / B / 监控
- 做了什么：按只读方式检查 `ladd4090-zw1` GPU、训练进程、队列进程、P15 两条 YOLO-init run、目标 A 已完成 run，以及 07:18 启动相关日志错误标记。
- 使用资源：`ladd4090-zw1`；GPU0 约 `9807/24564 MiB`、util `35%`，运行 P15 control；GPU1 约 `18675/24564 MiB`、util `88%`，运行 P15 method 与旧 OGSOD RGB baseline。未发现新的 DroneVehicle method-search 队列在等待或抢跑；只看到旧 OGSOD formal baseline queue/watcher。
- 结果：P15 control `c0_yoloinit_nofusion` rows=`33`，best/latest epoch=`33`，AP50/AP50-95=`0.43688/0.26436`，late5/late10/late20 AP50-95=`0.25539/0.25022/0.23984`，epoch25=`0.37370/0.21986`。P15 method `sum_mlp_cap2_ema_yoloinit` rows=`25`，best epoch=`22`，best AP50/AP50-95=`0.41010/0.24671`，latest epoch=`25`，latest=`0.40040/0.23762`，late5/late10/late20 AP50-95=`0.24160/0.23694/0.21847`，epoch25=`0.40040/0.23762`。与 RGB baseline final AP50-95 `0.35385` 相比，当前 P15 control best delta=`-0.08949`，method best delta=`-0.10714`；method best 低于 control best `0.01765`。目标 A 状态不变：CMDistill-from-YOLO-init 已完成 positive candidate（best `0.36215`，final `0.35933`），LD-from-YOLO-init 已完成 weak/marginal（best `0.36098`，final `0.35357`）。日志扫描无 OOM、Traceback、batch fallback、NaN、FileNotFound、CUDA assert。
- 判定：P15 为 `valid-running / inconclusive`。两条 YOLO-init 曲线从低 AP 正常爬升，但仍远低于 baseline final，且 method 当前落后 control；尚未到 50 epoch，不做正负结论。由于 GPU1 已到约 18.7G，继续加任务会增加混淆和显存风险，因此不追加新实验。
- 下一步：继续等 P15 到 50 epoch 后做首次 epoch-matched 比较；若 method 到 100+ epoch 仍低于 control 且低于 baseline final，则优先转向下一条更接近 LADD 的 YOLO-init/reload 候选，而不是过早判死。

### 2026-06-24 07:23 CST - Heartbeat 复查：P15 YOLO-init 双线有效爬升

- 目标：B / 监控
- 做了什么：同步本 MD 到 `ladd4090-zw1`；只读检查 GPU、训练进程、P15 `c0_yoloinit_nofusion` control 与 `sum_mlp_cap2_ema_yoloinit` method 的最新 `results.csv`，并扫描 07:18 启动相关日志错误标记。
- 使用资源：`ladd4090-zw1`；GPU0 约 `9791/24564 MiB`，GPU1 约 `18555/24564 MiB`，均低于 22G 危险线。当前 P15 control pid=`88367/88388` 在 GPU0；P15 method pid=`88390/88406` 在 GPU1，另有旧 OGSOD RGB baseline 仍在 GPU1。
- 结果：P15 control rows=`11`，best/latest epoch=`11`，AP50/AP50-95=`0.34300/0.20072`，late5/late10/late20 AP50-95=`0.19088/0.15772/0.14783`。P15 method rows=`8`，best/latest epoch=`8`，AP50/AP50-95=`0.32755/0.18897`，late5/late10/late20 AP50-95=`0.16077/0.12276/0.12276`。日志未发现 OOM、Traceback、batch fallback、NaN。
- 判定：`valid-running / inconclusive`。两条曲线均从低 AP 起步并向上爬升，符合 YOLO-init 观察前提；当前 epoch 仍太早，不能判断正负。method 当前低于 control，但差距可能来自额外模块早期优化滞后，需要看 50/100 epoch。
- 下一步：不追加新任务；继续等 P15 到 50 epoch 后做首次有判别力比较，重点看 method peak 是否超过 baseline final `0.35385`、baseline best `0.36087`，以及是否超过同组 control。

### 2026-06-24 07:20 CST - Heartbeat 巡检：P14/DSN decay 完成判负，启动 YOLO-init fused-shared 对照组

- 目标：B / 监控
- 做了什么：检查 `ladd4090-zw1` GPU、P14 fused-shared EMA、DSN decay 完整结果与日志；确认二者均已完成并释放 GPU 后，读取 B 阶段 reset/split-load 顺序，扩展 `reachable_fused_shared` 单次脚本，新增两个 YOLO-init 变体：`c0_yoloinit_nofusion` 作为 detector-only/split 架构 control，`sum_mlp_cap2_ema_yoloinit` 作为 fused-shared EMA 方法。
- 使用资源：`ladd4090-zw1`；P15 control 启动到 GPU0，run tag `reachable_fused_c0_yoloinit_nofusion_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_071822`，pid=`88367`；P15 method 启动到 GPU1，run tag `reachable_fused_sum_mlp_cap2_ema_yoloinit_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_071822`，pid=`88390`。脚本 `docs/experiments/dronevehicle_method_search_20260623/reachable_fused_shared/launch_single_reachable_fused_shared_20260624.sh` 已同步远端；`yolo11n.pt` 远端存在。
- 结果：P14 `sum_mlp_cap2_ema` 完成 rows=`200`，best epoch=`2`，best AP50/AP50-95=`0.56889/0.36335`，final/latest=`0.53755/0.34408`，late5/late10/late20 AP50-95=`0.34412/0.34479/0.34472`。DSN `w1p0_decay60_160_final0` 完成 rows=`200`，best epoch=`2`，best AP50/AP50-95=`0.56957/0.36435`，final/latest=`0.53710/0.34397`，late5/late10/late20 AP50-95=`0.34403/0.34467/0.34453`。二者日志均无 OOM、Traceback、batch fallback、NaN。YOLO-init 新组 07:20 CST 复查：`c0_yoloinit_nofusion` rows=`4`，best/latest AP50/AP50-95=`0.24038/0.12613`；`sum_mlp_cap2_ema_yoloinit` rows=`3`，best/latest AP50/AP50-95=`0.16596/0.09165`；两者日志无错误标记，复查 GPU0/GPU1 约 `9225/18429 MiB`。
- 判定：P14 与 DSN decay 均为 `negative` by reload criterion，只有 epoch 2 高点，后续未恢复，final/late-window 均低于 baseline final。YOLO-init 组是 `valid-started`：它们从低 AP 开始爬升，不再继承收敛 baseline 高点；后续按 YOLO-init 判据判断，method 必须超过 RGB baseline final `0.35385`，更强是超过 baseline best `0.36087`，并且还要超过同组 `c0_yoloinit_nofusion`。
- 下一步：继续监控 YOLO-init control/method 到 50/100/150/200 epoch；不在这两条刚启动时追加新任务，避免把真正 from-scratch 曲线的早期学习阶段搞混。

### 2026-06-24 07:01 CST - Heartbeat 巡检：P14 到 150+，DSN decay 接近完成，均未恢复

- 目标：B / 监控
- 做了什么：检查 `ladd4090-zw1` GPU、P14 fused-shared EMA、DSN `w1p0_decay60_160_final0`、队列尾部与日志错误标记。
- 使用资源：`ladd4090-zw1`；GPU0 约 `10635/24564 MiB`、util `77%`，运行 P14 `sum_mlp_cap2_ema`；GPU1 约 `17947/24564 MiB`、util `94%`，运行 DSN decay 与旧 OGSOD RGB baseline。两张卡均低于 22G 危险线；已查日志无 OOM、Traceback、batch fallback、NaN。
- 结果：P14 `sum_mlp_cap2_ema` rows=`152`，best epoch=`2`，best AP50/AP50-95=`0.56889/0.36335`，latest=`0.53840/0.34417`，late5/late10/late20 AP50-95=`0.34508/0.34591/0.34585`。DSN `w1p0_decay60_160_final0` rows=`190`，best epoch=`2`，best AP50/AP50-95=`0.56957/0.36435`，latest=`0.53810/0.34498`，late5/late10/late20 AP50-95=`0.34427/0.34440/0.34458`。
- 判定：P14 到 150+ epoch 后没有出现 reload 判据要求的恢复，latest/late-window 已显著低于 baseline final `0.35385`，基本可判 `negative-trending`，等 200 epoch 后归档。DSN decay 已过 decay 关键区间 60-160，190 epoch 仍无恢复，基本为 `negative`；epoch 2 高点不计正结果。
- 下一步：不追加新任务，等待 P14 与 DSN decay 跑满 200 epoch 后做最终归档；若下一轮二者都完成并释放 GPU，再选择下一条更接近 LADD 的候选，而不是在当前后期阶段打断或混入新变量。

### 2026-06-24 06:46 CST - Heartbeat 巡检：P14 与 DSN decay 均有效推进，但仍是回落形态

- 目标：B / 监控
- 做了什么：检查 `ladd4090-zw1` GPU、P14 fused-shared EMA、DSN `w1p0_decay60_160_final0`、队列尾部与日志错误标记。
- 使用资源：`ladd4090-zw1`；GPU0 约 `10595/24564 MiB`、util `77%`，运行 P14 `sum_mlp_cap2_ema`；GPU1 约 `17767/24564 MiB`、util `96%`，运行 DSN decay 与旧 OGSOD RGB baseline。两张卡均低于 22G 危险线，日志无 OOM、Traceback、batch fallback、NaN。
- 结果：P14 `sum_mlp_cap2_ema` rows=`70`，best epoch=`2`，best AP50/AP50-95=`0.56889/0.36335`，latest=`0.54981/0.35131`，late5/late10/late20 AP50-95=`0.35130/0.35021/0.35052`。DSN `w1p0_decay60_160_final0` rows=`107`，best epoch=`2`，best AP50/AP50-95=`0.56957/0.36435`，latest=`0.54094/0.34656`，late5/late10/late20 AP50-95=`0.34642/0.34722/0.34825`。DSN 队列仍在第二个 refine 变体中，尚未完成。
- 判定：P14 当前只能记为 `valid-running / negative-trending`：epoch 2 高点来自 reload early-high，后续未恢复，latest 和 late-window 仍低于 baseline final `0.35385`；需要看到 100/150/final 后才能归档。DSN decay 到 100+ epoch 后没有改善原始 DSN 的 late-collapse，latest/late-window 低于 baseline final，暂为 `negative-trending`；decay schedule 的关键区间是 60-160 epoch，仍需等 160/200 epoch 看是否有后程恢复。
- 下一步：不追加新任务。当前两条有效 run 正在分别占用 GPU0/GPU1，实验解释清晰；下一次 heartbeat 重点看 P14 到 100+ epoch、DSN decay 到 160+ epoch/是否完成，以及是否需要在其中一条释放 GPU 后补发下一个更接近 LADD 的候选。

### 2026-06-24 06:35 CST - Heartbeat 巡检：P11 完成，启动 P14 fused-shared EMA 变体

- 目标：B / 监控
- 做了什么：检查 `ladd4090-zw1` GPU、P11 A2 det-only/reach-KD 完整结果、DSN refine 队列、日志错误标记；P11 队列已完成且 GPU0 完全释放后，扩展 `reachable_fused_shared` 单次启动脚本，新增独立 `sum_mlp_cap2_ema` 变体，并在 GPU0 启动 P14 teacher-EMA target 版本。
- 使用资源：`ladd4090-zw1`；P14 启动前 GPU0 used=`1 MiB`，GPU1 运行 DSN decay 与旧 OGSOD RGB baseline。P14 脚本 `docs/experiments/dronevehicle_method_search_20260623/reachable_fused_shared/launch_single_reachable_fused_shared_20260624.sh` 已新增 `VARIANT=sum_mlp_cap2_ema`，启动 run tag `reachable_fused_sum_mlp_cap2_ema_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_063301`，pid=`81574`，outer log `logs/dronevehicle_method_search/sub2k_seed0_fullval/reachable_fused_shared/sum_mlp_cap2_ema/reachable_fused_sum_mlp_cap2_ema_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_063301_gpu0.outer.log`。
- 结果：P11 `a2_detonly_split_control` 完成 rows=`100`，best epoch=`38`，best AP50/AP50-95=`0.56158/0.36284`，final/latest=`0.55148/0.35463`，late5/late10/late20 AP50-95=`0.35621/0.35591/0.35650`。P11 `a2_reach_kd_lowlr` 完成 rows=`100`，best epoch=`38`，best AP50/AP50-95=`0.56140/0.36261`，final/latest=`0.55165/0.35481`，late5/late10/late20 AP50-95=`0.35645/0.35601/0.35667`。P11 日志无 OOM、Traceback、batch fallback、NaN，队列于 06:29 CST 正常结束。DSN `w0p25_nodecay` 完成 rows=`200`，best epoch=`2`，best AP50/AP50-95=`0.56875/0.36366`，final/latest=`0.53834/0.34516`，late5/late10/late20 AP50-95=`0.34480/0.34561/0.34542`；随后 DSN 队列自动启动 `w1p0_decay60_160_final0`，当前 rows=`43`，best epoch=`2`，best AP50/AP50-95=`0.56957/0.36435`，latest=`0.54992/0.35434`，late5/late10/late20 AP50-95=`0.35582/0.35568/0.35528`。P14 `sum_mlp_cap2_ema` 06:34 CST 复查 rows=`6`，best epoch=`2`，best AP50/AP50-95=`0.56889/0.36335`，latest=`0.55929/0.35593`，late5/late10/late20 AP50-95=`0.35690/0.35746/0.35746`；P14 与 DSN decay 日志均无错误标记，复查 GPU0/GPU1 约 `10215/17627 MiB`。
- 判定：P11 A2 det-only 是有效 weak-positive control，但 P11 reach/KD best `0.36261` 没有超过同结构 det-only best `0.36284`，因此 P11 reach/KD 不满足主线正结果；其 late-window 略高于 det-only（late20 `0.35667` vs `0.35650`）只能记为很弱的稳定性线索，不足以称正。DSN `w0p25_nodecay` 最终为 `negative` by reload criterion，epoch 2 高点后未恢复，final/late-window 低于 baseline final。DSN decay 与 P14 EMA 都是 valid-started，当前 early-high 仍按 reload 高点处理，必须等 100/150/final 和 late-window。
- 下一步：继续监控 DSN decay 到 100/150/200，P14 `sum_mlp_cap2_ema` 到 50/100/150/200；不再追加新任务，因为 GPU0+GPU1 已分别承载 P14 与 DSN decay，显存处于安全但清晰的双线并行状态。

### 2026-06-24 06:16 CST - Heartbeat 巡检：P11 reach/KD 已启动，暂未超过同结构 det-only

- 目标：B / 监控
- 做了什么：检查 `ladd4090-zw1` GPU、P11 A2 det-only/reach-KD、DSN refine `w0p25_nodecay`、队列尾部和日志错误标记。
- 使用资源：`ladd4090-zw1`；GPU0 约 `11833/24564 MiB`、util `35%`，运行 P11 A2 reach/KD；GPU1 约 `17409/24564 MiB`、util `91%`，运行 DSN refine `w0p25_nodecay` 与旧 OGSOD RGB baseline。均低于 22G 危险线。
- 结果：P11 `a2_detonly_split_control` 已完成 rows=`100`，best epoch=`38`，best AP50/AP50-95=`0.56158/0.36284`，final/latest=`0.55148/0.35463`，late5/late10/late20 AP50-95=`0.35621/0.35591/0.35650`，日志无错误标记。P11 `a2_reach_kd_lowlr` 已启动，当前 rows=`31`，best epoch=`30`，best AP50/AP50-95=`0.56386/0.36100`，latest=`0.55634/0.35631`，late5/late10/late20 AP50-95=`0.35758/0.35496/0.35445`，日志无错误标记。DSN refine `w0p25_nodecay` rows=`146`，best epoch=`2`，best AP50/AP50-95=`0.56875/0.36366`，latest=`0.54095/0.34685`，late5/late10/late20 AP50-95=`0.34720/0.34634/0.34618`，日志无错误标记；DSN decay 变体尚未启动，因为 `w0p25_nodecay` 仍未完成。
- 判定：P11 det-only control 的弱正信号仍成立，但它是 control。P11 reach/KD 当前 best AP50-95 `0.36100` 低于同结构 det-only best `0.36284`，也低于全局 det-only reload best `0.36279`，暂为 `inconclusive/negative-vs-control`；需要等 100 rows/final 判断是否后程反超。DSN `w0p25_nodecay` 到 146 epoch 后已经明显回落，当前 late-window 低于 baseline final，若最终不恢复则应判 negative；epoch 2 高点仍不能按 reload 判据算正结果。
- 下一步：不追加新任务。GPU0 虽然只有约 11.8G，但 P11 队列还在跑关键 reach/KD 对照，且 GPU1 的 DSN refine 队列后续会自动发 `w1p0_decay60_160_final0`；保持当前两条主线候选清晰推进。下一次 heartbeat 重点看 P11 reach/KD 到 70-100 rows、DSN `w0p25_nodecay` 是否完成并自动发 decay 变体。

### 2026-06-24 06:01 CST - Heartbeat 巡检：P8 完成判负，P11 A2 det-only 出现弱正 control 信号

- 目标：B / 监控
- 做了什么：检查 `ladd4090-zw1` GPU、P8 raw KD late-decay 最终结果、P11 oldsplit A2 det-only control、DSN refine `w0p25_nodecay`、队列尾部与日志错误标记。
- 使用资源：`ladd4090-zw1`；GPU0 约 `12341/24564 MiB`、util `77%`，主要运行 P11 A2 det-only control；GPU1 约 `17235/24564 MiB`、util `81%`，运行 DSN refine `w0p25_nodecay` 与旧 OGSOD RGB baseline。两张卡均低于 22G 危险线。
- 结果：P8 `rawkd_late_decay` 已完成 rows=`200`，best epoch=`2`，best AP50/AP50-95=`0.56856/0.36112`，final/latest=`0.53915/0.34467`，late5/late10/late20 AP50-95=`0.34407/0.34471/0.34429`，日志无 OOM、Traceback、batch fallback、NaN。P11 A1 `a1_shared_init` rows=`50`，AP50/AP50-95 保持 `0.56886/0.36087`；P11 `a2_detonly_split_control` rows=`63`，best epoch=`38`，best AP50/AP50-95=`0.56158/0.36284`，latest=`0.55330/0.35431`，late5/late10/late20 AP50-95=`0.35542/0.35655/0.35642`，日志无错误标记。DSN refine `w0p25_nodecay` rows=`66`，best epoch=`2`，best AP50/AP50-95=`0.56875/0.36366`，latest=`0.54945/0.35096`，late5/late10/late20 AP50-95=`0.34785/0.34994/0.35039`，日志无错误标记。
- 判定：P8 最终为 `negative`，因为它只有 epoch 2 reload 高点，后续 final/late-window 明显低于 baseline final `0.35385`，也没有恢复超过加载点。P11 A2 det-only control 出现一个值得记录的 `weak-positive control signal`：best AP50-95 `0.36284` 高于 A1 加载点 / RGB baseline best `0.36087` 约 `+0.00197`，高于 RGB baseline final `0.35385` 约 `+0.00899`，也略高于全局 low-LR det-only reload best `0.36279` 约 `+0.00005`；但它是同结构 det-only control，不是 reach/KD 方法效果，后续 P11 A2 reach/KD 必须超过它才有主线意义。DSN refine `w0p25_nodecay` 当前仍是 reload early-high 后回落，best 不能提前算正结果，latest/late-window 已低于 baseline final。
- 下一步：不追加新任务。虽然 P8 完成后 GPU0 降到约 12.3G，但 P11 队列即将完成 A2 det-only 并启动 A2 reach/KD，强行叠加新任务会增加 queue race 和解释混乱风险；GPU1 已有 DSN refine，显存约 17.2G。下一次 heartbeat 重点检查 P11 A2 det-only 是否跑满 100 并启动 reach/KD、DSN `w0p25_nodecay` 是否到 100+ epoch，以及是否出现 OOM/fallback。

### 2026-06-24 05:50 CST - Heartbeat 巡检：P11 进入 A2 det-only，补发 DSN refine 到 GPU1

- 目标：B / 监控
- 做了什么：检查 `ladd4090-zw1` GPU、P8 raw KD late-decay、P11 oldsplit A2-only controlled、DSN S2/refine 队列状态；发现旧 DSN refine 等待队列已不在运行，而 primary controls 与 DSN S2 均已满 100/200 rows，且 GPU1 低于 `8000 MiB`，因此补发 DSN refine 队列到 GPU1。
- 使用资源：`ladd4090-zw1`；补发前 GPU0 约 `21357/24564 MiB`，GPU1 约 `7227/24564 MiB`；补发后 GPU0 约 `21797/24564 MiB`，GPU1 约 `16529/24564 MiB`。DSN refine 队列 pid=`74304`，日志 `logs/dronevehicle_method_search/sub2k_seed0_fullval/dsn_shared_private/s2_refine_variants/queue_dsn_refine_variants_after_primary_20260624_0548.log`；首个 run `w0p25_nodecay` pid=`74327`，outer log `logs/dronevehicle_method_search/sub2k_seed0_fullval/dsn_shared_private/s2_refine_variants/w0p25_nodecay/dsn_s2_w0p25_nodecay_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_054826_gpu1.outer.log`。
- 结果：P8 `rawkd_late_decay` rows=`169`，best epoch=`2`，best AP50/AP50-95=`0.56856/0.36112`，latest=`0.53989/0.34470`，late5/late10/late20 AP50-95=`0.34412/0.34437/0.34417`，日志无 OOM、Traceback、batch fallback、NaN。P11 A1 `a1_shared_init` 已完成 rows=`50`，AP50/AP50-95 固定为 `0.56886/0.36087`；P11 已进入 `a2_detonly_split_control`，当前 rows=`22`，best epoch=`8`，best AP50/AP50-95=`0.56246/0.35981`，latest=`0.54717/0.35014`，late5/late10/late20 AP50-95=`0.35323/0.35360/0.35512`，日志无错误标记。DSN 原始 S2 已完成 rows=`200`，best epoch=`2`，best AP50/AP50-95=`0.56957/0.36435`，final/latest=`0.53861/0.34521`，late5/late10/late20 AP50-95=`0.34504/0.34566/0.34536`，按 reload 判据不能算正结果。新 DSN refine `w0p25_nodecay` 已有效启动到 GPU1，rows=`6`，best epoch=`2`，best AP50/AP50-95=`0.56875/0.36366`，latest=`0.56047/0.35749`，late5/late10/late20 AP50-95=`0.35752/0.35829/0.35829`；日志中只出现 `strict-batch` 参数文本，无 OOM/fallback/Traceback。
- 判定：P8 到 169 epoch 后 late-window 仍低于 baseline final，基本为 `negative-trending`，但等 200 epoch 后再归档。P11 当前 A2 det-only 是同结构 control，不是方法正结果；它的 best `0.35981` 低于 baseline best，后续要用它和 A2 reach/KD 分支比较。DSN 原始 S2 是 `negative` by reload criterion；新 `w0p25_nodecay` 只是 valid-started，epoch 2 高点仍按 reload 高起点处理，必须看后续是否能恢复并超过加载点 / baseline final / det-only reload control。
- 下一步：不再追加任务。GPU0 已约 21.8G，接近 22G 危险线；GPU1 加 DSN refine 后约 16.5G，处于安全区间。下一次 heartbeat 重点看 P8 是否完成 200、P11 A2 det-only 是否到 50+、DSN `w0p25_nodecay` 是否到 50+；若 P8 完成且释放 GPU0，仍需先确认 P11/DSN 队列不会发生抢卡或连发 race。

### 2026-06-24 05:36 CST - Heartbeat 巡检：P10/P7 完成判负，P11 oldsplit A2-only 启动

- 目标：B / 监控
- 做了什么：检查 `ladd4090-zw1` GPU、P10 reachability-weighted KD、P7 teacher-confidence gate、P8 raw KD late-decay、P11 oldsplit A2-only controlled 队列与日志错误标记。
- 使用资源：`ladd4090-zw1`；GPU0 约 `20579/24564 MiB`、util `91%`，运行 P8 `rawkd_late_decay` 与 P11 `oldsplit_a2only` A1；GPU1 约 `7079/24564 MiB`、util `11%`，运行旧 OGSOD RGB baseline。P11 队列 pid=`71150`，队列日志 `logs/dronevehicle_method_search/sub2k_seed0_fullval/oldsplit_a2only_controlled/queue_oldsplit_a2only_after_controls_20260624_0532.log`。
- 结果：P10 `splitkd_unweighted` rows=`200`，best epoch=`1`，best AP50/AP50-95=`0.56867/0.36186`，final/latest=`0.53847/0.34502`，late5/late10/late20 AP50-95=`0.34441/0.34499/0.34488`。P10 `reachgap_weighted` rows=`200`，best epoch=`1`，best AP50/AP50-95=`0.56888/0.36215`，final/latest=`0.53797/0.34450`，late5/late10/late20 AP50-95=`0.34411/0.34472/0.34468`。P7 `teacher_conf_gate` rows=`200`，best epoch=`1`，best AP50/AP50-95=`0.56768/0.36119`，final/latest=`0.53965/0.34400`，late5/late10/late20 AP50-95=`0.34376/0.34441/0.34408`。P8 `rawkd_late_decay` rows=`109`，best epoch=`2`，best AP50/AP50-95=`0.56856/0.36112`，latest=`0.53565/0.34118`，late5/late10/late20 AP50-95=`0.34419/0.34553/0.34644`。P11 队列在 05:31 CST 启动 `oldsplit_a2only_lowlr1e3_nowarmup_ir2rgb_yolo11n_b64_s0_20260624_053127`，但因为当时 GPU0 已低于阈值，实际落在 GPU0 而不是 GPU1；A1 `a1_shared_init` rows=`16`，best/latest AP50/AP50-95=`0.56886/0.36087`，late-window 均为 `0.36087`。已查 P8 与 P11 A1 日志均无 OOM、Traceback、batch fallback、NaN 标记。
- 判定：P10 `reachgap_weighted` 与其 split control 均为 `negative`，二者都是 reload epoch 1 高点后下降，reachgap final/late-window 还略低于 split control，不满足恢复并超过加载点 / baseline final 的 reload 正结果标准。P7 `teacher_conf_gate` 为 `negative`，late-window 低于 baseline final。P8 仍在运行，但 100+ epoch 后 late-window 已低于 baseline final，暂为 `inconclusive/negative-trending`。P11 A1 只是 oldsplit 的 shared init 阶段，AP 保持加载 baseline 水平，不作为正结果；真正需要看后续 A2 det-only split control 与 A2 reach-KD 的差异。
- 下一步：不追加新任务。GPU0 已约 20.6G，虽然低于 22G 危险线，但 P8 与 P11 同卡运行，继续叠任务会增加 OOM/race 和解释混乱风险；GPU1 仍有空余但本轮先保持静默。下一次 heartbeat 重点检查 P8 是否到 150/200 epoch、P11 A1 是否完成并进入 A2 det-only split control；如果 P11 的 GPU 选择继续不理想但显存仍安全，先不干预。

### 2026-06-24 05:16 CST - Heartbeat 巡检：P10 split 完成，P8 raw KD late-decay 自动发车

- 目标：B / 监控
- 做了什么：检查 `ladd4090-zw1` GPU、P10 split/reachgap、P7 teacher_conf、P8 rawkd_late_decay 与两个队列日志。
- 使用资源：`ladd4090-zw1`；GPU0 运行 P10 reachgap + P8 rawkd_late_decay；GPU1 运行 P7 teacher_conf + 旧 OGSOD RGB baseline。
- 结果：GPU0 约 `19559/24564 MiB`、util `97%`，GPU1 约 `16759/24564 MiB`、util `94%`。P10 split rows=`200`，best epoch=`1`，best AP50/AP50-95=`0.56867/0.36186`，final/latest=`0.53847/0.34502`，late5/late10/late20 AP50-95=`0.34441/0.34499/0.34488`，日志无错误。P10 reachgap rows=`174`，best epoch=`1`，best AP50/AP50-95=`0.56888/0.36215`，latest=`0.53776/0.34382`，late5/late10/late20 AP50-95=`0.34477/0.34475/0.34483`，日志无错误。P7 teacher_conf rows=`128`，best epoch=`1`，best AP50/AP50-95=`0.56768/0.36119`，latest=`0.54218/0.34658`，late5/late10/late20 AP50-95=`0.34664/0.34598/0.34568`，日志无错误。P7/P8 队列在 05:13 CST 检测到 GPU0 低于 `15000 MiB` 后自动启动 `rawkd_late_decay`，run tag `rawkd_late_decay_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_051317`，pid=`68211`；05:16 CST 复查 rows=`8`，best epoch=`2`，best AP50/AP50-95=`0.56856/0.36112`，latest=`0.55515/0.35151`，late5/late10/late20 AP50-95=`0.35029/0.35348/0.35348`，日志无错误。
- 判定：P10 split 已完成且为 negative；P10 reachgap 的 early-high 优势已消失，late20 `0.34483` 与 split `0.34488` 基本持平且均低于 baseline final，若最终不回升则 P10 也应判 negative。P7 teacher_conf 中期 late-window 略高于 B1/P10，但仍低于 baseline final，不可称正结果。P8 rawkd_late_decay 已有效启动，需等 60-160 epoch 的 decay 生效后再判断是否能改善 late-window。
- 下一步：不追加新任务；继续监控 P10 reachgap 到 200 epoch、P7 teacher_conf 到 200 epoch、P8 rawkd_late_decay 到 100/150/200 epoch。若 P10 reachgap 完成且 negative，下一轮可把 P10 作为失败候选归档；P8 是当前仍需等待的稳定化组件。

### 2026-06-24 05:01 CST - Heartbeat 巡检：P10 优势收窄，P8 队列继续等待空卡

- 目标：B / 监控
- 做了什么：检查 `ladd4090-zw1` GPU、compute app、B1 三变体最终结果、P10 split/reachgap、P7 teacher_conf 与 P7/P8 队列日志。
- 使用资源：`ladd4090-zw1`；GPU0 运行 P10 split + P10 reachgap；GPU1 运行 P7 teacher_conf + 旧 OGSOD RGB baseline。
- 结果：GPU0 约 `19847/24564 MiB`、util `98%`，GPU1 约 `16595/24564 MiB`、util `94%`。B1 三变体均已完成且日志无错误：c0 final AP50/AP50-95=`0.53773/0.34500`、late20=`0.34487`；sum final=`0.54409/0.34883`、late20=`0.34851`；concat final=`0.53886/0.34434`、late20=`0.34507`。P10 split rows=`156`，best epoch=`1`，best AP50/AP50-95=`0.56867/0.36186`，latest=`0.53635/0.34451`，late5/late10/late20 AP50-95=`0.34484/0.34551/0.34619`，日志无错误。P10 reachgap rows=`116`，best epoch=`1`，best AP50/AP50-95=`0.56888/0.36215`，latest=`0.54526/0.34858`，late5/late10/late20 AP50-95=`0.34674/0.34611/0.34691`，日志无错误。P7 teacher_conf rows=`55`，best epoch=`1`，best AP50/AP50-95=`0.56768/0.36119`，latest=`0.55942/0.35823`，late5/late10/late20 AP50-95=`0.35106/0.35035/0.35206`，日志无错误。P7/P8 队列仍在等待 GPU memory `<15000 MiB`，尚未启动 `rawkd_late_decay`。
- 判定：B1 彻底 negative：三个变体均是 reload early-high 后滑落，final/late-window 低于 baseline final/best。P10 reachgap 当前仍略高于 split 的 late20（`0.34691` vs `0.34619`），但优势已从 04:50 的约 `+0.0034` 收窄到约 `+0.0007`，且两者都低于 baseline final，不可称正结果。P7 teacher_conf 的 latest 较高但 late-window 没有恢复到 epoch1 加载点，更不能按 reload 判据算正结果。当前显存安全但不适合继续叠任务。
- 下一步：不追加新任务；继续等 P10 split/reachgap 跑满 200 epoch，观察 reachgap 是否能重新拉开 late-window。继续等 P7/P8 队列在空卡时自动启动 `rawkd_late_decay`；若下一轮 GPU 仍未低于 15G，则保持等待，不强行发车。

### 2026-06-24 04:50 CST - Heartbeat 巡检：B1 concat 完成，启动 P7 teacher-confidence gate

- 目标：B / 监控
- 做了什么：检查 `ladd4090-zw1` GPU、B1 concat、P10 split/reachgap 曲线与日志；等待 B1 concat 从 epoch 195 跑满 200 并释放 GPU1 后，读取 P7/P8 队列设计，确认其只会先发 `teacher_conf_gate`，第二个 `rawkd_late_decay` 需要继续等待空卡；随后在 GPU1 安全窗口启动 P7/P8 队列。
- 使用资源：`ladd4090-zw1`；GPU0 运行 P10 split + P10 reachgap；GPU1 从 B1 concat + 旧 OGSOD baseline 切换为 P7 `teacher_conf_gate` + 旧 OGSOD baseline。P7 队列 pid=`64234`，队列日志 `logs/dronevehicle_method_search/sub2k_seed0_fullval/teacher_conf_gated_kd/queue_teacher_conf_and_late_decay_after_primary_20260624_0449.log`；首个 run tag `teacher_conf_gate_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_044914`，pid=`64255`。
- 结果：04:46 CST 时 GPU0 约 `19847/24564 MiB`、GPU1 约 `17629/24564 MiB`，B1 concat rows=`195`，best epoch=`1`，best AP50/AP50-95=`0.56744/0.36116`，latest=`0.54070/0.34649`，late5/late10/late20 AP50-95=`0.34585/0.34530/0.34542`。04:48 CST，B1 concat 完成 rows=`200`，best epoch=`1`，best AP50/AP50-95=`0.56744/0.36116`，final/latest=`0.53886/0.34434`，late5/late10/late20 AP50-95=`0.34476/0.34530/0.34507`，GPU1 降到约 `6519 MiB`。同一时刻 P10 split rows=`110`，best epoch=`1`，best AP50/AP50-95=`0.56867/0.36186`，latest=`0.54468/0.34923`，late5/late10/late20 AP50-95=`0.34578/0.34620/0.34749`；P10 reachgap rows=`70`，best epoch=`1`，best AP50/AP50-95=`0.56888/0.36215`，latest=`0.55071/0.35251`，late5/late10/late20 AP50-95=`0.35222/0.35056/0.35090`。P7 `teacher_conf_gate` 于 04:49 CST 发车到 GPU1，04:50 CST 复查 GPU0/GPU1 约 `19847/16021 MiB`；teacher_conf rows=`5`，best epoch=`1`，best AP50/AP50-95=`0.56768/0.36119`，latest=`0.54412/0.34255`，late5 AP50-95=`0.35377`，日志无 OOM、Traceback、batch fallback、NaN 标记。P7/P8 队列会在 180 秒后继续等待 GPU memory `<15000 MiB` 才发 `rawkd_late_decay`。
- 判定：B1 三变体全部完成或接近完成后的趋势已清楚：c0/sum/concat 都是 reload early-high 后滑落，均不满足正结果标准；sum 的 late-window 最高但仍低于 baseline final/best。P10 reachgap 在 70 epoch 时 late-window 仍高于 split（late20 `0.35090` vs `0.34749`），是当前较有价值的机制线索，但还没有满足 reload 正结果标准。P7 teacher_conf 已有效启动，早期高点不计正结果。
- 下一步：不追加新任务；等待 P7/P8 队列是否在 GPU 释放后自动启动 `rawkd_late_decay`。继续监控 P10 split/reachgap 到 100/150/200 epoch、P7 teacher_conf 到 50/100 epoch；若 P10 reachgap 后续 late-window 继续高于 split，再重点比较 det-only/rawKD 同协议 control。

### 2026-06-24 04:31 CST - Heartbeat 巡检：B1 sum 完成，P10 reachgap 自动发车

- 目标：B / 监控
- 做了什么：检查 `ladd4090-zw1` GPU、compute app、B1 三变体、P10 split/reachgap、P10 队列日志与错误标记。
- 使用资源：`ladd4090-zw1`；GPU0 运行 P10 `splitkd_unweighted` + P10 `reachgap_weighted`；GPU1 运行 B1 `concat_mlp_cap2` + 旧 OGSOD RGB baseline。
- 结果：GPU0 约 `19469/24564 MiB`、util `53%`，GPU1 约 `16845/24564 MiB`、util `94%`。B1 c0 rows=`200`，best epoch=`2`，best AP50/AP50-95=`0.56869/0.36331`，final/latest=`0.53773/0.34500`，late5/late10/late20 AP50-95=`0.34461/0.34504/0.34487`。B1 sum rows=`200`，best epoch=`1`，best AP50/AP50-95=`0.56738/0.36138`，final/latest=`0.54409/0.34883`，late5/late10/late20 AP50-95=`0.34824/0.34873/0.34851`。B1 concat rows=`128`，best epoch=`1`，best AP50/AP50-95=`0.56744/0.36116`，latest=`0.54085/0.34616`，late5/late10/late20 AP50-95=`0.34753/0.34740/0.34726`。P10 split rows=`44`，best epoch=`1`，best AP50/AP50-95=`0.56867/0.36186`，latest=`0.55161/0.35348`，late5/late10/late20 AP50-95=`0.35612/0.35624/0.35533`。P10 队列在 04:29 CST 发现 GPU0 低于 `15000 MiB` 后自动启动 `reachgap_weighted`，run tag `reachgap_weighted_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_042935`，pid=`61088`；04:31 CST 时 rows=`4`，best epoch=`1`，best AP50/AP50-95=`0.56888/0.36215`，latest=`0.55477/0.35573`，late5/late10/late20 AP50-95=`0.35930/0.35930/0.35930`。已查日志均无 OOM、Traceback、batch fallback、NaN 标记。
- 判定：B1 sum 与 c0 均完成，二者都呈 reload 后 early high、后期下滑，未满足正结果标准；sum final/late-window 比 c0 高约 `+0.0035 AP50-95`，可作为机制线索但不是主线正结果。B1 concat 正在下滑，暂不判断。P10 reachgap 已有效启动；其 epoch 1 高点同样不能按 reload 判据称正结果，需要看后续是否能回升并超过加载点、det-only/rawKD/split control。
- 下一步：不追加新任务，因为 GPU0/GPU1 分别约 19.5G/16.8G 且正在跑有效实验。继续监控 B1 concat 到 200 epoch、P10 split/reachgap 到 50/100/150/200 epoch；下一次 heartbeat 若 reachgap 仍明显优于 split 的 late-window，再记录为 P10 候选线索。

### 2026-06-24 04:21 CST - 启动 P10 reachability-weighted KD 队列的首个 control

- 目标：B
- 做了什么：B1 c0 跑满 200 epoch 后 GPU0 释放到约 `10583 MiB`；读取 P10 `reachability_weighted_kd` 设计与队列脚本，确认其会先启动 `splitkd_unweighted`，再等待 180 秒且等任一 GPU 低于 `15000 MiB` 才启动 `reachgap_weighted`。因此在 GPU0 安全余量内启动 P10 队列，用于验证“可达性只做 KD token 权重，而不做强 reach loss”的 LADD-like 主线候选。
- 使用资源：`ladd4090-zw1` GPU0；脚本 `docs/experiments/dronevehicle_method_search_20260623/reachability_weighted_kd/queue_reachability_weighted_after_controls_20260624.sh`；队列日志 `logs/dronevehicle_method_search/sub2k_seed0_fullval/reachability_weighted_kd/queue_reachability_weighted_after_controls_20260624_0419.log`；首个 run tag `splitkd_unweighted_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_041934`，pid=`59253`。
- 结果：P10 队列确认 low-LR det/raw/CMDistill controls 均已满足 `MIN_ROWS=20`；首个 `splitkd_unweighted` 于 04:19 CST 发车，启动后 GPU0 约 `19939-20035/24564 MiB`，低于 22G 危险线。04:21 CST 复查，`splitkd_unweighted` rows=`6`，last epoch=`6`，日志无 OOM、Traceback、batch fallback、NaN 标记。队列进程仍在等待第二个 `reachgap_weighted`，只有当 GPU 显存低于 `15000 MiB` 时才会启动。
- 判定：`splitkd_unweighted` 为 `valid-started`；`reachgap_weighted` 尚未启动。当前 GPU0 约 20.0G，GPU1 约 16.7G，均不追加新任务。P10 的正结果必须等 `reachgap_weighted` 完成后同时超过 det-only reload、raw KD、以及本 `splitkd_unweighted` control，不能提前判断。
- 下一步：继续监控 B1 sum/concat 与 P10 split；如果 P10 队列自动发起 `reachgap_weighted`，记录 pid、run path 和启动时 GPU 显存；若队列没有自动发起但 GPU 释放，可手动补发 `reachgap_weighted`。

### 2026-06-24 04:18 CST - B1 c0 control 完成

- 目标：B / 监控
- 做了什么：等待 B1 `c0_nofusion_splitrec` 从 epoch 199 跑满 200，确认最终 control 曲线与 GPU0 释放情况。
- 使用资源：`ladd4090-zw1` GPU0；run `reachable_fused_c0_nofusion_splitrec_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_032714_b`。
- 结果：B1 c0 rows=`200`，best epoch=`2`，best AP50/AP50-95=`0.56869/0.36331`，final/latest=`0.53773/0.34500`，late5/late10/late20 AP50-95=`0.34461/0.34504/0.34487`。GPU0 从约 20.4G 降到约 `10583/24564 MiB`，说明 c0 进程已释放，只剩 B1 sum 占用约 10.5G。
- 判定：这是有效 strict-batch control，但按 reload 判据不是正结果：高点来自 epoch 2，后续没有恢复并超过加载点，final/late-window 明显回落。
- 下一步：在 GPU0 安全余量内考虑启动一个新的 LADD-like 主线候选，优先 P10 reachability-weighted KD；不再启动 B1 额外变体。

### 2026-06-24 04:15 CST - Heartbeat 巡检：B1 三变体均有效，暂不追加新任务

- 目标：A / B / 监控
- 做了什么：检查 `ladd4090-zw1` GPU、compute app、目标 A 完成曲线、B1 c0/sum/concat 的 `results.csv` 与错误日志。
- 使用资源：`ladd4090-zw1`；GPU0 运行 B1 c0 + B1 sum；GPU1 运行 B1 concat + 旧 OGSOD RGB baseline。
- 结果：GPU0 约 `20385/24564 MiB`、util `99%`，GPU1 约 `16643/24564 MiB`、util `85%`。CMDistill-from-YOLO-init rows=`200`，best epoch=`184`，best AP50/AP50-95=`0.56976/0.36215`，final/latest=`0.56725/0.35933`，late5/late10/late20 AP50-95=`0.35928/0.35959/0.35987`，日志无错误标记。LD-from-YOLO-init rows=`200`，best epoch=`141`，best AP50/AP50-95=`0.56220/0.36098`，final/latest=`0.55153/0.35357`，late5/late10/late20 AP50-95=`0.35457/0.35479/0.35468`，日志无错误标记。B1 c0 rows=`194`，best epoch=`2`，best AP50/AP50-95=`0.56869/0.36331`，latest=`0.53952/0.34551`，late5/late10/late20 AP50-95=`0.34531/0.34486/0.34509`，日志无错误标记。B1 sum rows=`154`，best epoch=`1`，best AP50/AP50-95=`0.56738/0.36138`，latest=`0.54464/0.34872`，late5/late10/late20 AP50-95=`0.34823/0.34893/0.34916`，日志无错误标记。B1 concat rows=`59`，best epoch=`1`，best AP50/AP50-95=`0.56744/0.36116`，latest=`0.55130/0.35201`，late5/late10/late20 AP50-95=`0.35404/0.35181/0.35302`，日志无错误标记。
- 判定：目标 A 中 CMDistill-from-YOLO-init 是正式 `positive`；LD-from-YOLO-init 只能记 `weak-positive / marginal`，因为 best 只比 baseline best `0.36087` 高 `+0.00011`，final 和 late-window 都低于 baseline best。B1 三变体当前均有效；但 reload 判据下早期 epoch 1/2 高点不能算正结果，后续都在下降，暂未满足“下降后涨回并超过加载点/最终 baseline”的正结果标准。当前两张卡都在 16G-21G 推荐区间内，且 util 较高，不追加新任务。
- 下一步：等待 B1 c0 完成 200 epoch 得到最终 control；继续观察 sum 到 200 epoch、concat 到至少 100 epoch。若 GPU0 在 c0 完成后实际释放且不超过 22G，再考虑启动下一条更接近 LADD 的主线候选；否则保持当前三变体，避免 OOM/race。

### 2026-06-24 04:01 CST - LD 完成后启动 B1 concat_mlp_cap2

- 目标：A / B
- 做了什么：LD-from-YOLO-init 已跑满 200 epoch，GPU1 释放到约 `5991 MiB`；确认旧 DSN refine queue 进程 `33404` 已不在运行，不会突然抢 GPU。随后使用单次脚本在 GPU1 启动 `reachable_fused_shared/concat_mlp_cap2`，作为 B1 第三个 fused shared 变体。
- 使用资源：`ladd4090-zw1` GPU1；脚本 `docs/experiments/dronevehicle_method_search_20260623/reachable_fused_shared/launch_single_reachable_fused_shared_20260624.sh`；run tag `reachable_fused_concat_mlp_cap2_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_040140`，pid=`56300`，outer log=`logs/dronevehicle_method_search/sub2k_seed0_fullval/reachable_fused_shared/concat_mlp_cap2/reachable_fused_concat_mlp_cap2_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_040140_gpu1.outer.log`。
- 结果：启动前 precheck GPU1 used_mb=`5991`，低于 `PRECHECK_MAX_USED_MB=12000`；脚本带 `STRICT_BATCH_SIZE=1`，若 batch 64 不成立会直接失败而不做 batch fallback。约 04:03 CST 复查，concat 已写出 `results.csv` 至 epoch 5，best epoch=`1`，best AP50/AP50-95=`0.56744/0.36116`，latest=`0.55589/0.35426`，late5/late10/late20 AP50-95=`0.35803/0.35803/0.35803`，日志无错误标记。GPU1 启动后约 `16149/24564 MiB`，在推荐安全区间内。
- 判定：`concat_mlp_cap2` 为 `valid-started`；它是 reload/split-load 曲线，epoch 1 高点不能算正结果，后续需按 reload 判据看是否能恢复并超过加载点和 baseline final。
- 下一步：监控 concat 到 50/100/150/200 epoch，并和 c0/sum 做 epoch-matched 与 late-window 对比。

### 2026-06-24 04:00 CST - Heartbeat 巡检：目标 A 完成，B1 到 100+ epoch

- 目标：A / B / 监控
- 做了什么：检查 `ladd4090-zw1` GPU、compute app、目标 A 完整曲线、B1 c0/sum 当前曲线与队列状态。
- 使用资源：`ladd4090-zw1`；GPU0 运行 B1 c0 + B1 sum；GPU1 只剩旧 OGSOD RGB baseline，占用约 6G。
- 结果：GPU0 约 `20325/24564 MiB`、util `98%`，GPU1 约 `5977/24564 MiB`、util `17%`。CMDistill-from-YOLO-init rows=`200`，best epoch=`184`，best AP50/AP50-95=`0.56976/0.36215`，final/latest=`0.56725/0.35933`，late5/late10/late20 AP50-95=`0.35928/0.35959/0.35987`。LD-from-YOLO-init rows=`200`，best epoch=`141`，best AP50/AP50-95=`0.56220/0.36098`，final/latest=`0.55153/0.35357`，late5/late10/late20 AP50-95=`0.35457/0.35479/0.35468`。B1 c0 rows=`136`，best epoch=`2`，best AP50/AP50-95=`0.56869/0.36331`，latest=`0.53826/0.34497`，late5/late10/late20 AP50-95=`0.34560/0.34587/0.34649`。B1 sum rows=`103`，best epoch=`1`，best AP50/AP50-95=`0.56738/0.36138`，latest=`0.54196/0.34651`，late5/late10/late20 AP50-95=`0.34975/0.35060/0.35087`。所有已查日志均无 OOM、Traceback、batch fallback 标记。
- 判定：目标 A 至少已有 CMDistill-from-YOLO-init 一个正式 positive，因此不触发 D3T-style。LD 只是边缘 weak-positive，不能作为强证据。B1 sum 在 100+ epoch 的 late-window 高于 c0，但两者都没有满足 reload 正结果标准。
- 下一步：先确认旧 DSN refine queue 是否仍活跃；若 GPU1 空闲且无队列 race，再用单次脚本启动 B1 `concat_mlp_cap2`，补完整 fused shared 的 sum/concat 对照。

### 2026-06-24 03:20 CST - 重新创建 active goal 并确认 automation 状态

- 目标：A / B / 监控
- 做了什么：用户反馈 ChatGPT app 中显示目标未开始，并已删除旧 goal；检查后确认当前 goal 为空，于是重新创建正式 active goal。随后读取本地 automation 配置，确认 `dronevehicle-night-method-search` 存在、`status=ACTIVE`、`rrule=FREQ=MINUTELY;INTERVAL=15`、`target_thread_id=019ef4a6-8811-7462-9f62-b0dce78f1411`。
- 使用资源：Codex goal 状态、`~/.codex/automations/dronevehicle-night-method-search/automation.toml`、本地 repo 文档目录。
- 结果：目标已重新启动，automation 无需重复创建；当前执行标准仍以本文档为准。
- 判定：waiting
- 下一步：等待下一次 heartbeat 自动巡检；若手动继续，则优先检查 CMDistill-from-YOLO-init 是否完成并补正结果对象，随后在 GPU0 释放后启动目标 B 候选。

### 2026-06-24 03:21 CST - 目标 A 中期确认与目标 B 单次启动脚本准备

- 目标：A / B / 监控
- 做了什么：再次检查 `ladd4090-zw1` GPU 与两个目标 A run；解析 `results.csv` 并扫描日志关键错误；同时为 `reachable_fused_shared` 创建单次启动脚本，避免旧连续队列自动连发 c0/sum/concat 造成 OOM 或 batch fallback。
- 使用资源：`ladd4090-zw1`；本地与远端脚本 `docs/experiments/dronevehicle_method_search_20260623/reachable_fused_shared/launch_single_reachable_fused_shared_20260624.sh`。
- 结果：GPU0 约 `9807/24564 MiB`，GPU1 约 `15289/24564 MiB`。CMDistill-from-YOLO-init rows=`175`，best epoch=`151`，best AP50/AP50-95=`0.57128/0.36213`，latest=`0.56791/0.36049`，late5/late10/late20 AP50-95=`0.36086/0.36056/0.35961`，日志无 OOM/Traceback/batch fallback 标记。LD-from-YOLO-init rows=`94`，best AP50/AP50-95=`0.53579/0.33945`，latest=`0.52326/0.32654`，late5/late10/late20 AP50-95=`0.32890/0.32676/0.31801`，日志无错误标记。单次 B1 启动脚本已同步到远端并通过 `bash -n`。
- 判定：CMDistill-from-YOLO-init 维持 `positive candidate`；LD 暂为 `inconclusive/negative-trending`；D3T-style 暂不启动。目标 B 已完成安全启动脚本准备，尚未发新 run。
- 下一步：等待 CMDistill 跑满 200 epoch 并释放 GPU0；随后优先用单次脚本启动 `reachable_fused_shared/c0_nofusion_splitrec` strict-batch control，确认有效后再启动 `sum_mlp_cap2` 或 `concat_mlp_cap2`。

### 2026-06-24 03:27 CST - CMDistill-from-YOLO-init 完成并启动 B1 c0 control

- 目标：A / B
- 做了什么：确认 CMDistill-from-YOLO-init 已完成 200 epoch，GPU0 释放；随后使用单次脚本启动 `reachable_fused_shared/c0_nofusion_splitrec`，作为 fused shared 主线候选的同结构无 fusion/no KD control。
- 使用资源：`ladd4090-zw1` GPU0；脚本 `docs/experiments/dronevehicle_method_search_20260623/reachable_fused_shared/launch_single_reachable_fused_shared_20260624.sh`；远端 run tag `reachable_fused_c0_nofusion_splitrec_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_032714`。
- 结果：CMDistill-from-YOLO-init rows=`200`，best epoch=`184`，best AP50/AP50-95=`0.56976/0.36215`，final/latest=`0.56725/0.35933`，late5/late10/late20 AP50-95=`0.35928/0.35959/0.35987`；它从 `yolo11n.pt` 出发，best AP50-95 超过 RGB baseline final `0.35385` 与 baseline best `0.36087`。LD-from-YOLO-init rows=`115`，best/latest AP50/AP50-95=`0.55050/0.34775`，仍未过 baseline final。B1 c0 已启动，pid=`50224`，outer log=`logs/dronevehicle_method_search/sub2k_seed0_fullval/reachable_fused_shared/c0_nofusion_splitrec/reachable_fused_c0_nofusion_splitrec_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_032714_gpu0.outer.log`。
- 判定：目标 A 已得到一个 `positive` 外部跨模态单模态推理结果，因此暂不启动 D3T-style；B1 c0 为 `waiting`，需确认 strict-batch 有效启动。
- 下一步：检查 B1 c0 的日志和 `results.csv`；如果无 OOM/batch fallback，继续等待若干 epoch 后再决定是否在安全资源下启动 `sum_mlp_cap2`。

### 2026-06-24 03:31 CST - 修复 fused_shared CAP2 参数 bug 并重启 sum_mlp_cap2

- 目标：B
- 做了什么：检查 `sum_mlp_cap2` 首次启动日志，发现不是 OOM，而是代码 bug：`_fused_shared_cap_loss()` 定义接收 `fused_map`，调用处误传 `fused_shared_map=`。本地修复一行调用参数，`py_compile` 通过后同步到远端，远端 `py_compile` 通过；随后重新启动新的 `sum_mlp_cap2` run。
- 使用资源：本地与远端 `ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/loss.py`；旧 invalid run tag `reachable_fused_sum_mlp_cap2_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_032837`；新 run tag `reachable_fused_sum_mlp_cap2_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_033127`，pid=`51559`。
- 结果：旧 `sum_mlp_cap2` 标记为 `invalid`，原因是 `TypeError: ... got an unexpected keyword argument 'fused_shared_map'`，不进入任何曲线比较。新 `sum_mlp_cap2` 启动时 GPU0 已用 `9793 MiB`，低于单次脚本 precheck 阈值 `12000 MiB`。
- 判定：代码 bug 已修复；新 sum 为 `waiting`，需要确认能够写出 `results.csv` 且无 OOM/batch fallback。
- 下一步：检查新 sum 的日志、GPU0 总显存和 `results.csv`；如果 c0+sum 总显存低于 22G 且训练有效，不再叠加 concat，等待 c0/sum 曲线形成。

### 2026-06-24 03:33 CST - 新 sum_mlp_cap2 有效启动，暂停继续叠加 concat

- 目标：A / B / 监控
- 做了什么：检查新 `sum_mlp_cap2` 启动状态、GPU 总显存和当前 `results.csv`；同时更新 LD-from-YOLO-init、B1 c0、B1 sum 的当前曲线摘要。
- 使用资源：`ladd4090-zw1` GPU0/1；B1 c0 run `reachable_fused_c0_nofusion_splitrec_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_032714_b`；B1 sum run `reachable_fused_sum_mlp_cap2_lowlr1e3_nowarmup_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_033127_b`。
- 结果：GPU0 在 c0+sum 并行后约 `19941/24564 MiB`，低于 22G 危险线且处于推荐利用区间；GPU1 约 `15435/24564 MiB`。新 sum 已写出 `results.csv` 至 epoch 5，无刚才的 TypeError。LD-from-YOLO-init rows=`141`，best/latest AP50/AP50-95=`0.56220/0.36098`，late5/late10/late20 AP50-95=`0.35604/0.35430/0.35228`。B1 c0 rows=`30`，best epoch=`2`，best AP50/AP50-95=`0.56869/0.36331`，latest=`0.55548/0.35499`，late5/late10/late20 AP50-95=`0.35606/0.35484/0.35463`。B1 sum rows=`5`，best epoch=`1`，best AP50/AP50-95=`0.56738/0.36138`，latest=`0.55651/0.35468`。
- 判定：新 sum 为 `valid-started`；GPU0 当前不适合再叠加 concat，否则预计超过 22G。LD-from-YOLO-init 更新为 `weak-positive candidate`，因为 AP50-95 peak `0.36098` 刚刚超过 RGB baseline best `0.36087`，但 margin 只有 `+0.00011`，需等待 200 epoch 完整曲线。B1 c0/sum 均为 reload/split-load 曲线，早期高点不能直接算正结果，后续必须看是否恢复并超过加载点/对照。
- 下一步：不启动 concat；继续监控 c0 与 sum 的 50/100/150/200 epoch 曲线，判断 sum 是否超过自身 c0 架构对照和 reload 判据。继续让 LD 跑满 200 epoch，完整记录目标 A 第二个候选。

### 2026-06-24 03:45 CST - Heartbeat 巡检：B1 双 run 正常推进，暂不追加任务

- 目标：A / B / 监控
- 做了什么：按 heartbeat 检查 `ladd4090-zw1` GPU、进程、LD-from-YOLO-init、B1 c0、B1 sum 的 `results.csv` 与错误日志。
- 使用资源：`ladd4090-zw1`；GPU0 运行 B1 c0 + B1 sum；GPU1 运行 LD-from-YOLO-init + 旧 OGSOD RGB baseline。
- 结果：GPU0 约 `20323/24564 MiB`、util `79%`，GPU1 约 `15577/24564 MiB`、util `78%`，均低于 22G 危险线。LD-from-YOLO-init rows=`186`，best epoch=`141`，best AP50/AP50-95=`0.56220/0.36098`，latest=`0.55029/0.35365`，late5/late10/late20 AP50-95=`0.35456/0.35531/0.35514`，日志无错误标记。B1 c0 rows=`77`，best epoch=`2`，best AP50/AP50-95=`0.56869/0.36331`，latest=`0.54974/0.35246`，late5/late10/late20 AP50-95=`0.34992/0.35035/0.34986`，日志无错误标记。B1 sum rows=`49`，best epoch=`1`，best AP50/AP50-95=`0.56738/0.36138`，latest=`0.55382/0.35265`，late5/late10/late20 AP50-95=`0.35328/0.35502/0.35523`，日志无错误标记。
- 判定：系统状态正常，`concat_mlp_cap2` 暂不启动，因为 GPU0 已约 20.3G，再叠加预计接近或超过 22G。LD 仍是 `weak-positive candidate`，但 final/late-window 未明显超过 baseline best，需要跑满 200 epoch 后定性。B1 sum 当前 late20 `0.35523` 高于 c0 late20 `0.34986`，出现早期机制性线索，但还不能按 reload 判据称正结果。
- 下一步：继续等待 LD 完成 200 epoch，并监控 B1 c0/sum 到至少 100 epoch；若 sum 继续稳定高于 c0，再考虑在 GPU 释放后补 `concat_mlp_cap2` 或 shuffled control。

### 2026-06-24 03:49 CST - Heartbeat 巡检：LD 接近完成，B1 继续等待 100 epoch

- 目标：A / B / 监控
- 做了什么：再次检查 `ladd4090-zw1` GPU、训练进程、目标 A/B 的 `results.csv` 与错误日志；同时尝试确认 Codex goal 状态。
- 使用资源：`ladd4090-zw1`；GPU0 运行 B1 c0 + B1 sum；GPU1 运行 LD-from-YOLO-init + 旧 OGSOD RGB baseline。
- 结果：GPU0 约 `20323/24564 MiB`、util `96%`，GPU1 约 `15617/24564 MiB`、util `88%`。CMDistill-from-YOLO-init 已完成，rows=`200`，best epoch=`184`，best AP50/AP50-95=`0.56976/0.36215`，final/latest=`0.56725/0.35933`，late5/late10/late20 AP50-95=`0.35928/0.35959/0.35987`，日志无错误标记。LD-from-YOLO-init rows=`197`，best epoch=`141`，best AP50/AP50-95=`0.56220/0.36098`，latest=`0.55188/0.35483`，late5/late10/late20 AP50-95=`0.35512/0.35481/0.35493`，日志无错误标记。B1 c0 rows=`88`，best epoch=`2`，best AP50/AP50-95=`0.56869/0.36331`，latest=`0.54997/0.35353`，late5/late10/late20 AP50-95=`0.34938/0.34987/0.35028`，日志无错误标记。B1 sum rows=`59`，best epoch=`1`，best AP50/AP50-95=`0.56738/0.36138`，latest=`0.55527/0.35499`，late5/late10/late20 AP50-95=`0.35518/0.35318/0.35410`，日志无错误标记。
- 判定：系统状态正常，不追加 `concat_mlp_cap2`，因为 GPU0 已约 20.3G，再叠加会接近或超过 22G 危险线。目标 A 中 CMDistill-from-YOLO-init 已是正式 `positive`；LD 仍为 `weak-positive candidate`，但 margin 只有 `+0.00011`，且 late-window 没有稳定超过 baseline best，需等待最后 3 epoch 完成后再定性。B1 sum 目前 latest/late5 高于 c0，但 late10/late20 优势缩小，仍只能记为早期机制线索，不能称 reload 正结果。Codex goal 工具返回 `blocked` 且无法重新 create 同一 goal，属于工具状态异常；heartbeat 与实验执行继续按本文档推进。
- 下一步：等待 LD 到 200 epoch 后补 final 判定；等待 B1 c0/sum 到至少 100 epoch 后做第一轮目标 B 对比。如果 GPU0 没释放，不启动 concat；如果 LD 完成后 GPU1 空余，再优先保持监控，避免对 GPU0 的 B1 对照产生资源干扰。

### 2026-06-24 03:13 CST - 启动夜间目标与 heartbeat automation

- 目标：A / B / 监控
- 做了什么：用户确认开始执行；检查 goal 系统时发现已有 paused goal，目标与当前任务一致，因此沿用该 goal，不新建重复目标；创建当前线程 heartbeat automation `dronevehicle-night-method-search`，15 分钟检查一次。
- 使用资源：Codex goal 状态、Codex app automation、本地 repo 文档目录。
- 结果：夜间搜索进入执行态；后续 heartbeat 需要按本文档检查 DroneVehicle 目标 A/B、服务器资源、results.csv、日志、队列状态，并持续追加执行日志。
- 判定：waiting
- 下一步：下一次 heartbeat 先检查 ladd4090-zw1 的 GPU/进程/队列，再解析 CMDistill-from-YOLO-init 与 LD-from-YOLO-init 的最新曲线；若 A1/A2 完成且失败，再准备 D3T-style；同时寻找/启动符合资源规则的目标 B 变体。

### 2026-06-24 03:15 CST - 第 0 次 heartbeat：检查 GPU 与目标 A 当前曲线

- 目标：A / 监控
- 做了什么：连接 `ladd4090-zw1`，检查 GPU 显存/进程，并解析 CMDistill-from-YOLO-init 与 LD-from-YOLO-init 的 `results.csv`。
- 使用资源：`ladd4090-zw1`；GPU0 当前约 `9807/24564 MiB`，GPU1 当前约 `15233/24564 MiB`。GPU0 运行 CMDistill-from-YOLO-init，GPU1 运行 LD-from-YOLO-init 与一个旧 OGSOD baseline。
- 结果：CMDistill-from-YOLO-init rows=`144`，best epoch=`134`，best AP50/AP50-95=`0.57127/0.36201`，latest=`0.56551/0.35638`，late5/late10/late20 AP50-95=`0.35843/0.35783/0.35687`。LD-from-YOLO-init rows=`74`，best/latest AP50/AP50-95=`0.50161/0.31163`，late5/late10/late20 AP50-95=`0.29976/0.29637/0.29217`。
- 判定：CMDistill-from-YOLO-init 当前为 `positive candidate`，因为它从 `yolo11n.pt` 出发，AP50-95 peak `0.36201` 已超过 RGB baseline final `0.35385` 和 RGB baseline best `0.36087`；LD 当前仍为 `inconclusive/negative-trending`。暂不启动 D3T-style，因为 A1 已出现正结果候选。
- 下一步：等待 CMDistill 跑满 200 epoch 后补 final、late-window、曲线和正结果对象表；继续监控 LD 到至少中后期。目标 B 不在 GPU0 立刻叠任务，因为 CMDistill 已接近完成，避免为短时间利用率强行增加 race/OOM 风险；下一次 heartbeat 优先在 CMDistill 完成后释放的 GPU0 上启动更接近 LADD 的候选。

### 2026-06-24 03:09 CST - 修正显存利用规则为推荐区间而非填满目标

- 目标：文档
- 做了什么：把显存利用规则从“尽量 20G-22G”改为“有合适任务时推荐 15G-21G，超过 22G 危险，不强行塞满”；补充 16G 当前占用但新增任务会到 23G 时应保持 16G 不动的示例。
- 使用资源：本地 repo 文档目录。
- 结果：后续调度应优先保证实验有效性和安全性；即使显存仍有空余，只要新增任务预计超过 22G 或引入 OOM/race/batch fallback 风险，就不加任务。
- 判定：waiting
- 下一步：等待用户确认是否按本文件启动夜间目标和 heartbeat。

### 2026-06-24 03:06 CST - 扩展夜间目标与正结果对象标准

- 目标：文档
- 做了什么：补充夜间总目标、目标完成状态、正结果对象必须包含的材料、明早交付内容、目标 A/B 的详细成功标准与成果形式。
- 使用资源：本地 repo 文档目录。
- 结果：目标描述从简略方向扩展为可执行任务书；后续自动化需要按这些标准判断 positive / weak-positive / negative / invalid。
- 判定：waiting
- 下一步：等待用户确认是否按本文件启动夜间目标和 heartbeat。

### 2026-06-24 03:05 CST - 记录服务器资源授权与调度边界

- 目标：文档
- 做了什么：补充三台服务器的可用资源、杀进程边界、加进程边界和显存调度目标。
- 使用资源：本地 repo 文档目录。
- 结果：明确 LADD4090 双卡服务器可自由使用和杀进程；LADD3090 与 AutoDL 可加进程但不杀原有进程；显存应在有合适任务时充分利用，推荐区间约 15G-21G，不是强行塞满要求，超过 22G 视为危险。
- 判定：waiting
- 下一步：等待用户确认是否按本文件启动夜间目标和 heartbeat。

### 2026-06-24 02:58 CST - 建立夜间目标草案与日志文件

- 目标：文档
- 做了什么：把当前讨论中的目标 A / 目标 B、YOLO-init 与 reload 正结果判据、三方法上限、夜间 heartbeat 节奏、日志模板写入本文件。
- 使用资源：本地 repo 文档目录。
- 结果：仅新增文档；未启动 goal，未启动 automation，未追加实验队列。
- 判定：waiting
- 下一步：等待用户确认是否按本文件启动夜间目标和 heartbeat。

### 2026-06-24 02:49 CST - CMDistill from YOLO-init 快照

- 目标：A
- 做了什么：检查已启动的 CMDistill-from-YOLO-init 训练。
- 使用资源：ladd4090-zw1 GPU0。
- 结果：rows=31，best AP50/AP50-95=`0.43014/0.25905`，latest=`0.42287/0.24550`，仍低于 RGB baseline final `0.35385`，处于早期爬升段。
- 判定：inconclusive
- 下一步：继续跑到至少 100 epoch 或完成 200 epoch 后再按 YOLO-init 判据判断；若明显平台且未超过 baseline final，则判失败。

### 2026-06-24 02:49 CST - LD from YOLO-init 快照

- 目标：A
- 做了什么：检查已启动的 LD-from-YOLO-init 训练。
- 使用资源：ladd4090-zw1 GPU1。
- 结果：rows=2，best AP50/AP50-95=`0.16299/0.08223`，刚启动，不能判断。
- 判定：inconclusive
- 下一步：继续观察；若完成后未出现高于 RGB baseline final 的 peak，则触发 A3 D3T-style。

### 2026-06-24 02:30 CST 左右 - 外部方法候选筛选结论

- 目标：A
- 做了什么：筛选可开源迁移的跨模态蒸馏方法。
- 使用资源：已查到的开源仓库与方法笔记。
- 结果：D3T 是当前最适合做 A3 的外部开源锚点；它是 RGB-to-thermal/domain-adaptive teacher-student 检测蒸馏，推理时为单模态 student。AMFD、M2D-LIF 等虽然有蒸馏或多模态设计，但推理仍依赖 RGB-T/fusion，不满足目标 A 的单模态 student 推理约束。
- 判定：waiting
- 下一步：仅当 CMDistill-from-YOLO-init 与 LD-from-YOLO-init 均失败时，启动 D3T-style output KD。

### 2026-06-24 08:46 CST - 检查 AutoDL 双卡服务器上的正向候选

- 目标：外部线索 / 旧主线对照
- 做了什么：按用户提示检查 `autodl-nmb1` 双 4090 服务器的 GPU、screen、进程命令和当前 `results.csv`；确认用户看到的正向趋势来自 AutoDL2 上的 OGSOD no-mosaic 旧主线实验，而不是 DroneVehicle 小风洞实验。
- 使用资源：AutoDL2 `/root/autodl-tmp/LADD_public`；GPU0 当前约 `12621/24564 MiB`，GPU1 当前约 `8904/24564 MiB`；5 个 screen 均为 `nl_nomosaic_*_20260623_194650`。
- 结果：这组实验是 OGSOD HBB no-mosaic / no-reload-existing-cache / yolo11n / seed0 / 700 epoch。`reload_detonly` 控制组 rows=`537`，best/latest AP50/AP50-95=`0.79515/0.52477`，late5/late10/late20 AP50-95=`0.524396/0.523695/0.522402`。`yolo_probeA` rows=`483`，best/latest AP50/AP50-95=`0.79301/0.52639`，late5/late10/late20 AP50-95=`0.52570/0.52505/0.52369`，当前 best 比 det-only best 高 `+0.00162` AP50-95；在共同 epoch 上，ep300/400/450/480 AP50-95 差值分别为 `+0.00856/+0.01503/+0.01653/+0.01702`。`warm100_probeA` rows=`461`，latest AP50-95=`0.51608`，相对 det-only 同 epoch ep461 为 `+0.01217`，但低于 det-only 当前 best；`warm100_dynamic` rows=`457`，latest=`0.51064`，同 epoch `+0.00773`；`yolo_dynamic` rows=`458`，latest=`0.51274`，同 epoch `+0.00956`。
- 判定：`yolo_probeA` 是 AutoDL2 当前最像正结果的候选，且是从 `yolo11n.pt` detector 初始化、加载 A1 decomposition cache，并使用 frozen/detached reach probe 的旧主线变体；不过它属于 OGSOD no-mosaic 旧主线对照，不是 DroneVehicle 小风洞结果。需要等 700 epoch final 后再确认是否最终稳定高于同组 det-only reload 控制。
- 下一步：继续监控 `yolo_probeA` 到 final；若它最终成立，可反向提取 `--ladd-b-frozen-reach-probe --ladd-b-detach-reach-probe`、YOLO-init detector、A1 decomposition cache、no-mosaic 训练设置这些关键差异，作为 DroneVehicle 目标 B 的近 LADD 方案候选。

### 2026-06-24 08:56 CST - heartbeat：P16 标准学习率复查并启动 P17 ProbeA 风洞版

- 目标：B / 监控
- 做了什么：检查 `ladd4090-zw1` GPU、活动进程、P15/P16/P17 与目标 A 完成结果；根据 AutoDL2 上 `yolo_probeA` 的正向线索，新增单独子目录脚本 `oldsplit_probeA_yoloinit_std/launch_oldsplit_probeA_yoloinit_std_20260624.sh`，并在 GPU1 启动一个 DroneVehicle sub2k full-val 的 YOLO-init + A1 decomposition cache + frozen/detached reach probe 变体。
- 使用资源：`ladd4090-zw1`。P17 启动前 GPU0/GPU1 约 `9839/9727 MiB`；P17 启动后 GPU0/GPU1 约 `9839/20145 MiB`，GPU1 低于 `22G` 危险线。P17 pid=`103246`，日志目录 `logs/dronevehicle_method_search/sub2k_seed0_fullval/oldsplit_probeA_yoloinit_std/oldsplit_probeA_yoloinit_std_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_085359_gpu1/`。
- 结果：目标 A 当前定稿：CMDistill-from-YOLO-init rows=`200`，best AP50/AP50-95=`0.56976/0.36215`，final/latest=`0.56725/0.35933`，late5/late10/late20 AP50-95=`0.35928/0.359586/0.3598665`；LD-from-YOLO-init rows=`200`，best=`0.56220/0.36098`，final=`0.55153/0.35357`，late20=`0.354683`。P15 low-lr YOLO-init control rows=`200`，best/latest AP50-95=`0.32541/0.32522`；method rows=`200`，best/latest=`0.33038/0.32647`，相对 low-lr control best `+0.00497`，但远低于 RGB baseline final `0.35385`，因此只算 schedule-confounded 弱线索，不算正结果。P16 std-lr control rows=`57`，best/latest AP50-95=`0.31110/0.30970`，仍在爬升；P16 std-lr fused method rows=`28`，best=`0.18930`，之后 ep29 训练 loss NaN，`last.pt` 被判定含 NaN/Inf，标记为 `invalid` 而不是负结果。P17 ProbeA 风洞版启动成功，strict batch 生效，无 batch fallback；当前 rows=`3`，best/latest AP50-95=`0.10353/0.07098`，仍是早期。
- 判定：不再使用 P16 fused std-lr method；P16 control 继续作为 YOLO-init/std-lr det-only 对照观察。P17 是当前最贴近 AutoDL2 正向线索、也最接近旧 LADD 主线的风洞候选，先观察到 50/100 epoch 再判断。
- 下一步：下一次 heartbeat 重点看 P16 control 与 P17 的 50-epoch 前后曲线；若 P17 出现 NaN 或明显平台低于 control，则停止该分支并准备更保守的 ProbeA 变体（例如降低 core loss 或改为 no-mixup 复刻 AutoDL2 条件）。

### 2026-06-24 08:58 CST - 用户确认 ProbeA 同 epoch 领先也应视为正向信号

- 目标：B / 判据修正
- 做了什么：按用户判断更新 ProbeA 风洞实验的观察标准：ProbeA 不只看 final/best 是否超过 baseline final；在同协议、同初始化、同 epoch 下持续领先同条件 det-only baseline，也应作为正向主线信号记录，尤其是 AutoDL2 `yolo_probeA` 这种先落后、后期追上的曲线形态。
- 使用资源：`ladd4090-zw1`，P16 YOLO-init/std-lr det-only control 与 P17 YOLO-init/std-lr ProbeA 风洞版。
- 结果：P17 已经是用户建议的 “YOLO-init ProbeA 小风洞版”，run 为 `oldsplit_probeA_yoloinit_std_ir2rgb_yolo11n_e200_b64_img512_s0_20260624_085359_b`，当前 rows=`8`，best/latest AP50/AP50-95=`0.29048/0.16145`。同 epoch 对照 P16 control 当前 rows=`64`，best=`0.51475/0.32309`，latest=`0.50336/0.30830`；P17 与 P16 的共同 epoch AP50-95 差值为 ep1 `+0.00043`、ep2 `+0.01884`、ep3 `-0.03845`、ep4 `-0.01753`、ep5 `-0.05274`、ep6 `-0.05469`、ep7 `-0.01713`、ep8 `-0.00774`。
- 判定：P17 启动有效、strict batch 生效、无 OOM/Traceback/NaN，目前仍太早；ep3-8 暂时未领先 control，但 AutoDL2 yolo_probeA 本身也是后期追上，因此继续观察到 50/100 epoch。后续报告必须同时给 best/final/late-window 和同 epoch delta。
- 下一步：不重复启动第二条 YOLO-init ProbeA，避免混淆；继续让 P17 跑，并在下一次自动检查时重点看 P17 是否开始缩小或反超 P16 control。

### 2026-06-24 09:00 CST - 收缩方法探索范围，只保留 ProbeA 与必要 control

- 目标：调度 / B
- 做了什么：用户判断其他方法探索暂时没有正结果，建议先停；检查 `ladd4090-zw1` 当前 DroneVehicle method-search 进程、GPU 和队列 PID；更新 heartbeat automation `dronevehicle-night-method-search`，把目标从“继续探索任意主线方法”收缩为“只监控 P16/P17 和 AutoDL2 ProbeA 线索，不再启动新方法”。
- 使用资源：`ladd4090-zw1`。当前 GPU0/GPU1 约 `9839/20551 MiB`；没有发现还活着的旧 method-search queue PID。保留中的训练为 P16 YOLO-init/std-lr det-only control 与 P17 YOLO-init/std-lr ProbeA。
- 结果：P16 control rows=`74`，best AP50/AP50-95=`0.52320/0.32796`，latest=`0.52095/0.32574`，late5/late10/late20 AP50-95=`0.320826/0.315705/0.311559`。P17 ProbeA rows=`14`，best=`0.33230/0.19672`，latest=`0.32232/0.19505`，late5/late10/late20 AP50-95=`0.190624/0.157928/0.131722`。同 epoch AP50-95 delta：ep1 `+0.00043`，ep2 `+0.01884`，ep3 `-0.03845`，ep4 `-0.01753`，ep5 `-0.05274`，ep6 `-0.05469`，ep7 `-0.01713`，ep8 `-0.00774`，ep9 `+0.00344`，ep10 `+0.00476`，ep11 `+0.00251`，ep12 `+0.00033`，ep13 `-0.00860`，ep14 `+0.00028`。
- 判定：其他方法族冻结，不再新开；P17 仍处于早期，但 ep9-12、ep14 出现小幅同 epoch 领先，继续观察是合理的。P16 control 是 P17 的必要同条件 control，也保留。
- 下一步：自动化只做监控和记录；除非用户明确要求，不启动 D3T、fused shared、DSN、teacher-conf、reachability-weighted、CMDistill/LD 之外的新变体。

### 2026-06-24 09:06 CST - 启动 DroneVehicle 小风洞 img256 baseline pair

- 目标：协议扩展 / baseline
- 做了什么：按用户建议新增 256 输入的小风洞 baseline 协议；先训练 RGB student baseline，完成后同一脚本自动训练 IR teacher baseline。训练入口 `baseline/code/train_ogsod_baseline.py` 新增 `--strict-batch-size`，256 baseline 脚本已启用该参数，防止 OOM 后静默降 batch 产生混淆结果。
- 使用资源：`ladd4090-zw1` GPU0。启动前 GPU0/GPU1 约 `9839/20601 MiB`；启动后训练进程确认生效，GPU0/GPU1 约 `13951/20651 MiB`，低于 `22G` 危险线。队列 PID=`105784`，当前 RGB baseline 子进程 PID=`105791`。
- 路径：脚本 `docs/experiments/dronevehicle_method_search_20260623/img256_baselines/launch_img256_baseline_pair_20260624.sh`；日志目录 `logs/dronevehicle_method_search/sub2k_seed0_fullval/img256_baselines/dronevehicle_sub2k_img256_baseline_pair_yolo11n_e200_b64_s0_20260624_090613_gpu0/`；RGB 结果目录 `runs_public/dronevehicle_method_search/sub2k_seed0_fullval/img256_baselines/student_rgb/dronevehicle_sub2k_student_rgb_yolo11n_cclkdproto_e200_b64_img256_mosaic0p0_close0_mixup0p1_s0_20260624_090613/`；IR teacher 结果目录将在 RGB 完成后写入 `runs_public/dronevehicle_method_search/sub2k_seed0_fullval/img256_baselines/teacher_ir/`。
- 结果：启动有效，`strict_batch_size=True`，协议为 `imgsz=256, epochs=200, batch=64, optimizer=SGD, lr0=0.01, lrf=0.01, cos_lr=True, mosaic=0.0, close_mosaic=0, mixup=0.1, seed=0, deterministic=True`。09:07 左右快照：img256 RGB rows=`3`，best AP50/AP50-95=`0.09378/0.03368`，latest=`0.09374/0.03190`，属于正常 YOLO-init 冷启动早期。
- 同时状态：P16 YOLO-init/std-lr det-only control rows=`111`，best/latest AP50/AP50-95=`0.56913/0.35990`；P17 YOLO-init/std-lr ProbeA rows=`41`，best=`0.45716/0.27970`，latest=`0.45048/0.27617`，仍明显低于同条件 control 早期曲线。
- 判定：img256 baseline pair 是独立协议，不与 img512 P16/P17 直接混用；在 img256 RGB baseline 和 IR teacher baseline 完成前，不启动 img256 ProbeA 或其他 img256 方法，避免没有同协议 control。
- 下一步：自动化监控范围新增 img256 baseline pair；若 RGB baseline 完成，确认最终/best/late-window 后等待 IR teacher 自动启动并完成。只有拿到 img256 baseline/teacher 后，才讨论是否启动 img256 ProbeA。

### 2026-06-24 09:09 CST - img256 baseline 启动核验

- 目标：baseline / 监控
- 做了什么：启动后短轮询，确认 256 RGB baseline 已写入 `results.csv`，自动化 prompt 已更新为同时监控 P16/P17、AutoDL2 ProbeA 线索和 img256 baseline pair。
- 使用资源：`ladd4090-zw1`，当前 GPU0/GPU1 约 `13971/21097 MiB`，GPU1 接近上限因此不追加新任务。
- 结果：img256 RGB rows=`10`，best AP50/AP50-95=`0.18955/0.08212`，latest=`0.16653/0.07238`；P16 control rows=`117`，best=`0.56913/0.35990`，latest=`0.55956/0.35630`；P17 ProbeA rows=`47`，best/latest=`0.47753/0.29391`。
- 判定：img256 RGB baseline 训练有效，strict batch 有效；P17 仍未接近同条件 control，继续仅监控。
- 下一步：继续等待 img256 RGB 完成；完成后脚本会自动进入 img256 IR teacher baseline。

### 2026-06-24 09:10 CST - heartbeat 监控快照

- 目标：B / img256 baseline / 监控
- 做了什么：按 heartbeat 要求检查 `ladd4090-zw1` GPU、活动进程、P16/P17 曲线、img256 baseline pair 曲线和日志有效性；未启动任何新实验。
- 使用资源：`ladd4090-zw1`。当前 GPU0/GPU1 约 `13971/21117 MiB`，均低于 `22G` 危险线但 GPU1 已接近上限，不追加任务。img256 baseline pair 队列 PID=`105784`，RGB baseline 子进程 PID=`105791` 仍在 GPU0 训练。
- 结果：P16 YOLO-init/std-lr det-only control rows=`124`，best AP50/AP50-95=`0.57120/0.36163` at ep118，latest=`0.56049/0.35358`，late5/late10/late20 AP50-95=`0.355596/0.355512/0.3534115`，未完成。P17 YOLO-init/std-lr ProbeA rows=`53`，best=`0.48527/0.30158` at ep51，latest=`0.48320/0.29783`，late5/late10/late20 AP50-95=`0.296908/0.293741/0.281347`，未完成。共同 epoch 1-53 上，P17-P16 AP50-95 latest delta=`-0.00069`，best delta=`+0.02200` at ep41，late5/late10/late20 delta=`+0.00404/+0.00166/-0.0013785`，正 delta epoch 数=`22/53`。
- img256：RGB baseline rows=`19`，best AP50/AP50-95=`0.25849/0.12197` at ep18，latest=`0.24570/0.11730`，late5/late10/late20 AP50-95=`0.11492/0.104988/0.0777426`；IR teacher baseline 尚未启动，符合 RGB 完成后自动排队的预期。
- 有效性：P16 日志无 Traceback/OOM/NaN/batch fallback；img256 RGB 日志无 Traceback/OOM/NaN/batch fallback，`strict_batch_size=True`。P17 结果仍持续写入且进程存活；本次未定位到对应 `b.log` 文件，因此有效性主要依据运行进程和 `results.csv` 连续更新，下一次 heartbeat 可补充查找实际日志路径。
- 判定：不通知用户；P17 在 ep41 附近和最近 late5/late10 有同 epoch 小幅正 delta，但绝对性能仍远低于 P16 当前收敛区间，不能称为正结果。img256 RGB baseline 正常推进。
- 下一步：继续仅监控；等待 P16/P17 至 100/150/final，并等待 img256 RGB 完成后自动进入 img256 IR teacher。

### 2026-06-24 09:13 CST - 按用户修正并行启动 img256 IR teacher baseline

- 目标：baseline / 调度
- 做了什么：用户指出 GPU0 显存明显可以同时跑另一个模态 baseline。检查后确认 GPU0 启动前约 `13971/24564 MiB`，因此新增并同步脚本 `docs/experiments/dronevehicle_method_search_20260623/img256_baselines/launch_img256_teacher_parallel_20260624.sh`，将 img256 IR teacher baseline 并行启动到 GPU0。为避免原串行 wrapper 在 RGB 结束后重复启动同名 teacher，已对 wrapper PID=`105784` 执行 `STOP`；RGB 子进程 PID=`105791` 保持运行；另有清理 watcher 在 RGB 退出后 kill stopped wrapper。
- 使用资源：`ladd4090-zw1` GPU0。IR teacher 并行 baseline PID=`106968`。并行启动后 GPU0/GPU1 约 `17662/21147 MiB`，低于 `22G` 危险线；GPU1 仍接近上限，不追加任务。
- 路径：IR teacher 结果目录 `runs_public/dronevehicle_method_search/sub2k_seed0_fullval/img256_baselines/teacher_ir/dronevehicle_sub2k_teacher_ir_yolo11n_cclkdproto_e200_b64_img256_mosaic0p0_close0_mixup0p1_s0_20260624_090613/`；日志 `logs/dronevehicle_method_search/sub2k_seed0_fullval/img256_baselines/dronevehicle_sub2k_img256_baseline_pair_yolo11n_e200_b64_s0_20260624_090613_gpu0/teacher_ir_parallel.log`。
- 结果：IR teacher 启动有效，`strict_batch_size=True`，首轮已写出 `results.csv`。09:13 快照：img256 RGB rows=`37`，best/latest AP50/AP50-95=`0.32198/0.16356`；img256 IR rows=`3`，best/latest=`0.14781/0.06096`。同时 P16 control rows=`138`，best=`0.57440/0.36940`，latest=`0.56550/0.35951`；P17 ProbeA rows=`65`，best=`0.50840/0.31748`，latest=`0.50386/0.31185`。
- 判定：用户判断正确，img256 baseline pair 应并行而非串行；当前没有 OOM、NaN 或 batch fallback 迹象。此动作只补齐 baseline，不属于新方法探索。
- 下一步：继续监控两个 img256 baseline 到 final；两者都完成并汇总后，再决定是否开 img256 ProbeA。

### 2026-06-24 09:28 CST - 研究小风洞 ProbeA 负增益并挂起 img256 ProbeA 队列

- 目标：B / 负增益诊断 / img256 ProbeA
- 做了什么：按用户要求重新检查 512 ProbeA 风洞版、同条件 det-only control、img256 双模态 baseline 进度；新增脚本 `docs/experiments/dronevehicle_method_search_20260623/img256_probeA/launch_img256_probeA_after_baselines_20260624.sh`，并同步到 `ladd4090-zw1`。该脚本不是立即训练，而是等待 img256 RGB baseline 与 img256 IR teacher baseline 都达到 `>=200` 行且 IR teacher `weights/best.pt` 存在后，再做 GPU0 显存守卫并启动一条 img256 ProbeA。
- 使用资源：`ladd4090-zw1`。当前 GPU0/GPU1 约 `17712/21309 MiB`，均未超过 `22G` 危险线；GPU1 接近上限，不追加任务。img256 ProbeA 等待队列 pid=`109546`，日志 `logs/dronevehicle_method_search/sub2k_seed0_fullval/img256_probeA/oldsplit_probeA_yoloinit_std_after_baselines_20260624_gpu0/queue.log`。队列首次记录为 `rgb_rows=101, ir_rows=65, teacher_best=1`，因此正在等待 baseline 完成，没有误启动训练。
- 512 结果：P16 YOLO-init/std-lr det-only control rows=`181`，best AP50/AP50-95=`0.57440/0.36940` at ep133，latest=`0.57227/0.36584`，late5/late10/late20 AP50-95=`0.367302/0.367390/0.367083`。P17 YOLO-init/std-lr ProbeA rows=`108`，best=`0.55374/0.35230` at ep101，latest=`0.54223/0.34086`，late5/late10/late20=`0.342634/0.342231/0.339912`。共同 epoch 上 P17-P16 AP50-95 latest delta=`-0.00940`，best delta=`+0.02200` at ep41，positive epochs=`38/108`，late5/late10/late20 delta=`-0.006834/-0.004113/-0.003547`。
- img256 baseline 最新快照：RGB rows=`104`，best AP50/AP50-95=`0.40705/0.22667` at ep91，latest=`0.40107/0.22109`，late20=`0.216949`；IR teacher rows=`68`，best=`0.47892/0.28210` at ep65，latest=`0.43306/0.24617`，late20=`0.248360`。
- 当前解释：512 小风洞 P17 不能称为正结果。它既没有超过 512 RGB baseline final AP50-95 `0.35385`，也没有在同 epoch 后段持续领先 P16；ep41 的短暂正 delta 后已经转成 late-window 负 delta。当前最可能的负增益来源按优先级拆成五类：1）数据域差异，AutoDL 正例是 OGSOD SAR/RGB，风洞是 DroneVehicle RGB/IR 子集；2）输入尺寸与协议差异，AutoDL 正例是 img256/no-mixup，而 P17 是 img512/mixup=0.1；3）A1 decomposition cache 来源差异，P17 复用了 low-lr oldsplit A1 cache，未必是 img256 当前协议最优；4）augmentation/optimizer 差异，P17 原 512 版本用 `optimizer auto`，而 baseline256/待跑 ProbeA256 使用 `optimizer SGD` 以贴近 baseline；5）实现有效性需要继续排查，但目前没有 OOM、batch fallback 或 NaN 迹象。
- 判定：其他方法族继续冻结。img256 ProbeA 是当前最干净的下一步，因为它能优先隔离 “img512/mixup 协议是否导致负迁移” 这一因素；该 run 仍必须和 img256 RGB baseline 同协议比较，不能和 512 baseline 混用。
- 下一步：等待 img256 RGB/IR baseline 都完成；队列会自动启动一条 `imgsz=256, batch=64, mixup=0.1, optimizer=SGD, YOLO-init detector, img256 IR teacher best.pt, frozen/detached reach probe` 的 ProbeA。heartbeat automation 已更新为允许并监控这一个等待队列，但不启动任何其他新方法。

### 2026-06-24 09:32 CST - heartbeat：P17 出现回升但仍未超过同条件 control

- 目标：B / ProbeA 监控 / img256 baseline
- 做了什么：按 heartbeat 检查 `ladd4090-zw1` GPU、P16/P17、img256 baseline pair、img256 ProbeA 等待队列和基础日志错误；未启动任何新方法，img256 ProbeA 队列仍按条件等待。
- 使用资源：`ladd4090-zw1`。GPU0/GPU1 约 `7881/21343 MiB`，GPU0 当前运行 img256 RGB/IR baselines，GPU1 运行 P17 512 ProbeA；P16 已完成 200 epoch。已知进程：img256 RGB PID=`105791`、img256 IR PID=`106968`、img256 ProbeA 等待队列 PID=`109546`；旧串行 wrapper PID=`105784` 仍为 stopped 状态，按设计不恢复。
- 512 结果：P16 det-only control rows=`200`，best AP50/AP50-95=`0.57440/0.36940` at ep133，final/latest=`0.57056/0.36524`，late5/late10/late20 AP50-95=`0.365368/0.365640/0.365227`。P17 ProbeA rows=`134`，best/latest=`0.56152/0.36268` at ep134，late5/late10/late20=`0.357430/0.355550/0.353690`。
- 同 epoch 对比：P17-P16 AP50-95 matched rows=`134`，latest delta=`-0.00150`，best delta=`+0.02200` at ep41，positive epochs=`43/134`，late5/late10/late20 delta=`-0.005360/-0.007701/-0.005691`。
- img256 baseline：RGB rows=`126`，best AP50/AP50-95=`0.42922/0.24136` at ep119，latest=`0.42019/0.23773`，late20=`0.233859`；IR teacher rows=`90`，best=`0.48603/0.28796` at ep84，latest=`0.47418/0.28517`，late20=`0.277267`。
- 队列状态：img256 ProbeA 尚无 `results.csv`；等待队列日志到 `09:31:38` 为 `rgb_rows=125, ir_rows=89, teacher_best=1`，符合“等两条 img256 baseline 完成后再启动”的预期。
- 有效性：img256 RGB log、img256 IR log、img256 queue outer log 最近 200 行未见 Traceback/OOM/NaN/batch fallback 关键字；strict batch 仍有效。
- 判定：P17 512 ProbeA 从低谷回升，当前 best AP50-95=`0.36268` 已超过固定 512 RGB baseline final `0.35385` 和 best `0.36087`，这是一个需要继续看的边界信号；但它仍低于同协议 P16 det-only control best `0.36940`，且 matched late-window delta 仍为负，所以暂不称为主线正结果。
- 下一步：继续等待 P17 到 final；继续等待 img256 RGB/IR baseline 完成，之后由已挂起队列自动启动一条 img256 ProbeA。若 P17 final 后仍低于 P16 late-window/peak，则优先看 img256 ProbeA 是否能复现 AutoDL img256 正向形态。

### 2026-06-24 09:43 CST - 用户查询当前进度与完成时间

- 目标：进度 / ETA
- 做了什么：按用户要求检查 `ladd4090-zw1` 当前 GPU、P16/P17、img256 RGB/IR baseline、img256 ProbeA 等待队列；按进程已运行时间和当前 rows 粗估剩余时间。
- 使用资源：`ladd4090-zw1`。09:42:55 CST 快照 GPU0/GPU1 为 `3737/22037 MiB`，util=`22%/62%`。GPU1 略高于 `22G` 警戒线但 P17 已接近 final，因此不追加任务、不主动中断。
- P16 512 det-only control：rows=`200/200`，已完成；best AP50/AP50-95=`0.57440/0.36940` at ep133，final=`0.57056/0.36524`，late20=`0.365227`。
- P17 512 ProbeA：rows=`180/200`，best AP50/AP50-95=`0.56600/0.36449` at ep178，latest=`0.56481/0.36327`，late5/late10/late20=`0.363484/0.362906/0.362344`。按当前平均约 `16.3 s/epoch`，预计约 `5.4 min` 后完成，即约 `09:48 CST`。
- P17 vs P16 同 epoch：matched rows=`180`，latest delta AP50-95=`-0.00483`，best delta=`+0.02200`，positive epochs=`44/180`，late5/late10/late20 delta=`-0.004230/-0.004535/-0.004691`。P17 绝对值已超过固定 512 RGB baseline best，但仍低于同条件 P16 control。
- img256 RGB baseline：rows=`200/200`，已完成；best AP50/AP50-95=`0.43646/0.24956` at ep179，final=`0.43565/0.24939`，late20=`0.248713`。
- img256 IR teacher baseline：rows=`183/200`，best AP50/AP50-95=`0.52231/0.32846` at ep150，latest=`0.51031/0.32315`，late20=`0.324553`。按当前平均约 `9.9 s/epoch`，预计约 `2.8 min` 后完成，即约 `09:45 CST`。
- img256 ProbeA 队列：尚未启动训练、尚无 `results.csv`；等待队列 PID=`109546` 仍存活。队列日志到 `09:41:38` 为 `rgb_rows=200, ir_rows=170, teacher_best=1`。预计 IR baseline 完成后约 `09:45 CST` 自动启动；按 512 ProbeA 速度折算，img256 ProbeA 200 epoch 粗估约 `10:15 CST` 完成。
- 有效性：img256 RGB/IR baseline log 与 queue outer log 最近未见 Traceback/OOM/NaN/batch fallback 关键词。
- 下一步：等待 IR baseline 完成和 img256 ProbeA 自动启动；下一轮监控重点是确认队列是否成功发起 ProbeA、是否 strict batch、以及早期 rows 是否正常写出。

### 2026-06-24 09:48 CST - heartbeat：256 baselines 完成，img256 ProbeA 已自动启动

- 目标：B / img256 ProbeA
- 做了什么：按 heartbeat 检查 `ladd4090-zw1` GPU、P16/P17、img256 baselines 和 img256 ProbeA。等待队列已经满足条件并自动启动一条 img256 ProbeA；未手动启动任何额外方法。
- 使用资源：`ladd4090-zw1`。09:47:56 CST 快照 GPU0/GPU1 为 `5445/22085 MiB`，util=`24%/77%`。img256 ProbeA 训练进程 PID=`112830`，结果目录 `runs_public/dronevehicle_method_search/sub2k_seed0_fullval/img256_probeA/oldsplit_probeA_yoloinit_std/ir_to_rgb/oldsplit_probeA_yoloinit_std_ir2rgb_yolo11n_e200_b64_img256_s0_20260624_094538_b/`。
- P16 512 det-only control：已完成 rows=`200/200`，best AP50/AP50-95=`0.57440/0.36940` at ep133，final=`0.57056/0.36524`，late20=`0.365227`。
- P17 512 ProbeA：已写满 rows=`200/200`，best AP50/AP50-95=`0.56600/0.36449` at ep178，final/latest=`0.56438/0.36404`，late5/late10/late20=`0.364072/0.363965/0.363800`。训练进程仍短暂存活，应是在收尾保存/验证。
- P17 vs P16 同 epoch：matched rows=`200`，latest delta AP50-95=`-0.00120`，best delta=`+0.02200`，positive epochs=`44/200`，late5/late10/late20 delta=`-0.001296/-0.001675/-0.001427`。判定仍为“超过固定 baseline，但未超过同管线 det-only control”。
- img256 RGB baseline：已完成 rows=`200/200`，best AP50/AP50-95=`0.43646/0.24956` at ep179，final=`0.43565/0.24939`，late20=`0.248713`。
- img256 IR teacher baseline：已完成 rows=`200/200`，best AP50/AP50-95=`0.52231/0.32846` at ep150，final=`0.51159/0.32321`，late20=`0.323573`。
- img256 ProbeA：已启动 rows=`10/200`，best/latest AP50/AP50-95=`0.19348/0.08493` at ep10，late5/late10/late20=`0.068822/0.047066/0.047066`。按当前早期平均速度粗估 ETA 约 `10:31 CST`，但前 10 epoch 初始化/缓存开销较大，后续 ETA 可能变化。
- 有效性：P17 outer、img256 RGB/IR baseline log、img256 ProbeA queue outer 最近未见 Traceback/OOM/NaN/batch fallback 关键词。
- 下一步：继续监控 img256 ProbeA 到 50/100/150/final；先看它是否能超过 img256 RGB baseline final `0.24939`，再看是否超过同协议 baseline best `0.24956`。若后续需要严格判定方法增益，还应补一个 img256 同管线 det-only control，否则只能先判断是否超过正式 img256 baseline。

### 2026-06-24 09:54 CST - 用户确认后启动 img256 同管线 det-only control

- 目标：B / img256 ProbeA 严格对照
- 做了什么：用户询问“256 的 probe 和对照可以开了吧”。检查后确认 img256 ProbeA 已由等待队列自动启动，因此只补开一条同管线 det-only control，脚本为 `docs/experiments/dronevehicle_method_search_20260623/img256_controls/launch_img256_detonly_yoloinit_std_control_20260624.sh`。该 control 使用与 ProbeA 相同的 LADD B 训练管线、paired data、imgsz=`256`、batch=`64`、SGD、mixup=`0.1`、strict batch、YOLO-init detector 和同一份 img256 IR teacher best.pt，但关闭所有 KD / reach / rec / taskL / inner rank-match 损失，并显式使用 `--ladd-b-det-only --b-reset-student-from-scratch --b-detector-source yolo11n.pt --no-b-load-student-split --no-b-load-student-reachability`。
- 使用资源：`ladd4090-zw1`。09:54 CST GPU0/GPU1 为 `10309/9771 MiB`，util=`74%/6%`，两卡均低于 `22G` 危险线。img256 ProbeA 主进程 PID=`112830`，img256 det-only control 主进程 PID=`114176`（外层 bash PID=`114172`）。
- 路径：img256 ProbeA 结果目录 `runs_public/dronevehicle_method_search/sub2k_seed0_fullval/img256_probeA/oldsplit_probeA_yoloinit_std/ir_to_rgb/oldsplit_probeA_yoloinit_std_ir2rgb_yolo11n_e200_b64_img256_s0_20260624_094538_b/`；img256 det-only control 结果目录 `runs_public/dronevehicle_method_search/sub2k_seed0_fullval/img256_controls/detonly_yoloinit_std/ir_to_rgb/detonly_yoloinit_std_ir2rgb_yolo11n_e200_b64_img256_s0_20260624_095103_b/`。
- 当前结果：img256 ProbeA rows=`44/200`，best/latest AP50/AP50-95=`0.32877/0.16992` at ep44，late5/late10/late20 AP50-95=`0.157006/0.156500/0.146645`。img256 det-only control rows=`12/200`，best AP50/AP50-95=`0.18458/0.08188` at ep9，latest=`0.18261/0.07986`，late5/late10/late20=`0.075202/0.059929/0.053828`。共同前 12 epoch 上 ProbeA-control latest delta AP50-95=`+0.008760`，但仍属极早期，不能据此判正。
- 固定 img256 baseline：RGB baseline 已完成，best/final AP50/AP50-95=`0.43646/0.24956` / `0.43565/0.24939`；IR teacher baseline 已完成，best/final=`0.52231/0.32846` / `0.51159/0.32321`。
- 有效性：ProbeA 与 control 日志当前未见 Traceback/OOM/NaN/batch fallback 关键词；两条 run 均按 strict batch 启动。512 P16/P17 已完成，P17 final AP50-95=`0.36404`，仍低于同管线 P16 final `0.36524` 和 best `0.36940`，所以 512 ProbeA 仍不判正。
- 判定：256 ProbeA 和严格对照都已经开齐。后续 img256 ProbeA 的正结果标准应至少同时满足：超过固定 img256 RGB baseline final/best AP50-95=`0.24939/0.24956`，并在同 epoch、late-window 或 final 上超过 img256 同管线 det-only control；只超过固定 baseline 但不超过同管线 control，仍只能算训练管线差异而不是方法增益。
- 下一步：只监控 img256 ProbeA 与 img256 det-only control，不再启动新的方法探索；重点在 rows 50/100/150/final 汇报两者同 epoch delta、late5/late10/late20 和日志有效性。

### 2026-06-24 10:02 CST - heartbeat：img256 ProbeA 暂未达到正结果标准

- 目标：B / img256 ProbeA 监控
- 做了什么：按自动化要求检查 `ladd4090-zw1` GPU、img256 ProbeA、img256 同管线 det-only control、固定 img256 RGB baseline，以及两条 active run 的日志错误；未启动任何新实验。
- 使用资源：`ladd4090-zw1`。10:02 CST GPU0/GPU1 为 `10319/9877 MiB`，util=`82%/16%`，两卡均低于 `22G` 危险线。img256 ProbeA 主进程 PID=`112830` 仍运行；img256 det-only control 主进程 PID=`114176` 仍运行。
- img256 ProbeA：rows=`90/200`，best AP50/AP50-95=`0.40000/0.22127` at ep89，latest=`0.39340/0.21646`，late5/late10/late20 AP50-95=`0.217072/0.212705/0.208108`。
- img256 det-only control：rows=`60/200`，best/latest AP50/AP50-95=`0.36972/0.19502` at ep60，late5/late10/late20=`0.184852/0.179801/0.170050`。
- 同 epoch 对比：共同前 60 epoch 上 ProbeA-control AP50-95 latest delta=`-0.011680`，best delta=`+0.030190`，positive epochs=`35/60`，late5/late10/late20 delta=`+0.003310/+0.002163/+0.001014`。也就是说局部 late-window 仍有一点正差，但最新 epoch 已反转为负。
- 固定 img256 RGB baseline：best/final AP50-95=`0.24956/0.24939`。当前 ProbeA latest/best AP50-95 相比固定 baseline final 分别为 `-0.032930/-0.028120`，相比固定 baseline best 分别为 `-0.033100/-0.028290`。
- 有效性：ProbeA 与 control 的 master/outer/queue 日志均未见 Traceback/OOM/NaN/batch fallback 关键词。
- 判定：当前不能通知正结果。ProbeA 还没有超过固定 img256 RGB baseline，更没有稳定超过同管线 det-only control；但实验还在中段，继续看 100/150/final。
- 下一步：继续只监控这两条 active run；若 ProbeA 在 100/150/final 超过固定 baseline 且同时领先 det-only control，再通知用户，否则保持安静记录。

### 2026-06-24 10:17 CST - heartbeat：img256 ProbeA 出现边界正信号但尚未 final 定论

- 目标：B / img256 ProbeA 监控
- 做了什么：按自动化要求检查 `ladd4090-zw1` GPU、img256 ProbeA、img256 同管线 det-only control、固定 img256 RGB baseline，以及两条 active run 的日志错误；未启动任何新实验。
- 使用资源：`ladd4090-zw1`。10:17 CST GPU0/GPU1 为 `10323/10087 MiB`，util=`88%/15%`，两卡均低于 `22G` 危险线。img256 ProbeA 主进程 PID=`112830` 仍运行；img256 det-only control 主进程 PID=`114176` 仍运行。
- img256 ProbeA：rows=`180/200`，best AP50/AP50-95=`0.45053/0.25660` at ep171，latest=`0.44600/0.25456`，late5/late10/late20 AP50-95=`0.254492/0.254986/0.254640`。
- img256 det-only control：rows=`156/200`，best AP50/AP50-95=`0.45060/0.25594` at ep140，latest=`0.44772/0.25302`，late5/late10/late20=`0.252516/0.252291/0.250570`。
- 同 epoch 对比：共同前 156 epoch 上 ProbeA-control AP50-95 latest delta=`-0.001900`，best delta=`+0.030190`，positive epochs=`77/156`，late5/late10/late20 delta=`+0.000836/+0.000510/+0.000019`。也就是说 latest 仍为负，但 late-window 已经回到极小正差。
- 固定 img256 RGB baseline：best/final AP50-95=`0.24956/0.24939`。当前 ProbeA latest/best AP50-95 相比固定 baseline final 分别为 `+0.005170/+0.007210`，相比固定 baseline best 分别为 `+0.005000/+0.007040`。
- 有效性：ProbeA 与 control 的 master/outer/queue 日志均未见 Traceback/OOM/NaN/batch fallback 关键词。
- 判定：这是第一个 img256 ProbeA 的边界正信号：它已经超过固定 img256 baseline，并且在 late-window 上相对同管线 control 是极小正差；但由于共同 latest delta 仍为负、control 尚未 final，不能把它表述为稳定正结果，只能表述为“边界候选，需要 final 确认”。
- 下一步：继续监控 final。若 ProbeA final/best/late-window 仍高于固定 baseline 且相对 control 不低，则通知用户；若 final 回落或 control 追平/超过，则按未稳定正结果记录。

### 2026-06-24 10:24 CST - img256 ProbeA 与同管线 det-only control 均完成

- 目标：B / img256 ProbeA final 判定
- 做了什么：按用户询问“应该结束了吧”后继续等待至两条 img256 run 都写满 `200/200`，检查 GPU/进程、最终 `results.csv` 与日志错误关键词；未启动任何新实验。
- 使用资源：`ladd4090-zw1`。10:24 CST GPU0 已回到约 `1 MiB` 空闲；GPU1 仍约 `10193 MiB`，来自其它旧进程，不是本轮 img256 ProbeA/control。两条 img256 run 已完成，未见对应训练进程继续运行。
- 固定 img256 RGB baseline：best/final AP50/AP50-95=`0.43646/0.24956` / `0.43565/0.24939`。
- img256 ProbeA final：rows=`200/200`，best AP50/AP50-95=`0.45053/0.25660` at ep171，final/latest=`0.44846/0.25568`，late5/late10/late20 AP50-95=`0.255082/0.255101/0.255157`。
- img256 同管线 det-only control final：rows=`200/200`，best AP50/AP50-95=`0.45242/0.25755` at ep182，final/latest=`0.45131/0.25568`，late5/late10/late20 AP50-95=`0.255732/0.255503/0.255431`。
- ProbeA vs 固定 baseline：ProbeA final AP50-95 比固定 RGB baseline final/best 分别高 `+0.00629/+0.00612`；ProbeA best 比固定 RGB baseline final/best 分别高 `+0.00721/+0.00704`。
- ProbeA vs 同管线 det-only control：共同 200 epoch 下 final delta AP50-95=`0.00000`，ProbeA best-control best=`-0.00095`，late5/late10/late20 delta=`-0.000650/-0.000402/-0.000275`。matched-epoch 轨迹里曾有 best delta=`+0.030190`，positive epochs=`105/200`，但最终与 late-window 都没有保持优势。
- 有效性：ProbeA 与 control 的 master/outer/queue 日志均未见 Traceback/OOM/NaN/batch fallback 关键词；strict batch 有效。
- 判定：img256 ProbeA 不能按严格标准称为方法正结果。它超过固定 img256 baseline，但同管线 det-only control 也超过固定 baseline，且 control 在 best 与 late-window 上略高，final AP50-95 打平。因此当前最稳妥结论是：`imgsz=256 + LADD B 同管线/协议` 本身比固定 baseline 更强，ProbeA 没有提供额外稳定增益。
- 下一步：本轮 narrowed ProbeA 监控已完成；停止 15 分钟自动化。若继续探索，下一步应优先分析为什么同管线 det-only control 超过固定 baseline，再决定是否把主线比较标准迁移到同管线 control。

### 2026-06-24 10:31 CST - 生成 512/256 ProbeA 曲线图

- 目标：可视化 / ProbeA 对照与 baseline 比较
- 做了什么：按用户要求，将 512 和 256 的 ProbeA 曲线同时与同管线 det-only control、固定 RGB baseline、IR teacher baseline 放在图中比较；每个输入尺寸还附带差值曲线 `ProbeA - det-only control` 与 `ProbeA - RGB baseline`，避免只看绝对曲线时误判。
- 数据来源：远端 `ladd4090-zw1` 的 `results.csv` 已拉取到本地 `docs/experiments/dronevehicle_method_search_20260623/plot_data/probeA_curves/`。512 baseline 使用 `runs_public/cross_dataset/cclkd_yolo11n/dronevehicle_sub2k_seed0/baselines/student_rgb/...` 与对应 IR teacher；512 ProbeA/control 使用本轮 `oldsplit_probeA_yoloinit_std` 与 `reachable_fused_shared/c0_yoloinit_std`；256 使用本轮 `img256_baselines`、`img256_probeA`、`img256_controls`。
- 输出文件：`docs/experiments/dronevehicle_method_search_20260623/figures/probeA_curves/dronevehicle_probeA_img512_diagnostic.svg`、`docs/experiments/dronevehicle_method_search_20260623/figures/probeA_curves/dronevehicle_probeA_img256_diagnostic.svg`、`docs/experiments/dronevehicle_method_search_20260623/figures/probeA_curves/dronevehicle_probeA_img512_img256_overview.svg`。生成脚本为 `docs/experiments/dronevehicle_method_search_20260623/gen_probeA_curve_figures.py`。
- 图中摘要：img512 ProbeA best/final AP50-95=`0.36449/0.36404`，control=`0.36940/0.36524`，RGB baseline=`0.36087/0.35385`；Probe-control final/best/late20 delta=`-0.00120/-0.00491/-0.00143`，Probe-RGB baseline best/final delta=`+0.00362/+0.01019`。img256 ProbeA best/final=`0.25660/0.25568`，control=`0.25755/0.25568`，RGB baseline=`0.24956/0.24939`；Probe-control final/best/late20 delta=`+0.00000/-0.00095/-0.00027`，Probe-RGB baseline best/final delta=`+0.00704/+0.00629`。
- 判定：512 与 256 的图都支持同一个结论：ProbeA 都超过固定 RGB baseline，但同管线 det-only control 也超过固定 baseline；ProbeA 没有稳定超过同管线 control。因此目前正向信号应归因于同管线协议/训练设置，而不是 ProbeA 方法增益。

### 2026-06-24 10:45 CST - AutoDL2 OGSOD ProbeA 正向结果差异核查

- 目标：对照 AutoDL 双卡服务器上“明显正”的 ProbeA 结果，分析它与 DroneVehicle 小风洞的协议差异。
- 做了什么：只读检查 `autodl-nmb1` 的 `runs_public/paper/ogsod_hbb_nomosaic/no_reload_existing_cache/yolo11n/seed0/20260623_194650/` 组，解析 `results.csv`、`manifest.txt` 和 `b.log` 中的实际命令；未启动或停止任何实验。
- 当前状态：AutoDL2 两张 4090 正在运行该组，10:40 CST GPU0/GPU1 分别约 `12621/8904 MiB`，util 均约 `99%`。
- 结果摘要：`yolo_probeA` rows=`554/700`，best/latest AP50/AP50-95=`0.81480/0.54460` / `0.81577/0.54460`，late20=`0.542625`。`reload_detonly` rows=`616/700`，best/latest=`0.80823/0.53660`，late20=`0.535418`。共同前 554 epoch 上 `yolo_probeA - reload_detonly` AP50-95 latest delta=`+0.015550`，late20 delta=`+0.016021`，positive epochs=`350/554`，run-best delta=`+0.008000`。`yolo_dynamic` rows=`525/700`，best/latest=`0.81237/0.53387`，共同前 525 epoch 上 `yolo_probeA - yolo_dynamic` latest delta=`+0.004470`，late20 delta=`+0.004338`，run-best delta=`+0.010730`。
- AutoDL 关键协议：数据集是 OGSOD HBB SAR/RGB，`data=configs/paper/datasets/ogsod_hbb_sar.yaml`，teacher 是 RGB baseline；`imgsz=256`，`epochs=700`，`batch=64`，`mosaic=0.0`，`mixup=0.0`，`optimizer=auto` 并在日志中实际选择 `MuSGD(lr=0.01, momentum=0.9)`，`strict_batch_size=False`。
- AutoDL ProbeA 开关：`yolo_probeA` 使用 `--model yolo11n.pt` 和 `--b-detector-source yolo11n.pt`，即 YOLO-init；同时使用 existing OGSOD A1 cache：`b_decomp_source=.../a1_decomp_cache/ogsod_hbb_ladd_a1_decomp_from_sar_baseline_yolo11n_s0_20260623_192641_img256_a1_e10_b64_s0_gpu0/weights/best.pt`，并开启 `--ladd-b-a2-core --ladd-b-frozen-reach-probe --ladd-b-detach-reach-probe --b-load-student-reachability`。
- 主要差异/混淆点：1）AutoDL 是 OGSOD SAR/RGB，不是 DroneVehicle RGB/IR sub2k；2）AutoDL 是 `imgsz=256 + mixup=0.0 + optimizer=auto/MuSGD + 700 epoch`，小风洞 256 是 `mixup=0.1 + explicit SGD momentum=0.937 + 200 epoch`；3）AutoDL 使用 img256 专门的 OGSOD A1 cache，小风洞 256 复用了当前风洞 oldsplit A1/cache；4）AutoDL 的明显正差主要是对 `reload_detonly/warm100 init`，缺少严格的 `YOLO-init det-only` 同管线 control；5）不过 AutoDL `yolo_probeA` 相对同为 YOLO-init 的 `yolo_dynamic` 也为正，说明 frozen/detached ProbeA 设计在 OGSOD 上可能确实优于 dynamic reach 版本。
- 判定：AutoDL 正结果可信地说明 “OGSOD + img256 + no-mixup + existing A1 cache + frozen/detached ProbeA” 是有效组合，但目前不能直接外推为 ProbeA 在 DroneVehicle 小风洞上有方法增益。要关闭混淆，AutoDL 需要补 `yolo_detonly` control；DroneVehicle 若要复刻 AutoDL 条件，则应优先试 `img256 + mixup=0.0 + optimizer=auto/MuSGD + img256-specific A1 cache + YOLO-init det-only control`。

### 2026-06-24 10:57 CST - 在 ladd4090-zw1 启动 AutoDL 条件补实验 1/2

- 目标：补实验 1 / OGSOD AutoDL 条件缺失对照；补实验 2 / DroneVehicle 小风洞复刻 AutoDL-like `img256 + mixup=0.0 + optimizer=auto/MuSGD` 条件。
- 做了什么：将 AutoDL2 上 OGSOD ProbeA 使用的 A1 cache 与 RGB teacher 权重复制到 `ladd4090-zw1`，并用脚本 `docs/experiments/dronevehicle_method_search_20260623/autodl_condition_followups/launch_ogsod_yolodet_and_drone_nomix_20260624.sh` 启动两条补实验链。脚本已加入 `--strict-batch-size`，避免 batch fallback 污染判断。
- 使用资源：`ladd4090-zw1`。启动前 GPU0 约 `1 MiB` 空闲，GPU1 约 `10663 MiB`，来自正在运行的 OGSOD RGB baseline。分配为 GPU0 跑 OGSOD `yolo_detonly`，GPU1 跑 DroneVehicle no-mix A1，A1 完成后自动并行启动 DroneVehicle no-mix ProbeA 与 det-only control。显存守卫设置 `MAX_AFTER_MB=22000`。
- 补实验 1 / OGSOD `yolo_detonly`：PID=`124542`（外层 bash），主训练进程 PID=`124547`。结果目录 `runs_public/dronevehicle_method_search/ogsod_nomosaic_autodl_condition/yolo_detonly/ogsod_nomix_yolo_detonly_existingcache_yolo11n_e700_b64_img256_s0_20260624_105706_b/`。关键协议：OGSOD SAR/RGB，`imgsz=256`，`epochs=700`，`batch=64`，`mixup=0.0`，`optimizer=auto`，`model=yolo11n.pt`，`b_detector_source=yolo11n.pt`，`b_decomp_source` 使用 AutoDL OGSOD img256 A1 cache，开启 `--ladd-b-det-only --b-load-student-reachability`。
- 补实验 2 / DroneVehicle no-mix chain：wrapper PID=`124548`，A1 主训练进程 PID=`124564`。A1 结果目录 `runs_public/dronevehicle_method_search/sub2k_seed0_fullval/img256_nomix_autodl_condition/a1_cache/dronevehicle_img256_nomix_a1_from_img256rgb_yolo11n_b64_s0_20260624_105707_a1/`。A1 完成后会自动启动 ProbeA 目录 `runs_public/dronevehicle_method_search/sub2k_seed0_fullval/img256_nomix_autodl_condition/probeA/ir_to_rgb/` 与 det-only control 目录 `runs_public/dronevehicle_method_search/sub2k_seed0_fullval/img256_nomix_autodl_condition/detonly/ir_to_rgb/`。
- 当前健康检查：启动后约 16 秒检查，OGSOD 日志已进入 Ultralytics 初始化并显示 `strict_batch_size=True`；DroneVehicle wrapper 日志显示已进入 no-mix A1。GPU 约为 `603/10679 MiB`，两个新训练进程均存活；暂时还未写出 `results.csv`，属于正常早期状态。
- 注意事项：补实验 1 用于关闭 AutoDL OGSOD ProbeA 缺少 `YOLO-init det-only` control 的混淆。补实验 2 的 DroneVehicle no-mix A1 暂时沿用已有 img256 RGB baseline / IR teacher 作为起点和 teacher，不是完整重训 no-mix baseline；因此它用于判断 AutoDL-like 训练设置是否改变 ProbeA/control 关系，不能单独替代正式 no-mix baseline。
- 下一步：开启 15 分钟监控。每次检查 GPU、PID、日志是否出现 Traceback/OOM/NaN/batch fallback，解析 OGSOD `yolo_detonly`、DroneVehicle no-mix A1/ProbeA/control 的 rows、best/latest/final、late5/late10/late20；OGSOD 重点比较 AutoDL `yolo_probeA` 与新增 `yolo_detonly`，DroneVehicle 重点比较 no-mix ProbeA 是否同时超过固定 img256 baseline 与 no-mix det-only control。

### 2026-06-24 11:00 CST - AutoDL 条件补实验启动后健康检查

- 目标：确认补实验 1/2 没有秒退、路径错误、OOM 或 batch fallback。
- 当前资源：`ladd4090-zw1` GPU0/GPU1 约 `3563/14249 MiB`，util 约 `21%/86%`。GPU1 同时包含原有 OGSOD RGB baseline 与 DroneVehicle no-mix A1，仍低于 `22G` 危险线。
- 进程状态：OGSOD `yolo_detonly` 外层 PID=`124542`、主训练 PID=`124547` 均存活；DroneVehicle no-mix wrapper PID=`124548`、A1 主训练 PID=`124564` 均存活。
- rows：OGSOD `yolo_detonly` 已写出 rows=`1/700`；DroneVehicle no-mix A1 已写出 rows=`14/50`。DroneVehicle ProbeA/control 尚未启动，符合“等 A1 best.pt 生成后自动启动”的链式逻辑。
- 有效性：OGSOD 日志显示 `strict_batch_size=True` 且已进入训练；DroneVehicle A1 正常写 `results.csv`。当前未见需要人工介入的错误。
- 下一步：等待 15 分钟 heartbeat 继续监控。A1 完成后确认 ProbeA/control 是否自动启动，并重点看是否出现 OOM、batch fallback 或同管线 control 混淆。

### 2026-06-24 11:02 CST - 按用户修正将 OGSOD 对照改为 800 epoch

- 目标：补实验 1 / OGSOD `yolo_detonly` 对照协议修正。
- 背景：用户指出 OGSOD 实验长度选 `800` 更好。此前启动的 OGSOD `e700` 对照只跑到 rows=`2/700`，仍处于极早期。
- 做了什么：只停止刚才启动的 OGSOD `e700` 对照进程树，未碰 GPU1 的 DroneVehicle no-mix chain，也未碰原有 OGSOD RGB baseline。`e700` 目录 `runs_public/dronevehicle_method_search/ogsod_nomosaic_autodl_condition/yolo_detonly/ogsod_nomix_yolo_detonly_existingcache_yolo11n_e700_b64_img256_s0_20260624_105706_b/` 标记为 INVALID / aborted，不用于任何结论。
- 新实验：重新启动 OGSOD `yolo_detonly` e800 对照，外层 PID=`126129`，主训练进程 PID=`126131`。结果目录 `runs_public/dronevehicle_method_search/ogsod_nomosaic_autodl_condition/yolo_detonly/ogsod_nomix_yolo_detonly_existingcache_yolo11n_e800_b64_img256_s0_20260624_110207_b/`，日志 `logs/dronevehicle_method_search/sub2k_seed0_fullval/autodl_condition_followups_20260624/ogsod_yolo_detonly_gpu0/ogsod_nomix_yolo_detonly_existingcache_yolo11n_e800_b64_img256_s0_20260624_110207.outer.log`。
- 协议确认：日志与进程参数均显示 `--epochs 800 --patience 800 --phase-min-epochs 800`，`strict_batch_size=True`，`imgsz=256`，`batch=64`，`mixup=0.0`，`optimizer=auto`，`model=yolo11n.pt`，`b_detector_source=yolo11n.pt`，`b_decomp_source` 仍为 AutoDL OGSOD img256 A1 cache。
- 当前资源：启动验证时 GPU0/GPU1 约 `603/14295 MiB`；GPU0 e800 正在初始化，GPU1 DroneVehicle no-mix A1 继续运行。
- 监控更新：15 分钟 heartbeat 已更新为监控 e800 路径，并明确忽略 e700 早停残留。后续 OGSOD 对比时以 e800 为准；若和 AutoDL 现有 e700 ProbeA 做同 epoch 对比，需要在报告里注明 epoch-length 差异。

### 2026-06-24 11:13 CST - heartbeat：小风洞 no-mix A1 完成并自动启动 B

- 目标：AutoDL 条件补实验监控，不启动新实验。
- 使用资源：`ladd4090-zw1` GPU0/GPU1 约 `3587/20332 MiB`，util 约 `7%/67%`。GPU1 接近但仍低于 `22G` 危险线；不追加任何任务。
- OGSOD e800 `yolo_detonly`：rows=`7/800`，best/latest AP50/AP50-95=`0.20598/0.08191` at ep7，late5/late10/late20 AP50-95=`0.05230/0.04930/0.04930`。与 AutoDL2 `yolo_probeA` 的共同前 7 epoch 粗比：AutoDL ProbeA ep7 AP50/AP50-95=`0.20754/0.07852`，`ProbeA - detonly` latest delta AP50-95=`-0.00339`，matched best delta=`+0.02061`；当前过早，不做结论。
- DroneVehicle no-mix A1：已完成 rows=`50/50`，`best.pt` 已生成，wrapper 日志显示 `11:04:57` 自动启动 B。
- DroneVehicle no-mix ProbeA：rows=`38/200`，best AP50/AP50-95=`0.35462/0.19202` at ep35，latest=`0.34017/0.17915`，late5/late10/late20 AP50-95=`0.179878/0.171866/0.165549`。
- DroneVehicle no-mix det-only control：rows=`38/200`，best AP50/AP50-95=`0.35726/0.19010` at ep37，latest=`0.33661/0.17984`，late5/late10/late20 AP50-95=`0.184416/0.177385/0.165340`。
- ProbeA vs det-only：matched rows=`38`，latest delta AP50-95=`-0.00069`，matched best delta=`+0.01850`，positive epochs=`16/38`，late5/late10/late20 delta=`-0.004538/-0.005519/+0.000210`。两条当前都低于固定 img256 RGB baseline final/best AP50-95=`0.24939/0.24956`，所以不能称正结果。
- 有效性：外层日志未见 Traceback/OOM/RuntimeError/batch fallback；`ladd_diagnostics.csv` 中出现的 `nan` 是 det-only/reach 关闭项的占位，不作为故障信号。
- 下一步：继续监控到 DroneVehicle no-mix B 的 50/100/150/final；若 ProbeA 超过固定 baseline 且同时超过 no-mix det-only control，再通知用户。当前只是正常推进，无需人工介入。

### 2026-06-24 11:30 CST - 在 ladd4090-zw1 补启动 OGSOD e800 YOLO-init ProbeA

- 目标：关闭 AutoDL2 下午可能关机导致 `yolo_probeA` 主实验跑不完的风险，并给 `ladd4090-zw1` 上已启动的 OGSOD e800 YOLO-init det-only control 补齐同服务器、同数据、同 A1 cache、同 epoch 的 ProbeA 主实验。
- 背景：用户指出 AutoDL2 双卡服务器下午会关机，因此不能只依赖 AutoDL2 上当前 rows 约 500+ 的 OGSOD `yolo_probeA` 作为主证据；应在 `ladd4090-zw1` 双 4090 上继续补充完整实验。这个判断是正确的。
- 做了什么：新增并同步脚本 `docs/experiments/dronevehicle_method_search_20260623/autodl_condition_followups/launch_ogsod_probeA_e800_20260624.sh`，在 `ladd4090-zw1` 的 GPU0 启动 OGSOD `yolo_probeA` e800 复刻。未停止任何现有任务，未触碰 GPU1 上的 OGSOD RGB baseline 与 DroneVehicle no-mix B 组。
- 新实验：外层 PID=`131299`，主训练进程 PID=`131304`。结果目录 `runs_public/dronevehicle_method_search/ogsod_nomosaic_autodl_condition/yolo_probeA/ogsod_nomix_yolo_probeA_existingcache_yolo11n_e800_b64_img256_s0_20260624_113044_b/`，日志 `logs/dronevehicle_method_search/sub2k_seed0_fullval/autodl_condition_followups_20260624/ogsod_yolo_probeA_gpu0/ogsod_nomix_yolo_probeA_existingcache_yolo11n_e800_b64_img256_s0_20260624_113044.outer.log`。
- 协议确认：OGSOD SAR/RGB，`imgsz=256`，`epochs=800`，`batch=64`，`strict_batch_size=True`，`mixup=0.0`，`optimizer=auto` 并在日志中实际选择 `MuSGD(lr=0.01, momentum=0.9)`，`model=yolo11n.pt`，`b_detector_source=yolo11n.pt`，`b_decomp_source` 为 AutoDL OGSOD img256 A1 cache；开启 `--ladd-b-a2-core --ladd-b-frozen-reach-probe --ladd-b-detach-reach-probe --b-load-student-reachability`。
- 健康检查：启动后约 1 分钟，GPU0/GPU1 约 `7123/20562 MiB`，util 约 `93%/99%`；GPU0 同时跑 OGSOD e800 det-only 与 e800 ProbeA，总显存远低于 `22G` 危险线。ProbeA 日志已进入第 1 epoch 训练，显示 split-load 初始化成功、`strict_batch_size=True`、无 Traceback/OOM/batch fallback。
- 下一步：后续 heartbeat 必须同时监控 `ladd4090-zw1` 上 OGSOD e800 `yolo_probeA` 与 e800 `yolo_detonly`，报告 rows、best/latest/final、late5/late10/late20，以及同 epoch `ProbeA - detonly` delta。AutoDL2 上的 e700 `yolo_probeA` 继续作为外部早期证据，但若 AutoDL2 关机，主结论转向 `ladd4090-zw1` e800 复刻组。

### 2026-06-24 11:35 CST - heartbeat：OGSOD e800 ProbeA/control 与小风洞 no-mix B 正常推进

- 目标：AutoDL 条件补实验监控，不启动新实验。
- 使用资源：`ladd4090-zw1` GPU0/GPU1 约 `7853/20606 MiB`，util 约 `92%/98%`。GPU0 同时跑 OGSOD e800 det-only 与 ProbeA，显存仍安全；GPU1 继续跑 OGSOD RGB baseline 与 DroneVehicle no-mix B 组，接近但低于 `22G` 危险线，不追加任务。
- OGSOD e800 `yolo_detonly`：rows=`22/800`，best AP50/AP50-95=`0.37248/0.16705` at ep21，latest=`0.36006/0.16528`，late5/late10/late20 AP50-95=`0.159306/0.148221/0.114987`。
- OGSOD e800 `yolo_probeA`：rows=`2/800`，best/latest AP50/AP50-95=`0.10241/0.03453` at ep2，late5/late10/late20 AP50-95=`0.028230/0.028230/0.028230`。
- OGSOD ProbeA vs det-only：共同 rows=`2`，latest delta AP50-95=`+0.00536`，positive epochs=`1/2`，late5/late10/late20 delta=`-0.013585/-0.013585/-0.013585`。当前过早，只记录健康状态，不做正负结论。
- DroneVehicle no-mix ProbeA：rows=`148/200`，best AP50/AP50-95=`0.45182/0.25857` at ep144，latest=`0.44396/0.25539`，late5/late10/late20 AP50-95=`0.255462/0.255022/0.253254`。
- DroneVehicle no-mix det-only control：rows=`154/200`，best AP50/AP50-95=`0.45577/0.26302` at ep143，latest=`0.45412/0.26023`，late5/late10/late20 AP50-95=`0.259028/0.259371/0.259381`。
- DroneVehicle ProbeA vs det-only：共同 rows=`148`，latest delta AP50-95=`-0.00509`，matched best delta=`+0.01888`，positive epochs=`53/148`，late5/late10/late20 delta=`-0.005058/-0.004445/-0.004940`。ProbeA 已超过固定 img256 RGB baseline best/final AP50-95=`0.24956/0.24939`，但同管线 det-only control 更高，因此仍不能称方法正结果。
- 有效性：OGSOD det-only、OGSOD ProbeA、DroneVehicle ProbeA、DroneVehicle det-only 日志均未见 Traceback/CUDA OOM/RuntimeError/NaN/batch fallback 关键词；主训练 PID 均存活。
- 下一步：继续 15 分钟 heartbeat 监控。重点等 OGSOD e800 ProbeA/control 到 50+ epoch 后再看同 epoch 与 late-window 差异；DroneVehicle no-mix B 继续等 final，当前趋势偏向同管线 det-only 更强。

### 2026-06-24 11:50 CST - heartbeat：DroneVehicle no-mix B final，ProbeA 未超过同管线 det-only

- 目标：AutoDL 条件补实验监控，不启动新实验。
- 使用资源：`ladd4090-zw1` GPU0/GPU1 约 `7853/11351 MiB`，util 约 `96%/17%`。GPU0 继续跑 OGSOD e800 det-only + ProbeA；GPU1 上 DroneVehicle no-mix B 组已结束，剩余主要是原 OGSOD RGB baseline。
- OGSOD e800 `yolo_detonly`：rows=`32/800`，best/latest AP50/AP50-95=`0.45615/0.22628` at ep32，late5/late10/late20 AP50-95=`0.211040/0.197517/0.172869`。
- OGSOD e800 `yolo_probeA`：rows=`10/800`，best/latest AP50/AP50-95=`0.28357/0.11765` at ep10，late5/late10/late20 AP50-95=`0.093058/0.060756/0.060756`。
- OGSOD ProbeA vs det-only：共同 rows=`10`，latest delta AP50-95=`-0.00015`，positive epochs=`4/10`，matched best delta=`+0.01057`，late5/late10/late20 delta=`-0.000612/-0.004810/-0.004810`。当前仍处于极早期，不做方法结论。
- DroneVehicle no-mix ProbeA final：rows=`200/200`，best AP50/AP50-95=`0.45158/0.25946` at ep182，final/latest=`0.45002/0.25816`，late5/late10/late20 AP50-95=`0.258192/0.258052/0.258452`。
- DroneVehicle no-mix det-only control final：rows=`200/200`，best AP50/AP50-95=`0.45701/0.26321` at ep159，final/latest=`0.45455/0.26221`，late5/late10/late20 AP50-95=`0.262108/0.261848/0.261803`。
- DroneVehicle ProbeA vs det-only：共同 rows=`200`，final delta AP50-95=`-0.00405`，matched best delta=`+0.01888`，positive epochs=`53/200`，late5/late10/late20 delta=`-0.003916/-0.003796/-0.003352`。
- 判定：DroneVehicle no-mix AutoDL-like 条件下，ProbeA 超过固定 img256 RGB baseline best/final AP50-95=`0.24956/0.24939`，但同管线 det-only control 更高且 final/late-window 均领先；因此仍不能称为 ProbeA 方法正结果。这个结果支持此前判断：小风洞上提升主要来自同管线训练设置，而不是 ProbeA 本身。
- 有效性：OGSOD det-only、OGSOD ProbeA、DroneVehicle ProbeA、DroneVehicle det-only 日志均未见 Traceback/CUDA OOM/RuntimeError/NaN/batch fallback 关键词；OGSOD 两个主训练 PID 存活，DroneVehicle no-mix B 两个主训练 PID 已正常退出。
- 下一步：继续 heartbeat，但监控重点收窄到 `ladd4090-zw1` 上 OGSOD e800 ProbeA/control。待 OGSOD 到 50/100 epoch 后再检查 ProbeA 是否出现 AutoDL2 类似的后期追上迹象。

### 2026-06-24 12:05 CST - heartbeat：OGSOD e800 ProbeA/control 正常推进，仍属早期

- 目标：OGSOD e800 ProbeA/control 监控，不启动新实验。
- 使用资源：`ladd4090-zw1` GPU0/GPU1 约 `7853/11577 MiB`，util 约 `88%/16%`。GPU0 继续跑 OGSOD e800 det-only 与 ProbeA；GPU1 主要剩余原 OGSOD RGB baseline。
- OGSOD e800 `yolo_detonly`：rows=`42/800`，best/latest AP50/AP50-95=`0.48889/0.25793` at ep42，late5/late10/late20 AP50-95=`0.253148/0.244155/0.220836`。
- OGSOD e800 `yolo_probeA`：rows=`19/800`，best/latest AP50/AP50-95=`0.35342/0.16245` at ep19，late5/late10/late20 AP50-95=`0.148274/0.136136/0.097435`。
- ProbeA vs det-only：共同 rows=`19`，latest delta AP50-95=`+0.00412`，positive epochs=`8/19`，matched best delta=`+0.01203`，late5/late10/late20 delta=`+0.001626/+0.000912/-0.002044`。这是早期轻微正差，但还不到 50 epoch，不按正结果报告。
- 有效性：OGSOD det-only 与 ProbeA 日志均未见 Traceback/CUDA OOM/RuntimeError/NaN/batch fallback 关键词；外层与主训练 PID 均存活。
- 下一步：继续等 50/100 epoch。若 ProbeA 在 50+ epoch 后还能保持同 epoch 与 late-window 优势，再作为 AutoDL2 正结果复刻信号通知用户。

### 2026-06-24 12:20 CST - heartbeat：OGSOD e800 ProbeA 早期同 epoch 领先，继续等 50 epoch

- 目标：OGSOD e800 ProbeA/control 监控，不启动新实验。
- 使用资源：`ladd4090-zw1` GPU0/GPU1 约 `7853/11791 MiB`，util 约 `97%/17%`。GPU0 两条 OGSOD e800 run 正常运行；GPU1 主要剩余原 OGSOD RGB baseline。
- OGSOD e800 `yolo_detonly`：rows=`52/800`，best/latest AP50/AP50-95=`0.53093/0.29035` at ep52，late5/late10/late20 AP50-95=`0.286746/0.278960/0.261558`。
- OGSOD e800 `yolo_probeA`：rows=`27/800`，best/latest AP50/AP50-95=`0.43345/0.21330` at ep27，late5/late10/late20 AP50-95=`0.199728/0.184965/0.154067`。
- ProbeA vs det-only：共同 rows=`27`，latest delta AP50-95=`+0.01591`，positive epochs=`16/27`，matched best delta=`+0.01946`，late5/late10/late20 delta=`+0.015734/+0.013315/+0.006157`。
- 判定：这是比 12:05 更清晰的早期正向轨迹，形态上与 AutoDL2 `yolo_probeA` “后续追上/领先”线索一致；但 ProbeA 自身仍只有 27 epoch，未到 50 epoch，因此暂不作为用户级正结果通知。
- 有效性：OGSOD det-only 与 ProbeA 日志均未见 Traceback/CUDA OOM/RuntimeError/NaN/batch fallback 关键词；外层与主训练 PID 均存活。
- 下一步：继续监控到 ProbeA 50 epoch。如果 50 epoch 时 latest/late10/late20 仍为正，再通知用户这是 OGSOD 复刻组的可信早期正信号。

### 2026-06-24 12:35 CST - heartbeat：OGSOD e800 ProbeA 36 epoch，早期优势继续存在

- 目标：OGSOD e800 ProbeA/control 监控，不启动新实验。
- 使用资源：`ladd4090-zw1` GPU0/GPU1 约 `7853/12017 MiB`，util 约 `91%/17%`。GPU0 两条 OGSOD e800 run 正常运行；GPU1 主要剩余原 OGSOD RGB baseline。
- OGSOD e800 `yolo_detonly`：rows=`63/800`，best/latest AP50/AP50-95=`0.55068/0.31171` at ep63，late5/late10/late20 AP50-95=`0.307510/0.303322/0.292671`。
- OGSOD e800 `yolo_probeA`：rows=`36/800`，best/latest AP50/AP50-95=`0.49403/0.25405` at ep36，late5/late10/late20 AP50-95=`0.242046/0.231673/0.205498`。
- ProbeA vs det-only：共同 rows=`36`，latest delta AP50-95=`+0.01254`，positive epochs=`25/36`，matched best delta=`+0.02335`，late5/late10/late20 delta=`+0.010120/+0.013079/+0.012636`。
- 判定：ProbeA 在 36 epoch 时仍保持同 epoch 与 late-window 正差，早期复刻信号继续增强；但仍未到 ProbeA 50 epoch，暂不作为用户级正结果通知。
- 有效性：OGSOD det-only 与 ProbeA 日志均未见 Traceback/CUDA OOM/RuntimeError/NaN/batch fallback 关键词；外层与主训练 PID 均存活。
- 下一步：继续等 ProbeA 50 epoch。若 50 epoch 时 latest/late10/late20 仍为正，通知用户 OGSOD e800 复刻组出现可信早期正信号。

### 2026-06-24 12:52 CST - 按用户要求补启动 OGSOD e800 YOLO-init dynamic

- 目标：AutoDL2 上 OGSOD `yolo_dynamic` 看起来与 `yolo_probeA` 不相上下，因此在 `ladd4090-zw1` 上补一个同服务器、同数据、同 A1 cache、同 e800 协议的 YOLO-init dynamic，避免 AutoDL2 下午关机导致证据中断。
- 做了什么：新增并同步脚本 `docs/experiments/dronevehicle_method_search_20260623/autodl_condition_followups/launch_ogsod_dynamic_e800_20260624.sh`，在 `ladd4090-zw1` GPU1 启动 OGSOD `yolo_dynamic` e800。
- 新实验：外层 PID=`143393`，主训练进程已出现 PID=`143398`。结果目录 `runs_public/dronevehicle_method_search/ogsod_nomosaic_autodl_condition/yolo_dynamic/ogsod_nomix_yolo_dynamic_existingcache_yolo11n_e800_b64_img256_s0_20260624_125211_b/`，日志 `logs/dronevehicle_method_search/sub2k_seed0_fullval/autodl_condition_followups_20260624/ogsod_yolo_dynamic_gpu1/ogsod_nomix_yolo_dynamic_existingcache_yolo11n_e800_b64_img256_s0_20260624_125211.outer.log`。
- 协议确认：OGSOD SAR/RGB，`imgsz=256`，`epochs=800`，`batch=64`，`strict_batch_size=True`，`mosaic=0.0`，`mixup=0.0`，`optimizer=auto`，`model=yolo11n.pt`，`b_detector_source=yolo11n.pt`，`b_decomp_source` 为现有 OGSOD img256 A1 cache。dynamic 只开启 `--ladd-b-a2-core --b-load-student-reachability`，不加 `--ladd-b-frozen-reach-probe` 和 `--ladd-b-detach-reach-probe`；这是它和 ProbeA 的关键差异。
- 使用资源：启动前 GPU0/GPU1 约 `7853/12269 MiB`，启动进入训练后约 `7853/15859 MiB`，util 约 `85%/87%`；GPU1 总显存仍明显低于 `22G` 危险线。
- 健康检查：dynamic 日志已通过 AMP、数据扫描、split-load 初始化并进入第 1 epoch 训练；`ladd_phase_diag` 显示 `ladd_b_a2_core=True`、`ladd_b_frozen_reach_probe=False`、`ladd_b_detach_reach_probe=False`、`trainable_params=5360421/5360421`。未见 Traceback/CUDA OOM/RuntimeError/NaN/batch fallback 关键词。
- 下一步：后续 heartbeat 同时监控 OGSOD e800 `yolo_detonly`、`yolo_probeA`、`yolo_dynamic`。报告 rows、best/latest/final、late5/late10/late20，以及 `ProbeA - detonly`、`dynamic - detonly`、必要时 `dynamic - ProbeA` 的同 epoch 差异。dynamic 未到 50 epoch 前只作为早期健康状态，不做正结果结论。

### 2026-06-24 12:59 CST - dynamic 启动后首轮结果确认

- 使用资源：`ladd4090-zw1` GPU0/GPU1 约 `7853/16607 MiB`，util 约 `96%/94%`。GPU1 同时跑 OGSOD RGB baseline 与新开的 OGSOD dynamic，总显存仍低于 `22G` 危险线。
- OGSOD e800 `yolo_detonly`：rows=`78/800`，best/latest AP50/AP50-95=`0.57333/0.32960` at ep78，late5/late10/late20 AP50-95=`0.328336/0.325642/0.318703`。
- OGSOD e800 `yolo_probeA`：rows=`49/800`，best/latest AP50/AP50-95=`0.53498/0.29559` at ep49，late5/late10/late20 AP50-95=`0.287622/0.278812/0.261257`。同 epoch 对 det-only：matched rows=`49`，latest delta AP50-95=`+0.010300`，positive epochs=`38/49`。下一次 heartbeat 大概率会到 50 epoch，应重点检查 50+ epoch 的 latest/late-window 是否继续为正。
- OGSOD e800 `yolo_dynamic`：rows=`2/800`，latest AP50/AP50-95=`0.08936/0.02956` at ep2，best=`0.16813/0.06597` at ep1，late5/late10/late20 AP50-95=`0.047765/0.047765/0.047765`。同 epoch 对 det-only：matched rows=`2`，latest delta AP50-95=`+0.000390`，positive epochs=`2/2`；样本太少，只能说明训练/验证链路跑通。
- 有效性：dynamic 外层日志未见 Traceback/CUDA OOM/RuntimeError/NaN/batch fallback 关键词；`strict_batch_size=True`，当前有效。
- 自动化：已把 `dronevehicle-ogsod-follow-up-monitor` 更新为 15 分钟 heartbeat，同时监控 OGSOD e800 det-only、ProbeA、dynamic 三条，并按同 epoch/late-window 差异报告。

### 2026-06-24 13:00 CST - heartbeat：OGSOD ProbeA 到 50 epoch，达到早期通知阈值

- 目标：OGSOD e800 det-only / ProbeA / dynamic 监控，不启动新实验。
- 使用资源：`ladd4090-zw1` GPU0/GPU1 约 `7853/16651 MiB`，util 约 `90%/95%`；三条 OGSOD e800 训练进程均存活，GPU1 总显存仍低于 `22G` 危险线。
- OGSOD e800 `yolo_detonly`：rows=`80/800`，best/latest AP50/AP50-95=`0.57501/0.33154` at ep80，late5/late10/late20 AP50-95=`0.329756/0.327732/0.321391`。
- OGSOD e800 `yolo_probeA`：rows=`50/800`，best AP50/AP50-95=`0.53498/0.29559` at ep49，latest=`0.53429/0.29384` at ep50，late5/late10/late20 AP50-95=`0.290270/0.281477/0.264832`。
- ProbeA vs det-only：matched rows=`50`，latest delta AP50-95=`+0.007510`，positive epochs=`39/50`，best matched delta=`+0.023350`，late5/late10/late20 delta=`+0.008664/+0.009141/+0.010022`。这满足此前设定的早期通知条件：候选超过 50 epoch，且 latest/late10/late20 同 epoch 差异仍为正；但最终正结果仍需 100/150/final 或更长窗口确认。
- OGSOD e800 `yolo_dynamic`：rows=`3/800`，best AP50/AP50-95=`0.16813/0.06597` at ep1，latest=`0.06645/0.02273` at ep3，late5/late10/late20 AP50-95=`0.039420/0.039420/0.039420`。同 epoch 对 det-only：matched rows=`3`，latest delta AP50-95=`+0.019820`，positive epochs=`3/3`；样本太少，只能作为链路跑通与极早期现象记录。
- 有效性：det-only、ProbeA、dynamic 日志均未见 Traceback/CUDA OOM/RuntimeError/NaN/batch fallback 关键词。
- 下一步：继续监控 ProbeA 100 epoch 与 dynamic 50 epoch；若 ProbeA 100 epoch 仍保持 late-window 正差，可信度会明显提升。dynamic 需要至少到 50 epoch 后再判断是否与 AutoDL2 的 `yolo_dynamic` 形态一致。

### 2026-06-24 13:15 CST - heartbeat：ProbeA 58 epoch 正差延续，dynamic 仍太早

- 目标：OGSOD e800 det-only / ProbeA / dynamic 监控，不启动新实验。
- 使用资源：`ladd4090-zw1` GPU0/GPU1 约 `7853/16875 MiB`，util 约 `89%/91%`；显存仍低于 `22G` 危险线，训练正常推进。
- OGSOD e800 `yolo_detonly`：rows=`90/800`，best/latest AP50/AP50-95=`0.58072/0.33815` at ep90，late5/late10/late20 AP50-95=`0.337018/0.335220/0.331476`。
- OGSOD e800 `yolo_probeA`：rows=`58/800`，best/latest AP50/AP50-95=`0.55400/0.31021` at ep58，late5/late10/late20 AP50-95=`0.306242/0.301763/0.288559`。
- ProbeA vs det-only：matched rows=`58`，latest delta AP50-95=`+0.008090`，positive epochs=`47/58`，best matched delta=`+0.023350`，late5/late10/late20 delta=`+0.007108/+0.007741/+0.008615`。50 epoch 后的早期正差仍延续，但还没到新的 100 epoch 决策点。
- OGSOD e800 `yolo_dynamic`：rows=`11/800`，best/latest AP50/AP50-95=`0.29250/0.12087` at ep11，late5/late10/late20 AP50-95=`0.099446/0.072273/0.071700`。
- dynamic vs det-only：matched rows=`11`，latest delta AP50-95=`-0.003310`，positive epochs=`6/11`，best matched delta=`+0.019820`，late5/late10/late20 delta=`-0.003878/-0.000265/+0.000805`。dynamic 仍处于极早期且近期窗口不稳定，暂不做正负结论。
- dynamic vs ProbeA：matched rows=`11`，latest delta AP50-95=`-0.001330`，positive epochs=`5/11`。
- 有效性：det-only、ProbeA、dynamic 日志均未见 Traceback/CUDA OOM/RuntimeError/NaN/batch fallback 关键词。
- 下一步：继续等 ProbeA 100 epoch 与 dynamic 50 epoch；当前无需要人工介入的新决策。

### 2026-06-24 13:30 CST - heartbeat：ProbeA 正差扩大，dynamic 早期转回正窗

- 目标：OGSOD e800 det-only / ProbeA / dynamic 监控，不启动新实验。
- 使用资源：`ladd4090-zw1` GPU0/GPU1 约 `7853/6819 MiB`，util 约 `92%/86%`；三个主训练进程 `126131/131304/143398` 均存活。GPU1 显存下降，说明先前同卡其他任务已结束或释放资源；当前仍按“不新增实验”约束继续监控。
- OGSOD e800 `yolo_detonly`：rows=`100/800`，best/latest AP50/AP50-95=`0.58624/0.34495` at ep100，late5/late10/late20 AP50-95=`0.343508/0.341678/0.338449`。
- OGSOD e800 `yolo_probeA`：rows=`67/800`，best/latest AP50/AP50-95=`0.57560/0.32859` at ep67，late5/late10/late20 AP50-95=`0.325452/0.320792/0.310343`。
- ProbeA vs det-only：matched rows=`67`，latest delta AP50-95=`+0.010670`，positive epochs=`56/67`，best matched delta=`+0.023350`，late5/late10/late20 delta=`+0.010834/+0.010687/+0.009287`。相较 13:15，latest 与 late5/late10 正差略有扩大；但 ProbeA 还没到 100 epoch 决策点。
- OGSOD e800 `yolo_dynamic`：rows=`19/800`，best AP50/AP50-95=`0.35653/0.16403` at ep18，latest=`0.35631/0.16036` at ep19，late5/late10/late20 AP50-95=`0.152882/0.139663/0.102494`。
- dynamic vs det-only：matched rows=`19`，latest delta AP50-95=`+0.002030`，positive epochs=`13/19`，best matched delta=`+0.019820`，late5/late10/late20 delta=`+0.006234/+0.004439/+0.003015`。dynamic 近期窗口转回正，但仍不足 50 epoch，暂不做方法结论。
- dynamic vs ProbeA：matched rows=`19`，latest delta AP50-95=`-0.002090`，positive epochs=`10/19`，late5/late10 delta=`+0.004608/+0.003527`。
- 有效性：det-only、ProbeA、dynamic 日志均未见 Traceback/CUDA OOM/RuntimeError/NaN/batch fallback 关键词。
- 下一步：继续等 ProbeA 100 epoch 与 dynamic 50 epoch；当前无需要人工介入的新决策。

### 2026-06-24 13:45 CST - heartbeat：ProbeA 正差稳定，dynamic 早期正差增强

- 目标：OGSOD e800 det-only / ProbeA / dynamic 监控，不启动新实验。
- 使用资源：`ladd4090-zw1` GPU0/GPU1 约 `7853/6869 MiB`，util 约 `94%/93%`；三个主训练进程 `126131/131304/143398` 均存活。
- OGSOD e800 `yolo_detonly`：rows=`110/800`，best/latest AP50/AP50-95=`0.59335/0.35150` at ep110，late5/late10/late20 AP50-95=`0.350010/0.348438/0.345058`。
- OGSOD e800 `yolo_probeA`：rows=`76/800`，best/latest AP50/AP50-95=`0.58558/0.33849` at ep76，late5/late10/late20 AP50-95=`0.336314/0.333416/0.326039`。
- ProbeA vs det-only：matched rows=`76`，latest delta AP50-95=`+0.010510`，positive epochs=`65/76`，best matched delta=`+0.023350`，late5/late10/late20 delta=`+0.009666/+0.009970/+0.010123`。正差继续稳定，但还没到 ProbeA 100 epoch 新决策点。
- OGSOD e800 `yolo_dynamic`：rows=`28/800`，best/latest AP50/AP50-95=`0.43674/0.21378` at ep28，late5/late10/late20 AP50-95=`0.198646/0.186044/0.159824`。
- dynamic vs det-only：matched rows=`28`，latest delta AP50-95=`+0.017210`，positive epochs=`21/28`，best matched delta=`+0.019820`，late5/late10/late20 delta=`+0.010624/+0.009230/+0.006691`。dynamic 早期正差增强，但仍不足 50 epoch。
- dynamic vs ProbeA：matched rows=`28`，latest delta AP50-95=`-0.004170`，positive epochs=`12/28`，late5/late10/late20 delta=`-0.005494/-0.005380/-0.000570`。
- 有效性：det-only、ProbeA、dynamic 日志均未见 Traceback/CUDA OOM/RuntimeError/NaN/batch fallback 关键词。
- 下一步：继续等 ProbeA 100 epoch 与 dynamic 50 epoch；当前无需要人工介入的新决策。

### 2026-06-24 14:00 CST - heartbeat：ProbeA 84 epoch 稳定正差，dynamic 36 epoch 仍为正

- 目标：OGSOD e800 det-only / ProbeA / dynamic 监控，不启动新实验。
- 使用资源：`ladd4090-zw1` GPU0/GPU1 约 `7853/7073 MiB`，util 约 `85%/89%`；三个主训练进程 `126131/131304/143398` 均存活。
- OGSOD e800 `yolo_detonly`：rows=`120/800`，best/latest AP50/AP50-95=`0.59884/0.35680` at ep120，late5/late10/late20 AP50-95=`0.355510/0.354293/0.351366`。
- OGSOD e800 `yolo_probeA`：rows=`84/800`，best/latest AP50/AP50-95=`0.59151/0.34459` at ep84，late5/late10/late20 AP50-95=`0.342924/0.341082/0.336046`。
- ProbeA vs det-only：matched rows=`84`，latest delta AP50-95=`+0.010800`，positive epochs=`73/84`，best matched delta=`+0.023350`，late5/late10/late20 delta=`+0.010118/+0.010188/+0.010113`。正差继续稳定，但还没到 ProbeA 100 epoch 新决策点。
- OGSOD e800 `yolo_dynamic`：rows=`36/800`，best/latest AP50/AP50-95=`0.49006/0.25015` at ep36，late5/late10/late20 AP50-95=`0.238408/0.227497/0.202009`。
- dynamic vs det-only：matched rows=`36`，latest delta AP50-95=`+0.008640`，positive epochs=`29/36`，best matched delta=`+0.019820`，late5/late10/late20 delta=`+0.006482/+0.008903/+0.009147`。dynamic 仍为早期正差，但不足 50 epoch。
- dynamic vs ProbeA：matched rows=`36`，latest delta AP50-95=`-0.003900`，positive epochs=`13/36`，late5/late10/late20 delta=`-0.003638/-0.004176/-0.003489`。
- 有效性：det-only、ProbeA、dynamic 日志均未见 Traceback/CUDA OOM/RuntimeError/NaN/batch fallback 关键词。
- 下一步：继续等 ProbeA 100 epoch 与 dynamic 50 epoch；当前无需要人工介入的新决策。

### 2026-06-24 14:15 CST - heartbeat：ProbeA 92 epoch、dynamic 44 epoch，均未到新阈值

- 目标：OGSOD e800 det-only / ProbeA / dynamic 监控，不启动新实验。
- 使用资源：`ladd4090-zw1` GPU0/GPU1 约 `7853/7279 MiB`，util 约 `90%/96%`；三个主训练进程 `126131/131304/143398` 均存活。
- OGSOD e800 `yolo_detonly`：rows=`131/800`，best/latest AP50/AP50-95=`0.60585/0.36319` at ep131，late5/late10/late20 AP50-95=`0.361936/0.360430/0.357645`。
- OGSOD e800 `yolo_probeA`：rows=`92/800`，best/latest AP50/AP50-95=`0.59658/0.34857` at ep92，late5/late10/late20 AP50-95=`0.347558/0.346216/0.342834`。
- ProbeA vs det-only：matched rows=`92`，latest delta AP50-95=`+0.009630`，positive epochs=`81/92`，best matched delta=`+0.023350`，late5/late10/late20 delta=`+0.009400/+0.009788/+0.009876`。正差仍稳定，但 ProbeA 还未到 100 epoch 新决策点。
- OGSOD e800 `yolo_dynamic`：rows=`44/800`，best/latest AP50/AP50-95=`0.52272/0.27872` at ep44，late5/late10/late20 AP50-95=`0.270708/0.261853/0.239206`。
- dynamic vs det-only：matched rows=`44`，latest delta AP50-95=`+0.009200`，positive epochs=`37/44`，best matched delta=`+0.019820`，late5/late10/late20 delta=`+0.011082/+0.010558/+0.009510`。dynamic 仍保持早期正差，但不足 50 epoch。
- dynamic vs ProbeA：matched rows=`44`，latest delta AP50-95=`+0.001870`，positive epochs=`17/44`，late5/late10/late20 delta=`+0.000706/-0.001012/-0.002939`。
- 有效性：det-only、ProbeA、dynamic 日志均未见 Traceback/CUDA OOM/RuntimeError/NaN/batch fallback 关键词。
- 下一步：继续等 ProbeA 100 epoch 与 dynamic 50 epoch；下一次 heartbeat 很可能达到两个决策点之一。

### 2026-06-24 14:30 CST - 手动查询：ProbeA 100 epoch 与 dynamic 50+ epoch 均出现同 epoch 正差

- 目标：回答用户“现在有什么现象”，不启动新实验。
- 使用资源：`ladd4090-zw1` GPU0/GPU1 约 `7853/7475 MiB`，util 约 `90%/94%`；三个主训练进程 `126131/131304/143398` 均存活。
- OGSOD e800 `yolo_detonly`：rows=`140/800`，best/latest AP50/AP50-95=`0.60990/0.36835` at ep140，late5/late10/late20 AP50-95=`0.367190/0.365797/0.362812`。
- OGSOD e800 `yolo_probeA`：rows=`100/800`，best/latest AP50/AP50-95=`0.60008/0.35205` at ep100，late5/late10/late20 AP50-95=`0.351392/0.350247/0.347648`。
- ProbeA vs det-only：matched rows=`100`，latest delta AP50-95=`+0.007100`，positive epochs=`89/100`，best matched delta=`+0.023350`，late5/late10/late20 delta=`+0.007884/+0.008569/+0.009199`。结论：到 100 epoch 后，ProbeA 的同 epoch 正差仍成立，但 gap 从 70-80 epoch 的约 `+0.010` 缩小到约 `+0.007`。
- OGSOD e800 `yolo_dynamic`：rows=`51/800`，best/latest AP50/AP50-95=`0.54485/0.29737` at ep51，late5/late10/late20 AP50-95=`0.291262/0.284490/0.267029`。
- dynamic vs det-only：matched rows=`51`，latest delta AP50-95=`+0.007590`，positive epochs=`44/51`，best matched delta=`+0.019820`，late5/late10/late20 delta=`+0.006678/+0.008772/+0.008674`。结论：dynamic 到 50+ epoch 后也满足早期正结果标准。
- dynamic vs ProbeA：matched rows=`51`，latest delta AP50-95=`+0.002230`，positive epochs=`21/51`，late5/late10/late20 delta=`-0.001428/+0.000099/-0.001263`。结论：dynamic 与 ProbeA 在共同早期几乎打平，latest 略高但窗口均值没有稳定优势。
- 有效性：det-only、ProbeA、dynamic 日志均未见 Traceback/CUDA OOM/RuntimeError/NaN/batch fallback 关键词。
- 当前现象归纳：OGSOD 上 ProbeA 和 dynamic 都复现出同 epoch 正向增益；这与 AutoDL 上 “dynamic 和 ProbeA 不相上下” 的观察一致。注意不能用 det-only 当前 ep140 的绝对值去直接压 ep100/ep51 的候选，必须同 epoch 比。

### 2026-06-24 14:31 CST - heartbeat：阈值后正差延续，无新决策

- 目标：OGSOD e800 det-only / ProbeA / dynamic 监控，不启动新实验。
- 使用资源：`ladd4090-zw1` GPU0/GPU1 约 `7853/7489 MiB`，util 约 `92%/93%`；三个主训练进程 `126131/131304/143398` 均存活。
- OGSOD e800 `yolo_detonly`：rows=`141/800`，best/latest AP50/AP50-95=`0.61045/0.36854` at ep141，late5/late10/late20 AP50-95=`0.367802/0.366332/0.363381`。
- OGSOD e800 `yolo_probeA`：rows=`101/800`，best/latest AP50/AP50-95=`0.60023/0.35228` at ep101，late5/late10/late20 AP50-95=`0.351730/0.350671/0.348141`。
- ProbeA vs det-only：matched rows=`101`，latest delta AP50-95=`+0.006570`，positive epochs=`90/101`，best matched delta=`+0.023350`，late5/late10/late20 delta=`+0.007520/+0.008280/+0.009031`。100 epoch 后正差延续，但 latest gap 继续轻微收窄。
- OGSOD e800 `yolo_dynamic`：rows=`52/800`，best/latest AP50/AP50-95=`0.54597/0.29963` at ep52，late5/late10/late20 AP50-95=`0.293980/0.287313/0.270375`。
- dynamic vs det-only：matched rows=`52`，latest delta AP50-95=`+0.009280`，positive epochs=`45/52`，best matched delta=`+0.019820`，late5/late10/late20 delta=`+0.007234/+0.008353/+0.008818`。50 epoch 后正差延续。
- dynamic vs ProbeA：matched rows=`52`，latest delta AP50-95=`-0.000140`，positive epochs=`21/52`，late5/late10/late20 delta=`-0.001192/-0.000092/-0.001155`。两者仍基本打平，暂无稳定高下。
- 有效性：det-only、ProbeA、dynamic 日志均未见 Traceback/CUDA OOM/RuntimeError/NaN/batch fallback 关键词。
- 下一步：继续等 ProbeA/dynamic 150 epoch 和更长 late-window；当前无新的人工决策点。

### 2026-06-24 14:45 CST - 手动复查：AutoDL2 e700 支持 YOLO-init 主线线索，但仍需 e800 严格对照

- 目标：回应用户“YOLO-init 主线至少在 OGSOD 上可能有正向效果，AutoDL 双卡只有 700 epoch 不够 ProbeA 收敛”的判断；只读检查 AutoDL2 与 `ladd4090-zw1`，未启动/停止实验。
- AutoDL2 `20260623_194650` no-mosaic existing-cache 组现状：`reload_detonly` 与 `yolo_probeA` 均已到 `700/700`；`yolo_dynamic` 到 `673/700`；`warm100_probeA/warm100_dynamic` 到 `678/666`，仍在运行。
- AutoDL2 `reload_detonly`：rows=`700`，latest AP50/AP50-95=`0.81025/0.53623`，best=`0.81235/0.54008` at ep646，late5/10/20/50/100 AP50-95=`0.536392/0.536554/0.537409/0.538501/0.538049`。
- AutoDL2 `yolo_probeA`：rows=`700`，latest=`0.82971/0.55439`，best=`0.83178/0.55885` at ep653，late5/10/20/50/100=`0.554308/0.554490/0.554746/0.556251/0.556545`。对 `reload_detonly` 的同 epoch 差值：latest AP50/AP50-95=`+0.01946/+0.01816`，positive epochs=`496/700`，best_delta AP50-95=`+0.02017`，late5/10/20/50/100=`+0.017916/+0.017936/+0.017337/+0.017751/+0.018495`。
- 同协议 no-mosaic baseline 参考：`BASELINE_STANDARD_CN.md` 记录 YOLO11n SAR 800-epoch baseline best AP50-95=`0.55654`。因此 AutoDL2 `yolo_probeA` 的 best=`0.55885` 对 baseline best 有 `+0.00231` 的边界正点，但 ep700/latest=`0.55439` 低于 baseline best `-0.00215`，需要 e800 严格复现实验判断能否站稳。
- AutoDL2 `yolo_dynamic`：rows=`673`，latest=`0.83530/0.55341`，best=`0.83498/0.55539` at ep641，late5/10/20/50/100=`0.553890/0.553855/0.554267/0.554525/0.552894`。对 `reload_detonly` 的同 epoch 差值：latest AP50/AP50-95=`+0.02605/+0.01474`，positive epochs=`414/673`，late20=`+0.014955`。`yolo_probeA - yolo_dynamic` 共同 673 epoch 的 latest AP50-95=`+0.00324`、late20=`+0.003275`，ProbeA 略高但两者接近。
- 对用户判断的细化：AutoDL2 明确支持“YOLO-init ProbeA/dynamic 在 OGSOD 上有正向同 epoch 增益”的线索；但 `yolo_probeA` 自身 best 在 ep653，ep700 已从 `0.55885` 回落到 `0.55439`，所以“700 不够”的更精确含义不是还在单调上升，而是 `700` 只能证明它显著高于 AutoDL 的 reload/warm-start 对照，仍不足以回答它相对 `800 epoch` formal baseline 和严格 YOLO-init det-only control 的最终收敛关系。
- `ladd4090-zw1` 严格 e800 复现实验当前资源：GPU0/GPU1 约 `7853/7685 MiB`，util 约 `89%/90%`，显存安全；未见需要调度的新动作。
- `ladd4090-zw1` e800 `yolo_detonly`：rows=`151/800`，latest/best AP50/AP50-95=`0.61488/0.37357` at ep151，late5/10/20=`0.372492/0.371237/0.368784`，估计剩余约 `16.1h`。
- `ladd4090-zw1` e800 `yolo_probeA`：rows=`109/800`，latest/best=`0.60370/0.35739` at ep109，late5/10/20=`0.356044/0.354596/0.352203`，估计剩余约 `20.7h`。对同服务器同协议 `yolo_detonly` 的共同 109 epoch 差值：latest AP50/AP50-95=`+0.01139/+0.00658`，positive epochs=`98/109`，best_delta AP50-95=`+0.02335`，late5/10/20=`+0.006752/+0.006813/+0.007812`。
- `ladd4090-zw1` e800 `yolo_dynamic`：rows=`60/800`，latest/best=`0.55846/0.31719` at ep60，late5/10/20=`0.312180/0.306849/0.294115`，估计剩余约 `23.4h`。对同 epoch `yolo_detonly` 的差值：latest AP50/AP50-95=`+0.01457/+0.01197`，positive epochs=`53/60`，best_delta AP50-95=`+0.01982`，late5/10/20=`+0.010200/+0.009144/+0.009094`。
- 当前判断：用户关于 OGSOD YOLO-init 主线可能有正向效果的怀疑是合理的；最强证据来自 AutoDL2 e700 的稳定正差和 `ladd4090-zw1` e800 strict control 的早期同向复现。最终仍需等 `ladd4090-zw1` 的 e800 同服务器同协议结果，尤其是 150/300/final 与 late-window，才能决定 ProbeA/dynamic 是否真能作为主线候选。

### 2026-06-24 14:49 CST - heartbeat：e800 strict control 正差延续，无新决策

- 目标：OGSOD e800 det-only / ProbeA / dynamic 监控，不启动新实验。
- 使用资源：`ladd4090-zw1` GPU0/GPU1 约 `7853/7745 MiB`，util 约 `92%/92%`；三条主训练均存活。日志关键词检查未见 Traceback/CUDA OOM/RuntimeError/NaN/batch fallback。
- OGSOD e800 `yolo_detonly`：rows=`154/800`，latest/best AP50/AP50-95=`0.61777/0.37488` at ep154，late5/10/20 AP50-95=`0.374090/0.372749/0.370307`，按近期速度估计剩余约 `16.0h`。
- OGSOD e800 `yolo_probeA`：rows=`112/800`，latest/best AP50/AP50-95=`0.60627/0.35928` at ep112，late5/10/20=`0.357938/0.356389/0.353769`，估计剩余约 `20.6h`。
- ProbeA vs det-only：matched rows=`112`，latest delta AP50/AP50-95=`+0.01218/+0.00674`，positive epochs=`101/112`，best_delta AP50-95=`+0.02335`，late5/10/20=`+0.006748/+0.006738/+0.007386`。100 epoch 后正差仍延续，但窗口差值仍需看 150/300。
- OGSOD e800 `yolo_dynamic`：rows=`62/800`，latest/best AP50/AP50-95=`0.56297/0.32161` at ep62，late5/10/20=`0.317244/0.311372/0.299343`，估计剩余约 `23.3h`。
- dynamic vs det-only：matched rows=`62`，latest delta AP50/AP50-95=`+0.01327/+0.01202`，positive epochs=`55/62`，best_delta AP50-95=`+0.01982`，late5/10/20=`+0.011652/+0.009941/+0.009147`。dynamic 50+ epoch 后仍保持正差。
- dynamic vs ProbeA：matched rows=`62`，latest delta AP50/AP50-95=`-0.00164/+0.00138`，positive epochs=`29/62`，late5/10/20=`+0.001112/+0.000998/+0.000453`。两者仍非常接近，dynamic 当前 AP50-95 略高但没有形成可靠优势。
- 下一步：继续等 ProbeA 150 epoch、dynamic 100/150 epoch 和更长 late-window；当前无新的人工决策点。

### 2026-06-24 15:06 CST - heartbeat：AutoDL2 已关机后只监控 ladd4090-zw1 三条 e800

- 目标：OGSOD e800 det-only / ProbeA / dynamic 监控；按用户最新消息确认 AutoDL2 已关机，因此本次开始不再检查 AutoDL2，只保留其 e700 结果为历史外部证据。
- 自动化更新：`dronevehicle-ogsod-follow-up-monitor` 的 prompt 已更新，去掉 AutoDL2 active monitoring，只保留 `ladd4090-zw1` 三条 OGSOD e800 与明确阈值节点。
- 使用资源：`ladd4090-zw1` GPU0/GPU1 约 `7853/7969 MiB`，util 约 `96%/96%`；三条主训练均存活。日志关键词检查未见 Traceback/CUDA OOM/RuntimeError/NaN/batch fallback。
- OGSOD e800 `yolo_detonly`：rows=`165/800`，latest/best AP50/AP50-95=`0.62278/0.38132` at ep165，late5/10/20 AP50-95=`0.379824/0.378398/0.375807`，按近期速度估计剩余约 `15.8h`。
- OGSOD e800 `yolo_probeA`：rows=`121/800`，latest/best AP50/AP50-95=`0.61067/0.36513` at ep121，late5/10/20=`0.363984/0.362415/0.359105`，估计剩余约 `20.3h`。
- ProbeA vs det-only：matched rows=`121`，latest delta AP50/AP50-95=`+0.01081/+0.00797`，positive epochs=`110/121`，best_delta AP50-95=`+0.02335`，late5/10/20=`+0.007982/+0.007555/+0.007168`。100 epoch 后正差仍延续。
- OGSOD e800 `yolo_dynamic`：rows=`71/800`，latest/best AP50/AP50-95=`0.57708/0.33355` at ep71，late5/10/20=`0.330982/0.327681/0.318427`，估计剩余约 `23.1h`。
- dynamic vs det-only：matched rows=`71`，latest delta AP50/AP50-95=`+0.01186/+0.01027`，positive epochs=`64/71`，best_delta AP50-95=`+0.01982`，late5/10/20=`+0.010738/+0.011083/+0.010375`。50+ epoch 后仍保持正差。
- dynamic vs ProbeA：matched rows=`71`，latest delta AP50/AP50-95=`-0.00286/+0.00094`，positive epochs=`37/71`，late5/10/20=`+0.000464/+0.000532/+0.000689`。两者仍非常接近，dynamic 当前 AP50-95 略高但差距很小。
- 下一步：继续等 ProbeA 150 epoch、dynamic 100/150 epoch 和更长 late-window；当前无新的人工决策点。

### 2026-06-24 15:45 CST - 新开 old-commit ProbeA：复核 AutoDL2 高增益是否来自代码版本

- 目标：按用户“新开一个”的确认，在 `ladd4090-zw1` 上新开一条严格复刻 AutoDL2 代码版本的 OGSOD YOLO-init ProbeA，用于判断 AutoDL2 早期更高增益是否由代码版本差异导致。
- 代码版本：单独创建 `/root/shared-nvme/LADD_public_commit_6f663c6b` 快照，commit=`6f663c6b4650a96f5308b5fe5d47fc7ca105b335`，与本地拉回的 AutoDL2 manifest 一致；不污染当前 `/root/shared-nvme/LADD_public` 工作树。
- 运行配置：OGSOD no-mosaic existing-cache，YOLO-init，`imgsz=256`，`batch=64`，`epochs=700`，GPU1，phase B，`--ladd-b-a2-core --ladd-b-frozen-reach-probe --b-load-student-reachability`；旧代码没有新的显式 `--ladd-b-detach-reach-probe` 参数，`frozen_reach_probe` 内部已包含 detach 语义。
- A1/decomp source：沿用当前 strict 复现实验使用的 A1 cache：`runs_public/paper/ogsod_hbb_mosaic100/no_reload_warm100/yolo11n/seed0/a1_decomp_cache/ogsod_hbb_ladd_a1_decomp_from_sar_baseline_yolo11n_s0_20260623_192641_img256_a1_e10_b64_s0_gpu0/weights/best.pt`。
- 结果目录：`/root/shared-nvme/LADD_public/runs_public/dronevehicle_method_search/ogsod_nomosaic_autodl_condition/yolo_probeA_oldcommit6f663c6b/ogsod_nomix_yolo_probeA_oldcommit6f663c6b_existingcache_yolo11n_e700_b64_img256_s0_20260624_154303_b/`。
- 日志目录：`/root/shared-nvme/LADD_public_commit_6f663c6b/logs/dronevehicle_method_search/sub2k_seed0_fullval/autodl_condition_followups_20260624/ogsod_yolo_probeA_oldcommit_gpu1/`。
- 当前状态：训练主进程存活，GPU0/GPU1 约 `7853/8571 MiB`，util 约 `96%/92%`；首个 epoch 已写入 `results.csv`，rows=`1`，ep1 AP50/AP50-95=`0.06804/0.02193`。
- 有效性：当前日志未见 Traceback/CUDA OOM/RuntimeError/NaN/batch fallback 关键词；已进入 epoch2 训练。
- 判读方式：若该 old-commit 复刻在 50/100/150 epoch 的同 epoch delta 回到 AutoDL2 的较高区间，说明当前 ZW1 ProbeA 增益偏小主要来自代码版本/ProbeA 语义差异；若仍接近当前 e800 ProbeA 的小幅增益，则需要继续看数据 cache、A1 source、机器环境或随机性。
