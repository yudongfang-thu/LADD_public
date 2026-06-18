# LADD B-Entrance 压缩调度归档（2026-06-14）

本文件归档 `2026-06-13` 到 `2026-06-14` 这一批 B-entrance 诊断结果。核心修正是：这些 `B100/B120` 结果不能再作为 `B800` 主线趋势证据使用，因为 `EPOCHS_B` 同时改变了 cosine learning-rate schedule 的时间尺度。

## 1. 关键判断

`B100 + cos_lr=True` 不是 `B800` 训练到 epoch 100 的等价截断。

| 设置 | epoch 100 的学习率含义 | 能解释什么 |
|---|---|---|
| `EPOCHS_B=100, cos_lr=True` | cosine 已经接近末尾，学习率接近最终值 | 只能说明压缩 schedule 下的短训练表现 |
| `EPOCHS_B=800, cos_lr=True`，观察前 100 epoch | 学习率仍接近初始值 | 才是主线 B800 的早期读数 |

因此，上一批 `N1/N2/N3/N4` 和 `S1/S2/S3/S4` 需要降级为 **compressed-schedule diagnostic**。它们仍有工程价值：验证加载、BN freeze、OOM、KD ramp、split-load 是否能跑；但不能判断 LADD B800 是否会在后期继续增长。

## 2. 已归档结果

完整表格见：

- `docs/experiments/ladd_b_entrance_compressed_schedule_archive_20260614.csv`
- 原始状态快照：`docs/experiments/LADD_B_ENTRANCE_DIAG_STATUS_20260613_CN.md`
- 曲线图目录：`docs/experiments/figures/ladd_b_entrance_trends_20260614/`
- 历史 B 阶段对比图：`docs/experiments/figures/ladd_b_stage_historical_compare_20260614/`
- no-mosaic baseline 对比图：`docs/experiments/figures/ladd_nomosaic_baseline_b_compare_20260614/`

| run | 结果 | 当前解释 |
|---|---|---|
| `N1 baseline cont B100` | best `0.56615@99`, last `0.56594` | 只能作为压缩 schedule 的 detection continuation 对照 |
| `N2 A2-best B100` | best/last `0.55872@100` | 不能证明 A2-best 主线 B800 无后劲 |
| `N3 SAR-base + A2-last decomp B100` | best/last `0.55722@100` | 不能证明 split-load decomp 在 B800 无收益 |
| `N4 KD ramp B120` | best `0.56379@113`, last `0.56311` | KD warmup 与 B120 退火末期耦合，解释受限 |
| `S1 baseline cont B100` | best `0.62493@62`, last `0.62238` | 只能作为压缩 schedule 对照 |
| `S2 A2-best B100` | best `0.62599@54`, last `0.62174` | 不能否定 S 的 B800 后期冲劲 |
| `S3 SAR-base + A2-last decomp B100` | best `0.62553@65`, last `0.62262` | split-load 能跑，但不形成 B800 结论 |
| `S4 KD ramp B120` | best `0.62521@62`, last `0.62111` | KD-only warmup 结论受压缩调度混淆 |

## 3. 已停止的 N2-last

刚启动的 `N2_a2last_continue_b100` 已停止。它原本用于比较 `A2 best` 与 `A2 last`，但因为仍是 `B100` 压缩 schedule，会继续混淆主线判断。

服务器上已处理：

- 停止 N2-last 训练进程；
- 删除该中断 run 的 `weights/*.pt`；
- 写入 `ABORTED_DO_NOT_USE.txt`；
- 保留 `args.yaml/results.csv/ladd_diagnostics.csv` 作为“错误设计被中止”的轻量证据。

该 run 不纳入任何正式结果表。

## 4. 重新开始的原则

后续主线相关 B 实验必须满足：

1. `EPOCHS_B=800`，保持与正式主线一致的 cosine schedule。
2. 可以先读 epoch 100/200 的中间结果，但不能把 `EPOCHS_B` 改成 100/200。
3. `B100/B120` 只保留为 smoke 或压缩 schedule 诊断，不能作为主线方法成败依据。
4. KD warmup、core LADD warmup 必须放在 B800 schedule 下解释。
5. 从 YOLO 初始 detector 开始的 split-load 实验，必须配套 YOLO-init det-only 对照。

## 5. 对曲线现象的重新解释

上一批曲线“很快平台”并不一定说明 LADD 没有后劲。更可能是短 B 把 cosine schedule 压缩后，学习率过早降到很低。历史 no-mosaic 主线中，`n/s` 的 LADD B 阶段收益常出现在几百 epoch 以后；在 B800 schedule 下，epoch 100 仍处于高学习率探索期。

这也解释了“早期受冲击下跌、后期冲劲更大”的现象：LADD B 可能需要先离开 SAR baseline 的 detection-only basin，再通过 KD/decomposition 约束进入更适合跨模态蒸馏的区域。这个过程需要足够长的高学习率窗口。若 B100 压缩 schedule 很快退火，模型可能只是安全地停在原附近，而不是完成 basin transition。
