# DSN Shared-Private

日期：2026-06-23

## 作用

记录 frozen dual-detector shared/private latent 方案。S1 只训练 projector，不训练 detector；S2 后续用于 shared-latent student distillation。

## 当前 S1

```text
code: tools/train_dsn_shared_private_projector.py
run: runs_public/cross_dataset/dsn_shared_private/dronevehicle_sub2k_seed0/dronevehicle_sub2k_rgb_ir_dsn_s1_e80_b32_ld256_h512_seed0_20260623_2304
log: logs/cross_dataset/dsn_shared_private/dronevehicle_sub2k_seed0/dronevehicle_sub2k_rgb_ir_dsn_s1_e80_b32_ld256_h512_seed0_20260623_2304.log
```

当前 run 是 legacy path；后续 S2/control 应放到：

```text
runs_public/dronevehicle_method_search/sub2k_seed0_fullval/dsn_shared_private/
logs/dronevehicle_method_search/sub2k_seed0_fullval/dsn_shared_private/
```

S1 完成快照：

```text
epoch=80
train_loss=0.1687
batch_top1=0.9834
val_top1=0.2301
val_top5=0.5371
```

解释：随机 top1 约为 `1/1469` 或 `1/2048` 量级，S1 shared latent 已经学到明显跨模态配对信号。

## S2 代码路径

2026-06-24 已在 `current_hbb` 训练链路中加入一个默认关闭的 DSN shared-latent KD：

```text
ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/loss.py
ladd/code_versions/current_hbb/src/teacher_student_decomposition_kd_hbb/trainer.py
ladd/code_versions/current_hbb/tools/train_ladd_hbb.py
ladd/code_versions/current_hbb/scripts/ogsod_public/run_ladd_phase.sh
```

新增参数：

```text
--dsn-projector-weights <S1 best.pt>
--dsn-kd-weight <float>
--dsn-student-projector rgb|peer
--dsn-teacher-projector rgb|peer
--teacher-batch-roll <int>
```

当前 S2 版本是第一版低侵入验证：

```text
student raw multi-scale feature -> global pool concat -> frozen S1 rgb_projector.shared
teacher raw multi-scale feature -> global pool concat -> frozen S1 peer_projector.shared
loss = normalized latent MSE / cosine alignment
```

该版本不改 detector head，不重写 YOLO training loop，也不引入 student private 分支；它用于回答最小问题：`DSN shared latent 是否比 raw feature KD 更适合作为跨模态监督目标`。

本地和远端语法检查均已通过：

```text
python -m py_compile loss.py trainer.py train_ladd_hbb.py
bash -n run_ladd_phase.sh
```

## 已启动：S2 shared-latent KD

远端队列：

```text
logs/dronevehicle_method_search/sub2k_seed0_fullval/dsn_shared_private/s2_student_distill/queue_dsn_s2_after_lowlr_controls_20260624_001641.sh
logs/dronevehicle_method_search/sub2k_seed0_fullval/dsn_shared_private/s2_student_distill/queue_dsn_s2_after_lowlr_controls_20260624_001641.log
```

队列 pid：`17909`

启动条件：

```text
1. low-LR no-warmup det-only control 结果文件存在且至少 20 epoch
2. low-LR no-warmup raw feature KD 结果文件存在且至少 20 epoch
3. det-only control best AP50-95 >= 0.358 且 latest AP50-95 >= 0.340
4. 任意 GPU 显存低于 15000 MB
```

该条件已满足，2026-06-24 01:16:42 CST 启动：

```text
runs_public/dronevehicle_method_search/sub2k_seed0_fullval/dsn_shared_private/s2_student_distill/dsn_s2_sharedlatent_lowlr1e3_nowarmup_ir2rgb_yolo11n_w1p0_e200_b64_img512_s0_20260624_011642_b
logs/dronevehicle_method_search/sub2k_seed0_fullval/dsn_shared_private/s2_student_distill/dsn_s2_sharedlatent_lowlr1e3_nowarmup_ir2rgb_yolo11n_w1p0_e200_b64_img512_s0_20260624_011642_gpu1.outer.log
```

2026-06-24 01:18 CST 快照：S2 刚启动，尚未生成 `results.csv`。后续判定以同协议 det-only low-LR best `0.36279` 为主门槛。

2026-06-24 01:25 CST 快照：

```text
rows: 23
best AP50/AP50-95: 0.56957 / 0.36435
latest AP50/AP50-95: 0.55069 / 0.35249
late5 AP50-95: 0.35284
```

当前解释：DSN S2 的 early-best 比 det-only low-LR best `0.36279` 高 `+0.00156`，是目前最明显的早期正向信号；但 best 出现在 epoch 2，latest/late5 已回落，因此必须继续观察 final/late-window。同 epoch 20 对照为 DSN `0.35284` vs det-only `0.35196`，仍略高但差距很小。

2026-06-24 01:28 CST 快照：

```text
rows: 32
best AP50/AP50-95: 0.56957 / 0.36435
latest AP50/AP50-95: 0.55032 / 0.35200
late5/late10/late20 AP50-95: 0.35561 / 0.35476 / 0.35484
```

与 det-only low-LR 对照：

```text
best delta: +0.00156
latest delta: +0.00123
late5/late10/late20 delta: +0.00372 / +0.00456 / +0.00500
epoch 2 AP50-95: DSN 0.36435 vs det-only 0.36279
epoch 16 AP50-95: DSN 0.35948 vs det-only 0.35837
epoch 20 AP50-95: DSN 0.35284 vs det-only 0.35196
```

当前解释：DSN S2 仍是当前最强 early-best 候选，并且同 epoch 16/20 没有输给 det-only；不过优势主要来自早期 epoch 2，是否能成为稳定主线要等 50+ epoch、final 和 late-window。

2026-06-24 01:35 CST 快照：

```text
rows: 54
best AP50/AP50-95: 0.56957 / 0.36435
latest AP50/AP50-95: 0.54667 / 0.34953
late5/late10/late20 AP50-95: 0.35017 / 0.35107 / 0.35351
```

与 det-only low-LR 对照：

```text
best delta: +0.00156
latest delta: +0.00280
late5/late10/late20 delta: +0.00150 / +0.00316 / +0.00509
epoch 50 AP50-95: DSN 0.35021 vs det-only 0.34889
epoch 52 AP50-95: DSN 0.35169 vs det-only 0.34977
epoch 54 AP50-95: DSN 0.34953 vs det-only 0.34968
```

当前解释：DSN 的全局 best 仍是当前最高，但 50+ epoch 的 epoch-matched 优势已经很小，epoch 54 甚至略低于 det-only。它还不能称为稳定主线；更合理的下一步是验证较弱 DSN KD 权重、KD decay 或 shuffled-pair control，确认 early-best 是否是真 shared latent 信号而不是 reload/噪声。

2026-06-24 01:41 CST 快照：

```text
rows: 73
best AP50/AP50-95: 0.56957 / 0.36435
latest AP50/AP50-95: 0.54475 / 0.34851
late5/late10/late20 AP50-95: 0.35209 / 0.35105 / 0.35162
```

与 det-only low-LR 当前窗口对照：

```text
best delta: +0.00156
latest delta: +0.00275
late5/late10/late20 delta: +0.00473 / +0.00476 / +0.00572
```

当前解释：DSN S2 的窗口均值仍高于 det-only 当前窗口，但它尚未到 100 rows/final，且 50+ epoch 的同 epoch 优势很小。仍需等待 primary 完成，并看已排队的弱权重/decay 变体。

S2 协议：

```text
student init: RGB baseline best
teacher weights: IR baseline best
S1 projector: runs_public/cross_dataset/dsn_shared_private/dronevehicle_sub2k_seed0/dronevehicle_sub2k_rgb_ir_dsn_s1_e80_b32_ld256_h512_seed0_20260623_2304/best.pt
YOLO11n, imgsz=512, batch=64, epochs=200, seed=0
lr0/lrf: 0.001 / 0.1
warmup_epochs: 0.0
profile_kd_replace_base: 1
dsn_kd_weight: 1.0
student_projector: rgb
teacher_projector: peer
```

## 已准备：S2 refine variants

2026-06-24 01:35 CST 根据当前 S2 形态准备一个等待型 refinement 队列：

```text
docs/experiments/dronevehicle_method_search_20260623/dsn_shared_private/queue_dsn_refine_variants_after_primary_20260624.sh
logs/dronevehicle_method_search/sub2k_seed0_fullval/dsn_shared_private/s2_refine_variants/queue_dsn_refine_variants_after_primary_20260624.sh
logs/dronevehicle_method_search/sub2k_seed0_fullval/dsn_shared_private/s2_refine_variants/queue_dsn_refine_variants_after_primary_20260624.log
```

触发条件：

```text
PRIMARY_MIN_ROWS=100  # det_low/raw_low/cmd_low/dsn_s2 都至少 100 rows
FREE_GPU_THRESHOLD_MB=8000
STRICT_BATCH_SIZE=1
```

远端等待队列已于 2026-06-24 01:39 CST 启动：

```text
pid: 33404
log: logs/dronevehicle_method_search/sub2k_seed0_fullval/dsn_shared_private/s2_refine_variants/queue_dsn_refine_variants_after_primary_20260624.log
```

启动后首个检查显示：`det_rows=114`、`raw_rows=99`、`cmd_rows=68`、`dsn_rows=67`，因此尚未放行训练。

2026-06-24 01:41 CST 检查：refine queue 仍在等待，尚未启动 `w0p25_nodecay` 或 `w1p0_decay60_160_final0`；两张 GPU 仍接近满显存。

队列按顺序只跑两个低风险变体，每个变体结束后才等待空卡启动下一个：

```text
1. w0p25_nodecay
   DSN_KD_WEIGHT=0.25
   LADD_KD_DECAY_MODE=none

2. w1p0_decay60_160_final0
   DSN_KD_WEIGHT=1.0
   LADD_KD_DECAY_MODE=linear
   LADD_KD_DECAY_START_EPOCH=60
   LADD_KD_DECAY_END_EPOCH=160
   LADD_KD_FINAL_MULT=0.0
```

动机：当前 `w1p0_nodecay` 的 early-best 很强，但 50+ epoch 优势缩小；弱权重测试是否减少后期干扰，late decay 测试是否只保留早期 shared-latent 指导、后期回到 detector self-training。

## 已准备：S2 shuffled-pair control

2026-06-24 01:41 CST 补充了一个默认关闭的 shuffled-pair 控制开关：

```text
--teacher-batch-roll 1
```

它在训练 batch 内把 `teacher_img` 沿 batch 维循环错位一位，破坏 paired teacher/student 对应关系，但保持同一 batch、同一数据分布、同一 schedule 和同一 teacher。该开关默认 `0`，不会影响正在运行的 primary/refine run。

已准备后置等待队列：

```text
docs/experiments/dronevehicle_method_search_20260623/dsn_shared_private/queue_dsn_shuffled_control_after_refine_20260624.sh
logs/dronevehicle_method_search/sub2k_seed0_fullval/dsn_shared_private/s2_shuffled_controls/queue_dsn_shuffled_control_after_refine_20260624.sh
logs/dronevehicle_method_search/sub2k_seed0_fullval/dsn_shared_private/s2_shuffled_controls/queue_dsn_shuffled_control_after_refine_20260624.log
```

触发条件：

```text
REFINE_MIN_ROWS=200  # w0p25_nodecay 与 w1p0_decay60_160_final0 都完成
FREE_GPU_THRESHOLD_MB=8000
STRICT_BATCH_SIZE=1
```

启动 run：

```text
w1p0_shuffle_roll1
DSN_KD_WEIGHT=1.0
TEACHER_BATCH_ROLL=1
LADD_KD_DECAY_MODE=none
```

这个队列不会和当前 active refine queue 抢卡；它只有在两个 DSN refine 变体完成后才等待空卡启动。

## 判定规则

S2 必须同时有：

```text
s2_student_distill
c0_detonly_reload
c1_shuffled_pair
```

只有 `s2_student_distill` 超过同初始化、同 schedule 的 `c0_detonly_reload`，并且 shuffled-pair 不同样涨点，才认为 shared latent distillation 有独立收益。
