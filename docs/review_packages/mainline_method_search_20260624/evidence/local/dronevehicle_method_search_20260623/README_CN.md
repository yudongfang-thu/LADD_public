# DroneVehicle 方法搜索专区

创建日期：2026-06-23

本目录是 DroneVehicle 系列实验的唯一入口，用来避免把 DroneVehicle 方法搜索、OGSOD 主线、CCLKD/VEDAI 复现和旧 debug 证据混在一起解释。

主线搜索队列见：

```text
docs/experiments/dronevehicle_method_search_20260623/MAINLINE_CANDIDATE_QUEUE_CN.md
```

baseline / CMDistill / reload 曲线快照见：

```text
docs/experiments/dronevehicle_method_search_20260623/curves_20260624/CURVE_SUMMARY_CN.md
docs/experiments/dronevehicle_method_search_20260623/curves_20260624/map5095_curves.svg
docs/experiments/dronevehicle_method_search_20260623/curves_20260624/lowlr_map5095_zoom.svg
```

## 1. 定位

DroneVehicle 当前作为 LADD 主线重构的“小风洞”使用：

- 用途：快速验证跨模态蒸馏/表示路径是否真的能提升 student-only detector。
- 当前推荐数据：`DroneVehicle_cclkd_hbb_sub2k_seed0`，train 2000 paired samples，val full 1469 paired samples。
- 当前方向：默认 CCLKD 方向 `IR teacher -> RGB student`。
- 当前 baseline 协议：CCLKD cross-dataset YOLO11n 协议，`imgsz=512, epochs=200, SGD lr0=0.01, momentum=0.937, mosaic=0.0, close_mosaic=0, mixup=0.1`；本轮为了提效使用 `batch=64`。
- 当前 B-only/reload 协议修正：从已收敛 RGB baseline best 继续训练时使用 `lr0=0.001, lrf=0.1, warmup_epochs=0.0, warmup_bias_lr=0.0`。low-LR det-only control 已成为当前主门槛。

不要把本目录下的结果和 OGSOD paper main table、OGSOD reload control、VEDAI CMDistill native 结果直接混表。

## 2. 当前数据入口

远端 `ladd4090-zw1`：

```text
/root/shared-nvme/LADD_public/comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0
```

YAML：

```text
comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0/configs/dronevehicle_rgb_hbb.yaml
comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0/configs/dronevehicle_ir_hbb.yaml
```

生成脚本：

```text
tools/prepare_dronevehicle_subset_hbb.py
```

子集 manifest：

```text
comparison/cmdistill/native_reproduction/data/processed/DroneVehicle_cclkd_hbb_sub2k_seed0/prepare_manifest.json
```

## 3. Baseline 结果

远端服务器：`ladd4090-zw1`

| run | role | best epoch | best AP50 | best AP50-95 | final AP50 | final AP50-95 | 结果目录 |
|---|---|---:|---:|---:|---:|---:|---|
| `dronevehicle_sub2k_student_rgb_yolo11n...20260623_221620` | RGB student baseline | 141 | `0.56886` | `0.36087` | `0.55255` | `0.35385` | `runs_public/cross_dataset/cclkd_yolo11n/dronevehicle_sub2k_seed0/baselines/student_rgb/...20260623_221620` |
| `dronevehicle_sub2k_teacher_ir_yolo11n...gpu0_20260623_221936` | IR teacher baseline | 142 | `0.63800` | `0.43299` | `0.62123` | `0.42480` | `runs_public/cross_dataset/cclkd_yolo11n/dronevehicle_sub2k_seed0/baselines/teacher_ir/...gpu0_20260623_221936` |

说明：这两个 run 已完成，可作为 DroneVehicle sub2k seed0 当前方案的 S0 起点。

## 4. 当前候选方案：DSN shared-private

本轮先把“已收敛双模态 detector -> shared/private latent extraction”作为方案 A。代码入口：

```text
tools/train_dsn_shared_private_projector.py
```

远端已启动 S1：

```text
runs_public/cross_dataset/dsn_shared_private/dronevehicle_sub2k_seed0/dronevehicle_sub2k_rgb_ir_dsn_s1_e80_b32_ld256_h512_seed0_20260623_2304
logs/cross_dataset/dsn_shared_private/dronevehicle_sub2k_seed0/dronevehicle_sub2k_rgb_ir_dsn_s1_e80_b32_ld256_h512_seed0_20260623_2304.log
```

S1 已完成：`val_retrieval_top1=0.2301`，`val_retrieval_top5=0.5371`，随机 top1 约为 `1/1469` 或 `1/2048` 量级，因此 S1 已经学到明显跨模态配对信号。

S2 shared-latent KD 已于 2026-06-24 01:16:42 CST 启动，使用 low-LR/no-warmup reload protocol。当前刚启动，尚未有 `results.csv`；后续以同协议 det-only low-LR best AP50-95 `0.36279` 为主门槛。

## 5. 当前候选方案：90 oldsplit HBB adaptation

按用户要求，已把 90 服务器旧 Sixiang split iterative 方案搬到 DroneVehicle sub2k 小风洞中并行验证。该 run 不是 OBB 原版逐字复刻，而是 HBB adaptation：

- 方法参数：复刻 90 旧方案的 A1/A2/C 三段、split student branch、decomposed teacher feature、adapter reach、`lambda_reach=1.0`、`lambda_match_inner=1.0`、`lambda_rank_inner=1.0`、`delta=0.3`、`lambda_rec=0.10`、`lambda_taskL=0.0`。
- 训练协议：沿用 DroneVehicle CCLKD baseline 协议的 SGD/cosine/augmentation，避免与本目录 baseline 不可比；并发下 batch 先用 `32`。
- 方向：`IR teacher -> RGB student`，student 起点为 RGB baseline best，teacher 为 IR baseline best。

远端启动信息：

```text
runs_public/dronevehicle_method_search/sub2k_seed0_fullval/oldsplit_90_hbb/ir_to_rgb/oldsplit90_hbb_cclkdproto_ir2rgb_from_rgbbase_P1_20260623_2313_a1
logs/dronevehicle_method_search/sub2k_seed0_fullval/oldsplit_90_hbb/oldsplit90_hbb_cclkdproto_ir2rgb_from_rgbbase_P1_20260623_2313_gpu0
```

2026-06-24 00:58 CST 快照：A2 已完成，best AP50/AP50-95 为 `0.56322 / 0.36326`，略高于 RGB baseline AP50-95 `0.36087`；但 C 阶段到 epoch 133 时 best 仅 `0.55536 / 0.35518`、latest `0.54902 / 0.35164`，说明 A2 的短程正信号没有自然保持到 C。已准备 `oldsplit_a2only_controlled` 队列，用 A2-as-final 与同结构 A2 det-only split control 重新验证该信号。

## 6. 当前候选方案：CMDistill-style sanity

按照“先证明风洞有正向潜力”的策略，已在 `ladd4090-zw1` GPU1 启动 CMDistill-style B-only transferred KD：

```text
runs_public/dronevehicle_method_search/sub2k_seed0_fullval/cmdistill_style/ir_to_rgb/cmdistill_ir2rgb_yolo11n_e200_b64_img512_mosaic0p0_mixup0p1_s0_20260623_235356_b
```

协议为 `YOLO11n, imgsz=512, batch=64, epochs=200, mosaic=0.0, close_mosaic=0, mixup=0.1, seed=0`，student 为 RGB baseline best，teacher 为 IR baseline best。启动后已确认训练进入 epoch 4，batch 64 显存可承受。详细配置与早期快照见 `cmdistill_style/README_CN.md`。

2026-06-24 00:58 CST 快照：CMDistill high-LR 到 epoch 184，best AP50/AP50-95 仍为 `0.56564 / 0.35835`，latest 为 `0.55998 / 0.35437`。它仍低于 RGB baseline best `0.56886 / 0.36087` 与 high-LR det-only best `0.56705 / 0.35876`；因此 CMDistill high-LR 不能证明风洞有正向潜力。

2026-06-24 01:28 CST 快照：CMDistill low-LR 已跑到 37 rows，best AP50/AP50-95 `0.56913 / 0.36286`，latest `0.55743 / 0.35434`；best 比同协议 det-only low-LR best `0.56818 / 0.36279` 只高 `+0.00007` AP50-95，属于擦线正向。同 epoch 16/20 对照分别为 CMDistill `0.36286/0.35721` vs det-only `0.35837/0.35196`，可作为风洞有正信号的 sanity。

2026-06-24 01:35 CST 快照：CMDistill low-LR 已跑到 56 rows，best 仍为 `0.56913 / 0.36286`，latest `0.56210 / 0.35745`；epoch50/52/54 AP50-95 分别为 `0.35170/0.35458/0.35223`，均高于 det-only 同 epoch 的 `0.34889/0.34977/0.34968`。这说明 CMDistill 在风洞上有稳定一些的 sanity 正信号，但全局 best 仍只是擦线超过 det-only，不作为 LADD 主线。

2026-06-24 01:41 CST 快照：CMDistill low-LR 已跑到 73 rows，best 仍为 `0.56913 / 0.36286`，latest `0.54839 / 0.35023`，late20 AP50-95 `0.35431`，继续明显高于 det-only 当前 late20 `0.34590`；仍作为 sanity-positive，而不是主线。

2026-06-24 01:47 CST 曲线检查：原始 RGB baseline 是从头训练慢慢涨到 best AP50/AP50-95 `0.56886 / 0.36087`；det-only lowLR reload 则在 epoch 2 达到 `0.36279` 后持续下滑，到 epoch 139 为 `0.34300`；CMDistill lowLR 在 epoch 16 达到 `0.36286`，之后也下滑，但 epoch 50/54/73/90 均高于 det-only 同 epoch。因此 CMDistill 的意义不是显著提高 best，而是证明该风洞中跨模态 KD 可以延缓 reload 下滑。

## 7. 当前候选方案：Raw feature KD

已在 `ladd4090-zw1` GPU1 追加低改动 P1：

```text
runs_public/dronevehicle_method_search/sub2k_seed0_fullval/raw_feature_kd/ir_to_rgb/rawfeatkd_ir2rgb_yolo11n_a0p25_affine_e200_b64_img512_mosaic0p0_mixup0p1_s0_20260624_000336_b
```

协议同 CMDistill / det-only，方法上只保留 `raw teacher feature -> raw student feature` 的 foreground KD，`alpha_kd=0.25`，`KD_CALIBRATION_MODE=affine`，并关闭 reach/rec/taskL。epoch 1 AP50-95 为 `0.35732`，接近 det-only control `0.35876`，但尚未超过 control。

2026-06-24 00:07 CST 快照：raw feature KD 到 epoch 6 后 latest AP50-95 掉到 `0.25078`；det-only control 到 epoch 21 latest AP50-95 也掉到 `0.28183`。这说明当前最大混杂是从 baseline best 继续训练时仍使用 `lr0=0.01 + default warmup`。

2026-06-24 01:28 CST 快照：low-LR/no-warmup det-only 已跑到 79 rows，best AP50/AP50-95 `0.56818 / 0.36279`；low-LR raw feature KD 已跑到 68 rows，best `0.56875 / 0.36265`，仍略低于 det-only `0.00014` AP50-95。rawKD late20 略高于 det-only 当前窗口，但不足以成为主线。

2026-06-24 01:35 CST 快照：low-LR/no-warmup det-only 已跑到 101 rows，best `0.56818 / 0.36279`，latest `0.54235 / 0.34673`；low-LR raw feature KD 已跑到 87 rows，best `0.56875 / 0.36265`，latest `0.54428 / 0.34767`。rawKD 当前窗口略好于 det-only，但全局 best 仍未过 det-only。

## 7.5. 当前候选方案：DSN shared-latent S2

2026-06-24 01:35 CST 快照：DSN S2 已跑到 54 rows，best AP50/AP50-95 `0.56957 / 0.36435`，是当前最高 early-best；但 latest `0.54667 / 0.34953`、epoch54 AP50-95 `0.34953` 已接近 det-only epoch54 的 `0.34968`。因此 DSN 仍是最值得改造的主线候选，但当前 S2 版本还不能判定为稳定。

对比 CMDistill：CMDistill low-LR 到 56 rows 时 best `0.56913 / 0.36286`，虽然全局 best 只擦线超过 det-only，但 epoch50/52/54 都明显高于 det-only。它更适合作为“风洞有正向潜力”的 sanity，而 DSN 更像“可改造成 LADD 新主线”的候选。

2026-06-24 01:41 CST 快照：DSN S2 已跑到 73 rows，best 仍为 `0.56957 / 0.36435`，latest `0.54475 / 0.34851`，late20 AP50-95 `0.35162`。它仍是最高 best 候选，但稳定性还没过关；弱权重/decay refine queue 仍在等待，尚未启动新变体。

## 8. 当前候选方案：Reachable fused shared

按用户提出的“两个 share 融合后用 CAP2/reachable 约束，再蒸馏 student share”思路，已在 HBB 主线加入默认关闭的 fused shared 开关，并在 `ladd4090-zw1` 启动 P3 队列：

```text
logs/dronevehicle_method_search/sub2k_seed0_fullval/reachable_fused_shared/queue_reachable_fused_shared_after_controls_20260624.sh
```

该队列曾等待 low-LR/no-warmup reload controls 至少跑出 20 epoch 后启动 `c0_nofusion_splitrec`，但 2026-06-24 01:13:55 的首次 c0 启动与 GPU0 上的 rawKD/CMDistill 并发冲突，发生 OOM 与 batch fallback，结果无效。队列 shell 已停止，后续 P3 需要在空卡上以 `STRICT_BATCH_SIZE=1` 重启。详细设计、代码入口与成败标准见 `reachable_fused_shared/README_CN.md`。

## 9. 已准备但尚未启动的低风险候选

这些队列已经在本地/远端 ready，但当前不抢卡；它们都等待 low-LR det-only、low-LR raw KD、low-LR CMDistill 至少 20 rows 后再放行。

```text
teacher_conf_gated_kd/
  teacher_conf_gate
  rawkd_late_decay

reachability_weighted_kd/
  splitkd_unweighted
  reachgap_weighted

oldsplit_a2only_controlled/
  a2_detonly_split_control
  a2_reach_kd_lowlr
```

用途：

- `teacher_conf_gate`：用 A2 teacher task head 的 confidence 给 KD token 加权，测试跨模态监督是否需要可靠性门控。
- `rawkd_late_decay`：把 KD 从 epoch 60 到 160 线性衰减到 0，测试 late collapse 是否来自持续 KD。
- `reachgap_weighted`：不优化强 reach loss，只用可达 gap 做 KD 权重，测试 learnability 信号是否应从“结构约束”改成“token weighting”。
- `oldsplit_a2only_controlled`：针对旧方案 A2 曾出现的 `AP50-95=0.36326` 正信号，单独验证 A2-as-final 是否超过同结构 A2 det-only split control。

## 10. 后续 run 放置规范

从下一批 DroneVehicle 方法搜索开始，统一使用专用 root，且每一种方法单独一个子目录，避免继续散落到 OGSOD 或 generic cross-dataset 目录。已有正在运行的 DSN / oldsplit run 不移动，只在本 README 中作为 legacy path 索引；后续 S2 和 controls 按下面结构落盘。

```text
runs_public/dronevehicle_method_search/<dataset_variant>/<method>/<stage_or_control>/<run_name>
logs/dronevehicle_method_search/<dataset_variant>/<method>/<stage_or_control>/<run_name>
debug/dronevehicle_method_search_20260623/<method>/<case_name>
docs/experiments/dronevehicle_method_search_20260623/<method>/
```

推荐 dataset variant 命名：

```text
sub2k_seed0_fullval
full_trainval
sub1k_seed0_fullval
sub2k_seed0_val300
```

推荐方法命名：

```text
baselines
dsn_shared_private
oldsplit_90_hbb
cmdistill_style
raw_feature_kd
object_aware_dsn
shared_only_ablation
reload_controls
shuffled_pair_controls
```

每个方法目录内部固定使用以下 stage/control 名称：

```text
s0_baseline_or_inputs
s1_probe_or_pretrain
s2_student_distill
c0_detonly_reload
c1_shuffled_pair
c2_raw_feature_kd
ablations
notes
```

当前建议目录树：

```text
runs_public/dronevehicle_method_search/sub2k_seed0_fullval/
  baselines/
    rgb_student/
    ir_teacher/
  dsn_shared_private/
    s1_probe/
    s2_student_distill/
    c0_detonly_reload/
    c1_shuffled_pair/
    ablations/
  oldsplit_90_hbb/
    ir_to_rgb/
    c0_detonly_reload/
  cmdistill_style/
    ir_to_rgb/
    c0_detonly_reload/
  raw_feature_kd/
    ir_to_rgb/
    c0_detonly_reload/
  reload_controls/
    lr1e-3_nowarmup/
```

对应日志：

```text
logs/dronevehicle_method_search/sub2k_seed0_fullval/<method>/<stage_or_control>/<run_name>
```

对应说明文档：

```text
docs/experiments/dronevehicle_method_search_20260623/
  dsn_shared_private/README_CN.md
  oldsplit_90_hbb/README_CN.md
  cmdistill_style/README_CN.md
  raw_feature_kd/README_CN.md
```

如果继续使用已有 paper launcher，请通过环境变量覆盖 `PROJECT` 和 `LOG_DIR` 到上述 root。

## 10. 解释规则

1. baseline-first：没有同子集、同方向、同 batch 的 RGB student 和 IR teacher baseline，不启动 LADD/CMDistill 结论性实验。
2. 子集与全量分开：`sub2k_seed0_fullval` 只能和同一子集结果比较，不能直接声称等价于 full DroneVehicle。
3. 方向分开：`IR->RGB` 与 `RGB->IR` 必须分表；reverse 只用于诊断方向敏感性。
4. batch 标注：本轮 `batch=64` 是提效设置，不等同 CCLKD paper reported `batch=16`。
5. 推理模态标注：LADD/CMDistill 结果必须写清楚 inference-only modality；当前默认是 RGB student-only inference。
6. 旧 LADD 在 DroneVehicle 上的 no-gain 结果只能作为失败诊断，不作为新方法负结果的最终结论。
7. DSN S2 必须使用同初始化、同 schedule 的 detector-only continued-training control；只有超过该 control 与 shuffled-pair distill，才可认为 shared latent distillation 有独立收益。

## 10. 既有证据入口

本地 debug 证据：

```text
debug/dronevehicle_no_gain_20260622/
debug/dronevehicle_reverse_20260623/
debug/current_grouped_curves_20260623/
```

关键结论：

- 默认 `IR teacher -> RGB student`：旧 LADD main 基本不提升 RGB student baseline。
- reverse `RGB teacher -> IR student`：旧 LADD B 对 AP50-95 也没有稳定增益。
- 这说明 DroneVehicle 适合作为方法搜索风洞：没有 OGSOD reload 自涨点那种强混杂，但数据本身有跨模态文献信号。

## 11. 下一步

当前 baseline 已完成；下一步在本目录下追加：

```text
baseline_dsn_s1_oldsplit90_sub2k_seed0_b64_20260623.md
```

记录：

- RGB student best / final AP50 与 AP50-95
- IR teacher best / final AP50 与 AP50-95
- DSN S1 best retrieval 指标和 checkpoint
- 90 oldsplit HBB adaptation 的 A1/A2/C best / final 指标
- 训练时长、显存、是否 OOM
- 后续 CMDistill-style 与新 LADD candidate 的目标线
