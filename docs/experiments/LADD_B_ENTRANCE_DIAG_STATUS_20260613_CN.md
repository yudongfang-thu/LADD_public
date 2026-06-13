# LADD B-Entrance 诊断快照（2026-06-13）

本快照整理了当前 B-entrance 诊断中已经完成、正在运行、以及因 OOM/自动降 batch 失效的实验。证据来自双卡 4090 服务器与 AutoDL 的轻量同步包；本地生成时间：`2026-06-13 19:21 CST`。

只同步轻量证据：`results.csv`、`args.yaml`、`ladd_diagnostics.csv`、`initial_val_metrics.json`、`b_split_load_manifest.json`、`manifest/source_manifest` 和关键词 log extract。未同步 checkpoint/weights、TensorBoard event、wandb、完整 run 目录或完整大日志。

## 1. 诊断目标

- `N1/S1`: baseline best 继续纯 detection 训练，估计“多训练 100 epoch”本身能带来多少收益。
- `N2/S2`: A2 selected/best checkpoint 进入 B，测试 warm-start 是否比 baseline continuation 更好。
- `N3/S3`: baseline detector + A2 decomposition split-load，测试 decomposition 是否在不继承 A2 detector 的情况下带来方法收益。
- `N4/S4`: split-load + KD-only warmup/ramp，测试延迟 KD 是否缓解 B 入口冲击；注意这不是 core LADD loss warmup。

## 2. 有效完成结果

|run_id|server|model|planned_epochs|best_map50_95|best_epoch|last_map50_95|best_final_drop|init_eval_map50_95|best_minus_baseline_best|last_minus_baseline_final|notes|
|---|---|---|---|---|---|---|---|---|---|---|---|
|N1_basecontinue_b100|ladd4090|n|100|0.56615|99|0.56594|0.00021|0.55584|0.00961||baseline best 纯 detection continuation；N 组最强有效 B-entrance 参考。|
|N2_a2best_continue_b100|ladd4090|n|100|0.55872|100|0.55872|0.00000|0.56102|0.00218||A2 selected/best checkpoint 继续 B100；低于 N1，说明 N 上 A2 warm-start 不优于 baseline continuation。|
|N3_base_a2last_decomp_b100|ladd4090|n|100|0.55722|100|0.55722|0.00000|0.55584|0.00068||baseline detector + A2 decomposition split-load；低于 N1，当前 split-load decomp 未证明增益。|
|N4_base_a2last_kdramp_b120|ladd4090|n|120|0.56379|113|0.56311|0.00068|0.55584|0.00725||baseline detector + A2 decomposition + KD-only ramp B120；低于 N1，但高于 N2/N3。|
|S1_basecontinue_b100_autodl|autodl|s|100|0.62493|62|0.62238|0.00255|0.62800|-0.00404|0.00005|s baseline best 纯 detection continuation；AutoDL 完成，作为 S 组 continuation 参考。|
|S3_base_a2last_decomp_b100|ladd4090|s|100|0.62553|65|0.62262|0.00291|0.62799|-0.00344|0.00029|S baseline detector + A2 decomposition split-load；完成，best 接近 S2 retry 当前 best，但 final 回落到 SAR baseline final 附近。|

## 3. 正在运行

|run_id|server|model|planned_epochs|rows|progress_pct|best_map50_95|best_epoch|last_map50_95|eta_cst_approx|remaining_hours_approx|batch|strict_batch_size|auto_batch_reduction_seen|notes|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|S2_a2best_continue_b100_retry2_running|ladd4090|s|100|86|86.0|0.62599|54|0.62251|2026-06-13 19:44 CST|0.38|64|false|0|S2 strict b64 retry 正在跑；目前 best 接近但未超过 s safe threshold。|
|S4_base_a2last_kdramp_b120_retry_strict_b64_running|ladd4090|s|120|8|6.7|0.62038|8|0.62038|2026-06-13 22:43 CST|3.36|64|true|0|S4 strict b64 KD-only ramp 正在跑；早期 AP finite，暂未见 auto-batch/NaN。|

## 4. 失败/无效但保留的诊断证据

|run_id|server|status|rows|best_map50_95|last_map50_95|oom_seen_in_log_extract|auto_batch_reduction_seen|traceback_seen_in_log_extract|notes|
|---|---|---|---|---|---|---|---|---|---|
|S2_a2best_continue_b100_oom_old|ladd4090|failed_oom_invalid|23|0.61945|0.61945|1|0|1|旧 S2 因 OOM 中断，只作为失败证据，不作为有效结果。|
|S4_base_a2last_kdramp_b120_autobatch_old|ladd4090|auto_batch_invalid|24|0.62103|0.61618|1|1|0|旧 S4 触发 auto batch reduction 到 b32，实验无效；只保留作失败/资源证据。|

## 5. Comparison 中间状态（辅助证据，不作为 B-entrance 结论）

|run_id|server|model|planned_epochs|rows|progress_pct|best_map50_95|best_epoch|last_map50_95|eta_cst_approx|remaining_hours_approx|notes|
|---|---|---|---|---|---|---|---|---|---|---|---|
|comparison_n_fgd_running|ladd4090|n|800|636|79.5|0.52561|631|0.52546|2026-06-14 05:15 CST|9.89|comparison FGD n from YOLO pretrain 正在跑；中间结果远低于 n SAR baseline。|
|comparison_s_fgd_running|ladd4090|s|800|452|56.5|0.58990|450|0.58972|2026-06-14 18:49 CST|23.46|comparison FGD s from YOLO pretrain 正在跑；中间结果远低于 s SAR baseline。|
|comparison_n_hallucidet_b16_running|autodl|n|800|315|39.4|0.32472|313|0.31930|||AutoDL HalluciDet standalone n b16 正在跑；中间结果远低于 n SAR baseline。|

## 6. 当前读数

- N 组已经形成比较清晰的结论：`N1 baseline continuation` best/last 为 `0.56615/0.56594`，明显高于 `N2/N3/N4`。这说明当前 N 的 B-entrance 里，A2 warm-start 或 split-load decomposition 暂时没有超过纯 detection continuation。
- S 组还不能定论：`S1` 和 `S3` 已完成，`S2 retry2` 与 `S4 strict b64` 仍在跑。当前 `S3` best `0.62553`、final `0.62262`，final 接近 SAR baseline final `0.62233`，但没有超过 s safe threshold `0.62697`。
- `S2 retry2` 当前 best `0.62599@54`，仍略低于 s safe threshold `0.62697`；是否有后期改善需要等 100 epoch 完成。
- 旧 `S2` 的 OOM 和旧 `S4` 的 auto batch reduction 都不应作为有效结果；它们只说明前一批资源条件会污染实验解释。当前 S4 retry 使用 strict batch size 64，截至快照未出现 auto-batch。
- `S4` 是 KD-only warmup：`LADD_KD_DECAY_MODE=warmup_linear`，`LADD_B_LOSS_WARMUP_MODE=none`。后续如果要测 core LADD warmup，应单独命名为 `*_core_ladd_warmup_*`，不要混写成 LADD warmup。
- FGD/HalluciDet comparison 仍在中途，当前 best 明显低于对应 SAR baseline；先保留进度，不做最终结论。

## 7. 后续建议

1. 先等 `S2 retry2` 和 `S4 strict b64` 完成，再决定 S 组是否进入 `S4_core_ladd_warmup_B120`。
2. 如果 `S4 KD-only warmup` 优于 `S1/S3`，再开 core LADD warmup；如果不优，优先分析 split-load/KD 分支是否本身带来负收益。
3. N 组当前优先级降低：除非有新的 B-entrance 机制，否则 N 上继续扩大 B100/B120 不太划算。

## 8. 文件位置

- 状态表：`docs/experiments/ladd_b_entrance_diag_status_20260613.csv`
- 证据包：`ladd/results/b_entrance_20260613/evidence/`
- 结果摘要：`ladd/results/b_entrance_20260613/summary/`
