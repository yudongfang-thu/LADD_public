# LADD B-Entrance 诊断快照（2026-06-13）

本记录更新于 `2026-06-13 20:01 CST`。本轮成功刷新 AutoDL；双卡服务器 `ladd4090` 在 19:58-19:59 CST 刷新时出现 `Permission denied (password,publickey)`，没有可复用 `~/.ssh/codex_cm/ladd4090.sock`，因此双卡行仍是上一版已同步快照，双卡 running 行不再给出新的 ETA。

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

## 3. 正在运行 / 待刷新

|run_id|server|model|planned_epochs|rows|progress_pct|best_map50_95|best_epoch|last_map50_95|eta_cst_approx|remaining_hours_approx|batch|strict_batch_size|auto_batch_reduction_seen|notes|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|S2_a2best_continue_b100_retry2_running|ladd4090|s|100|86|86.0|0.62599|54|0.62251|||64|false|0|S2 strict b64 retry；ladd4090 本轮无法刷新，仍为上一版快照。|
|S4_base_a2last_kdramp_b120_retry_strict_b64_running|ladd4090|s|120|8|6.7|0.62038|8|0.62038|||64|true|0|S4 strict b64 KD-only ramp；ladd4090 本轮无法刷新，仍为上一版快照。|

## 4. 失败/无效但保留的诊断证据

|run_id|server|status|rows|best_map50_95|last_map50_95|oom_seen_in_log_extract|auto_batch_reduction_seen|traceback_seen_in_log_extract|notes|
|---|---|---|---|---|---|---|---|---|---|
|S2_a2best_continue_b100_oom_old|ladd4090|failed_oom_invalid|23|0.61945|0.61945|1|0|1|旧 S2 因 OOM 中断，只作为失败证据，不作为有效结果。|
|S4_base_a2last_kdramp_b120_autobatch_old|ladd4090|auto_batch_invalid|24|0.62103|0.61618|1|1|0|旧 S4 触发 auto batch reduction 到 b32，实验无效；只保留作失败/资源证据。|

## 5. Comparison 中间状态（辅助证据，不作为 B-entrance 结论）

|run_id|server|model|planned_epochs|rows|progress_pct|best_map50_95|best_epoch|last_map50_95|eta_cst_approx|remaining_hours_approx|notes|
|---|---|---|---|---|---|---|---|---|---|---|---|
|comparison_n_fgd_running|ladd4090|n|800|636|79.5|0.52561|631|0.52546|||comparison FGD n from YOLO pretrain；ladd4090 本轮无法刷新，仍为上一版快照。|
|comparison_s_fgd_running|ladd4090|s|800|452|56.5|0.58990|450|0.58972|||comparison FGD s from YOLO pretrain；ladd4090 本轮无法刷新，仍为上一版快照。|
|comparison_n_hallucidet_b16_running|autodl|n|800|329|41.1|0.32911|318|0.32522|2026-06-14 22:44 CST|26.71|AutoDL HalluciDet standalone n b16；本轮已刷新。|

## 6. 当前读数

- N 组结论保持不变：`N1 baseline continuation` best/last 为 `0.56615/0.56594`，高于 `N2/N3/N4`。当前 N 的 B-entrance 里，A2 warm-start 或 split-load decomposition 暂时没有超过纯 detection continuation。
- S 组仍需等双卡刷新后确认：上一版快照中 `S2 retry2` 为 `86/100`，best `0.62599@54`；`S4 strict b64` 为 `8/120`，best `0.62038@8`。本轮无法确认它们是否已经完成。
- AutoDL HalluciDet n b16 已刷新到 `329/800`，best `0.32911@318`，last `0.32522@328`，按当前速度粗略预计约 `2026-06-14 22:44 CST` 完成。
- 旧 `S2` 的 OOM 和旧 `S4` 的 auto batch reduction 仍只作为无效资源条件证据，不作为有效结果解释。
- `S4` 是 KD-only warmup：`LADD_KD_DECAY_MODE=warmup_linear`，`LADD_B_LOSS_WARMUP_MODE=none`。后续 core LADD warmup 必须单独命名为 `*_core_ladd_warmup_*`。

## 7. 后续建议

1. 恢复 `ladd4090` SSH/ControlMaster 后，优先刷新 `S2 retry2`、`S4 strict b64`、FGD n/s。
2. 等 `S4 KD-only warmup` 完成后，再决定是否开 `S4_core_ladd_warmup_B120`。
3. N 组当前优先级降低；S 组需要以完成后的 S2/S4 为准。

## 8. 文件位置

- 状态表：`docs/experiments/ladd_b_entrance_diag_status_20260613.csv`
- 证据包：`ladd/results/b_entrance_20260613/evidence/`
- 双卡访问状态：`ladd/results/b_entrance_20260613/evidence/ladd4090/access_status_20260613_1959.txt`
- 结果摘要：`ladd/results/b_entrance_20260613/summary/`
