# LADD B-Entrance 诊断快照（2026-06-13）

本记录更新于 `2026-06-13 23:56 CST`。本轮已恢复 `ladd4090` ControlMaster，并刷新双卡 4090 与 AutoDL 的 lightweight evidence。只同步 `results.csv`、`args.yaml`、`ladd_diagnostics.csv`、`initial_val_metrics.json`、`b_split_load_manifest.json`、manifest/source_manifest 与关键词 log extract；未同步 checkpoint/weights、TensorBoard event、wandb、完整 run 目录或完整大日志。

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
|S2_a2best_continue_b100_retry2|ladd4090|s|100|0.62599|54|0.62174|0.00425|0.62964|-0.00298|-0.00059|S2 strict b64 retry 已完成；best 仍低于 s safe threshold，final 略低于 SAR baseline final。|
|S3_base_a2last_decomp_b100|ladd4090|s|100|0.62553|65|0.62262|0.00291|0.62799|-0.00344|0.00029|S baseline detector + A2 decomposition split-load；完成，best 接近 S2 retry 当前 best，但 final 回落到 SAR baseline final 附近。|


## 3. 正在运行 / 待完成

|run_id|server|model|planned_epochs|rows|progress_pct|best_map50_95|best_epoch|last_map50_95|eta_cst_approx|remaining_hours_approx|batch|strict_batch_size|notes|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|S4_base_a2last_kdramp_b120_retry_strict_b64_running|ladd4090|s|120|114|95.0|0.62521|62|0.62106|2026-06-14 00:14 CST|0.30|64|true|S4 strict b64 KD-only ramp，最新 114/120；当前 best/final 仍低于 safe threshold，等待最后 6 epoch。|
|comparison_n_fgd_running|ladd4090|n|800|657|82.1|0.52806|657|0.52806|2026-06-14 09:18 CST|9.37|64|false|comparison FGD n from YOLO pretrain；仍显著低于 n SAR baseline best。|
|comparison_s_fgd_running|ladd4090|s|800|585|73.1|0.61655|584|0.61652|2026-06-14 12:58 CST|13.03|64|false|comparison FGD s from YOLO pretrain；持续上升，但仍低于 s SAR baseline best/safe。|
|comparison_n_hallucidet_b16_running|autodl|n|800|402|50.2|0.35714|380|0.35093|2026-06-14 22:25 CST|22.48|||AutoDL HalluciDet standalone n b16；约半程，当前 AP 仍很低。|
|comparison_s_hallucidet_b16_running|ladd4090|s|800|65|8.1|0.18017|64|0.18017|2026-06-16 06:00 CST|54.07|16|false|dual4090 HalluciDet standalone s b16；仍很早期，不做结论。|


## 4. 失败/无效但保留的诊断证据

|run_id|server|status|rows|best_map50_95|last_map50_95|oom_seen_in_log_extract|auto_batch_reduction_seen|traceback_seen_in_log_extract|notes|
|---|---|---|---|---|---|---|---|---|---|
|S2_a2best_continue_b100_oom_old|ladd4090|failed_oom_invalid|23|0.61945|0.61945|1|0|1|旧 S2 因 OOM 中断，只作为失败证据，不作为有效结果。|
|S4_base_a2last_kdramp_b120_autobatch_old|ladd4090|auto_batch_invalid|24|0.62103|0.61618|1|1|0|旧 S4 触发 auto batch reduction 到 b32，实验无效；只保留作失败/资源证据。|


## 5. 当前读数

- `S2 retry2` 已完成：best `0.62599@54`，final `0.62174@100`。它低于 s safe threshold `0.62697`（best 差 `-0.00098`），final 也略低于 SAR baseline final `0.62233`（差 `-0.00059`），所以 A2 selected continuation 没有形成有效修复。
- `S4 KD-only ramp` 仍在跑：`114/120`，best `0.62521@62`，last `0.62106@114`。目前还低于 safe threshold，约 `2026-06-14 00:14 CST` 完成；除非最后 6 epoch 有明显回升，否则 KD-only warmup 的收益也偏弱。
- N 组结论保持：`N1 baseline continuation` best/last 为 `0.56615/0.56594`，高于 `N2/N3/N4`，说明 N 组 B-entrance 里纯 detection continuation 仍是最强参考。
- comparison 中间状态：FGD n `657/800` best `0.52806`，FGD s `585/800` best `0.61655`；HalluciDet n AutoDL `402/800` best `0.35714`；HalluciDet s dual4090 `65/800` best `0.18017`。这些只作为 comparison 进度，不作为 B-entrance 结论。
- `S4` 当前是 KD-only warmup：`LADD_KD_DECAY_MODE=warmup_linear`，`LADD_B_LOSS_WARMUP_MODE=none`。后续 core LADD warmup 必须单独命名为 `*_core_ladd_warmup_*`。

## 6. 后续建议

1. 等 `S4 KD-only ramp` 结束后立即刷新最终 best/final；如果仍低于 safe，S4 KD-only warmup 不值得扩到 B400/B800。
2. 若要继续 B entrance 修复，优先验证 `S4_core_ladd_warmup_B120`，但要明确它与 KD-only warmup 不同。
3. comparison FGD/HalluciDet 继续观察即可；当前 FGD s 还在上升，但距离 s baseline best 仍有明显差距。

## 7. 文件位置

- 状态表：`docs/experiments/ladd_b_entrance_diag_status_20260613.csv`
- 证据包：`ladd/results/b_entrance_20260613/evidence/`
- 结果摘要：`ladd/results/b_entrance_20260613/summary/`
- 最新双卡状态：`ladd/results/b_entrance_20260613/evidence/ladd4090/gpu_process_status_20260613_2355.txt`
- 最新 AutoDL 状态：`ladd/results/b_entrance_20260613/evidence/autodl/gpu_process_status_20260613_2356.txt`
