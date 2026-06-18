# LADD repair 诊断实验结果归档

日期：2026-06-13

本文整理已经跑完或已明确终止的 repair/A2 诊断实验轻量证据，并同步到 GitHub。仍在运行的实验不纳入完成结果表。

## 1. 证据位置

```text
ladd/results/repair_experiments_20260613/
```

包含内容：

- `results.csv`、`args.yaml`、`ladd_diagnostics.csv`。
- 每个 run 的 source manifest 与 compact log tail。
- 合并 summary：`summary/ladd_repair_experiments_summary_20260613.csv` 与 `summary/ladd_repair_phase_summary_20260613.csv`。

不包含：

- checkpoint 权重：`.pt`、`.pth`。
- TensorBoard event 文件、`wandb`、完整 run 目录、完整大日志。

## 2. 判据

| 模型 | SAR baseline best | SAR baseline final | safe threshold |
|---|---:|---:|---:|
| YOLO11n | 0.55654 | 0.55127 | NA |
| YOLO11s | 0.62897 | 0.62233 | 0.62697 |
| YOLO11m | 0.65580 | 0.64903 | 0.65380 |

## 3. Run 级结果

| ID | 实验 | 状态 | 主阶段 | A2 best/last | B best/last | 主阶段 best/last | vs baseline best/final | 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | s A2 lr3e-4 short13 + B weakKD0.1 B120 b32 retry2 | completed | b | 0.63331000/0.62608000 | 0.62827000/0.62267000 | 0.62827000/0.62267000 | -0.00070000/0.00034000 | no |
| S2 | s A2 lr3e-4 short13 + B det-only lr1e-4 B120 | completed | b | 0.63057000/0.63057000 | 0.63125000/0.62556000 | 0.63125000/0.62556000 | 0.00228000/0.00323000 | candidate-best-only |
| N1 | n A2 lr3e-4 short13 + B weakKD0.25 B200 | completed | b | 0.56217000/0.56217000 | 0.56476000/0.56419000 | 0.56476000/0.56419000 | 0.00822000/0.01292000 | secondary |
| M1 | m A2 lr3e-4 short4 + B1 placeholder | completed | a2 | 0.64325000/0.64325000 | 0.62839000/0.62839000 | 0.64325000/0.64325000 | -0.01255000/-0.00578000 | no |
| M2 | m A2 lr1e-4 short5 + B1 placeholder | completed | a2 | 0.64735000/0.64416000 | 0.62830000/0.62830000 | 0.64735000/0.64416000 | -0.00845000/-0.00487000 | no |
| M3 | m A2 lr3e-4 short4 freezeBN + B1 placeholder on AutoDL | completed | a2 | 0.64189000/0.64189000 | 0.62713000/0.62713000 | 0.64189000/0.64189000 | -0.01391000/-0.00714000 | no |
| M1_auto | m A2 lr3e-4 short4 duplicate on AutoDL | completed-duplicate | a2 | 0.64189000/0.64189000 | 0.62713000/0.62713000 | 0.64189000/0.64189000 | -0.01391000/-0.00714000 | no |
| Hallu_b8_aborted | HalluciDet n b8 AutoDL aborted | terminated | train |  |  | 0.10057387/0.08794306 | -0.45596613/-0.46332694 | no |

## 4. Phase 级原始索引

| ID | server | phase | rows | best@epoch | last | drop | local_results_csv |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | main_4090 | a1 | 10 | 0.62859000@10 | 0.62859000 | 0.00000000 | ladd/results/repair_experiments_20260613/evidence/main_4090/runs/s_weakkd0p1_b120_b32retry2/unknown/ladd_hbb_ogsod11s_formal_nomosaic_yolo11s_cap2_s0_fix_s_s0_a2lr3e4_short13_bweakKD0p1_b120_b32retry2_a1_e10_b32_s0_gpu0/results.csv |
| S1 | main_4090 | a2 | 13 | 0.63331000@4 | 0.62608000 | 0.00723000 | ladd/results/repair_experiments_20260613/evidence/main_4090/runs/s_weakkd0p1_b120_b32retry2/unknown/ladd_hbb_ogsod11s_formal_nomosaic_yolo11s_cap2_s0_fix_s_s0_a2lr3e4_short13_bweakKD0p1_b120_b32retry2_a2_e13_b32_s0_gpu0/results.csv |
| S1 | main_4090 | b | 120 | 0.62827000@10 | 0.62267000 | 0.00560000 | ladd/results/repair_experiments_20260613/evidence/main_4090/runs/s_weakkd0p1_b120_b32retry2/unknown/ladd_hbb_ogsod11s_formal_nomosaic_yolo11s_cap2_s0_fix_s_s0_a2lr3e4_short13_bweakKD0p1_b120_b32retry2_b_e120_b32_s0_gpu0/results.csv |
| S2 | main_4090 | a1 | 10 | 0.62878000@10 | 0.62878000 | 0.00000000 | ladd/results/repair_experiments_20260613/evidence/main_4090/runs/s_detonly_lr1e4_b120/unknown/ladd_hbb_ogsod11s_formal_nomosaic_yolo11s_cap2_s0_fix_s_s0_a2lr3e4_short13_bdetonly_lr1e4_b120_a1_e10_b64_s0_gpu1/results.csv |
| S2 | main_4090 | a2 | 13 | 0.63057000@13 | 0.63057000 | 0.00000000 | ladd/results/repair_experiments_20260613/evidence/main_4090/runs/s_detonly_lr1e4_b120/unknown/ladd_hbb_ogsod11s_formal_nomosaic_yolo11s_cap2_s0_fix_s_s0_a2lr3e4_short13_bdetonly_lr1e4_b120_a2_e13_b64_s0_gpu1/results.csv |
| S2 | main_4090 | b | 120 | 0.63125000@13 | 0.62556000 | 0.00569000 | ladd/results/repair_experiments_20260613/evidence/main_4090/runs/s_detonly_lr1e4_b120/unknown/ladd_hbb_ogsod11s_formal_nomosaic_yolo11s_cap2_s0_fix_s_s0_a2lr3e4_short13_bdetonly_lr1e4_b120_b_e120_b64_s0_gpu1/results.csv |
| N1 | main_4090 | a1 | 10 | 0.55593000@10 | 0.55593000 | 0.00000000 | ladd/results/repair_experiments_20260613/evidence/main_4090/runs/n_weakkd0p25_b200/unknown/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_fix_n_s0_a2lr3e4_short13_bweakKD0p25_b200_a1_e10_b64_s0_gpu1/results.csv |
| N1 | main_4090 | a2 | 13 | 0.56217000@13 | 0.56217000 | 0.00000000 | ladd/results/repair_experiments_20260613/evidence/main_4090/runs/n_weakkd0p25_b200/unknown/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_fix_n_s0_a2lr3e4_short13_bweakKD0p25_b200_a2_e13_b64_s0_gpu1/results.csv |
| N1 | main_4090 | b | 200 | 0.56476000@197 | 0.56419000 | 0.00057000 | ladd/results/repair_experiments_20260613/evidence/main_4090/runs/n_weakkd0p25_b200/unknown/ladd_hbb_ogsod11n_formal_nomosaic_yolo11n_cap2_s0_fix_n_s0_a2lr3e4_short13_bweakKD0p25_b200_b_e200_b64_s0_gpu1/results.csv |
| M1 | main_4090 | a1 | 10 | 0.65533000@10 | 0.65533000 | 0.00000000 | ladd/results/repair_experiments_20260613/evidence/main_4090/runs/m_lr3e4_short4_b1/unknown/ladd_hbb_ogsod11m_formal_nomosaic_yolo11m_cap2_s0_fix_m_s0_a2lr3e4_short4_b1_a1_e10_b32_s0_gpu1/results.csv |
| M1 | main_4090 | a2 | 4 | 0.64325000@4 | 0.64325000 | 0.00000000 | ladd/results/repair_experiments_20260613/evidence/main_4090/runs/m_lr3e4_short4_b1/unknown/ladd_hbb_ogsod11m_formal_nomosaic_yolo11m_cap2_s0_fix_m_s0_a2lr3e4_short4_b1_a2_e4_b32_s0_gpu1/results.csv |
| M1 | main_4090 | b | 1 | 0.62839000@1 | 0.62839000 | 0.00000000 | ladd/results/repair_experiments_20260613/evidence/main_4090/runs/m_lr3e4_short4_b1/unknown/ladd_hbb_ogsod11m_formal_nomosaic_yolo11m_cap2_s0_fix_m_s0_a2lr3e4_short4_b1_b_e1_b32_s0_gpu1/results.csv |
| M2 | main_4090 | a1 | 10 | 0.65533000@10 | 0.65533000 | 0.00000000 | ladd/results/repair_experiments_20260613/evidence/main_4090/runs/m_lr1e4_short5_b1/unknown/ladd_hbb_ogsod11m_formal_nomosaic_yolo11m_cap2_s0_fix_m_s0_a2lr1e4_short5_b1_a1_e10_b32_s0_gpu1/results.csv |
| M2 | main_4090 | a2 | 5 | 0.64735000@2 | 0.64416000 | 0.00319000 | ladd/results/repair_experiments_20260613/evidence/main_4090/runs/m_lr1e4_short5_b1/unknown/ladd_hbb_ogsod11m_formal_nomosaic_yolo11m_cap2_s0_fix_m_s0_a2lr1e4_short5_b1_a2_e5_b32_s0_gpu1/results.csv |
| M2 | main_4090 | b | 1 | 0.62830000@1 | 0.62830000 | 0.00000000 | ladd/results/repair_experiments_20260613/evidence/main_4090/runs/m_lr1e4_short5_b1/unknown/ladd_hbb_ogsod11m_formal_nomosaic_yolo11m_cap2_s0_fix_m_s0_a2lr1e4_short5_b1_b_e1_b32_s0_gpu1/results.csv |
| M3 | autodl | a1 | 10 | 0.65533000@10 | 0.65533000 | 0.00000000 | ladd/results/repair_experiments_20260613/evidence/autodl/runs/autodl_m_short4_freezebn/unknown/ladd_hbb_ogsod11m_formal_nomosaic_yolo11m_cap2_s0_fix_m_s0_a2lr3e4_short4_freezeBN_b1_autodl_retry1_a1_e10_b32_s0_gpu0/results.csv |
| M3 | autodl | a2 | 4 | 0.64189000@4 | 0.64189000 | 0.00000000 | ladd/results/repair_experiments_20260613/evidence/autodl/runs/autodl_m_short4_freezebn/unknown/ladd_hbb_ogsod11m_formal_nomosaic_yolo11m_cap2_s0_fix_m_s0_a2lr3e4_short4_freezeBN_b1_autodl_retry1_a2_e4_b32_s0_gpu0/results.csv |
| M3 | autodl | b | 1 | 0.62713000@1 | 0.62713000 | 0.00000000 | ladd/results/repair_experiments_20260613/evidence/autodl/runs/autodl_m_short4_freezebn/unknown/ladd_hbb_ogsod11m_formal_nomosaic_yolo11m_cap2_s0_fix_m_s0_a2lr3e4_short4_freezeBN_b1_autodl_retry1_b_e1_b32_s0_gpu0/results.csv |
| M1_auto | autodl | a1 | 10 | 0.65533000@10 | 0.65533000 | 0.00000000 | ladd/results/repair_experiments_20260613/evidence/autodl/runs/autodl_m_short4_duplicate/unknown/ladd_hbb_ogsod11m_formal_nomosaic_yolo11m_cap2_s0_fix_m_s0_a2lr3e4_short4_b1_autodl_retry1_a1_e10_b32_s0_gpu0/results.csv |
| M1_auto | autodl | a2 | 4 | 0.64189000@4 | 0.64189000 | 0.00000000 | ladd/results/repair_experiments_20260613/evidence/autodl/runs/autodl_m_short4_duplicate/unknown/ladd_hbb_ogsod11m_formal_nomosaic_yolo11m_cap2_s0_fix_m_s0_a2lr3e4_short4_b1_autodl_retry1_a2_e4_b32_s0_gpu0/results.csv |
| M1_auto | autodl | b | 1 | 0.62713000@1 | 0.62713000 | 0.00000000 | ladd/results/repair_experiments_20260613/evidence/autodl/runs/autodl_m_short4_duplicate/unknown/ladd_hbb_ogsod11m_formal_nomosaic_yolo11m_cap2_s0_fix_m_s0_a2lr3e4_short4_b1_autodl_retry1_b_e1_b32_s0_gpu0/results.csv |
| Hallu_b8_aborted | autodl | train | 5 | 0.10057387@1 | 0.08794306 | 0.01263081 | ladd/results/repair_experiments_20260613/evidence/autodl/runs/autodl_hallucidet_n_b8_terminated/train/hallucidet_yolo11n_s0_800ep_img256_b8_autodl_20260613/results.csv |

## 5. 主要结论

1. YOLO11s 的 A2 仍能产生较强中间 checkpoint，但 B 阶段稳定性没有解决。S1 的 A2 best 为 `0.63331@4`，进入 weakKD0.1 B120 后 best 降到 `0.62827`，final 为 `0.62267`；S2 的 det-only 低 LR B120 best 达到 `0.63125@13`，但 final 仍降到 `0.62556`，低于 s safe threshold `0.62697`。

2. YOLO11n 的 N1 是正向结果但不是当前最强主线。N1 B200 best/final 为 `0.56476/0.56419`，高于 n SAR baseline `0.55654/0.55127`，但低于此前 n BN-freeze/full mainline best 约 `0.57615`，因此只作为 short-A2 + weakKD 的辅助证据。

3. YOLO11m 当前 repair 方向仍失败。M1、M2、M3 的 A2 best 分别为 `0.64325`、`0.64735`、`0.64189`，均低于 m SAR final `0.64903` 和 safe threshold `0.65380`。短 A2、降低 A2 LR、A2 freeze BN 都没有把 m 拉回安全区间。

4. AutoDL 的 HalluciDet n b8 只记录为 terminated 证据，不作为 comparison 结果。它只跑到 5 行，best/final 为 `0.10057/0.08794`，log tail 显示进程 terminated；HalluciDet b16 仍在运行，未纳入本完成归档。

## 6. 对后续实验的含义

- s：可以保留 S2 的 best checkpoint 作为“B det-only low-LR 有短期增益”的证据，但不能声称 final protocol 已稳定。下一步如果继续 s，应围绕 B 阶段 early-stop/selection 或更短 B，而不是继续拉长 B。
- n：N1 支持 n 对 short-A2 + weakKD 有正收益，但不替代已知更强 n 主线。
- m：当前 m 不建议继续 full B；应先定位 m A2 目标或容量适配问题。

## 7. B-stage warmup 命名规则

后续 B-stage warmup 实验必须区分 KD-only warmup 与 core LADD warmup，避免把实验解释混在一起。

| 类型 | 关键配置 | 命名规则 | 解释 |
|---|---|---|---|
| KD-only warmup | `LADD_KD_DECAY_MODE=warmup_linear`；`LADD_B_LOSS_WARMUP_MODE=none` | `*_kd_warmup_*` | 只延迟打开 `alpha_kd` |
| Core LADD warmup | `LADD_KD_DECAY_MODE=none`；`LADD_B_LOSS_WARMUP_MODE=linear`；`LADD_B_LOSS_WARMUP_SCOPE=core` | `*_core_ladd_warmup_*` | 同时延迟打开 `alpha_kd`、`alpha_s_rec`、`alpha_sep`、`lambda_residual_aux` |

禁止把 KD-only warmup 写成 LADD warmup。若同时打开 KD warmup 和 core LADD warmup，`alpha_kd` 会同时乘 `kd_multiplier` 与 `b_loss_warmup_multiplier`，该配置只应用于明确的诊断，不作为默认主线。

## 8. 同步说明

本次同步只提交 lightweight evidence 与 summary。权重、TensorBoard event、wandb、完整 run 目录、完整大日志均已排除。
