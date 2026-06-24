# LADD HBB capR Audit 与 capR-Gated KD 记录

日期：2026-06-24  
分支：`codex/ladd-capr-audit-and-gated-kd-v1`  
实现位置：`ladd/code/src/teacher_student_decomposition_kd_hbb/`，同步快照：`ladd/code_versions/current_hbb/`

## 1. 改动摘要

本轮只做训练期 loss/diagnostics 与离线 audit，不改变最终 SAR YOLO student 的 inference/export graph。

新增能力：

- 在 `ladd_diagnostics.csv` 中记录 capR/reachability 的距离、gap、rank active、cap saturation、FG/BG 和 P3/P4/P5 摘要。
- 新增 `kd_weight_mode=cap_reachability_gap`，用 capR 一致的 reachable margin 对 KD token 加权。
- 新增 `--kd-target-branch {z,u,shuffled_z}` 作为负控制，默认仍为 `z`。
- 新增 `--shuffle-teacher-pairs` 作为 paired RGB-SAR 负控制。
- 新增离线 `audit_ladd_learnability_hbb.py`，检查 z_t/u_t 对 SAR student 是否形成 learnable/unlearnable 分工。
- 新增 `inspect_ladd_run_args.py` 和 `monitor_ladd_capr_gatedkd_20260624.py` 便于确认 run 参数与早筛状态。

本轮没有加入默认 z/u 强正交、private anti-task、adversarial suppression，也没有新增推理分支。

## 2. 新增训练参数

`train_ladd_hbb.py` 和 `current_hbb/tools/train_ladd_hbb.py` 新增：

```bash
--kd-weight-mode cap_reachability_gap
--kd-reach-margin 0.0
--kd-reach-tau 0.2
--kd-reach-use-capped-gap / --no-kd-reach-use-capped-gap
--kd-reach-detach-weight / --no-kd-reach-detach-weight
--kd-reach-min-weight 0.0
--kd-reach-conf-power 1.0
--kd-reach-active-threshold 0.5
--kd-target-branch z|u|shuffled_z
--shuffle-teacher-pairs
```

主线默认仍是 `--kd-target-branch z`。`u` 和 `shuffled_z` 只用于负控制，不作为主线正结果。

## 3. 不改变推理结构

新增逻辑全部位于：

- `loss.py`：训练期 loss weighting 与 diagnostics。
- `trainer.py`：训练 batch 负控制、args 透传、epoch diagnostics。
- `tools/`：离线分析脚本。

没有在 `model.py` 中新增 inference module；没有修改 `student_detect_mode` 为新推理分支；test/export 时仍只依赖原 SAR YOLO student。

## 4. capR Audit 字段

配置字段：

- `rank_d_neg_cap`
- `normalize_reach`
- `capR_effectively_enabled`
- `reach_delta`
- `reach_input_mode`
- `use_fg_mask_for_reach`
- `kd_weight_mode`
- `kd_reach_use_capped_gap`

距离字段：

- `d_pos_*`
- `d_neg_*`
- `d_neg_eff_*`
- `gap_raw_*`
- `gap_capped_*`
- `reachable_margin_*`

rank/cap 状态：

- `cap_saturation_ratio`
- `rank_active_ratio`
- `cap_blocked_active_ratio`
- `zero_loss_feasible_ratio`

FG/BG 与 per-level：

- `fg_gap_capped_mean` / `bg_gap_capped_mean`
- `fg_cap_saturation_ratio` / `bg_cap_saturation_ratio`
- `P3/P4/P5_gap_capped_mean`
- `P3/P4/P5_cap_saturation_ratio`
- `P3/P4/P5_rank_active_ratio`

解释重点：

- `rank_d_neg_cap=4.0` 且 `normalize_reach=True` 时基本等价 no-cap。
- `cap_saturation_ratio` 接近 0：capR 很可能没有实际作用。
- `cap_blocked_active_ratio` 长期高：u_t 已被 cap 截断，但 q_s/z_t 仍不够近。

## 5. KD Gate 字段

`kd_weight_mode=cap_reachability_gap` 使用：

```text
d_pos = ||normalize(q_s)-normalize(z_t)||^2_sum
d_neg = ||normalize(q_s)-normalize(u_t)||^2_sum
d_neg_gate = min(d_neg, rank_d_neg_cap)  # capR enabled 时
reachable_margin = d_neg_gate - d_pos - delta
w = sigmoid((reachable_margin - kd_reach_margin) / kd_reach_tau)
```

再与 teacher confidence / target score 权重相乘，并归一化到 mean=1。

诊断字段：

- `kd_reach_weight_mean/std/min/max/q25/q50/q75`
- `kd_reach_active_ratio`
- `kd_reach_conf_mean`
- `kd_reach_reachable_margin_mean/q25/q75`
- `P3/P4/P5_kd_active_ratio`

健康预期：`kd_reach_active_ratio` 不应长期接近 0 或 1，应能筛出 high-confidence/reachable token。

## 6. Learnability Audit

脚本：

```bash
python ladd/code/tools/audit_ladd_learnability_hbb.py \
  --weights <ladd_or_dynamic.pt> \
  --data shared/configs/datasets_public/ogsod1_sar_detect.yaml \
  --teacher-data shared/configs/datasets_public/ogsod1_rgb_detect.yaml \
  --teacher-weights <rgb_teacher.pt> \
  --split val --imgsz 256 --batch 16 --device 0 \
  --max-batches 20 --max-tokens-per-level 4096 \
  --output-dir runs_public/ogsod/hbb/audits/<run_name>
```

输出：

- `learnability_audit_summary.csv`
- `learnability_audit_per_level.csv`
- `learnability_audit_per_batch.csv`
- `learnability_audit_config.yaml`
- `learnability_audit_notes.md`

核心指标：

- `learnability_gap_direct = d(q_s,u_t)-d(q_s,z_t)`，正值表示 z_t 更接近 SAR q_s。
- `learnability_gap_probe = R2(q_s->z_t)-R2(q_s->u_t)`，正值支持 z_t 更 SAR-learnable。
- `task_auc_z/u` 只说明 z_t/u_t 的 task utility；`u_t` task 高不自动代表失败。

paired-vs-shuffled 通过同一 checkpoint 跑两次 audit 比较：

```bash
# paired
python ladd/code/tools/audit_ladd_learnability_hbb.py ... --output-dir audits/run_paired

# shuffled
python ladd/code/tools/audit_ladd_learnability_hbb.py ... --shuffle-teacher-pairs --output-dir audits/run_shuffled
```

## 7. Inspect 与 Monitor

检查单个 run 的 capR 参数：

```bash
python ladd/code/tools/inspect_ladd_run_args.py \
  --run-dir <run_dir>
```

输出：`run_args_capr_summary.json`。

监控候选和 det-only control：

```bash
python docs/experiments/monitor_ladd_capr_gatedkd_20260624.py \
  --baseline <detonly_run_dir> \
  --run dynamic_capR2=<run_dir> \
  --run gatedKD=<run_dir>
```

表格字段：

- `run_name`
- `rows`
- `latest_map5095`
- `best_map5095`
- `late20_map5095`
- `latest_delta`
- `late20_delta`
- `capR_enabled`
- `cap_saturation_ratio`
- `rank_active_ratio`
- `kd_active_ratio`
- `status`

## 8. Smoke 命令

正式跑实验前建议先做 1 epoch / 2 batch smoke：

```bash
# det-only smoke
python ladd/code/train_ladd_hbb.py \
  --phase b --model yolo11n.pt \
  --data shared/configs/datasets_public/ogsod1_sar_detect.yaml \
  --teacher-data shared/configs/datasets_public/ogsod1_rgb_detect.yaml \
  --teacher-weights <rgb_teacher.pt> \
  --imgsz 256 --epochs 1 --batch 64 --device 0 \
  --ladd-b-det-only --max-train-batches 2 \
  --project runs_public/ogsod/hbb/smoke_capr --name detonly_smoke

# capR-gated KD smoke
python ladd/code/train_ladd_hbb.py \
  --phase b --model yolo11n.pt \
  --data shared/configs/datasets_public/ogsod1_sar_detect.yaml \
  --teacher-data shared/configs/datasets_public/ogsod1_rgb_detect.yaml \
  --teacher-weights <rgb_teacher.pt> \
  --imgsz 256 --epochs 1 --batch 64 --device 0 \
  --rank-d-neg-cap 2.0 \
  --kd-weight-mode cap_reachability_gap \
  --kd-reach-use-capped-gap --kd-reach-tau 0.2 \
  --max-train-batches 2 \
  --project runs_public/ogsod/hbb/smoke_capr --name capr_gatedkd_smoke

# shuffled teacher smoke
python ladd/code/train_ladd_hbb.py ... \
  --rank-d-neg-cap 2.0 --kd-weight-mode cap_reachability_gap \
  --shuffle-teacher-pairs \
  --project runs_public/ogsod/hbb/smoke_capr --name shuffledT_smoke

# KD-to-u smoke
python ladd/code/train_ladd_hbb.py ... \
  --rank-d-neg-cap 2.0 --kd-weight-mode cap_reachability_gap \
  --kd-target-branch u \
  --project runs_public/ogsod/hbb/smoke_capr --name kd_to_u_smoke
```

通过标准：

- 进入第 1 epoch；
- 生成 `results.csv` 和 `ladd_diagnostics.csv`；
- diagnostics 包含新增字段；
- 无 Traceback / CUDA OOM / NaN / batch fallback；
- args.yaml 正确记录新增参数。

## 9. 第一批正式实验命令模板

所有实验必须 YOLO-init，同机 same-pipeline det-only control。

```bash
# dynamic_capR2_yoloinit
python ladd/code/train_ladd_hbb.py ... \
  --rank-d-neg-cap 2.0 \
  --name dynamic_capR2_yoloinit

# dynamic_capR4_yoloinit
python ladd/code/train_ladd_hbb.py ... \
  --rank-d-neg-cap 4.0 \
  --name dynamic_capR4_yoloinit

# dynamic_capR2_gatedKD_yoloinit
python ladd/code/train_ladd_hbb.py ... \
  --rank-d-neg-cap 2.0 \
  --kd-weight-mode cap_reachability_gap \
  --kd-reach-use-capped-gap \
  --kd-reach-tau 0.2 \
  --kd-reach-margin 0.0 \
  --kd-reach-detach-weight \
  --name dynamic_capR2_gatedKD_yoloinit

# dynamic_capR2_gatedKD_wo_srec_yoloinit
python ladd/code/train_ladd_hbb.py ... \
  --rank-d-neg-cap 2.0 \
  --kd-weight-mode cap_reachability_gap \
  --alpha-s-rec 0.0 \
  --name dynamic_capR2_gatedKD_wo_srec_yoloinit

# negative controls
python ladd/code/train_ladd_hbb.py ... \
  --rank-d-neg-cap 2.0 \
  --kd-weight-mode cap_reachability_gap \
  --shuffle-teacher-pairs \
  --name dynamic_capR2_gatedKD_shuffledT_yoloinit

python ladd/code/train_ladd_hbb.py ... \
  --rank-d-neg-cap 2.0 \
  --kd-weight-mode cap_reachability_gap \
  --kd-target-branch u \
  --name dynamic_capR2_gatedKD_toU_yoloinit
```

## 10. 负控制解释

- KD-to-z 正常，KD-to-u 弱：支持 z_t 更适合作为 SAR KD target。
- KD-to-u 也涨：u_t 中也有 SAR 可学习知识，当前 z/u 分解 claim 不成立或角色混淆。
- shuffled teacher 也涨：方法可能主要是 auxiliary regularization，没有真正利用 paired RGB-SAR teacher。
- paired 明显优于 shuffled：支持 paired teacher feature alignment 有实际贡献。

## 11. 已知风险与下一步

- `cap_reachability_gap` 是否有效依赖 `rank_d_neg_cap` 和 `normalize_reach` 的真实配置，必须先看 diagnostics。
- 若 `cap_blocked_active_ratio` 长期过高，瓶颈可能不是 u_t 不够远，而是 q_s/z_t 不够近。
- 100 epoch 只做 early screen，最终主线 claim 仍需要 e800、late-window、final/best 和 seed。
- 如果 learnability audit 显示 `R2(q_s->u_t) >= R2(q_s->z_t)`，应回到 A-stage decomposition 目标，而不是继续调 KD gate。
