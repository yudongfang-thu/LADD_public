# Direct-400 public experiment ledger

日期: 2026-07-09 CST

本文是 direct-400 阶段的公开实验账本。它从本地 PM / Coordinator 记录中抽取 compact facts, 去掉私有路径和远端连接信息, 用于 GitHub 读者理解当前实验矩阵。

状态标签:

- `FINAL_FACT_READY`: 已审计为可用事实, 但不等于 claim-ready。
- `FINAL_PENDING`: 自然完成或 monitor 显示完成, 仍需 artifact / protocol / health audit。
- `RUNNING_PROGRESS_ONLY`: 正在跑或只读到中间进度, 不能进 final table。
- `BLOCKED`: 因访问、路径、协议或 provenance 缺口不能验收。
- `DIAGNOSTIC_ONLY`: 只用于定位问题, 不作为主 claim evidence。

## 1. Current direct-400 gate

| gate item | value |
|---|---|
| Current protocol bucket | `pure direct-400` |
| LADD expansion gate | seed0 must approach or exceed strong comparison seed0 first |
| FGD seed0 floor | AP50-95 `0.55147` |
| LD seed0 threshold | AP50-95 `0.56048` |
| CMDistill seed0 threshold | AP50-95 `0.56240` |
| Current best LADD row | `singleproj_plus_ld_replace_base`, seed0, 3090 |
| Current best LADD AP50-95 | `0.50350` |
| Current gap to FGD floor | `-0.04797` |
| Current route | `DISPATCH_FAILURE_LOCALIZATION` after latest 3090 final fact audit |

## 2. Comparison seed0 anchors

These rows define the direct-400 comparison floor / threshold used by the current PM gate.

| server | method | model | seed | AP50-95 | role |
|---|---|---|---:|---:|---|
| 90 | FGD | YOLO11n | 0 | 0.55147 | floor |
| 90 | LD | YOLO11n | 0 | 0.56048 | competitive threshold |
| 90 | CMDistill | YOLO11n | 0 | 0.56240 | strongest current seed0 comparison |

Boundary: these are comparison lines, not LADD matched controls.

## 3. 90 det-only controls

| server | method | model | seed | AP50-95 | status | use |
|---|---|---|---:|---:|---|---|
| 90 | det-only | YOLO11n | 42 | 0.48657 | `FINAL_FACT_READY` | 90 comparison seed42 control |
| 90 | det-only | YOLO11n | 123 | 0.48589 | `FINAL_FACT_READY` | 90 comparison seed123 control |

Boundary: these controls help analyze 90 comparison seed42 / seed123 rows. They are not LADD matched gain controls for 3090 LADD rows.

## 4. 3090 LADD seed0 final fact audit

The latest completed-row final fact audit accepted the five 3090 LADD seed0 rows below as audit-clean candidates. All remain below FGD floor.

| family | variant | seed | final AP50 | final AP50-95 | best AP50-95 | status | next route |
|---|---|---:|---:|---:|---:|---|---|
| reset-v2 | `singleproj_kd_only_no_decomp_aux` | 0 | 0.75473 | 0.49190 | 0.49198 | `FINAL_FACT_READY` | failure localization |
| reset-v2 | `singleproj_plus_ld_profile` | 0 | 0.75973 | 0.49831 | 0.49957 | `FINAL_FACT_READY` | failure localization |
| reset-v2 | `singleproj_plus_cmdistill_profile` | 0 | 0.76372 | 0.49159 | 0.49185 | `FINAL_FACT_READY` | failure localization |
| reset-v2 | `plain_plus_cmdistill_profile` | 0 | 0.75749 | 0.49056 | 0.49156 | `FINAL_FACT_READY` | failure localization |
| fusion-v1 | `singleproj_plus_ld_replace_base` | 0 | 0.77631 | 0.50350 | 0.50361 | `FINAL_FACT_READY` | failure localization |

Audit facts:

- row count: 400 for all five rows.
- final epoch: 400 for all five rows.
- protocol: `imgsz=256`, `epochs=400`, `mosaic=0.0`, `close_mosaic=0`, `cos_lr=true`, batch 64, seed0.
- health scan: no fatal / OOM / Traceback / NaN found in audited 3090 artifacts.
- claim boundary: these rows are facts for failure localization, not positive LADD evidence.

## 5. 90 CMDistill / LD seed42 / seed123 rows

Monitor values looked strong, but latest final fact audit could not inspect artifacts because the 90 route was blocked by SSH/TUN access.

| method | seed | monitor AP50 | monitor AP50-95 | latest audit status | next route |
|---|---:|---:|---:|---|---|
| CMDistill | 42 | 0.83796 | 0.56520 | `BLOCKED_BY_90_SSH_ACCESS` | scoped 90 artifact audit |
| CMDistill | 123 | 0.83720 | 0.56550 | `BLOCKED_BY_90_SSH_ACCESS` | scoped 90 artifact audit |
| LD | 42 | 0.83501 | 0.56463 | `BLOCKED_BY_90_SSH_ACCESS` | scoped 90 artifact audit |
| LD | 123 | 0.82935 | 0.56183 | `BLOCKED_BY_90_SSH_ACCESS` | scoped 90 artifact audit |

Boundary: monitor values cannot be promoted to final facts until result path, args, weights listing, logs, protocol and risk scan are verified.

## 6. Progress-only rows

| server | family | method / variant | seed | latest epoch | latest AP50-95 | status |
|---|---|---|---:|---:|---:|---|
| 3090 | LADD fusion-v1 | `singleproj_plus_fgd_profile` | 0 | 262 | 0.46514 | `RUNNING_PROGRESS_ONLY` in accepted monitor |
| 90 | comparison | FGD | 42 | 98 | 0.44346 | `RUNNING_PROGRESS_ONLY` in accepted monitor |
| 90 | comparison | FGD | 123 | 200 | 0.50461 | `RUNNING_PROGRESS_ONLY` in accepted monitor |

Boundary: progress-only rows must stay out of final comparison tables.

## 7. Previously planned or diagnostic direct-400 work

| item | purpose | status / lesson |
|---|---|---|
| 3090 dense-pack YOLO11n rows | Fill GPUs with pure direct-400 n-level rows | useful for throughput, but extra seed3/4 was later rejected; default official seed set is 0/42/123 |
| registered seed panel 0/42/123 | Align seeds with user policy | source policy and strict matched-control boundaries remain important |
| BN-freeze diagnostics | Investigate possible B-stage BN pollution | deprioritized by user as low-value for current LADD loss-surface question |
| 400 vs 800 curve analysis | Understand why row400-from-800 and pure 400 diverge | learning-rate / schedule / protocol differences remain a key concern; do not mix protocols |
| launcher / CLI parity audit | Check whether launch scripts pass args unsupported by trainer | active blocker for future LADD / LADD-fusion launches |

## 8. Current interpretation

Current direct-400 evidence is negative for the active claim:

```text
best LADD AP50-95 0.50350 < FGD floor 0.55147
```

This suggests the project should prioritize:

1. Failure localization on the five audited 3090 LADD rows.
2. Scoped 90 artifact audit to unblock comparison stability.
3. Launcher / CLI parity review before any new LADD / fusion launch.
4. A small implementation rescue decision step only after failure localization, not before.

## 9. What not to conclude

- Do not claim LADD beats FGD / LD / CMDistill under direct-400.
- Do not use 90 monitor values as final facts until artifact audit succeeds.
- Do not use progress-only FGD seed42 / seed123 as final stability evidence.
- Do not expand LADD seed42 / seed123 from the current seed0 evidence.
- Do not mix direct-400 with row400-from-800, 800, 1600, reload, or external reported rows.
