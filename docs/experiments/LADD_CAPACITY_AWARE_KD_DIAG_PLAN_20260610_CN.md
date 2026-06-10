# LADD Capacity-aware / Gap-aware KD 诊断计划

日期：2026-06-10

用途：在不改变 formal no-mosaic 协议、不重写 LADD 主方法的前提下，定位 YOLO11s/m 容量上的 late degradation 是否来自 KD/aux 权重过强、late-stage negative distillation，或 A2 阶段已经损伤 detector。

## 1. 当前假设

YOLO11n 已在 `cap2 + A2/B MuSGD lr0=1e-3 no warmup + B FREEZE_BN_STATS=1` 下形成 seed0/42/123 正向证据。n 的 RGB-SAR gap 约 `0.07013`，蒸馏收益余量较大。

YOLO11s/m 的 RGB-SAR gap 明显更小：

| 容量 | seed0 SAR | seed0 RGB | gap |
|---|---:|---:|---:|
| YOLO11n | 0.55654 | 0.63018 | 0.07364 |
| YOLO11s | 0.62897 | 0.65768 | 0.02871 |
| YOLO11m | 0.65580 | 0.67909 | 0.02329 |

因此当前假设是：

- n 稳住不是因为完全没有退化，而是 teacher-student gap 大，KD 正收益足够覆盖轻微后期目标错配；
- s/m gap 小，固定 `alpha_kd=1.0` 和固定 aux 权重更容易在后期变成 negative distillation；
- m 可能不是 B 后期才退化，而是在 A2 阶段已经被 reach/aux 或优化冲击损伤。

## 2. 不考虑 Mosaic 的原因

当前正式主线仍是 full no-mosaic baseline protocol。旧 90 服务器 `mosaic=1.0, close_mosaic=700` 结果只作为历史可行性和反崩溃证据，说明 LADD 方法本身并非必然 B 阶段崩溃；它不作为本轮下一步主线实验协议。

本轮所有诊断都保持：

```text
imgsz=256
full no-mosaic
default Albumentations
A1 -> A2 -> B
cap2
A2/B MuSGD lr0=1e-3 no warmup
B FREEZE_BN_STATS=1 for s B diagnostics
```

## 3. 新增代码开关

本轮新增：

| 开关 | 作用 |
|---|---|
| `--ladd-kd-decay-mode {none,linear,cosine,step}` | 仅 B 阶段调度 KD multiplier |
| `--ladd-kd-decay-start-epoch` / `--ladd-kd-decay-end-epoch` | 1-based phase epoch，与 `results.csv` 对齐 |
| `--ladd-kd-final-mult` | KD 退火或 step 后的最终 multiplier |
| `--ladd-kd-stop-after-epoch` | B 阶段到达该 epoch 后强制 KD multiplier=0 |
| `--ladd-b-det-only` | B 阶段 trainability 不变，但关闭所有非 detection loss |
| `--ladd-a2-det-only` | A2 阶段 trainability 不变，但关闭所有非 detection loss |

`ladd_diagnostics.csv` 新增 effective weight 字段，用于确认每个 epoch 的实际 KD/aux 权重。

## 4. 本轮最小实验

| 优先级 | 实验 | 目的 | 如何解释 |
|---|---|---|---|
| P0 | dry-run | 确认新增变量能从 launcher 进入 Python CLI/manifest | dry-run command 中必须出现新增参数 |
| P1 | `s alpha_kd=0.5 B400` | 判断 s late degradation 是否来自 KD 强度过大 | last 明显改善，尤其接近 `0.62697`，说明 KD 强度是主因 |
| P1 | `m A2 probe, B=1` | 判断 m 是否 A2 已经低于 baseline | A2 best < `0.65380` 则不进 m full B |
| P2 | `s alpha_kd=0.25 B400` | 进一步寻找 safe KD 强度 | 0.25 稳于 0.5 说明需要 capacity-aware KD |
| P2 | `s B det-only B400` | 判断无 KD/aux 是否仍退化 | det-only 稳而 KD 掉，说明 KD/aux 负迁移；det-only 也掉，说明 B 长训练自身有问题 |
| P3 | `s KD decay 200->300 B400` | 判断 late-stage KD 是否导致过峰后下降 | 过峰后不再掉，late KD 负迁移成立 |
| P3 | `s KD stop@220 B400` | 直接切断 late KD | 比 decay 更激进，验证停止 KD 是否保住泛化 |
| P4 | `m A2 det-only` | 判断 A2 aux 是否伤 m | det-only 恢复说明 A2 reach/aux 是主因 |
| P4 | `m A2 lr3e-4` | 判断 A2 lr 是否冲击大容量 | 改善说明大容量需要更低 A2 lr |
| P4 | `m A2 aux-half` | 判断 reach/match/rank 是否过强 | 改善说明 A2 需要 capacity-aware aux 权重 |

## 5. 合格判据

YOLO11s B400：

```text
baseline = 0.62897
PASS threshold = 0.62697
```

- `last AP50-95 >= 0.62697`：可认为当前设置基本缓解 late degradation；
- `best > baseline` 但 `last < 0.62697`：仍为 late degradation；
- `best <= baseline`：该设置不支持 s 容量收益。

YOLO11m A2 probe：

```text
baseline = 0.65580
A2 safe threshold = 0.65380
```

- A2 best 不低于 `0.65380`，才考虑进入 B；
- A2 best 低于 `0.65380`，优先做 A2 det-only / lr3e-4 / aux-half，不开 m full B。

## 6. 汇总工具

新增工具：

```bash
python tools/summarize_ladd_capacity_diag.py <run_dir> [...]
```

默认输出：

```text
docs/experiments/ladd_capacity_kd_diag_20260610_summary.csv
docs/experiments/LADD_CAPACITY_KD_DIAG_RESULTS_20260610_CN.md
```

输出字段包括 best/last、baseline gain、late window loss delta、effective KD policy、det-only 标记和 PASS/WEAK/FAIL/DIAG 状态。
