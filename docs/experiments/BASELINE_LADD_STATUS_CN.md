# Baseline 与 LADD 主方法状态

最后更新：2026-06-05 16:45 CST

用途：给导师快速查看当前 formal no-mosaic baseline、LADD 主方法和可启动条件。对比方法来源与 DOI 见 [`COMPARISON_METHODS_RECORD_CN.md`](COMPARISON_METHODS_RECORD_CN.md)。

## 1. 正式协议

当前正式 baseline 协议：

```text
OGSOD-1.0 HBB
imgsz=256
800 epochs
cos_lr
full no-mosaic
default Albumentations
SAR-only detector for student/inference
RGB teacher uses same capacity and same seed when available
```

服务器口径：

| 服务器 | 路径 | 当前作用 |
|---|---|---|
| 90 | `/mnt/dataY/ydf/projects/LADD_og` | baseline 主参考；当前受控对比和部分 LADD 见缝插针 |
| 4090D | `/root/autodl-tmp/LADD` | 无卡模式/历史结果恢复；当前不作为新实验状态来源 |
| 双卡 4090 | `/root/shared-nvme/ladd` | 2026-06-05 发现 `nc=5` yaml 错误，相关结果作废；当前不启动新实验 |
| 117 | 暂停 | 文件 IO/网络过慢，暂不作为当前受控实验主力 |

## 2. Baseline 最新结果

结果来自 90 服务器 `runs_public/ogsod/hbb/formal_nomosaic_20260528/baselines`。表中数值为 `metrics/mAP50-95(B)` 的 best。

| Model | seed | batch | SAR baseline | RGB teacher | teacher-student gap | 状态 |
|---|---:|---:|---:|---:|---:|---|
| YOLO11n | 0 | 64 | 0.55654@734 | 0.63018@723 | 0.07364 | 完成 |
| YOLO11n | 42 | 64 | 0.55794@739 | 0.62664@739 | 0.06870 | 完成 |
| YOLO11n | 123 | 64 | 0.56128@797 | 0.62933@789 | 0.06805 | 完成 |
| YOLO11s | 0 | 64 | 0.62897@729 | 0.65768@647 | 0.02871 | 完成 |
| YOLO11s | 42 | 64 | 0.62879@735 | 0.66218@683 | 0.03339 | 完成 |
| YOLO11s | 123 | 64 | 0.62357@750 | 0.65987@710 | 0.03630 | 完成 |
| YOLO11m | 0 | 32 | 0.65580@704 | 0.67909@663 | 0.02329 | 完成 |
| YOLO11l | 0 | 32 | 0.65427@735 | 0.68356@618 | 0.02929 | 完成 |
| YOLO11x | 0 | 16 | 0.65867@685 | 0.68284@539 | 0.02417 | 附录容量点 |

多 seed 汇总：

| Model | SAR mean | SAR std | RGB mean | RGB std | 平均 gap | 结论 |
|---|---:|---:|---:|---:|---:|---|
| YOLO11n | 0.55859 | 0.00244 | 0.62872 | 0.00185 | 0.07013 | 主实验最重要容量，蒸馏空间最大 |
| YOLO11s | 0.62711 | 0.00307 | 0.65991 | 0.00225 | 0.03280 | 第二主容量，baseline 已三 seed 齐 |
| YOLO11m | 0.65580 | — | 0.67909 | — | 0.02329 | seed0 已齐，需补 42/123 |
| YOLO11l | 0.65427 | — | 0.68356 | — | 0.02929 | seed0 已齐，需补 42/123 |

## 3. Baseline 缺口

| 优先级 | 缺口 | 影响 |
|---|---|---|
| P1 | YOLO11m seed42/123 SAR+RGB | m 多 seed 主表/稳定性证据 |
| P2 | YOLO11l seed42/123 SAR+RGB | l 多 seed 主表/稳定性证据 |
| P3 | YOLO11x 多 seed | 仅作附录容量趋势，不是主表必须 |

当前 n/s 的 baseline 条件已经充分，可以支撑 LADD 主实验和当前四个受控对比方法先在 n/s 上跑通。

## 4. LADD 主方法最新状态

主线配置：

```text
A1=10 -> A2=50 -> B=800
cap2 reach-rank
A2/B 使用 MuSGD lr0=0.001 no warmup
对 YOLO11n seed0/123 的 B 塌缩诊断新增 FREEZE_BN_STATS=1 修正版
```

| Model | seed | 服务器/版本 | epoch | 当前 AP50-95 | best AP50-95 | vs SAR baseline | 状态判断 |
|---|---:|---|---:|---:|---:|---:|---|
| YOLO11n cap2 | 0 | 90 `a2mu1e3` | 800 | 0.57504 | 0.57662@725 | +0.02008 | 完成，可作为主表有效点 |
| YOLO11n cap2 | 42 | 90 `a2mu1e3` | 800 | 0.57293 | 0.57420@735 | +0.01626 | 完成，可作为主表有效点 |
| YOLO11n cap2 | 42 | 4090D r2 | 800 | 0.55222 | 0.57191@420 | +0.01397 | 跑满但后期退化，跨机器 sanity |
| YOLO11n cap2 | 123 | 90 `bstable1e3` | 800 | 0.52875 | 0.56161@165 | +0.00033 | 能防 NaN，但后期退化，不作为主线 |
| YOLO11n cap2 | 0 | 90 BN-freeze | 800 | 0.57254 | 0.57276@793 | +0.01622 | 完成，稳定候选 |
| YOLO11n cap2 | 123 | 90 BN-freeze | 800 | 0.57219 | 0.57269@779 | +0.01141 | 完成，修复 seed123 退化 |
| YOLO11s cap2 | 0 | 90 `a2mu1e3` | 608 | 0.63527 | 0.63551@605 | +0.00654 | 未满 800，但已有正向收益 |
| YOLO11s cap2 | 0 | 4090D r2 | 800 | 0.58962 | 0.61787@570 | -0.01110 | 跑满但低于 baseline |
| YOLO11s cap2 | 42 | 4090D r2 | 800 | 0.58895 | 0.60838@638 | -0.02041 | 跑满但低于 baseline |
| YOLO11s cap2 | 123 | 4090D r2 | 800 | 0.58143 | 0.60849@513 | -0.01408 | 跑满但低于 baseline |
| YOLO11m cap2 | 0 | 90 `a2mu1e3` | 121 | 0.52361 | 0.59796@1 | -0.05784 | 当前异常，暂不纳入主表 |

## 5. 当前判断

YOLO11n 是目前最稳的主线证据：seed0 和 seed42 的 `a2mu1e3` 已经完成且分别提升约 +2.0 和 +1.6 个 AP，说明 LADD 在蒸馏空间最大的 n 容量上确实有效。seed123 的旧 B 和 `bstable1e3` 证明只降低学习率不能解决后期退化；BN-freeze 版本已经跑满 seed0/123，并将 seed123 修复到 `0.57269@779`，相对 SAR baseline 提升 +0.01141。当前最合理的统一主线候选是 `cap2 + A2/B MuSGD 1e-3 + B FREEZE_BN_STATS=1`。

YOLO11s 的 baseline 三 seed 已齐。90 上 seed0 的 LADD 跑到 epoch 608，best `0.63551@605`，相对 SAR baseline 0.62897 有 +0.00654，方向是正的；4090D 上 s 三 seed 已跑满但全部低于 baseline，需要复核协议/实现差异，不能作为主线证据。

YOLO11m/l seed0 baseline 已齐，但 m 的 LADD 当前异常，l 尚未启动。下一阶段应优先保持 n/s 主线和当前受控对比方法跑完 seed0，再补 n 三 seed闭环，最后扩展到 m/l。

## 6. 阶段性目标

当前阶段目标不是一次性铺满所有容量，而是先保证：

1. 补 YOLO11n seed42 BN-freeze，使当前稳定主线候选形成严格三 seed 同协议闭环。
2. YOLO11s LADD 至少 seed0 在 public 最终代码上跑满，并确认 4090D/90 协议差异。
3. FGD、LD、CCLKD-style、HalluciDet-style 四个受控方法先完成 smoke，再至少在 YOLO11n seed0 闭环；其中 s seed0 作为第二容量优先补。
4. CoLD/CrossKD 与无效旧结果只保留在统一归档中，不进入 controlled main table。

更细的 LADD 崩溃证据见 [`../../ladd/diagnostics/b_collapse/LADD_CRASH_EVIDENCE_20260604_CN.md`](../../ladd/diagnostics/b_collapse/LADD_CRASH_EVIDENCE_20260604_CN.md)，当前受控对比方法代码映射见 [`../../comparison/METHOD_CODE_MAP_CN.md`](../../comparison/METHOD_CODE_MAP_CN.md)。
