# CCLKD YOLO11n 消融计划

最后更新：2026-06-06

本页用于把 CCLKD 原文消融表逐项映射到当前公开仓库代码。目标不是先追求多
seed 完整均值，而是用 YOLO11n 单 seed 的模块增益方向判断实现是否符合原文。

## 原文参照

CCLKD 原文 Table 12 在 OGSOD-1.0 上给出的模块消融如下：

| 配置 | mAP50 | mAP |
|---|---:|---:|
| Baseline | 80.9 | 46.3 |
| LLD | 83.4 | 48.5 |
| LLD + FLD | 84.2 | 49.3 |
| LLD + FLD + RLD | 84.9 | 50.1 |
| LLD + FLD + RLD + PATM | 87.0 | 55.1 |
| CCL only | 85.9 | 54.4 |
| Full CCLKD | 88.7 | 57.3 |

判断标准应以相对趋势为主：LLD 应高于 baseline，FLD/RLD 应继续带来小幅增益，
PATM 应带来明显增益，CCL only 应有较大独立收益，Full 应高于 ATKD-only 和
CCL-only。绝对数值仍受 YOLO11 适配、数据划分、增强细节和实现差异影响。

## 当前代码映射

消融 launcher：

```bash
bash cclkd_reproduction/code/launch_cclkd_n_ablation_job.sh <ablation> <seed> <gpu_id>
```

支持的 `<ablation>`：

| 名称 | 对应原文行 | 关键开关 |
|---|---|---|
| `lld` | LLD | `lld=1, fld/rld/ccl=0, T=1` |
| `lld_fld` | LLD + FLD | `lld=fld=1, rld/ccl=0, T=1` |
| `lld_fld_rld` | LLD + FLD + RLD | `lld=fld=rld=1, ccl=0, T=1` |
| `atkd` | LLD + FLD + RLD + PATM | `lld=fld=rld=1, ccl=0, T=[0.5,5.0]` |
| `ccl_only` | CCL only | `lld=fld=rld=0, ccl=1` |
| `full` | Full CCLKD | `lld=fld=rld=ccl=1, T=[0.5,5.0]` |
| `full_ccl05` | 旧默认诊断 | `full` 但 `ccl=0.5` |

原文没有报告 Fixed-Temp KD 的具体固定温度值。本仓库使用 `T=1.0` 作为无 PATM
时的中性固定温度，并在结果表中明确记录。

注意：`ccl_only` 的唯一 KD 信号来自 COP 触发后的 CCL。teacher 与 student 都从
COCO 初始化时，早期 teacher 对 OGSOD RGB 的置信度可能低于 `min_confidence`，
导致 COP 正样本为空、CCL 返回 0。因此 `ccl_only` 前若干 epoch 与 SAR baseline
曲线接近属于预期空窗期，记录结果时必须标注。

## 已修正的复现权重

原文 Table 4 的最佳配置为 `lambda_kd=1.0, lambda_cc=1.0`。此前代码默认
`--ccl-weight 0.5`，因此旧的 full CCLKD 运行只能作为诊断结果，不能视为严格
Table 4 / Table 12 的 full 配置。本次已将原文复现与正式 online CCLKD launcher
默认值改为 `--ccl-weight 1.0`，并在 paper reproduction launcher 中显式传入。

## 建议启动顺序

先用 YOLO11n seed0 跑下面 7 个配置：

```bash
for ablation in lld lld_fld lld_fld_rld atkd ccl_only full full_ccl05; do
  bash cclkd_reproduction/code/launch_cclkd_n_ablation_job.sh "$ablation" 0 <gpu_id>
done
```

其中 `full_ccl05` 只用于解释旧结果，不进入论文主表。若 GPU 时间不足，优先级为：
`full`、`atkd`、`ccl_only`、`lld_fld_rld`、`lld`、`lld_fld`、`full_ccl05`。
