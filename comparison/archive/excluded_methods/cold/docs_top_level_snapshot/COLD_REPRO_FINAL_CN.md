# CoLD 复现总结

最后更新：2026-06-02

## 1. 论文要点

CoLD (Category-Oriented Localization Distillation), TGRS 2023. DOI: `10.1109/TGRS.2023.3291356`

**核心机制**：CPM (Category Prior Matching) 将 teacher 候选框按类别分为 target (m_t) 和 nontarget (m_hat_t) 两组，分别做 TCLD 和 NCLD 蒸馏。IWM (IoU Weighting Module) 对候选框加权。OKD (Online KD) 让 teacher/student 在线同步训练。

**关键消融 (Table III)**：

| TCLD | NCLD | AP |
|---|---|---|
| off | off | 0.463 (baseline) |
| on | off | 0.502 |
| off | on | **0.563** |
| on | on | 0.567 |

NCLD 是主导项（+0.100），TCLD 贡献很小（+0.039）。

**Table IV (模块消融)**：

| CPM | IWM | OKD | AP |
|---|---|---|---|
| off | off | off | 0.463 |
| on | off | off | 0.497 |
| off | off | on | 0.519 |
| on | on | off | 0.534 |
| on | on | on | 0.567 |

OKD 单独贡献 +0.056。训练时间 42.59 GPUHour vs YOLOv5 41.11 (仅 3.6% 开销)。

## 2. 实现概述

基于 YOLOv5 v5.0 (CSPDarkNet-X, 86.23M params), OGSOD-1.0 HBB, 256x256, batch=64, T=20, alpha=2.

代码入口：
- `scripts/ogsod_public/cold_baseline_repro_20260528/train_cold_v5p0_hbb.py`
- `scripts/ogsod_public/cold_baseline_repro_20260528/run_cold_v5p0_hbb.sh`

两种 CPM 模式：
- `candidate`：基于 teacher confidence 选 topk=1000 候选框，逐图逐类 Python 循环计算 KL
- `matched`：用 YOLOv5 anchor matching 的正样本，速度快但覆盖率低

关键参数：`COLD_LOSS_MODE=candidate`, `CANDIDATE_TOPK=1000`, `COLD_IWM_MODE=none` (CPM 机制验证阶段不开启 IWM)。

## 3. 实验结果

### 3.1 4090D (matched 模式, 400ep)

| 实验 | AP | 备注 |
|---|---|---|
| YOLOv5x baseline | 0.445 | 同协议 |
| CoLD matched (OKD on) | 0.470 | +0.025，远低于原文 +0.104 |

matched 模式只在 GT positive anchors 上蒸馏，无法复现 NCLD 的大规模 nontarget 蒸馏。

### 3.2 5880 Ada (candidate 模式, 50ep)

使用 4090D 同协议 YOLOv5x baseline @50ep 作为参照 (mAP@.5=0.429, mAP=0.200)：

| 实验 | mAP@.5 | mAP | vs baseline |
|---|---|---|---|
| online NCLD (OKD on) | 0.520 | 0.260 | **+30%** |
| online TCLD (OKD on) | 0.604 | 0.323 | **+61%** |
| offline NCLD (frozen teacher) | ~0.26 | ~0.11 | 低于 baseline |

### 3.3 关键发现

1. **在线 OKD 有效**：NCLD +30% over baseline，相对增益对齐论文
2. **TCLD > NCLD 与论文相反**：可能 50ep 太短，NCLD 收敛更慢；也可能是实现差异
3. **离线 frozen teacher 无效**：比 baseline 还差。OKD 的贡献 (+0.056) 比 CPM 单开 (+0.034) 更大
4. **candidate 模式速度瓶颈**：逐图逐类 Python 循环导致 ~2 s/it（正常 YOLOv5x ~0.5 s/it），比论文报告的 3.6% 开销高 10 倍

## 4. 实现偏差

1. **bbox distribution 近似**：YOLOv5 v5.0 没有 DFL bins，用候选框集合上的 softmax KL 近似。不等于原文 DFL-bin 实现
2. **candidate 循环**：Python 逐图逐类循环（3 layers x batch x 3 classes），不是向量化实现
3. **速度**：~2 s/it vs 论文 ~1.67 s/it。瓶颈在 Python 循环，不在 GPU

## 5. 最终判断

- **CPM 机制方向是对的**：在线 NCLD 相对 baseline 有明确正向增益
- **不能声称复现成功**：TCLD/NCLD 趋势与论文相反，candidate 模式速度远差于论文
- **OKD 是关键组件**：不可省略
- **适合放在 90 服务器慢慢跑**：CPU-bound 任务，不占用珍贵 GPU 算力
- **不建议在 5880 Ada 上继续投入**：应优先主线实验和对比方法
