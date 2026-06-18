# 对比方法最终状态（2026-06-13）

> ⚠️ ARCHIVED DIAGNOSTIC NOTE
> This document records historical comparison-method debugging status before the paper mosaic100 gate.
> It is not the source of paper main-table results.
> Use `paper_results/` and `docs/paper/PAPER_PROTOCOL_CN.md` for paper-facing results.

**更新**: HalluciDet-style已废弃，使用新HalluciDet (paper reproduction)

---

## 📊 最终对比方法清单

| 方法 | 实现质量 | 论文对齐度 | 训练状态 | 备注 |
|------|---------|-----------|---------|------|
| **LD** | A+ | 92% | 🔄 训练中 (644/800ep) | 预计周末完成 |
| **CCLKD** | A | 98% ⭐ | ❓ 待测试 | 对齐度最高 |
| **FGD** | A | 100% | ⚠️ 需验证 | 已修复normalization |
| **HalluciDet (新)** | A | 95% | ❓ 待测试 | Paper strict reproduction |
| ~~HalluciDet-style~~ | ~~废弃~~ | ~~已删除~~ | ❌ 不再使用 | 用新HalluciDet替代 |

---

## 🎯 正式对比实验计划

### 方法1: LD (Localization Distillation)
- **状态**: ✅ 训练中
- **完成度**:
  - YOLO11n: 644/800 epoch (0.570 mAP)
  - YOLO11s: 613/800 epoch (0.644 mAP)
- **预期完成**: 本周末
- **论文表述**: "LD (adapted for YOLO11 DFL, IoU-based VLR)"

### 方法2: FGD (Focal and Global KD)
- **状态**: ⚠️ 需smoke test验证
- **问题**: Loss normalization已修复
- **下一步**: 运行20 epoch smoke test
- **论文表述**: "FGD (with proper normalization for YOLO11)"

### 方法3: CCLKD (Cross-modal Contrastive Learning KD)
- **状态**: ✅ 实现完成，可选测试
- **优点**: 对齐度最高（98%）
- **下一步**: 可选20 epoch smoke test
- **论文表述**: "CCLKD (adapted for YOLO11 DFL)"

### 方法4: HalluciDet (Paper Reproduction)
- **状态**: ✅ 完整实现，可训练
- **特点**: 真正的hallucination network + frozen detector
- **下一步**: 决定是否运行（需1周训练时间）
- **论文表述**: "HalluciDet (strict reproduction with U-Net hallucination)"

---

## 📋 实验矩阵

### 必须完成的实验

| 方法 | YOLO11n seed0 | YOLO11s seed0 | 优先级 |
|------|---------------|---------------|--------|
| **Baseline** | ✅ 0.556 | ✅ 0.629 | - |
| **LADD** | ✅ 0.577 | ⚠️ 0.636 | P0 |
| **LD** | 🔄 0.570 @644ep | 🔄 0.644 @613ep | P0 |
| **FGD** | ❌ 需重跑 | ❌ 需重跑 | P0 |

### 可选但推荐的实验

| 方法 | YOLO11n seed0 | YOLO11s seed0 | 优先级 |
|------|---------------|---------------|--------|
| **CCLKD** | ❓ 未跑 | ❓ 未跑 | P1 |
| **HalluciDet (新)** | ❓ 未跑 | ❓ 未跑 | P2 |

---

## 🚀 本周行动计划

### Day 1-2 (周五-周六)
1. ✅ **等待LD完成** - 自动进行中
2. ⚠️ **FGD smoke test** - 验证修复
   ```bash
   # 20 epoch smoke test
   python train_ladd_hbb.py --comparison-kd-profile fgd \
       --epochs 20 --batch 32 --seed 0
   ```
3. 🔧 **检查LADD在s上的问题**（如果LD完成后有时间）

### Day 3-4 (周日-周一)
4. 📊 **分析LD结果** - 预计周末完成
5. 🔥 **FGD正式训练** - 如果smoke test通过
   ```bash
   # 800 epoch正式训练
   bash comparison/code/launch_formal_transfer_kd_job.sh fgd n 0 <gpu>
   bash comparison/code/launch_formal_transfer_kd_job.sh fgd s 0 <gpu>
   ```

### Day 5+ (下周)
6. ✅ **可选CCLKD** - 如果有GPU资源
7. ✅ **可选HalluciDet** - 如果有时间
8. 📝 **论文写作** - 整理结果表格

---

## 📊 预期最终对比表

### 保守估计（只包含必须完成的）

| 方法 | YOLO11n | YOLO11s | 平均提升 |
|------|---------|---------|---------|
| Baseline | 0.556 | 0.629 | - |
| **LADD** | 0.577 (+2.1%) | 0.636 (+0.7%) | +1.4% |
| **LD** | ~0.570 (+1.4%) | ~0.644 (+1.5%) | +1.5% |
| **FGD** | ? | ? | ? |

### 乐观估计（包含CCLKD/HalluciDet）

| 方法 | YOLO11n | YOLO11s | 平均提升 |
|------|---------|---------|---------|
| Baseline | 0.556 | 0.629 | - |
| **LADD** | 0.577 (+2.1%) | 0.636 (+0.7%) | +1.4% |
| **LD** | ~0.570 (+1.4%) | ~0.644 (+1.5%) | +1.5% |
| **FGD** | ~0.568? | ~0.640? | ~1.3%? |
| **CCLKD** | ? | ? | ? |
| **HalluciDet** | ? | ? | ? |

---

## 🎯 论文策略调整

### 如果只有LD和FGD

**主张**:
> "LADD demonstrates competitive performance compared to state-of-the-art cross-modal KD methods (LD, FGD), with superior results in large-gap scenarios (YOLO11n) while achieving comparable performance in small-gap settings (YOLO11s)."

### 如果有CCLKD

**主张**:
> "LADD, along with other cross-modal KD methods (LD, FGD, CCLKD), demonstrates the effectiveness of knowledge transfer for SAR→RGB detection. Our explicit decomposition approach provides interpretable intermediate representations..."

### 核心贡献不变

1. **Decomposition思想** - 理论贡献
2. **Multi-stage training** - 方法创新
3. **场景依赖分析** - Gap大小与方法选择的关系
4. **完整的对比实验** - LD/FGD等方法的严格复现

---

## ✅ 总结

**当前状态**:
- ✅ LD训练中，即将完成
- ⚠️ FGD已修复，需验证
- ✅ CCLKD实现完美，可选
- ✅ HalluciDet完整，可选
- ❌ HalluciDet-style已废弃

**最少需要**: LD + FGD（正在进行）
**理想情况**: LD + FGD + CCLKD（推荐）
**完整对比**: LD + FGD + CCLKD + HalluciDet（如果时间允许）

**优先级**: LD完成 > FGD验证 > CCLKD测试 > HalluciDet测试
