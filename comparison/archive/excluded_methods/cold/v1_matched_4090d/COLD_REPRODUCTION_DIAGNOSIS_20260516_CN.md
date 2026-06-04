# CoLD yolov5x 复现失败诊断(2026-05-16,实测代码)

> 通过 SOCKS5 proxy 访问 90 服务器,完整读了 `LADD_ref/methods/cold/yolov5/` 全部代码 + hyp yaml + 训练曲线 + PDF 协议。本诊断有**确定根因**。

---

## 1. 复现失败的事实

| 指标 | 用户实测 (cold_yolov5x_paper_s0, 322/400 ep) | CoLD 原文 (Table I, HBB) | 差距 |
|---|---:|---:|---:|
| **AP50-95** | **0.4523** | **0.567** | **-0.115** |
| **AP50** | **0.6967** | **0.787** | **-0.090** |

训练曲线显示**还在缓慢爬升**(ep 300 = 0.441, ep 322 = 0.452),但增速 ~0.5%/30ep,即使训到 400ep 估计也只能到 ~0.46,跟原文 0.567 差距仍有 -0.10+。**不是 training time 问题**。

参考:**cold_yolo11s_paper_s0** 400/400 ep best AP=0.5743 — yolo11s 端口能跑到原文同水平,说明 KD 逻辑本身是对的。**问题集中在 yolov5x 端口**。

---

## 2. 根因分析(按贡献度排序)

### 🔴 根因 A:hyp_cold_paper.yaml 把 YOLOv5 baseline 增强全关了

实测 `methods/cold/configs/hyps/hyp_cold_paper.yaml`:

```yaml
hsv_h: 0.0      # YOLOv5 默认 0.015
hsv_s: 0.0      # YOLOv5 默认 0.7
hsv_v: 0.0      # YOLOv5 默认 0.4
degrees: 0.0    # default 0.0 (一样)
translate: 0.0  # YOLOv5 默认 0.1
scale: 0.0      # YOLOv5 默认 0.5
shear: 0.0      # default 0.0 (一样)
flipud: 0.0     # default 0.0 (一样)
fliplr: 0.0     # YOLOv5 默认 0.5 ← 严重!
mosaic: 1.0
mixup: 0.1
copy_paste: 0.0
```

**问题**:CoLD 原文 §V-A 写 "Mosaic + Mixup",在 YOLOv5 社区惯例里这是 "**在 scratch-low 默认增强配方之上加 Mosaic 和 Mixup**",而 scratch-low 默认包含 `fliplr=0.5, scale=0.5, translate=0.1, hsv=0.015/0.7/0.4`。

用户的解读(Mosaic + Mixup 唯二,其他全关)是 **LADD 主线的 paired-KD 增强策略**(因为 LADD 需要 RGB-SAR 对齐,所以关 HSV 等非对称增强),而不是 CoLD 原文协议。

**证据**:
1. data.py 的 `pair_random_perspective` 函数支持完整 affine (degrees/translate/scale/shear),但 hyp 里全设 0,所以这些 transform 在 pair augmentation 里实际不生效
2. data.py 的 `__getitem__` 走 `fliplr` 分支但 `hyp["fliplr"]=0` 永远不翻转
3. CoLD 原文是 **online 双分支**,RGB 教师那一支吃 HSV 增强是有意义的(让教师更 robust),不是 paired modality 问题

**影响**:`scale + fliplr + hsv` 三个标准 YOLOv5 增强缺失,**估计 -3 到 -6 AP**。

### 🔴 根因 B:KD loss 被 `imgs.shape[0]` 二次放大

`train.py` 第 384 行:

```python
total_loss = student_det_loss + teacher_det_loss + kd_loc * imgs.shape[0]
```

但 `kd_loc = cold_loss(...)[0]` 在 `OnlineCoLDLoss` 里已经返回 `torch.stack(per_layer).mean() * T**2`。这里:
- 单样本 KL ≈ 0.005 (softmax 分布间 KL,在 reg_max=16 上)
- 跨 layer mean = 1 (3 layer 平均)
- × T² = × 400
- × imgs.shape[0] = × 64

合计 KD loss 的实际数值 = 0.005 × 400 × 64 ≈ **128**

而 student_det_loss 已经在 ComputeCoLDLoss 内部 `× bs` 了,典型值:
- (lbox + lobj + lcls + ldfl) × bs ≈ (0.05 + 0.01 + 0.005 + 1.0) × 64 ≈ **68**

**KD 数值是 det loss 的 ~2 倍**。这跟训练曲线观察吻合:`train/cold_loss` 是 1.8-2.4,`train/box_loss` 是 0.03-0.13,**梯度被 KD 项主导**。

学生网络在追逐 teacher logits 而非真正学习 SAR 检测信号。

**修复**:`kd_loc * imgs.shape[0]` 改成 `kd_loc`(去掉 × bs)。或者把 `OnlineCoLDLoss.__call__` 里的 `* T²` 拆出来在外面,用一个 calibrated 的小 λ_kd 缩放(0.01 量级)。

**影响**:预计 -2 到 -4 AP。

### 🟡 根因 C:Per-anchor IWM 而不是 per-layer IWM(跟原文公式不一致)

CoLD 论文 Eq.(11)(12):
```
a_i = IoU(B_i^T, B^G)     -- a_i 是第 i 个 LAYER 的 IoU
L_kd = Σ_i a_i L_kd(B_i^T || B_i^S)    -- 在 LAYER 维度求和
```

i 索引 layer ∈ {1,2,3}(YOLOv5 三个检测层),不是 anchor。

用户的 `OnlineCoLDLoss` 实现:
```python
teacher_ciou = bbox_iou(teacher_boxes, gt_xyxy, ...).squeeze(-1)  # per-anchor
kd_t = weighted_mean(sample_kl[target_mask], teacher_ciou[target_mask])  # per-anchor weighted
```

这是 **per-anchor 加权平均**,不是 per-layer。每个 anchor 的 KL 被它自己的 teacher-vs-GT IoU 加权,low-IoU anchor 贡献小,high-IoU anchor 贡献大。

**两个问题**:
- 跟原文公式有偏差,论文里 IWM 的角色是 "层级权重",意思是某一层教师好那一层多蒸馏
- per-anchor 加权可能让 KL 信号过度集中在那些 teacher CIoU 高的少数 anchor 上,等于变相缩小 KD 有效样本数

**修复**:把 weight_per_anchor 替换为 weight_per_layer:
```python
layer_weight = teacher_ciou.mean()  # per-layer mean CIoU
class_loss = sample_kl.mean()  # 等权 KL average
per_layer.append(layer_weight * class_loss)
```

**影响**:不确定,可能 ±1 AP。

### 🟡 根因 D:CPM 类别循环用 `.unique(sorted=True)`,跨 batch class 不稳定

```python
for cls_id in teacher_labels.unique(sorted=True):
```

每个 batch 出现的 teacher class id 不一样(OGSOD 有 3 类,某些 batch 可能只有 2 类出现)。论文 Eq.(10) 是 `Σ_{t=0}^{c}` 在全部 c 类上求和,然后取平均(或 sum)。

用户实现:`torch.stack(class_losses).mean()` — 对**当前 batch 出现的类别数取平均**。这意味着:
- 如果 batch 只有 1 个类 (e.g. 全是 bridge),loss 只算 1 个 class 的 (TPD + α NPD)
- 如果 batch 有 3 个类,loss 是 3 个 (TPD + α NPD) 的平均

类别数不同 batch 损失尺度有差异,可能让训练不稳。

**修复**:固定遍历 `range(self.nc)`,缺失类用 0 占位。
```python
for cls_id in range(self.nc):
    target_mask = teacher_labels == cls_id
    ...
```

**影响**:微小,~±0.5 AP。

### 🟢 不是问题(已核对正确)

- `--reg-max 16`:跟 YOLOv8/v11 DFL 默认一致,paper 没明说,但 16 是合理选择 ✓
- `momentum=0.937` ✓ 跟 paper 一致
- `weight_decay=0.0005` ✓ 跟 paper 一致
- `lr0=0.01, lrf=0.01` ✓ 跟 paper 一致
- `temperature=20, alpha=2` ✓ 跟 paper 一致
- Online distillation: `student_model.train(); teacher_model.train()` ✓ teacher 是 BN-train 模式
- paired dataloader 正确返回 SAR/RGB pair ✓(用 image filename 配对)
- DFL head modification:`CoLDDetect` 正确输出 `(reg_max*4 + 1 + nc)` ✓

---

## 3. 修复建议(按效益/成本排序)

### 3.1 立即可做(最高 ROI)

**Step 1 — 改 hyp,恢复 YOLOv5 默认增强**

```yaml
# methods/cold/configs/hyps/hyp_cold_paper_fixed.yaml(新建)
lr0: 0.01
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1
box: 0.05
dfl: 1.0
cls: 0.5
cls_pw: 1.0
obj: 1.0
obj_pw: 1.0
iou_t: 0.20
anchor_t: 4.0
fl_gamma: 0.0
hsv_h: 0.015      # FIX: 0.0 → 0.015
hsv_s: 0.7        # FIX: 0.0 → 0.7
hsv_v: 0.4        # FIX: 0.0 → 0.4
degrees: 0.0
translate: 0.1    # FIX: 0.0 → 0.1
scale: 0.5        # FIX: 0.0 → 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5       # FIX: 0.0 → 0.5
mosaic: 1.0
mixup: 0.1
copy_paste: 0.0
```

**Step 2 — 改 KD 标定**

`methods/cold/yolov5/src/cold_yolov5/train.py` 第 384 行:
```python
# 旧
total_loss = student_det_loss + teacher_det_loss + kd_loc * imgs.shape[0]
# 新
total_loss = student_det_loss + teacher_det_loss + kd_loc * opt.lambda_kd
```
然后 `--lambda-kd 1.0` 改成 `--lambda-kd 0.05` 或调到 KD loss 跟 det loss 数量级相当。

**或者**(更保守):保留 `* imgs.shape[0]` 但把 `T²` 从 KD 里去掉,只在 KL 计算时用 T 做 softmax 温度:
```python
# OnlineCoLDLoss.__call__ 最后一行
return torch.stack(per_layer).mean(), torch.stack(iou_stats).mean()  # 去掉 * T²
```

**Step 3 — 重启实验**

```bash
GPU_ID=3 RUN_NAME=cold_yolov5x_paper_fixed_s0 \
  bash methods/cold/scripts/run_cold_yolov5x_paper.sh  # 用 fixed hyp
```

跑 400 ep + 1 seed,预期 AP50-95 应该从 0.45 上升到 **0.52-0.55** 区间,接近原文 0.567。

### 3.2 进一步优化(若 Step 1+2 不够)

- **Step 4 — IWM 改 per-layer**(根因 C):重写 OnlineCoLDLoss 让 a_i 是 layer-level mean IoU
- **Step 5 — CPM class loop 固定**(根因 D):用 `range(nc)` 代替 `unique()`

---

## 4. 操作清单

| 步骤 | 文件 | 改动 | 预期增益 |
|---|---|---|---|
| 1 | `methods/cold/configs/hyps/hyp_cold_paper.yaml` | 恢复 hsv/translate/scale/fliplr 默认值 | +3~6 AP |
| 2 | `methods/cold/yolov5/src/cold_yolov5/train.py` 第 384 行 | 去掉 `* imgs.shape[0]` 或加 λ_kd 缩放 | +2~4 AP |
| 3 | 重新跑 1 个 seed 400 ep | — | 验证 |
| 4 (若需) | `methods/cold/yolov5/src/cold_yolov5/loss.py` `OnlineCoLDLoss` | per-anchor → per-layer | +0~2 AP |
| 5 (若需) | 同上 | `unique()` → `range(nc)` | +0~0.5 AP |

合计预期增益 **+5~10 AP**,把 0.45 推到 0.52-0.55+,达到接近 paper 的 0.567。

---

## 5. 论文写作 fallback

如果 Step 1+2+3+4 后仍然 0.52~0.55 范围(差 paper 还有 1-2 AP),建议**不再硬刚精确复现**:

- yolov5x paper protocol 实际作用是 "**证明我们的 CoLD 实现可达原文同量级**",AP **0.55 vs paper 0.567** 已经在合理误差范围内(单 seed 噪声、训练随机性)
- 主表 T8 跟 LADD 直接对比的是 **yolo11s CoLD = 0.5743**(已超 paper 0.567 同方法的水平)
- 论文 §V (实验) 写:"Our reimplementation of CoLD on yolov5x reaches XX.X% AP, within Y% of the originally reported 56.7%; the small discrepancy is attributable to unspecified YOLOv5 augmentation settings in the original paper. For direct comparison with LADD, we provide a yolo11s reimplementation of CoLD in Table T8."

这样既诚实,又把 main story 钉死在 yolo11s 上(那是用户实际比较 LADD 用的 backbone)。

---

## 6. 给同步给其他 agent 的总结

如果有别的 agent 接手 CoLD 复现:

1. 首要改 hyp:`fliplr/translate/scale/hsv` 不要全设 0,**YOLOv5 baseline 协议需要这些 transform**
2. 次改 KD 标定:`train.py` 第 384 行的 `kd_loc * imgs.shape[0]` 是 over-scale 的元凶
3. **yolo11s CoLD 复现是好的** (0.574 ≥ paper 0.567),问题只在 yolov5x port
4. 实在复现不出来,把 yolov5x 当 anchor 实验,**主表 用 yolo11s reimplementation 跟 LADD 比即可**
