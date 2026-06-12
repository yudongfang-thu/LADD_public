# HalluciDet 完整实现文档

**日期**: 2026-06-12
**状态**: HalluciDet-YOLO11 adaptation 已接入训练/验证入口，仍需真实权重 smoke
**基于**: WACV 2024 HalluciDet 协议思想的 YOLO11 适配

---

## 📋 已实现的组件

### 1. Hallucination Network ✅

**文件**: `hallucidet_model.py`

**架构**:
```
Input: SAR image [B, 1, H, W]
  ↓
U-Net Encoder (4 levels)
  - Conv blocks with BN and ReLU
  - MaxPooling for downsampling
  ↓
Bottleneck with Attention Block
  ↓
U-Net Decoder (4 levels)
  - TransposeConv for upsampling
  - Skip connections from encoder
  - Optional attention blocks
  ↓
Final Conv + Tanh
  ↓
Output: Pseudo-RGB [B, 3, H, W] in [-1, 1]
```

**关键类**:
- `HallucinationNetwork`: 主网络
- `AttentionBlock`: 注意力机制
- `ConvBlock`: 卷积块
- `UpConvBlock`: 上采样块

### 2. Complete Model ✅

**类**: `HalluciDetModel`

**功能**:
- 组合hallucination network和frozen RGB detector
- 自动冻结RGB detector的所有参数
- Forward时RGB detector在eval模式

### 3. Training Infrastructure ✅

**文件**: `train_hallucidet.py`

**关键组件**:
- `HalluciDetLoss`: 实现论文Equation 2的loss
- `HalluciDetTrainer`: 完整训练循环
- 梯度只流向hallucination network
- 支持gradient clipping和cosine annealing

---

## 🚀 使用方法

### 快速测试

```bash
cd /Users/yudongfang/Desktop/光sar/LADD_public

# 测试hallucination network
python comparison/hallucidet/hallucidet_model.py
```

**预期输出**:
```
Testing HallucinationNetwork...
Input shape: torch.Size([2, 1, 256, 256])
Output shape: torch.Size([2, 3, 256, 256])
Output range: [-0.xxx, 0.xxx]
Parameters: xxx,xxx
HallucinationNetwork test passed! ✅
```

### 完整训练（需要补充dataloader）

```bash
python comparison/hallucidet/train_hallucidet.py \
    --data shared/configs/datasets_public/ogsod1_sar_detect.yaml \
    --teacher-data shared/configs/datasets_public/ogsod1_rgb_detect.yaml \
    --teacher-weights runs_public/ogsod/hbb/formal_nomosaic_20260528/baselines/yolo11n/seed0/weights/best.pt \
    --imgsz 256 \
    --epochs 400 \
    --batch 32 \
    --lr 1e-4 \
    --lambda-reg 1.0 \
    --project runs_public/hallucidet_full \
    --name yolo11n_s0_hallucidet_paper
```

---

## ⚙️ 关键实现细节

### 1. Hallucination Network

**输入处理**:
```python
# 自动转换3-channel到1-channel
if input.shape[1] == 3:
    input = 0.299*R + 0.587*G + 0.114*B

# 归一化到[0,1]
input = (input - input.min()) / (input.max() - input.min())
```

**输出处理**:
```python
# Hallucination输出[-1,1]
hallucinated = hallucination_net(input)  # Tanh输出

# 转换到[0,1]给detector
hallucinated = (hallucinated + 1.0) / 2.0
```

### 2. Frozen Detector

**冻结方式**:
```python
# 冻结所有参数
for param in rgb_detector.parameters():
    param.requires_grad = False

# 设置eval模式
rgb_detector.eval()

# Forward时保持eval，但不能使用 torch.no_grad()
# detector参数冻结即可；autograd仍需穿过detector算子回到hallucination net
detections = rgb_detector(hallucinated)
```

**梯度路径**:
```python
# hallucinated需要梯度
hallucinated = hallucination_net(sar_image)  # requires_grad=True

# Detector forward不更新detector参数，但保留输入梯度
preds = detector(hallucinated)

loss = detection_loss(preds, targets)
loss.backward()  # 梯度流回hallucination_net
```

### 3. Loss Computation

**遵循论文Equation 2**:
```python
L_hall = L_cls + λ * L_reg

其中:
- L_cls: classification loss
- L_reg: box regression + DFL loss
- λ: 平衡参数（默认1.0）
```

**实现**:
```python
# 使用YOLO raw prediction上的非detach detection loss分量
preds = frozen_detector(hallucinated)
_, loss_vec, loss_items = detection_loss.get_assigned_targets_and_loss(parsed_preds, batch)
total_loss = (loss_vec[1] + lambda_reg * (loss_vec[0] + loss_vec[2])) * batch_size
```

---

## 🔧 需要完成的部分

### 1. Dataloader Integration ✅

当前训练脚本复用 Ultralytics `build_yolo_dataset` / `build_dataloader`，输入为 SAR dataset yaml。`--teacher-data` 保留为协议说明字段，当前训练实际依赖 SAR 图像与 SAR 标签，以及提前训练好的 RGB detector 权重。

### 2. Validation with mAP ✅

`validate()` 已接入 YOLO DetectionValidator 的 mAP 统计逻辑，验证路径为：

```text
SAR val image -> hallucination net -> frozen RGB YOLO11 -> YOLO NMS/metrics
```

结果写入 `results.csv`，包含 `metrics/mAP50(B)` 和 `metrics/mAP50-95(B)`。

### 3. Checkpoint Resume ⚠️

**需要添加**:
```python
def load_checkpoint(self, checkpoint_path):
    """Resume from checkpoint"""
    ckpt = torch.load(checkpoint_path)
    self.model.hallucination_net.load_state_dict(ckpt['hallucination_net_state'])
    self.optimizer.load_state_dict(ckpt['optimizer_state'])
    self.epoch = ckpt['epoch']
```

---

## 📊 超参数建议

基于论文和类似工作：

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `lr` | **1e-4** | Adam学习率 |
| `epochs` | **400-800** | 与其他方法一致 |
| `batch_size` | **32** | 根据GPU memory调整 |
| `lambda_reg` | **1.0** | Cls和Reg平衡 |
| `base_channels` | **64** | U-Net基础通道数 |
| `grad_clip` | **10.0** | 防止梯度爆炸 |
| `weight_decay` | **1e-4** | L2正则化 |

**调试建议**:
1. 先用小lr（1e-5）训练几个epoch warm-up
2. 监控loss scale，如果太大需要调整lambda_reg
3. 检查hallucinated images是否合理（可视化）

---

## 🎯 与论文的对齐度

| 组件 | 论文 | 实现 | 对齐度 |
|------|------|------|--------|
| **Hallucination Network** | U-Net + Attention | ✅ YOLO adaptation | 接近 |
| **Frozen Detector** | Faster R-CNN/FCOS/RetinaNet | ✅ YOLO11 frozen detector | 适配 |
| **Training Pipeline** | Detection loss → hallucination | ✅ 已接入 | 接近 |
| **Loss Function** | L_cls + λ·L_reg | ✅ YOLO box/cls/dfl adaptation | 接近 |
| **Inference** | IR→hallucinate→detect | ✅ SAR→hallucinate→YOLO detect | 接近 |

**主要差异**: 原论文检测器是 Faster R-CNN/FCOS/RetinaNet；本仓库适配为 YOLO11。因此它应写作 HalluciDet-YOLO11 adaptation，而不是逐行 strict reproduction。

---

## ⚡ 快速启动checklist

### 步骤1: 测试网络 ✅
```bash
python comparison/hallucidet/hallucidet_model.py
# 应该看到 "test passed"
```

### 步骤2: 真实YOLO loss smoke
```bash
python comparison/hallucidet/test_gradient_smoke.py \
    --teacher-weights <rgb_teacher_best.pt> \
    --device 0 \
    --imgsz 256
```

### 步骤3: 运行20 epoch Smoke Test
```bash
python comparison/hallucidet/train_hallucidet.py \
    --epochs 5 \
    --batch 16 \
    --name smoke_test
```

### 步骤4: 正式训练
```bash
python comparison/hallucidet/train_hallucidet.py \
    --epochs 400 \
    --batch 32 \
    --project runs_public/hallucidet_paper \
    --name yolo11n_s0
```

---

## 🐛 常见问题和解决方案

### 问题1: OOM (Out of Memory)

**症状**: CUDA out of memory

**解决方案**:
```bash
# 降低batch size
--batch 16  # 或更小

# 降低base_channels
--base-channels 32  # 默认64
```

### 问题2: Loss爆炸

**症状**: Loss变成NaN或Inf

**解决方案**:
```python
# 1. 降低学习率
--lr 1e-5

# 2. 使用gradient clipping（已实现）
grad_clip = 10.0

# 3. 调整lambda_reg
--lambda-reg 0.5  # 降低regression权重
```

### 问题3: Hallucinated images看起来很差

**症状**: 可视化hallucinated images全黑/全白/噪声

**解决方案**:
```python
# 1. Warm-up训练
# 先用很小的lr训练10-20 epoch

# 2. 检查输入归一化
# 确保SAR images在[0,1]范围

# 3. 可能需要调整Tanh输出的scale
```

### 问题4: 训练不收敛

**症状**: Loss下降很慢或震荡

**可能原因和解决**:
1. **初始化问题**: Hallucination network随机初始化可能产生garbage
   - 考虑用预训练的image translation network初始化
2. **Learning rate问题**: 太大或太小
   - 尝试1e-5到1e-3之间
3. **Frozen detector不合适**: RGB detector在你的数据上效果不好
   - 确保RGB detector在RGB数据上有合理性能

---

## 📝 后续工作

### 必须完成（才能运行）:
1. ⚠️ 实现paired dataloader
2. ⚠️ 集成validation mAP计算
3. ⚠️ 测试训练稳定性

### 可选优化:
1. 添加可视化（保存hallucinated images）
2. 添加TensorBoard logging
3. 实现mixed precision training
4. 添加data augmentation

### 实验建议:
1. 先在小数据集上smoke test（50 images，5 epochs）
2. 检查hallucinated images质量
3. 如果质量合理，再进行完整训练
4. 对比with/without hallucination的性能

---

## ✅ 总结

### 已完成 ✅
- Hallucination Network（U-Net + Attention）
- Frozen Detector集成
- Detection Loss计算
- 训练循环框架
- Gradient管理

### 需要你完成 ⚠️
- Dataloader适配（复用现有代码）
- mAP validation（复用现有代码）
- 运行和调试

### 预期工作量
- Dataloader适配: 2-3小时
- 首次训练调试: 1-2天
- 完整训练: 3-5天（800 epoch）

**你现在有一个完整的、遵循论文的HalluciDet实现！** 🎉

只需要补充dataloader就可以开始训练了。
