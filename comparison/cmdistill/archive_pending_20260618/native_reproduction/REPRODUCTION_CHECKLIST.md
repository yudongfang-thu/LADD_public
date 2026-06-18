# CMDistill Native 复现 Checklist

## 0. 数据下载

- [x] VEDAI 512 release 下载完成。
- [x] VEDAI 512 解压完成，并确认 RGB/IR 成对文件数量。
- [x] VEDAI 512 转换为 YOLO HBB RGB/IR 数据集。
- [x] VEDAI 官方 DevKit / 同数据集论文口径 audit 完成。
- [ ] DroneVehicle Train/Validation/Test 通过百度网盘下载完成。
- [ ] DroneVehicle 解压完成，并确认 RGB/IR pair、label、split 结构。

## 1. 数据协议固定

- [x] 记录 VEDAI 使用 512 或 1024 release。
- [x] 固定 VEDAI train/test split 文件列表，避免随机切分不可复现。
- [x] 明确 VEDAI 的 OBB -> HBB 转换规则。
- [x] 写出 YOLOv5 dataset YAML：RGB student、IR teacher 各一份。
- [x] 保存 VEDAI 类别顺序映射。

## 2. 原文 baseline

- [x] AutoDL 上准备 YOLOv5 v6.2 环境。
- [x] AutoDL 上启动 VEDAI IR/RGB YOLOv5s baseline：与当前 LADD GPU 任务并行，但两个模态按 `IR -> RGB` 串行、保持 `batch=64`。
- [x] 训练 IR teacher YOLOv5s。
- [x] 训练 RGB student YOLOv5s baseline。
- [x] 验证 VEDAI RGB YOLOv5s 是否接近论文 Table I 的 0.702 mAP。
- [ ] 验证 DroneVehicle 单模态 baseline 是否接近论文 Table II 的 70.7 mAP 量级。

## 3. CMDistill native

- [x] 实现原生 YOLOv5 v6.2 CMDistill 训练入口：冻结 teacher、paired teacher input、feature/relation/output KD。
- [x] 以冻结 IR teacher 训练 RGB student 的 VEDAI smoke 通过。
- [x] 使用 PCC-style feature loss + deepest relation loss + output logit/box loss 的 first-pass 近似。
- [x] 输入尺寸使用 640 x 640。
- [x] SGD：lr 0.01、momentum 0.937、weight decay 5e-4、batch 64、cosine decay。
- [ ] 增强至少覆盖 random rotation、random crop、color dithering。
- [x] 第一轮 loss 权重使用 `feature=1, relation=1, logit=1`。
- [ ] VEDAI `RGB student <- IR teacher` 300 epoch formal run 完成并汇总。

## 4. 验收标准

- [ ] VEDAI CMDistill mAP 接近 0.740，至少应复现出相对 RGB YOLOv5s baseline 的正增益。
- [ ] DroneVehicle CMDistill mAP 接近 74.3，至少应高于单模态 YOLOv5s baseline。
- [ ] 所有结果记录完整命令、git commit、dataset YAML、split manifest、args、metrics CSV。
- [ ] 大文件、权重和数据集不提交，只提交脚本、manifest、结果摘要。

当前状态：

- IR baseline 已完成，但它不是 CMDistill Table I 的 RGB baseline 目标。
- RGB baseline 已按 CMDistill Table I track 完成，best.pt validation `mAP@0.5=0.695`，与论文 RGB YOLOv5s `0.702` 差约 `0.007`。
- 本次 RGB run 被 YOLOv5 默认 early stopping 截到 231 epochs；baseline 启动脚本已修正为默认 `PATIENCE=EPOCHS`，后续正式 native run 不再默认提前停止。
- CMDistill native 正确方向为 `IR teacher -> RGB student`。`batch=64` smoke 已通过，`mAP@0.5=0.703`；300 epoch formal run `cmdi_rgb_ir_e300_20260618_133714` 已启动。
- 当前 formal run 为 aligned no-geo conservative variant，用于避免 RGB/IR teacher-student 特征错位；若未达到 `0.740`，下一步补同步 paired augmentation。
- `VEDAI_PROTOCOL_AUDIT.md` 已确认：官方 VEDAI DevKit、CMDistill-like 8:2 HBB、同数据集 YOLO/HBB 论文口径不能混为一谈。

## 5. 论文可用结论模板

当上述验收通过时，可谨慎表述为：

> To validate that our CMDistill-style implementation is not merely tuned for OGSOD, we additionally reproduced it under its native RGB-IR AAV setting on VEDAI/DroneVehicle. The reproduction follows the original IR-teacher-to-RGB-student protocol and reaches the same performance range as the reported native benchmark.

如果只能复现出正增益但未达到原文绝对值，则只能写作 partial reproduction / sanity validation，不能写 full reproduction。
