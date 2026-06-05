# CCLKD 复现实验问题记录

最后更新：2026-06-05

本目录整理 CCLKD-style 在 OGSOD HBB 上的两类运行记录：

- 90 服务器：尽量贴近 CCLKD 原文设置的 YOLO11s / 400 epoch 复现实验。
- 双卡 4090 服务器：正式 controlled comparison 协议下的 YOLO11n / 800 epoch 运行中实验。

注意：本仓库中的实现是 `CCLKD-style` port。原论文没有公开可运行代码；当前实现保留 AT-KD + 类别约束对比蒸馏的主要思想，但把原文候选框级 CCL 近似到 YOLO task-aligned foreground anchor token 上。因此这里的问题记录不应写成“官方代码复现失败”，而应写成“CCLKD-style port 在当前协议下未观察到正收益，需要进一步排查”。

## 一、文件结构

```text
comparison/cclkd/reproduction_issue_20260605/
├── README_CN.md
├── metrics_summary.csv
├── curve_checkpoints.csv
├── 90_paperclosest_cclkd/
│   ├── args.yaml
│   └── results.csv
├── 90_sar_baseline_paperclosest/
│   ├── args.yaml
│   └── results_partial.csv
├── 90_reference_baselines/
│   ├── sar_yolo11s_hbb_400ep_s0_legacy_results.csv
│   └── rgb_yolo11s_hbb_400ep_s0_teacher_results.csv
├── dual4090_formal_cclkd/
│   ├── args.yaml
│   └── results_partial.csv
└── logs_excerpt/
    ├── 90_cclkd_paperclosest_head_tail.txt
    ├── 90_cclkd_paperclosest_key_lines.txt
    ├── 90_sar_baseline_paperclosest_current_tail.txt
    └── dual4090_cclkd_formal_current_logs.txt
```

原始 90 CCLKD `.outer.log` 约 51 MB，主要是 tqdm/progress-bar 重复刷新；这里提交清洗后的 head/tail/key-lines 摘要和完整 `results.csv`。如需完整日志，应从服务器原路径取回。

## 二、90 服务器：paper-closest CCLKD 结果

配置：

- 模型：YOLO11s
- student data：SAR HBB
- teacher：RGB YOLO11s 400ep best
- epochs：400
- batch：64
- imgsz：256
- optimizer：SGD
- lr0/lrf：0.01 / 0.01
- scheduler：cos_lr
- augmentation：mosaic=1.0, mixup=0.0, close_mosaic=10
- seed：0

结果：

| run | rows | last epoch | best mAP50-95 | best mAP50 |
| --- | ---: | ---: | ---: | ---: |
| CCLKD paper-closest | 400 | 400 | 0.48567 | 0.76397 |
| SAR YOLO11s legacy 400ep baseline | 400 | 400 | 0.53255 | 0.80835 |
| RGB YOLO11s 400ep teacher | 400 | 400 | 0.59771 | 0.91844 |

关键现象：

- CCLKD paper-closest 完整跑完 400 epoch，不是中途失败。
- `train/kd_loss` 非零，epoch400 为 `2.88319`，说明 KD 分支不是静默失效。
- 相比已有 SAR YOLO11s 400ep baseline，CCLKD 低 `0.04688` mAP50-95。
- 差距不是只在末期出现。checkpoint 曲线中，CCLKD 从 50/100/200/300/400 epoch 均低于旧 SAR baseline。

曲线摘录：

| epoch | CCLKD mAP50-95 | SAR baseline mAP50-95 | delta |
| ---: | ---: | ---: | ---: |
| 50 | 0.27269 | 0.28323 | -0.01054 |
| 100 | 0.33150 | 0.34914 | -0.01764 |
| 200 | 0.39501 | 0.41986 | -0.02485 |
| 300 | 0.45159 | 0.48348 | -0.03189 |
| 400 | 0.48567 | 0.53255 | -0.04688 |

## 三、90 服务器：正在补完全对齐 SAR-only baseline

为了排除“旧 baseline 协议并非完全一致”的问题，已启动完全对齐 CCLKD paper-closest 训练协议的 SAR-only baseline：

- 目录：`90_sar_baseline_paperclosest/`
- 当前状态：运行中，已提交 partial `results_partial.csv`
- 当前 partial：13 rows，epoch13 best mAP50-95 `0.13600`

这个实验完成后，最直接的判断标准是：

```text
CCLKD paper-closest 400ep vs SAR-only paper-closest 400ep
```

如果 SAR-only paper-closest 仍接近或超过旧 SAR baseline，则 CCLKD-style port 的负收益基本成立；如果 SAR-only paper-closest 本身显著低于旧 baseline，则需要先解释 baseline protocol 差异。

## 四、双卡 4090：formal comparison CCLKD 当前状态

配置：

- 模型：YOLO11n
- 协议：formal no-mosaic controlled comparison
- 初始化：from YOLO pretrain
- epochs：800
- seed：0
- 状态：运行中

当前 partial：

| run | rows | last epoch | best mAP50-95 | best mAP50 |
| --- | ---: | ---: | ---: | ---: |
| dual4090 CCLKD formal partial | 512 | 512 | 0.45225 | 0.64247 |

这组不是 CCLKD 原文对齐复现实验，而是我们论文主表 controlled comparison 的正式运行。它用于比较 CCLKD-style 与 LD / FGD / HalluciDet-style / LADD 等方法在统一协议下的表现。

## 五、当前排查判断

目前能确定的事实：

1. CCLKD paper-closest 在 90 上完整跑完，结果低于已有 SAR YOLO11s baseline。
2. KD loss 非零，不能简单归因于 loss 没接上。
3. 双卡 4090 formal CCLKD 也能正常训练，说明实现至少不是启动级别错误。
4. 正在补完全相同协议的 SAR-only baseline，这是判断 CCLKD paper-closest 是否真正负收益的关键实验。

可能原因需要继续排查：

- 原文 CCLKD 的候选框级类别约束对比学习与当前 anchor-token 近似不完全一致。
- 当前实现的 AT-KD / CCL 权重可能过强，使 student 被 RGB teacher 的跨模态偏差牵制。
- OGSOD 中 SAR-only YOLO11s baseline 已经较强，直接对齐 RGB teacher token 未必带来正向迁移。
- 原文训练细节没有代码公开，数据增强、teacher/student 初始化、候选框筛选细节可能存在未披露差异。

## 六、建议老师优先查看

1. `metrics_summary.csv`：各 run 最终/best 指标。
2. `curve_checkpoints.csv`：关键 epoch 曲线对比。
3. `90_paperclosest_cclkd/args.yaml`：90 CCLKD paper-closest 训练协议。
4. `90_paperclosest_cclkd/results.csv`：完整训练曲线。
5. `logs_excerpt/90_cclkd_paperclosest_key_lines.txt`：训练日志关键行。
6. `90_sar_baseline_paperclosest/results_partial.csv`：正在补的完全对齐 baseline。
