# LADD 主表对比方法调研草案

更新时间：2026-05-28  
目标：从 2018-2026 年知识迁移 / 蒸馏方法中筛选 LADD 主表对比实验方法，重点权衡 **SOTA 性**、**可复现性**、**OGSOD 相关性** 和 **部署设定一致性**。

## 0. 当前结论

主表应该分清两件事：

1. **Controlled main table**：所有方法在同一个 OGSOD-1.0 HBB、同一个 YOLO11n/YOLO11s、同一训练协议下重跑。这里应该只放可复现或可合理重实现的方法。
2. **External reported table**：列 CoLD / CCLKD / FED-CHDistill / GaLD / fusion 方法的论文报告数值，但明确 backbone、输入尺寸、epoch、是否 SAR-only inference、是否开源。这个表用于证明我们知道 SOTA 地图，不能替代公平主表。

初步推荐的 4 个主表方法：

| 位置 | 方法 | 选择理由 | 当前风险 |
|---|---|---|---|
| 普通检测 KD #1 | LD | 定位蒸馏，和检测任务强相关；代码开源；本地已有 YOLO11 HBB 适配记录 | 年份较早，但仍是检测 KD 常用强基线 |
| 普通检测 KD #2 | CrossKD | CVPR 2024，近年 detection KD 强方法；官方代码完整；SOTA 性强 | 基于 MMDetection/GFL，迁移到 YOLO11 有工程成本 |
| 跨模态 KD #1 | CoLD | TGRS 2023，OGSOD-1.0 原始 benchmark 与 optical-guided SAR KD anchor | 官方仓库偏数据入口，完整训练代码复现性弱；需使用本项目重实现 |
| 跨模态 KD #2 | CCLKD | 2026，直接 incomplete-modality object detection，OGSOD-1.0 + YOLOv11 扩展 | 未见稳定 runnable repo；若不能复现，应进 external reported table，而不是 controlled main table |

保守替代方案：

| 如果... | 替代 |
|---|---|
| CrossKD 迁移成本超预算 | 用 FGD 或 MGD 替代 CrossKD |
| CCLKD 无法按论文重实现 | controlled main table 放 CoLD + CMDistill-style / CCLKD-lite 重实现；external table 单列 CCLKD reported |
| 只能跑最小对比 | LD + CoLD 是最低限度；一个普通检测 KD，一个 OGSOD 直接跨模态 KD |

## 1. 主表资格线

一个方法进入 controlled main table，需要同时满足：

- **SAR-only inference**：推理时只能用 SAR，不需要 RGB / optical / thermal。
- **训练期可用 paired RGB-SAR**：可以使用 optical teacher 或 RGB branch，但部署时必须剥离。
- **可统一协议复现**：能在 YOLO11n/s HBB 上用我们的 OGSOD 数据和训练协议重跑。
- **不改变部署模型容量**：若方法引入 inference-time extra module，需要单独标注；最好不进主表。
- **论文与代码可解释**：至少能从论文公式和公开实现中重建 loss。

不满足这些条件的方法可以进 related work 或 external reported table，但不应和 LADD 在主表里直接横向比。

## 2. 评分字段

建议每个候选方法都记录以下字段：

| 字段 | 用途 |
|---|---|
| Year / venue | 新近性和发表质量 |
| 类别 | 普通 KD / 跨模态 KD / fusion |
| 训练模态 | SAR only / optical+SAR / RGB-T / RGB-D |
| 推理模态 | SAR only 还是多模态 |
| Benchmark | COCO / OGSOD-1.0 / OGSOD-2.0 / DroneVehicle / VEDAI / RGB-T |
| Box 类型 | HBB / OBB / rotated |
| Backbone / detector | YOLOv5、YOLO11、RetinaNet、GFL、FCOS、Faster R-CNN 等 |
| Params / FLOPs | 防止大模型优势伪装成方法优势 |
| Input size / epoch / batch | OGSOD 上尤其敏感 |
| Reported baseline | raw AP 不够，必须看相对同表 baseline 的 lift |
| Reported delta | `method - student baseline` |
| Gap closure | `(method - SAR) / (RGB teacher - SAR)` |
| Code status | full runnable / partial / data-only / no code |
| Porting cost | 低 / 中 / 高 |
| 主表建议 | controlled main / external reported / related only |

## 3. 普通检测蒸馏候选

这些方法的 benchmark 主要是 COCO / VOC / MMDetection 系，不是跨模态。它们的作用是回答：**普通强检测 KD 是否足以解决 optical -> SAR？**

| 方法 | 年份 / venue | 核心机制 | 常见 benchmark | 开源情况 | 主表适配性 | 备注 |
|---|---|---|---|---|---|---|
| Hinton KD | 2015 / arXiv | soft target / logit KD | 分类为主，可迁移到检测 head | 容易自实现 | C | 太基础，适合作为 sanity baseline，不占 4 个核心名额 |
| FitNets / AT | 2015 / 2017 | hidden feature / attention transfer | 分类为主 | 容易自实现 | C | 年份偏早，可在 appendix 或不跑 |
| GID / GI imitation | 2021 / CVPR | general instance selection，feature + relation + response | COCO / VOC，RetinaNet 等 | 论文公开；本地已有实现记录 | B- | 本地 OGSOD 旧协议不稳定，后期曾崩 |
| FGD | 2022 / CVPR | foreground/background focal + global relation distillation | COCO，RetinaNet/Faster R-CNN/RepPoints/FCOS/Mask R-CNN/GFL/YOLOX | 官方 GitHub，MMDet 2.11 | A- | 强、经典、可复现；若 CrossKD 工程成本高，用它 |
| MGD | 2022 / ECCV | mask student feature，生成 teacher full feature | 分类/检测/分割/实例分割 | 官方 GitHub，MMDet | A- | 泛化好，工程比 CrossKD 简单；不是最新 |
| LD | 2022 / CVPR; 2023 / TPAMI | bbox distribution / localization dark knowledge | dense HBB + rotated detection | 官方 GitHub；已入 MMDetection；有 rotated LD | A | 非常适合 OGSOD HBB/OBB；本地已有 YOLO11 HBB 适配记录 |
| DKD | 2022 / CVPR | TCKD + NCKD decoupled logit KD | 分类为主，有 detection 目录 | 官方 GitHub mdistiller | B | 不是检测专用；本地 OGSOD 旧协议不强 |
| PKD | 2022 / NeurIPS | Pearson correlation feature distillation，支持异构 detector | COCO，RetinaNet/FCOS 等 | 通过 OpenMMLab/MMRazor | B+ | SOTA 性不错，和 CMDistill 的 PCC 思路相关；可作 CrossKD 备选 |
| CrossKD | 2024 / CVPR | student head features 送入 teacher head，cross-head prediction mimic | COCO，GFL/RetinaNet/FCOS/ATSS，异构 teacher | 官方 GitHub，MMDet 3.0rc6 | A- / 工程 B | SOTA 性强；迁移 YOLO11 需要谨慎处理 head |
| DetKDS | 2024 / ICML | detection distillation policy search | COCO，多 detector | 官方 GitHub，MMDet 2.19 | B- | 很新，但像搜索框架，跑 OGSOD 成本高；不适合主表 4 选 1 |

普通 KD 侧优先级：

1. **LD**：最稳，检测定位相关，OGSOD 上本地已有适配结果。
2. **CrossKD**：SOTA 性最好，但 porting 成本高。
3. **FGD / MGD**：强可复现备选。
4. **DKD / Hinton KD**：基础 sanity，不建议占 4 方法名额。

## 4. 跨模态 / incomplete-modality 候选

这些是最接近 LADD 的对手。主表优先级高于普通 KD，但复现风险也更大。

| 方法 | 年份 / venue | 数据 / 任务 | 推理模态 | 核心机制 | 开源情况 | 主表适配性 | 备注 |
|---|---|---|---|---|---|---|---|
| CoLD | 2023 / TGRS | OGSOD-1.0 HBB，optical-guided SAR detection | SAR only | Category-oriented partition + IoU weighting localization distillation | 数据/benchmark GitHub；完整现代训练代码不清楚 | A / 必选 | OGSOD 原始 anchor；必须至少有本项目重实现 |
| CMDistill | 2025 / JSTARS | VEDAI / AAV RGB-IR detection | single modality | PCC feature + semantic relation + IoU binary classification logic distillation | 未见稳定官方 GitHub | B-/C | 不是 SAR/OGSOD 主场，但 CCLKD 表中把它当对比 |
| FED-CHDistill | 2025 / IJRS | OGSOD-1.0 HBB optical-guided SAR | SAR only | frequency enhancement + dynamic mask + cross-head distillation | 未检索到可靠 runnable repo | C | reported 很强，但 baseline/backbone 坐标疑点大，只进 external |
| GaLD | 2025 / ICASSP | OGSOD-2.0 OBB / rotated | SAR only | Gaussian localization / angle distillation | GitHub 公开，但主要是 OGSOD-2.0 数据下载和说明 | C | 偏 OBB/角度；不适合 OGSOD-1.0 HBB 主表 |
| CCLKD | 2026 / Geo-spatial Information Science | OGSOD-1.0、DroneVehicle、VEDAI；含 YOLOv11 扩展 | hard-to-detect modality only | adaptive-temperature KD + category-constrained contrastive learning | 有 supplement；未见稳定 runnable repo | B/C | 最接近近期竞品；若复现不了，只进 external reported table |
| C2KD | 2024 / CVPR | AV / image-text / RGB-depth 等跨模态 | single modality | customized cross-modal KD + on-the-fly sample selection | 论文公开；未见直接 OGSOD代码 | C | 概念相关，但不是 object detection SAR 主场 |
| DisCoM-KD | 2024 / BMVC/arXiv | 多模态到单模态分类 | single modality | disentanglement + adversarial CMKD | 未确认完整检测代码 | related only | 和 LADD 的 shared/private 叙事近，但不是 OGSOD 检测主表 |
| CRKD | 2024 / CVPR | camera-radar 3D/object detection | camera/radar domain | camera-radar cross-modal KD | 官方项目页 + code | related only | 传感器和任务不同，不建议主表 |

跨模态侧优先级：

1. **CoLD**：必须跑。
2. **CCLKD**：最值得追，但需要先判定能否复现。
3. **CMDistill-style**：若 CCLKD 无法复现，可只作为外部对比和 related work。
4. **FED-CHDistill / GaLD**：只放 external reported table，不进 controlled main。

## 5. OGSOD reported 坐标

注意：这些数值来自不同论文、不同模型和不同训练协议，不能直接混成主表。这里只用于 SOTA 地图。

| 方法 / 论文 | OGSOD 版本 | Box | Backbone / 容量 | 输入/训练 | Reported result | 代码状态 | 对主表含义 |
|---|---|---|---|---|---|---|---|
| CoLD, TGRS 2023 | OGSOD-1.0 | HBB | YOLOv5 / CSPDarkNet-X，约 86.2M | 256, batch64, 400ep, SGD, Mosaic+MixUp | YOLOv5 `AP50/AP=80.9/46.3`; CoLD `87.6/56.7` | 数据仓库可见；训练代码不清楚 | 原始 anchor，必须引用/重实现 |
| CCLKD, GIS 2026 | OGSOD-1.0 | HBB | YOLOv5 + YOLOv11 扩展 | 256, batch32, 400ep, SGD | YOLOv5: CoLD `86.5/55.4`, CMDistill `87.5/56.2`, CCLKD `88.7/57.3`; YOLO11s CCLKD `87.5/55.1` | 未见 runnable repo | 最接近 reported SOTA，但只适合 external 或重实现后进主表 |
| FED-CHDistill, IJRS 2025 | OGSOD-1.0 | HBB | 表中 YOLOv5 student 约 13.7M；坐标和 CoLD 原文不一致 | 256, batch64, 400ep, SGD | 摘要声称 AP50/AP 分别提升 9.9/23.8%；本地 PDF 记录 Ours `90.8/70.1` | 未见可靠 repo | reported 极强，但协议风险大；external only |
| GaLD, ICASSP 2025 | OGSOD-2.0 | OBB | CSPDarkNet-M | 256, batch128, 300ep | CSL baseline `73.8/40.0`; CoLD `80.9/46.0`; GaLD `85.2/49.5` | GitHub 主要发布数据，不是完整训练代码 | OBB / OGSOD-2.0 相关，不进 HBB 主表 |
| MAIENet, Remote Sensing 2025 | OGSOD-1.0 | HBB | multimodal detector | 256，具体协议需细读 | 表中列 KD `88.4/48.4`, LD `90.1/51.9`, CoLD `93.5/56.7`, MAIENet `90.8/61.0` | fusion 方法 | 需要 RGB+SAR 推理，非同部署；external / upper-bound only |

## 6. 当前本地复现线索

本地文档已有以下记录，后续应以正式 `800ep/cos/nomosaic/albu` 协议重跑：

| 方法 | 旧协议记录 | 含义 |
|---|---|---|
| KD / LD / DKD / GI / CoLD 11n | 见 `docs/TEACHER_MEETING_REPORT_20260526_CN.md` §8.1 | 旧 `YOLO11n/256/400ep` 下 preliminary；CoLD 11n 旧协议最强 |
| CoLD YOLO11s | 旧记录 best AP `0.57429` | 统一协议强对比候选 |
| CoLD YOLOv5x | 复现诊断显示 head / hyp / 坐标系不一致 | 不建议作为主表核心，只作原文复现诊断 |

## 6.1 跨模态方法仓库审计：CoLD 对比证据链

本节只关心一个问题：公开仓库里是否能看到训练代码、CoLD baseline 的实现 / 配置 / 日志，从而判断论文里的 CoLD 对比是公开可复现，还是作者内部复现 / 表格引用。

| 方法 | 论文/页面给出的公开材料 | 仓库或页面实际内容 | 是否包含可运行训练代码 | 是否包含 CoLD 对比实现/配置/日志 | 判断 |
|---|---|---|---|---|---|
| CoLD | `mmic-lcl/Datasets-and-benchmark-code` | GitHub 顶层仅见 `LICENSE`、`README.md`、`index.html`；语言统计 HTML 100%；README 指向资源网页 | 否 | 不适用；它本身是 CoLD，但未见完整训练流水线 | 原文数值可信度来自论文，不来自可运行仓库 |
| GaLD | `wchao0601/GaLD` | GitHub 顶层仅见 `img/`、`.gitignore`、`LICENSE`、`README.md`；README 是 OGSOD-2.0 下载、目录结构、citation | 否 | 未见 CoLD / PseKD / GaLD 训练代码或对比 config | 仓库是数据入口，不是方法代码 |
| CCLKD | Taylor 页面有 full text 和 44.5MB supplement zip | 页面列出 OGSOD / VEDAI / DroneVehicle 数据可用性；未在正文页面看到 GitHub 代码入口 | 未确认；目前未见稳定 public repo | 论文正文明确把 CoLD、DetKDS、GKD、SSR-Net、CMDistill 等作为对比，并说 cross-modal KD 方法使用 YOLOv5 backbone；但这更像论文内部复现表，不是仓库证据 | 值得细读 supplement；未审完前不能当公开可复现 |
| FED-CHDistill | Taylor 页面 | 页面摘要列 FEM / DMM / CHD 机制和 OGSOD-1.0 结果；未见代码入口 | 否 | 未见 CoLD 复现证据；本地 PDF 记录其表格更多对比普通 KD/MGD/LSKD/SDD，不是清晰 CoLD 公开复现 | external only |
| CMDistill | DOI / DOAJ / IEEE 页面 | 公开页面描述 PCCFD、SLRD、IBCLD；未见 GitHub 入口 | 否 | CCLKD 将其列为对比方法，但 CMDistill 本身不是 OGSOD 代码仓证据来源 | related / external |

对 CoLD 对比方式的当前判断：

1. **CoLD 原文**：公开仓库更像 dataset / benchmark landing page；没有足够代码支撑“官方可复现”。
2. **CCLKD**：是目前最有用的“CoLD 之后横向表”，因为它直接列 CoLD、CMDistill、GKD、DetKDS 等；但公开页面没有给出 runnable repo，CoLD 对比来源更可能是作者内部复现。
3. **GaLD**：和 CoLD 作者线接近，因此它在 OGSOD-2.0 OBB 上列 CoLD/PseKD 的可信度可能高于普通第三方复现；但仓库没有代码，不能作为可复现证据。
4. **FED-CHDistill / CMDistill**：目前没有看到能回答“他们如何与 CoLD 对比”的仓库证据。

写作建议：external reported table 可以有一列 `CoLD comparison source`：

| 来源类型 | 写法 |
|---|---|
| 原文报告 | `reported by original CoLD paper; public repository is dataset-only` |
| 作者内部复现 | `reported by follow-up paper; no runnable baseline code found` |
| 仓库可复现 | `official code/config/log available` |
| 不同设置 | `different dataset/task/backbone; not controlled` |

下一轮正式主表应至少重跑：

- SAR baseline / RGB teacher；
- LD；
- CrossKD 或 FGD；
- CoLD；
- CCLKD 若可复现，否则先跑一个 CCLKD-lite/ATKD+CCL ablation，再把原文 CCLKD 放 external。

## 7. 主表设计建议

### 7.1 Controlled main table

建议列：

| Method | Type | Train modality | Test modality | Code source | AP | AP50 | Delta | Gap closure |
|---|---|---|---|---|---:|---:|---:|---:|
| SAR baseline | baseline | SAR | SAR | ours | TBD | TBD | - | - |
| RGB teacher | upper bound | RGB | RGB | ours | TBD | TBD | - | 100% |
| LD | generic detector KD | RGB teacher + SAR student | SAR | official + ours port | TBD | TBD | TBD | TBD |
| CrossKD / FGD | generic detector KD | RGB teacher + SAR student | SAR | official + ours port | TBD | TBD | TBD | TBD |
| CoLD | optical-SAR KD | RGB teacher + SAR student | SAR | ours reimplementation | TBD | TBD | TBD | TBD |
| CCLKD / CCLKD-lite | cross-modal KD | RGB teacher + SAR student | SAR | reimplementation if possible | TBD | TBD | TBD | TBD |
| LADD | ours | RGB teacher + SAR student | SAR | ours | TBD | TBD | TBD | TBD |

### 7.2 External reported table

建议列：

| Paper | Method | Dataset | Box | Backbone | Input / epoch | Test modality | Reported AP/AP50 | Code status | Fairness note |
|---|---|---|---|---|---|---|---|---|---|
| CoLD 2023 | CoLD | OGSOD-1.0 | HBB | YOLOv5/CSPDarkNet-X | 256/400 | SAR | 56.7 / 87.6 | partial/data | original anchor |
| CCLKD 2026 | CCLKD | OGSOD-1.0 | HBB | YOLOv5/YOLO11 | 256/400 | SAR | 57.3 / 88.7 (YOLOv5); 55.1 / 87.5 (YOLO11s) | no stable repo found | close competitor but not controlled |
| FED-CHDistill 2025 | FED-CHDistill | OGSOD-1.0 | HBB | ambiguous YOLOv5 | 256/400 | SAR | reported very high | no stable repo found | baseline coordinate ambiguous |
| GaLD 2025 | GaLD | OGSOD-2.0 | OBB | CSPDarkNet-M | 256/300 | SAR | 49.5 / 85.2 | data repo only | different dataset/task |
| MAIENet 2025 | MAIENet | OGSOD-1.0 | HBB | multimodal | 256 | RGB+SAR | 61.0 / 90.8 | paper | fusion, not SAR-only |

## 8. 推荐行动顺序

1. **先定普通 KD 的第二名额**：CrossKD vs FGD。
   - 如果要 SOTA 性：CrossKD。
   - 如果要稳复现：FGD。
2. **CoLD 必跑**：作为 OGSOD 原始 optical-guided SAR KD anchor。
3. **CCLKD 先做可复现性审计**：
   - 下载 supplemental；
   - 查是否包含训练代码；
   - 若无代码，按论文实现 ATKD + CCL 的最小版；
   - 如果实现细节仍不充分，不进 controlled main。
4. **external reported table 单独成表**：把 CCLKD/FED/GaLD/fusion 的 reported 数字放进去，防止审稿人觉得我们没看最新工作。
5. **所有 controlled 方法统一用 delta 和 gap closure 排序**，不要只按 raw AP。

## 9. 主要来源

- CoLD / OGSOD-1.0：TGRS 2023, DOI `10.1109/TGRS.2023.3291356`; dataset link `https://github.com/mmic-lcl/Datasets-and-benchmark-code`
- CCLKD：Geo-spatial Information Science 2026, DOI `10.1080/10095020.2026.2633014`
- FED-CHDistill：International Journal of Remote Sensing 2025, DOI `10.1080/01431161.2025.2529005`
- GaLD：ICASSP 2025, GitHub `https://github.com/wchao0601/GaLD`
- LD：CVPR 2022 / TPAMI 2023, GitHub `https://github.com/HikariTJU/LD`
- FGD：CVPR 2022, GitHub `https://github.com/yzd-v/FGD`
- MGD：ECCV 2022, GitHub `https://github.com/yzd-v/MGD`
- DKD：CVPR 2022, GitHub `https://github.com/megvii-research/mdistiller`
- PKD：NeurIPS 2022, implementation through OpenMMLab/MMRazor
- CrossKD：CVPR 2024, GitHub `https://github.com/jbwang1997/CrossKD`
- DetKDS：ICML 2024, GitHub `https://github.com/lliai/DetKDS`
- 本地记录：`docs/TEACHER_PROGRESS_REPORT_20260518_CN.md`, `docs/TEACHER_MEETING_REPORT_20260526_CN.md`, `docs/RELATED_WORK_CN.md`
