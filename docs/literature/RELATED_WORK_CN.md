# 相关工作与推荐阅读(Reading List)

**用途**:论文定位 + 应该读哪些相关研究。不是实验文档,不是主线手册(见 [`TSKD_METHOD_MAINLINE_CN.md`](TSKD_METHOD_MAINLINE_CN.md))。

**适用问题**:训练期可访问配对 `RGB-SAR`、部署期仅保留 SAR 检测器的 incomplete-modality 跨模态蒸馏。

---

## 1. 任务定位与论文对比分类

`M4-SAR` 相关文献可分为三组:

| 组 | 代表方法 | 训练模态 | 测试模态 | 是否作为直接 baseline? | 为什么重要 |
|---|---|---|---|---|---|
| 光学-SAR 融合检测 | [M4-SAR / E2E-OSDet](https://arxiv.org/abs/2505.10931),官方 [repo](https://github.com/wchao0601/m4-sar) 里的 `CFT / CLANet / CSSA / CMADet / ICAFusion / MMIDet` | 光学 + SAR | 光学 + SAR | **否** | 任务上界参考,但部署条件不匹配(它们推理要两种模态,我们只用 SAR) |
| 光学引导 SAR 蒸馏 / incomplete-modality | [CoLD](https://openreview.net/forum?id=rlD7aV7UFD),[Cross-modal Gaussian LD](https://dblp.org/rec/conf/icassp/WangLFY25),[CCLKD](https://www.tandfonline.com/doi/abs/10.1080/10095020.2026.2633014),[CMDistill](https://dblp.org/rec/journals/staeors/TongGSGSZ25) | 多模态 | 单一模态 | **主要对比**,光学→SAR 的优先 | 部署故事最接近我们 |
| Shared/private 解耦式 CMKD | [DisCoM-KD](https://arxiv.org/abs/2408.07080) | 多模态 | 单一模态 | **机制邻近**,不进主表 | 概念相关,但不在遥感 SAR 检测上 |

---

## 2. 必读文献(Tier 1 — 任务定位与直接对比)

| # | 文献 | 场景 | 为什么必读 |
|---|---|---|---|
| [1] | [**M4-SAR** (arXiv 2025→2026)](https://arxiv.org/abs/2505.10931) | 光学-SAR benchmark | 公开基准,**必引 + 必跑**。`112,174` 配对,`~1M` 实例。代码 [wchao0601/m4-sar](https://github.com/wchao0601/m4-sar) |
| [2] | [**CoLD** (TGRS 2023)](https://openreview.net/forum?id=rlD7aV7UFD) | Category-oriented localization distillation | 最强 optical-guided SAR KD,**必进主对比表**。我们已复现(`0.43980 clean / 0.53442 online` on Sixiang) |
| [3] | [**Cross-modal Gaussian Localization Distillation** (ICASSP 2025)](https://dblp.org/rec/conf/icassp/WangLFY25) | Localization 优先的光学→SAR KD | 比 CoLD 更新,若能复现可进主表 |
| [4] | [**DisCoM-KD** (BMVC 2024)](https://arxiv.org/abs/2408.07080) | Disentanglement + adversarial CMKD | 我们 shared/private 建模最近的机制级先例 |

---

## 3. 方案 X/Y/B 对应文献(Tier 2)

主线手册 §8 提出的架构候选 X/Y/B 对应的先前工作。

### 3.1 方案 X(删 `r_s`、单一投影式 KD)谱系

| # | 文献 | 做法 | 对应关系 |
|---|---|---|---|
| [5] | [**FitNets** (ICLR 2015)](https://arxiv.org/abs/1412.6550) | regressor 对齐 hidden feature | 最早的 hint-based feature KD |
| [6] | [**SimKD** (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Knowledge_Distillation_With_the_Reused_Teacher_Classifier_CVPR_2022_paper.html) | projector + 单一 L2 + 共享 teacher classifier | **方案 X 的直接谱系,必读** |
| [7] | [**ReviewKD** (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Distilling_Knowledge_via_Knowledge_Review_CVPR_2021_paper.html) | 跨 stage review path | 架构参考 |

### 3.2 方案 B(Cross-attention 分解)谱系

| # | 文献 | 做法 | 对应关系 |
|---|---|---|---|
| [8] | [**UniKD** (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/html/Lao_UniKD_Universal_Knowledge_Distillation_for_Mimicking_Homogeneous_or_Heterogeneous_Object_ICCV_2023_paper.html) | query-based cross-attention,异构 detector KD | **方案 B 最近先例,必读** |
| [9] | [**CanKD** (WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/html/Sun_CanKD_Cross-Attention-based_Non-local_Operation_for_Feature-based_Knowledge_Distillation_WACV_2026_paper.html) | student attend to teacher | 方案 B 的另一条线 |
| [10] | [**C2KD** (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.html) | cross-modal customized teacher knowledge | 跨模态 motivation |

### 3.3 检测 KD baseline

| # | 文献 | 做法 | 为什么看 |
|---|---|---|---|
| [11] | [**FGD** (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.html) | 前景/背景分离 focal + 全局相关性 | **我们"前景 KD"最接近先验,必读** |
| [12] | [**CrossKD** (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_CrossKD_Cross-Head_Knowledge_Distillation_for_Object_Detection_CVPR_2024_paper.html) | student feature 送进 teacher head | 了解即可 |
| [13] | [**DeiT** (ICML 2021)](https://proceedings.mlr.press/v139/touvron21a.html) | distillation token | 了解即可 |

---

## 4. 概念源头与外围(Tier 3)

| # | 文献 | 为什么 |
|---|---|---|
| [14] | [**DSN: Domain Separation Networks** (NeurIPS 2016)](https://proceedings.neurips.cc/paper/2016/hash/45fbc6d3e05ebd93369ce542e8f2322d-Abstract.html) | **软正交源头**。我们主线冲突(主线手册 §7.1)就是照搬 DSN 失败 |
| [15] | [**Attention Is All You Need** (NeurIPS 2017)](https://arxiv.org/abs/1706.03762) | QKV 类比源头,方案 B 模板 |
| [16] | [**CCLKD** (Geocarto 2026)](https://www.tandfonline.com/doi/abs/10.1080/10095020.2026.2633014) | incomplete-modality detection (OGSOD / DroneVehicle / VEDAI) |
| [17] | [**CMDistill** (JSTARS 2025)](https://dblp.org/rec/journals/staeors/TongGSGSZ25) | RGB→IR train-time CMKD |

---

## 5. 阅读顺序建议

- **第一周**(定位 + 架构决策):M4-SAR → CoLD → SimKD → UniKD + C2KD → FGD
- **第二周**(narrative + 机制):DisCoM-KD → CanKD → DSN → Cross-modal Gaussian
- **如有余力**:FitNets / ReviewKD / CrossKD / DeiT

---

## 6. 每篇必答的 4 个问题

1. 它的 teacher / student 是什么?(同 / 跨 modality,detector / classifier?)
2. 它怎么表达"不同但互补"?(正交 / 对抗 / 残差 / cross-attention / 不分解?)
3. 和我们的差异在哪(novelty 对比)?
4. 审稿人会拿它问我们什么(防守预判)?

---

## 7. 论文对比表结构建议

### 7.1 主表(deployment-matched baseline)

必须与我们训练 / 测试条件一致:

- `SAR-only baseline`
- `Vanilla KD`
- `Logit KD`
- `CoLD`
- `Cross-modal Gaussian Localization Distillation`(若可复现)
- `Our method`

### 7.2 机制邻近(补充分析用)

放附录或补充分析,不进主表:

- `D2AD-like disentanglement baseline`
- `CCLKD`(若适配到 M4-SAR 可行)
- `CMDistill`(若适配成本可接受)
- `DisCoM-KD`(citation-only,除非可以直接做成检测适配)

### 7.3 上界 / 参考结果

**明确标注**不同部署设置:

- `RGB teacher-only`
- `M4-SAR` 官方 benchmark 里的 fusion 方法(`E2E-OSDet / MMIDet / ...`)

---

## 8. 引用顺序建议(空间紧时)

1. `M4-SAR`
2. `CoLD`
3. `Cross-modal Gaussian Localization Distillation`
4. `CCLKD`
5. `CMDistill`
6. `DisCoM-KD`

---

## 9. 论文定位一句话

> Unlike optical-SAR fusion detectors that require both modalities at inference, our method targets the incomplete-modality setting on M4-SAR: it uses optical data only during training, explicitly decomposes teacher knowledge into learnable and private components, and deploys a pure SAR detector at test time.

中文版:
> 与需要推理时同时访问光学和 SAR 的融合检测器不同,我们针对 M4-SAR 上 incomplete-modality 的场景:训练期用配对光学数据并显式把教师知识分解为可迁移与私有两部分,部署期只保留一个纯 SAR 检测器。

---

## 10. Related Work 必须说清楚的 3 件事

1. `Fusion` 和 `distillation under incomplete modalities` **不是**同一个问题
2. 以前的光学引导 SAR 方法主要在回答 "蒸馏什么 target"(定位 / 分类),我们在回答 "教师哪部分知识 SAR 学生实际可学"
3. 以前的 shared/private CMKD 论文 motivate 了 disentanglement,我们的贡献是把五件事**合起来**:
   - 教师侧 learnable / private 分解
   - 学习分支上的 task anchoring
   - 两个私有分支的弱结构保留
   - 早期阶段的可达性建模
   - 检测路径始终 raw
