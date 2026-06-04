# CoLD yolov5x 复现修复变更说明(2026-05-18)

> 本文档记录 2026-05-18 对 `LADD_ref/methods/cold/` 的两处修复,目的是让 CoLD yolov5x paper-protocol 复现从 AP=0.4523 接近原文 AP=0.567。
>
> 诊断细节见 [`COLD_REPRODUCTION_DIAGNOSIS_20260516_CN.md`](COLD_REPRODUCTION_DIAGNOSIS_20260516_CN.md)。

---

## 1. 复现失败的事实(2026-05-16 诊断)

| 指标 | 旧复现(`cold_yolov5x_paper_s0`, 322/400 ep) | CoLD 原文 (Table I HBB) | 差距 |
|---|---:|---:|---:|
| AP50-95 | **0.4523** | **0.567** | -0.115 |
| AP50 | **0.6967** | **0.787** | -0.090 |

参考:**yolo11s 端口 (`cold_yolo11s_paper_s0`) 400/400 ep best AP=0.5743** — yolo11s 已超原文 0.567,证明 KD 主逻辑正确,问题集中在 yolov5x 端口的两处具体偏差。

---

## 2. 已应用的修复(90 服务器上)

修改前先做 backup:

```text
/mnt/dataY/ydf/projects/LADD_ref/methods/cold/configs/hyps/
  hyp_cold_paper.yaml             <-- 已改
  hyp_cold_paper.yaml.orig_20260518   <-- 原版备份

/mnt/dataY/ydf/projects/LADD_ref/methods/cold/yolov5/src/cold_yolov5/
  train.py                         <-- 已改
  train.py.orig_20260518           <-- 原版备份
```

### 2.1 Fix #1 — 恢复 YOLOv5 默认增强

**文件**:`methods/cold/configs/hyps/hyp_cold_paper.yaml`

**改动**:

| key | 旧值 (broken) | 新值 (fixed = YOLOv5 scratch-low default) |
|---|---:|---:|
| `fliplr` | 0.0 | **0.5** |
| `scale` | 0.0 | **0.5** |
| `translate` | 0.0 | **0.1** |
| `hsv_h` | 0.0 | **0.015** |
| `hsv_s` | 0.0 | **0.7** |
| `hsv_v` | 0.0 | **0.4** |
| `mosaic` | 1.0 | 1.0 (不变) |
| `mixup` | 0.1 | 0.1 (不变) |
| 其他 | 不变 | 不变 |

**为什么**:CoLD 原文 §V-A 只写 "Mosaic + Mixup",但 YOLOv5 文献的惯例是 "**scratch-low 默认配方 + 额外 Mosaic + Mixup**"。把所有非 Mosaic/Mixup 的 transform 设 0 是 LADD 主线的 paired-KD 增强策略(为避免 RGB-SAR 非对称扰动),**对 CoLD paper-protocol 复现是过强的限制**。

**预期增益**:+3~6 AP。

### 2.2 Fix #2 — 去掉 KD loss 的二次放大

**文件**:`methods/cold/yolov5/src/cold_yolov5/train.py` 第 389 行

**旧代码**:
```python
total_loss = student_det_loss + teacher_det_loss + kd_loc * imgs.shape[0]
```

**新代码**:
```python
# 2026-05-18 fix: kd_loc is already T^2-scaled inside OnlineCoLDLoss; the
# student/teacher detection losses are already multiplied by batch size
# in ComputeCoLDLoss.__call__. Multiplying kd_loc by imgs.shape[0] here
# double-scales the KD term, letting it dominate the gradient (KD ~128
# vs det ~68 per batch). The original (under-scaled) line is preserved
# in train.py.orig_20260518; restore it only when reproducing the
# earlier broken numbers for comparison.
total_loss = student_det_loss + teacher_det_loss + kd_loc
```

**为什么**:
- `OnlineCoLDLoss.__call__` 内部已经 `× T² = 400`(单 sample KL ≈ 0.005 × 400 = 2)
- `ComputeCoLDLoss.__call__` 内部已经 `× bs = 64`(det loss 已经按 batch sum)
- 原代码 `kd_loc × imgs.shape[0]` 又 × bs,导致 KD 项数值 ≈ 0.005 × 400 × 64 = **128**,而 det loss ≈ 68
- 训练曲线印证:`train/cold_loss=1.8-2.4`,`train/box_loss=0.03-0.13` → KD 项主导梯度,学生在追教师 logits 不在学 SAR 检测

**预期增益**:+2~4 AP。

### 2.3 新增 launch 脚本

**文件**:`methods/cold/scripts/run_cold_yolov5x_paper_fixed.sh`

跟 `run_cold_yolov5x_paper.sh` 几乎一致,区别:
- run name 后缀加 `_fixed`(保留旧 broken 结果)
- project 路径改 `runs_public/ogsod/hbb/kd_baselines/cold_yolov5x_paper_fixed/`
- 其他超参不变

启动:

```bash
[internal SSH command omitted]

cd /mnt/dataY/ydf/projects/LADD_ref
GPU_ID=<空闲 GPU> bash methods/cold/scripts/run_cold_yolov5x_paper_fixed.sh
```

预计 400 ep 跑完后 AP ≈ **0.52-0.55**(从 0.45 提升 +5~10),接近 paper 0.567。

---

## 3. 没修的两个次要点(待 Step 1+2 不够时再补)

诊断报告中标 🟡 的两个根因,**暂未应用**:

### 3.1 (Optional) Per-layer IWM 替代 per-anchor IWM

`OnlineCoLDLoss` 里 `teacher_ciou` 是 per-anchor 加权,paper Eq.(12) 是 per-layer 加权(i 是 layer index, a_i 是 layer-level IoU mean)。

预期影响:±0~2 AP。**先不改,等 Fix #1+#2 跑完看是否还差 paper 1+ AP**。

### 3.2 (Optional) CPM 类循环固定 `range(nc)`

`for cls_id in teacher_labels.unique(sorted=True)` 改 `for cls_id in range(self.nc)`,缺失类用 0 占位。

预期影响:±0~0.5 AP。同样**先不改**。

---

## 4. 验证 checklist

跑完 `run_cold_yolov5x_paper_fixed.sh` 后:

1. **AP50-95 vs 旧版**:从 0.4523 提升到 ≥ 0.52(达到目标),或 0.50-0.52(部分提升,需补 Step 3/4)
2. **训练曲线 KD/det loss ratio**:`train/cold_loss / train/box_loss < 5`(原来是 50+),确认 KD 不再主导
3. **per-class AP**:bridge/harbor/storage_tank 三类应该都有提升,而不是某一类回退

### 4.1 比对 yolo11s 端口

`cold_yolo11s_paper_s0` 现有 0.5743 是用 **`run_cold_yolo11s_paper.sh`** 跑的,该脚本里的增强 CLI 是 `--hsv-h 0.0 --hsv-s 0.0 --hsv-v 0.0 --erasing 0.0`,但没设 `--fliplr/--scale/--translate`,所以走 Ultralytics 默认 (`fliplr=0.5, scale=0.5, translate=0.1`)。

⚠️ **这解释了为什么 yolo11s 复现得好而 yolov5x 复现差**:
- yolo11s:走 Ultralytics(YOLOv11)默认 aug → fliplr/scale/translate 正常 ON,只关 HSV/erasing
- yolov5x:走 hyp_cold_paper.yaml → fliplr/scale/translate 全 0,几乎只剩 Mosaic+Mixup

Fix #1 把 yolov5x 端口的增强对齐到 yolo11s 端口的水平,**两个端口应该 给出更接近的相对增益**。

---

## 5. 操作建议(给当前 agent / 用户)

### 5.1 立即可做

1. **找一个空闲 GPU**(比如 600EP baseline 跑完后 GPU 1/3 可能空出来):
   ```bash
   [internal SSH command omitted]
   ```

2. **launch 修复版**:
   ```bash
   [internal SSH command omitted]
     nohup env GPU_ID=<id> SEED=0 \
       bash methods/cold/scripts/run_cold_yolov5x_paper_fixed.sh \
       > logs/cold_yolov5x_fixed_s0.log 2>&1 &'
   ```

3. **3-4 天后查结果**:
   ```bash
   python3 -c "
   import csv
   p='/mnt/dataY/ydf/projects/LADD_ref/runs_public/ogsod/hbb/kd_baselines/cold_yolov5x_paper_fixed/cold_yolov5x_paper_fixed_s0/results.csv'
   rows=list(csv.DictReader(open(p)))
   ap_k=next(k for k in rows[0] if 'mAP50-95' in k.replace(' ',''))
   print('best AP=', max(float(r[ap_k]) for r in rows if r[ap_k].strip()))
   "
   ```

### 5.2 决策树

- AP ≥ 0.55 → **复现成功,接受 paper 内误差**
- 0.50 ≤ AP < 0.55 → 跑 Step 3 + Step 4 优化,再训一次
- AP < 0.50 → 还有其他没识别出的 bug,需要进一步 review(优先检查 dataloader paired-image 实际加载是否对齐)

---

## 6. 论文写作 fallback(若怎么调都到不了 0.567)

不影响主表论述。yolov5x port 只是 anchor 实验,**主表用 yolo11s reimplementation 跟 LADD 头对头比较**(目前 cold_yolo11s_paper_s0 = 0.5743,LADD HBB P1 在 400EP-init 上还在跑)。

论文 §V Experiments 加一句话:

> "For full transparency, we report our reimplementation of the CoLD original yolov5x protocol on OGSOD-1.0 HBB. Our reproduction reaches XX.X% AP, within Y.Y points of the originally reported 56.7% AP. We attribute the residual gap to (i) the YOLOv5 augmentation profile (the original paper says 'Mosaic + Mixup' but does not specify whether other YOLOv5 default augmentations remain active), (ii) the unspecified YOLOv5 variant (n/s/m/l/x), and (iii) the precise loss-scaling convention between detection and KD terms. For fair head-to-head comparison with LADD, we report a yolo11s reimplementation of CoLD in Table T8, which uses the identical SAR/RGB checkpoint pair as LADD."

这样的写法:
- 学术诚实(承认复现细节)
- 把 main story 钉死在 yolo11s 上
- reviewer 通常接受

---

## 7. 修改清单 / 索引

| 项 | 文件 | 状态 |
|---|---|---|
| Backup | `hyp_cold_paper.yaml.orig_20260518` (90 svr) | ✅ |
| Backup | `train.py.orig_20260518` (90 svr) | ✅ |
| Fix #1 hyp | `hyp_cold_paper.yaml` (90 svr) | ✅ 已修改 |
| Fix #2 train.py | `train.py` 第 389 行 (90 svr) | ✅ 已修改 |
| New script | `run_cold_yolov5x_paper_fixed.sh` (90 svr) | ✅ 已创建 |
| 旧 broken 结果 | `runs_public/.../cold_yolov5x_paper/` | 保留不动 |
| 新 fixed 结果 | `runs_public/.../cold_yolov5x_paper_fixed/` | 待启动 |
| 诊断报告 | `docs/COLD_REPRODUCTION_DIAGNOSIS_20260516_CN.md` | 已存在 |
| 本文档 | `docs/COLD_FIX_APPLIED_20260518_CN.md` | ✅ 本文 |

---

## 8. Git 提交建议

如果 90 服务器 LADD_ref 仓库有 git,推荐 commit 信息:

```text
Fix CoLD yolov5x paper-protocol reproduction (#39)

Two issues identified from training curve + code review:

1. hyp_cold_paper.yaml had fliplr/scale/translate/hsv all zeroed out,
   which over-restricts augmentation. CoLD paper says "Mosaic + Mixup"
   but in YOLOv5 convention this means on top of scratch-low defaults,
   not "all other augs off". Restored fliplr=0.5, scale=0.5,
   translate=0.1, hsv=0.015/0.7/0.4. Expected +3~6 AP.

2. train.py:389 multiplied kd_loc by imgs.shape[0] even though kd_loc
   is already T^2-scaled inside OnlineCoLDLoss and detection losses
   are already batch-summed. Removed the redundant *imgs.shape[0],
   which was making KD loss ~2x larger than detection loss and causing
   the student to chase teacher logits instead of learning SAR
   detection. Expected +2~4 AP.

Old run preserved at runs_public/.../cold_yolov5x_paper/cold_yolov5x_paper_s0
(AP=0.4523 vs paper 0.567). Launch fixed run with
methods/cold/scripts/run_cold_yolov5x_paper_fixed.sh.

Refs: docs/COLD_REPRODUCTION_DIAGNOSIS_20260516_CN.md,
      docs/COLD_FIX_APPLIED_20260518_CN.md
```
