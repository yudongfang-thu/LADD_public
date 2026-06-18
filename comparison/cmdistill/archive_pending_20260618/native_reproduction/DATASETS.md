# CMDistill Native 数据集清单

## 总览

| 数据集 | 原文用途 | 官方入口 | 当前下载判断 |
|---|---|---|---|
| VEDAI | 主实验、消融实验 | `https://downloads.greyc.fr/vedai/` | 512 release 已下载到本机，详见 `LOCAL_DATA_STATUS.md` |
| DroneVehicle | 泛化/补充实验 | `https://github.com/VisDrone/DroneVehicle` | 官方只给百度网盘 Train/Validation/Test，可能需要百度云下载 |

## VEDAI

CMDistill 原文信息：

- 1246 aligned RGB-IR image pairs。
- 11 类车辆目标，例如 car、pickup、camper、truck。
- 论文使用约 8:2 train/test split。
- 每个像素约对应 12.5 cm x 12.5 cm 物理空间。
- CMDistill 将图像 resize 到 640 x 640 后训练。

VEDAI 官方页提供 512 和 1024 两套 release。CMDistill 原文没有明确说明使用 512 还是 1024，因此第一轮建议先下载 512 release 做 smoke 和 pipeline 对齐；如果 baseline/CMDistill 量级偏差明显，再补 1024 release。

已确认的官方文件大小：

| 文件 | 大小 |
|---|---:|
| `Annotations512.tar` | 1,753,088 bytes |
| `Vehicules512.tar.001` | 699,400,192 bytes |
| `Vehicules512.tar.002` | 593,733,632 bytes |
| `Annotations1024.tar` | 1,768,960 bytes |
| `Vehicules1024.tar.001` | 699,400,192 bytes |
| `Vehicules1024.tar.002` | 699,400,192 bytes |
| `Vehicules1024.tar.003` | 699,400,192 bytes |
| `Vehicules1024.tar.004` | 699,400,192 bytes |
| `Vehicules1024.tar.005` | 93,268,992 bytes |
| `DevKit.tar` | 543,232 bytes |
| `TermsandConditionsofUseVeDAI2014.pdf` | 53,320 bytes |

下载命令：

```bash
bash comparison/cmdistill/native_reproduction/scripts/download_vedai.sh 512
```

本机当前已完成 512 release 下载。tar sanity check 显示严格 `*_co.png` / `*_ir.png` 各 1246 张，和 CMDistill 原文的 1246 aligned RGB-IR pairs 对齐；但官方包包含额外 `*.png2.png` / `copie` 重复项，转换时必须过滤。

可选环境变量：

```bash
CMDISTILL_NATIVE_DATA_ROOT=/path/to/cmdistill_native_data \
  bash comparison/cmdistill/native_reproduction/scripts/download_vedai.sh 512
```

## DroneVehicle

官方仓库信息：

- 28,439 RGB-Infrared image pairs，共 56,878 images。
- 5 类：car、truck、bus、van、freight car。
- 标注为 oriented bounding boxes。
- 下载图像尺度为 840 x 712，四周有 100 px white border；官方建议训练前裁掉边界并转为 640 x 512。
- 官方 README 仍写着 Google Drive link will be released soon；实际可用入口是 BaiduYun。

官方百度网盘入口：

| Split | URL | 提取码 |
|---|---|---|
| Train | `https://pan.baidu.com/s/1ptZCJ1mKYqFnMnsgqEyoGg` | `ngar` |
| Validation | `https://pan.baidu.com/s/1e6e9mESZecpME4IEdU8t3Q` | `jnj6` |
| Test | `https://pan.baidu.com/s/1JlXO4jEUQgkR1Vco1hfKhg` | `tqwc` |

建议落盘结构：

```text
comparison/cmdistill/native_reproduction/data/raw/DroneVehicle/
  train/
  val/
  test/
```

如果放在外部盘，设置：

```bash
export CMDISTILL_NATIVE_DATA_ROOT=/path/to/cmdistill_native_data
```

然后将数据放到：

```text
$CMDISTILL_NATIVE_DATA_ROOT/raw/DroneVehicle/
  train/
  val/
  test/
```

下载后可运行：

```bash
bash comparison/cmdistill/native_reproduction/scripts/check_dronevehicle_manual_download.sh
```

## 当前未公开/需复核点

| 问题 | 影响 | 当前处理 |
|---|---|---|
| VEDAI 使用 512 还是 1024 release | 影响小目标尺度和 mAP 可比性 | 先跑 512，必要时补 1024 |
| VEDAI 8:2 split 是否固定 | 影响复现实验方差 | 优先查 DevKit；若没有固定 split，记录随机种子和文件列表 |
| 原文是否把 OBB 转 HBB | 影响 YOLOv5 detect 训练格式 | CMDistill 公式使用 rectangular boxes，第一版按 HBB 转换并记录假设 |
| CMDistill 三个 loss 权重 lambda | 影响复现精度 | 论文只定义 lambda，未在实验段公开数值；先从 1/1/1 开始，再调参 |
| DroneVehicle 官方 test 标注是否公开 | 影响可复现评价 | 下载后检查；若 test label 缺失，先用 validation 作为公开可复现验证集 |

## VEDAI 转换策略

第一版转换使用论文 Table I 的 8 个类别列：

| YOLO index | VEDAI class id | name |
|---:|---:|---|
| 0 | 1 | car |
| 1 | 11 | pickup |
| 2 | 5 | camper |
| 3 | 2 | truck |
| 4 | 10 | other |
| 5 | 4 | tractor |
| 6 | 23 | boat |
| 7 | 9 | van |

官方聚合标注还包含少量 class id `7`, `8`, `31`，但 CMDistill Table I 没有对应列，第一版复现先过滤并在转换 manifest 中记录。

OBB/HBB 处理：使用四点标注的 min/max 外接矩形生成 YOLO HBB label，并裁剪到图像边界。该选择与 CMDistill 公式中的 rectangular boxes 更一致，但仍需在复现报告中标注为转换假设。
