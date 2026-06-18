#!/usr/bin/env python3
"""Native YOLOv5-v6.2 CMDistill-style training for paired VEDAI RGB/IR.

This script intentionally stays independent from the OGSOD/YOLO11 comparison
profile. It trains one YOLOv5 student modality with a frozen paired teacher
modality and adds three CMDistill-like terms: feature PCC, spatial relation,
and output logit/box distillation.
"""

import argparse
import math
import os
import random
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path


def _bootstrap_yolov5_path():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--yolov5-dir", default=os.environ.get("YOLOV5_DIR", "/root/autodl-tmp/yolov5-v6.2"))
    known, _ = parser.parse_known_args()
    yolov5_dir = Path(known.yolov5_dir).expanduser().resolve()
    if not (yolov5_dir / "train.py").is_file():
        raise FileNotFoundError(f"YOLOv5 v6.2 train.py not found under {yolov5_dir}")
    sys.path.insert(0, str(yolov5_dir))
    return yolov5_dir


YOLOV5_DIR = _bootstrap_yolov5_path()
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402
from torch import nn  # noqa: E402
from torch.optim import lr_scheduler  # noqa: E402
from tqdm import tqdm  # noqa: E402

import val  # noqa: E402
from models.experimental import attempt_download, attempt_load  # noqa: E402
from models.yolo import Model  # noqa: E402
from utils.callbacks import Callbacks  # noqa: E402
from utils.dataloaders import create_dataloader  # noqa: E402
from utils.augmentations import letterbox  # noqa: E402
from utils.general import (  # noqa: E402
    LOGGER,
    check_amp,
    check_dataset,
    check_img_size,
    check_suffix,
    colorstr,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    one_cycle,
    strip_optimizer,
    yaml_save,
)
from utils.loss import ComputeLoss  # noqa: E402
from utils.metrics import fitness  # noqa: E402
from utils.torch_utils import (  # noqa: E402
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_optimizer,
)

warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.*")


class DetectFeatureHook:
    """Captures the feature list entering YOLOv5 Detect without cloning tensors."""

    def __init__(self, model):
        self.features = None
        self.handle = de_parallel(model).model[-1].register_forward_pre_hook(self._hook)

    def _hook(self, _module, inputs):
        x = inputs[0]
        self.features = list(x) if isinstance(x, (list, tuple)) else [x]

    def clear(self):
        self.features = None

    def close(self):
        self.handle.remove()


class FeatureAdapters(nn.Module):
    """1x1 student feature adapters used before PCCFD."""

    def __init__(self, student_channels, teacher_channels, indices):
        super().__init__()
        self.adapters = nn.ModuleDict()
        for idx in indices:
            conv = nn.Conv2d(student_channels[idx], teacher_channels[idx], kernel_size=1, bias=False)
            if student_channels[idx] == teacher_channels[idx]:
                nn.init.zeros_(conv.weight)
                eye = torch.eye(student_channels[idx]).view(student_channels[idx], student_channels[idx], 1, 1)
                with torch.no_grad():
                    conv.weight.copy_(eye)
            self.adapters[str(idx)] = conv

    def forward_one(self, idx, x):
        key = str(idx)
        return self.adapters[key](x) if key in self.adapters else x


def load_hyp(path):
    with open(path, errors="ignore") as f:
        hyp = yaml.safe_load(f)
    return hyp


def make_aligned_hyp(hyp, keep_color_aug=False):
    hyp = hyp.copy()
    for k in ("degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup",
              "copy_paste"):
        hyp[k] = 0.0
    if not keep_color_aug:
        for k in ("hsv_h", "hsv_s", "hsv_v"):
            hyp[k] = 0.0
    return hyp


def resolve_teacher_path(student_path, student_token="/images/ir/", teacher_token="/images/rgb/"):
    text = str(student_path)
    if student_token in text:
        teacher_path = Path(text.replace(student_token, teacher_token, 1))
    else:
        parts = list(Path(student_path).parts)
        try:
            i = parts.index("images")
            student_name = student_token.strip("/").split("/")[-1]
            teacher_name = teacher_token.strip("/").split("/")[-1]
            if parts[i + 1] == student_name:
                parts[i + 1] = teacher_name
                teacher_path = Path(*parts)
            else:
                raise ValueError
        except Exception as exc:
            raise FileNotFoundError(f"Cannot infer paired teacher image for {student_path}") from exc
    if not teacher_path.is_file():
        raise FileNotFoundError(f"Paired teacher image not found: {teacher_path}")
    return teacher_path


def load_paired_teacher_batch(paths, imgsz, device, student_token="/images/ir/", teacher_token="/images/rgb/"):
    ims = []
    for path in paths:
        teacher_path = resolve_teacher_path(path, student_token=student_token, teacher_token=teacher_token)
        im = cv2.imread(str(teacher_path))
        if im is None:
            raise FileNotFoundError(f"Failed to read paired teacher image: {teacher_path}")
        h0, w0 = im.shape[:2]
        r = imgsz / max(h0, w0)
        if r != 1:
            im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        im = letterbox(im, imgsz, auto=False, scaleup=True)[0]
        im = im.transpose((2, 0, 1))[::-1]
        ims.append(np.ascontiguousarray(im))
    batch = torch.from_numpy(np.stack(ims, axis=0)).to(device, non_blocking=True).float() / 255.0
    return batch


def load_student_model(weights, cfg, nc, hyp, device, resume=False):
    check_suffix(weights, ".pt")
    pretrained = str(weights).endswith(".pt")
    if not pretrained:
        return Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)

    weights = str(attempt_download(weights))
    ckpt = torch.load(weights, map_location="cpu")
    model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)
    exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []
    csd = ckpt["model"].float().state_dict()
    csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
    model.load_state_dict(csd, strict=False)
    LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")
    return model


def load_teacher_model(weights, device):
    check_suffix(weights, ".pt")
    teacher = attempt_load(weights, device=device)
    teacher.float().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def standardize_feature(x, eps=1e-6):
    x = x.float()
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (x - mean) / std


def feature_pcc_loss(student_features, teacher_features, layer_indices, adapters=None):
    loss = torch.zeros((), device=student_features[0].device)
    used = 0
    for idx in layer_indices:
        s, t = student_features[idx], teacher_features[idx]
        if adapters is not None:
            s = adapters.forward_one(idx, s)
        t = t.detach()
        if s.shape[-2:] != t.shape[-2:]:
            t = F.interpolate(t, size=s.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss + F.mse_loss(standardize_feature(s), standardize_feature(t))
        used += 1
    return loss / max(used, 1)


def relation_loss(student_features, teacher_features, max_tokens=256, layer=-1):
    s = student_features[layer].float()
    t = teacher_features[layer].detach().float()
    if s.shape[-2:] != t.shape[-2:]:
        t = F.interpolate(t, size=s.shape[-2:], mode="bilinear", align_corners=False)
    s = s.flatten(2).transpose(1, 2)
    t = t.flatten(2).transpose(1, 2)
    n = s.shape[1]
    if max_tokens > 0 and n > max_tokens:
        idx = torch.linspace(0, n - 1, max_tokens, device=s.device).long()
        s = s.index_select(1, idx)
        t = t.index_select(1, idx)
    s = F.normalize(s, dim=-1)
    t = F.normalize(t, dim=-1)
    rel_s = torch.bmm(s, s.transpose(1, 2))
    rel_t = torch.bmm(t, t.transpose(1, 2))
    return F.l1_loss(rel_s, rel_t)


def output_distill_loss(student_preds, teacher_preds, min_conf=0.05, topk=128, temperature=4.0):
    device = student_preds[0].device
    total = torch.zeros((), device=device)
    used = 0
    temp = max(float(temperature), 1e-6)
    for s, t in zip(student_preds, teacher_preds):
        t = t.detach()
        t_obj = torch.sigmoid(t[..., 4:5])
        t_cls = torch.sigmoid(t[..., 5:] / temp)
        score = t_obj * t_cls.max(dim=-1, keepdim=True).values
        mask = score[..., 0] >= min_conf
        if topk > 0:
            b = score.shape[0]
            flat = score.view(b, -1)
            k = min(int(topk), flat.shape[1])
            top_idx = flat.topk(k, dim=1).indices
            top_mask = torch.zeros_like(flat, dtype=torch.bool)
            top_mask.scatter_(1, top_idx, True)
            mask = mask | top_mask.view_as(mask)
        if not mask.any():
            continue
        s_sel = s[mask]
        t_sel = t[mask]
        box = F.smooth_l1_loss(s_sel[:, :4].float(), t_sel[:, :4].float())
        obj = F.binary_cross_entropy_with_logits((s_sel[:, 4:5] / temp).float(),
                                                 torch.sigmoid(t_sel[:, 4:5] / temp).float()) * (temp ** 2)
        if s_sel.shape[1] > 5:
            cls = F.binary_cross_entropy_with_logits((s_sel[:, 5:] / temp).float(),
                                                     torch.sigmoid(t_sel[:, 5:] / temp).float()) * (temp ** 2)
        else:
            cls = torch.zeros((), device=device)
        total = total + box + obj + cls
        used += 1
    return total / max(used, 1)


def unpack_teacher_output(output):
    if isinstance(output, tuple):
        return output[1]
    return output


def normalize_layer_indices(spec, n):
    aliases = {
        "all": list(range(n)),
        "*": list(range(n)),
        "shallow_deep": [0, n - 1],
        "shallow+deep": [0, n - 1],
        "table4_best": [0, n - 1],
        "deepest": [n - 1],
        "deep": [n - 1],
        "shallowest": [0],
        "shallow": [0],
        "middle": [n // 2],
    }
    spec = (spec or "shallow_deep").strip().lower()
    if spec in aliases:
        return sorted(set(aliases[spec]))

    indices = []
    for token in spec.replace("+", ",").split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in ("p2", "p3", "low", "small", "shallow", "shallowest"):
            idx = 0
        elif token in ("p4", "mid", "middle"):
            idx = n // 2
        elif token in ("p5", "c5", "high", "large", "deep", "deepest"):
            idx = n - 1
        else:
            idx = int(token)
            if idx < 0:
                idx += n
        if not 0 <= idx < n:
            raise ValueError(f"Feature layer index {idx} out of range for {n} Detect inputs")
        indices.append(idx)
    if not indices:
        raise ValueError(f"No feature layers parsed from {spec!r}")
    return sorted(set(indices))


def meshgrid_ij(y, x):
    try:
        return torch.meshgrid(y, x, indexing="ij")
    except TypeError:
        return torch.meshgrid(y, x)


def xywh_to_xyxy(x):
    y = x.clone()
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def bbox_iou_xyxy_aligned(box1, box2, eps=1e-7):
    inter_x1 = torch.maximum(box1[:, 0], box2[:, 0])
    inter_y1 = torch.maximum(box1[:, 1], box2[:, 1])
    inter_x2 = torch.minimum(box1[:, 2], box2[:, 2])
    inter_y2 = torch.minimum(box1[:, 3], box2[:, 3])
    inter = (inter_x2 - inter_x1).clamp_min(0) * (inter_y2 - inter_y1).clamp_min(0)
    area1 = (box1[:, 2] - box1[:, 0]).clamp_min(0) * (box1[:, 3] - box1[:, 1]).clamp_min(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp_min(0) * (box2[:, 3] - box2[:, 1]).clamp_min(0)
    return inter / (area1 + area2 - inter + eps)


def decode_yolov5_train_outputs(preds, detect_layer):
    decoded = []
    strides = detect_layer.stride.to(preds[0].device).float()
    anchors = detect_layer.anchors.to(preds[0].device).float()
    for i, pred in enumerate(preds):
        y = pred.float().sigmoid()
        _, na, ny, nx, no = y.shape
        yv, xv = meshgrid_ij(torch.arange(ny, device=y.device), torch.arange(nx, device=y.device))
        grid = torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float() - 0.5
        anchor_grid = (anchors[i] * strides[i]).view(1, na, 1, 1, 2)
        out = y.clone()
        out[..., 0:2] = (y[..., 0:2] * 2 + grid) * strides[i]
        out[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid
        decoded.append(out.view(out.shape[0], -1, no))
    return torch.cat(decoded, dim=1)


def decoded_output_distill_loss(student_preds,
                                teacher_preds,
                                student_detect,
                                teacher_detect,
                                min_conf=0.05,
                                topk=128):
    device = student_preds[0].device
    total = torch.zeros((), device=device)
    used = 0
    student_decoded = decode_yolov5_train_outputs(student_preds, student_detect)
    teacher_decoded = decode_yolov5_train_outputs(teacher_preds, teacher_detect).detach()
    teacher_obj = teacher_decoded[..., 4:5]
    teacher_cls = teacher_decoded[..., 5:]
    teacher_score = teacher_obj * teacher_cls.max(dim=-1, keepdim=True).values

    for b in range(student_decoded.shape[0]):
        mask = teacher_score[b, :, 0] >= min_conf
        if topk > 0:
            k = min(int(topk), teacher_score.shape[1])
            top_idx = teacher_score[b, :, 0].topk(k).indices
            top_mask = torch.zeros_like(mask)
            top_mask[top_idx] = True
            mask = mask | top_mask
        if not mask.any():
            continue

        s_sel = student_decoded[b, mask]
        t_sel = teacher_decoded[b, mask]
        iou = bbox_iou_xyxy_aligned(xywh_to_xyxy(s_sel[:, :4]), xywh_to_xyxy(t_sel[:, :4]))
        box = (1.0 - iou).mean()
        cls = F.binary_cross_entropy(s_sel[:, 5:].clamp(1e-6, 1 - 1e-6),
                                     t_sel[:, 5:].clamp(1e-6, 1 - 1e-6))
        total = total + box + cls
        used += 1
    return total / max(used, 1)


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8") as f:
        if not exists:
            f.write(",".join(row.keys()) + "\n")
        f.write(",".join(f"{v:.8g}" if isinstance(v, float) else str(v) for v in row.values()) + "\n")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolov5-dir", default=str(YOLOV5_DIR), help="YOLOv5 v6.2 root")
    parser.add_argument("--weights", type=str, required=True, help="student initial weights")
    parser.add_argument("--teacher-weights", type=str, required=True, help="frozen IR teacher weights")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--data", type=str, required=True, help="RGB dataset yaml")
    parser.add_argument("--hyp", type=str, default=str(YOLOV5_DIR / "data/hyps/hyp.scratch-low.yaml"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", "--batch", type=int, default=64)
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640)
    parser.add_argument("--cache", type=str, nargs="?", const="ram", default=None)
    parser.add_argument("--rect", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD")
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--patience", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/vedai_yolov5_cmdistill")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    parser.add_argument("--noval", action="store_true")
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--single-cls", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--freeze", nargs="+", type=int, default=[0])
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--amp-check", action="store_true",
                        help="run YOLOv5 check_amp(); this may download yolov5n.pt on a fresh server")
    parser.add_argument("--aligned-no-geo", action="store_true",
                        help="disable mosaic/geometric/flip aug so paired IR teacher stays spatially aligned")
    parser.add_argument("--keep-color-aug", action="store_true",
                        help="with --aligned-no-geo, keep HSV color augmentation on the RGB student")
    parser.add_argument("--rgb-token", default="/images/rgb/")
    parser.add_argument("--ir-token", default="/images/ir/")
    parser.add_argument("--student-token", default=None)
    parser.add_argument("--teacher-token", default=None)
    parser.add_argument("--feature-weight", type=float, default=1.0)
    parser.add_argument("--relation-weight", type=float, default=1.0)
    parser.add_argument("--logit-weight", type=float, default=1.0)
    parser.add_argument("--feature-layers", default="shallow_deep",
                        help="feature levels for PCCFD: shallow_deep/all/deepest or comma list such as p3,p5")
    parser.add_argument("--relation-layer", default="deepest",
                        help="feature level for SLRD: deepest/shallowest/middle or an index")
    parser.add_argument("--no-feature-adapt", action="store_true",
                        help="disable the 1x1 adaptive feature layer before PCCFD")
    parser.add_argument("--raw-output-kd", action="store_true",
                        help="legacy diagnostic: use raw-output SmoothL1/BCE instead of decoded IoU/class logical KD")
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--output-topk", type=int, default=128)
    parser.add_argument("--kd-warmup-epochs", type=float, default=0.0)
    parser.add_argument("--kd-gain", type=float, default=1.0)
    return parser.parse_args()


def train(opt):
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    last = weights_dir / "last.pt"
    best = weights_dir / "best.pt"
    best_map50 = weights_dir / "best_map50.pt"

    hyp = load_hyp(opt.hyp)
    if opt.aligned_no_geo:
        hyp = make_aligned_hyp(hyp, keep_color_aug=opt.keep_color_aug)
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    yaml_save(save_dir / "hyp.yaml", hyp)
    yaml_save(save_dir / "opt.yaml", vars(opt))

    device = select_device(opt.device, batch_size=opt.batch_size)
    cuda = device.type != "cpu"
    init_seeds(opt.seed + 1, deterministic=True)

    data_dict = check_dataset(opt.data)
    train_path, val_path = data_dict["train"], data_dict["val"]
    nc = 1 if opt.single_cls else int(data_dict["nc"])
    names = ["item"] if opt.single_cls and len(data_dict["names"]) != 1 else data_dict["names"]
    assert len(names) == nc, f"{len(names)} names found for nc={nc} dataset in {opt.data}"

    model = load_student_model(opt.weights, opt.cfg, nc, hyp, device)
    amp = check_amp(model) if opt.amp_check else cuda
    kd_active = opt.kd_gain != 0 and (opt.feature_weight != 0 or opt.relation_weight != 0 or opt.logit_weight != 0)
    teacher = load_teacher_model(opt.teacher_weights, device) if kd_active else None
    teacher_hook = DetectFeatureHook(teacher) if kd_active else None

    detect_layer = de_parallel(model).model[-1]
    teacher_detect_layer = de_parallel(teacher).model[-1] if kd_active else None
    feature_indices = normalize_layer_indices(opt.feature_layers, detect_layer.nl)
    relation_layer = normalize_layer_indices(opt.relation_layer, detect_layer.nl)[-1]
    if opt.feature_weight != 0 and not opt.no_feature_adapt:
        student_channels = [m.in_channels for m in detect_layer.m]
        teacher_channels = [m.in_channels for m in teacher_detect_layer.m]
        model.cmdistill_adapters = FeatureAdapters(student_channels, teacher_channels, feature_indices).to(device)
    else:
        model.cmdistill_adapters = None

    freeze = [f"model.{x}." for x in (opt.freeze if len(opt.freeze) > 1 else range(opt.freeze[0]))]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            LOGGER.info(f"freezing {k}")
            v.requires_grad = False

    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)
    batch_size = opt.batch_size

    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    hyp["weight_decay"] *= batch_size * accumulate / nbs
    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])
    if opt.cos_lr:
        lf = one_cycle(1, hyp["lrf"], opt.epochs)
    else:
        lf = lambda x: (1 - x / opt.epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    ema = ModelEMA(model)
    student_hook = DetectFeatureHook(model)

    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size,
                                              gs,
                                              opt.single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=opt.cache,
                                              rect=opt.rect,
                                              rank=-1,
                                              workers=opt.workers,
                                              image_weights=False,
                                              quad=False,
                                              prefix=colorstr("train: "),
                                              shuffle=True)
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {opt.data}"

    val_loader = create_dataloader(val_path,
                                   imgsz,
                                   batch_size * 2,
                                   gs,
                                   opt.single_cls,
                                   hyp=hyp,
                                   cache=opt.cache if not opt.noval else None,
                                   rect=True,
                                   rank=-1,
                                   workers=opt.workers * 2,
                                   pad=0.5,
                                   prefix=colorstr("val: "))[0]

    nl = de_parallel(model).model[-1].nl
    hyp["box"] *= 3 / nl
    hyp["cls"] *= nc / 80 * 3 / nl
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc = nc
    model.hyp = hyp
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    model.names = names

    compute_loss = ComputeLoss(model)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    callbacks = Callbacks()
    student_token = opt.student_token or opt.rgb_token
    teacher_token = opt.teacher_token or opt.ir_token

    nb = len(train_loader)
    nw = max(round(hyp["warmup_epochs"] * nb), 100)
    last_opt_step = -1
    maps = np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0)
    best_fitness = 0.0
    best_map50_value = 0.0
    csv_path = save_dir / "results.csv"
    t0 = time.time()

    LOGGER.info(f"Image sizes {imgsz} train, {imgsz} val\n"
                f"Using {train_loader.num_workers} dataloader workers\n"
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f"KD active={kd_active}, feature_layers={feature_indices}, relation_layer={relation_layer}, "
                f"feature_adapt={model.cmdistill_adapters is not None}, raw_output_kd={opt.raw_output_kd}\n"
                f"Starting CMDistill native training for {opt.epochs} epochs...")

    try:
        for epoch in range(opt.epochs):
            model.train()
            mloss = torch.zeros(7, device=device)
            LOGGER.info(("\n" + "%10s" * 11) %
                        ("Epoch", "gpu_mem", "box", "obj", "cls", "feat", "rel", "out", "labels", "img_size",
                         "kd_w"))
            pbar = tqdm(enumerate(train_loader), total=nb, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
            optimizer.zero_grad()
            for i, (imgs, targets, paths, _) in pbar:
                ni = i + nb * epoch
                imgs = imgs.to(device, non_blocking=True).float() / 255.0
                targets = targets.to(device)

                if ni <= nw:
                    xi = [0, nw]
                    accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                    for j, x in enumerate(optimizer.param_groups):
                        x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0,
                                                     x["initial_lr"] * lf(epoch)])
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

                student_hook.clear()
                with torch.cuda.amp.autocast(amp):
                    pred = model(imgs)
                    det_loss, det_items = compute_loss(pred, targets)
                    feat = torch.zeros((), device=device)
                    rel = torch.zeros((), device=device)
                    out = torch.zeros((), device=device)
                    kd_loss = torch.zeros((), device=device)

                    if kd_active:
                        teacher_imgs = load_paired_teacher_batch(paths,
                                                                 imgsz,
                                                                 device,
                                                                 student_token=student_token,
                                                                 teacher_token=teacher_token)
                        with torch.no_grad():
                            teacher_hook.clear()
                            with torch.cuda.amp.autocast(amp):
                                teacher_raw = unpack_teacher_output(teacher(teacher_imgs))
                            teacher_features = [f.detach() for f in teacher_hook.features]

                        student_features = student_hook.features
                        if opt.feature_weight != 0:
                            feat = feature_pcc_loss(student_features,
                                                    teacher_features,
                                                    feature_indices,
                                                    adapters=model.cmdistill_adapters)
                        if opt.relation_weight != 0:
                            rel = relation_loss(student_features,
                                                teacher_features,
                                                max_tokens=opt.max_tokens,
                                                layer=relation_layer)
                        if opt.logit_weight != 0:
                            if opt.raw_output_kd:
                                out = output_distill_loss(pred,
                                                          teacher_raw,
                                                          min_conf=opt.min_confidence,
                                                          topk=opt.output_topk,
                                                          temperature=opt.temperature)
                            else:
                                out = decoded_output_distill_loss(pred,
                                                                  teacher_raw,
                                                                  detect_layer,
                                                                  teacher_detect_layer,
                                                                  min_conf=opt.min_confidence,
                                                                  topk=opt.output_topk)
                    if opt.kd_warmup_epochs > 0:
                        kd_w = min(1.0, (epoch + (i + 1) / nb) / opt.kd_warmup_epochs)
                    else:
                        kd_w = 1.0
                    if kd_active:
                        kd_loss = opt.kd_gain * kd_w * (opt.feature_weight * feat + opt.relation_weight * rel +
                                                        opt.logit_weight * out)
                    loss = det_loss + kd_loss

                scaler.scale(loss).backward()

                if ni - last_opt_step >= accumulate:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    ema.update(model)
                    last_opt_step = ni

                mloss = (mloss * i + torch.cat((det_items, feat.detach().view(1), rel.detach().view(1),
                                                out.detach().view(1), kd_loss.detach().view(1)))) / (i + 1)
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
                pbar.set_description(("%10s" * 2 + "%10.4g" * 9) %
                                     (f"{epoch}/{opt.epochs - 1}", mem, *mloss[:6], targets.shape[0], imgs.shape[-1],
                                      kd_w))

            lr = [x["lr"] for x in optimizer.param_groups]
            scheduler.step()

            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            final_epoch = (epoch + 1 == opt.epochs) or stopper.possible_stop
            do_val = (not opt.noval and ((epoch + 1) % max(opt.val_interval, 1) == 0 or final_epoch))
            if do_val:
                results, maps, _ = val.run(data_dict,
                                           batch_size=batch_size * 2,
                                           imgsz=imgsz,
                                           half=amp,
                                           model=ema.ema,
                                           single_cls=opt.single_cls,
                                           dataloader=val_loader,
                                           save_dir=save_dir,
                                           plots=False,
                                           callbacks=callbacks,
                                           compute_loss=compute_loss)

            fi = fitness(np.array(results).reshape(1, -1))
            map50 = float(results[2])
            stop = stopper(epoch=epoch, fitness=fi)
            if fi > best_fitness:
                best_fitness = fi
            if map50 > best_map50_value:
                best_map50_value = map50

            append_csv(csv_path, {
                "epoch": epoch,
                "train/box_loss": float(mloss[0]),
                "train/obj_loss": float(mloss[1]),
                "train/cls_loss": float(mloss[2]),
                "train/feat_loss": float(mloss[3]),
                "train/rel_loss": float(mloss[4]),
                "train/out_loss": float(mloss[5]),
                "train/kd_loss": float(mloss[6]),
                "metrics/precision": float(results[0]),
                "metrics/recall": float(results[1]),
                "metrics/mAP_0.5": float(results[2]),
                "metrics/mAP_0.5:0.95": float(results[3]),
                "val/box_loss": float(results[4]),
                "val/obj_loss": float(results[5]),
                "val/cls_loss": float(results[6]),
                "x/lr0": float(lr[0]),
                "x/lr1": float(lr[1]) if len(lr) > 1 else float(lr[0]),
                "x/lr2": float(lr[2]) if len(lr) > 2 else float(lr[-1]),
            })

            if not opt.nosave or final_epoch:
                student_hook.clear()
                if teacher_hook is not None:
                    teacher_hook.clear()
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "best_map50": best_map50_value,
                    "model": deepcopy(de_parallel(model)).half(),
                    "ema": deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "date": datetime.now().isoformat(),
                }
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if best_map50_value == map50:
                    torch.save(ckpt, best_map50)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, weights_dir / f"epoch{epoch}.pt")
                del ckpt

            if stop:
                break

    finally:
        if teacher_hook is not None:
            teacher_hook.close()
        student_hook.close()

    LOGGER.info(f"\n{epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
    for f in (last, best, best_map50):
        if f.exists():
            strip_optimizer(f)
    torch.cuda.empty_cache()
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")
    return results


def main():
    opt = parse_opt()
    train(opt)


if __name__ == "__main__":
    main()
