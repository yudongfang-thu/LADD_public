#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from torch.cuda import amp
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Native YOLOv5 v5.0 online CoLD-style HBB training for OGSOD SAR student + RGB teacher."
    )
    parser.add_argument("--yolov5-root", type=Path, default=Path.cwd())
    parser.add_argument("--data", type=Path, required=True, help="SAR HBB dataset YAML.")
    parser.add_argument("--teacher-data", type=Path, required=True, help="RGB HBB dataset YAML.")
    parser.add_argument("--cfg", default="models/yolov5x.yaml")
    parser.add_argument("--weights", default="yolov5x.pt", help="Student init weights.")
    parser.add_argument("--teacher-weights", default="yolov5x.pt", help="Teacher init weights.")
    parser.add_argument("--hyp", default="data/hyp.cold_paper.yaml")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32, help="Physical batch size. Use 32 on 24GB 4090D.")
    parser.add_argument("--effective-batch-size", type=int, default=64)
    parser.add_argument("--img-size", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/ogsod_cold_repro")
    parser.add_argument("--name", default="cold_v5p0_yolov5x_coco_mixup010")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--single-cls", action="store_true")
    parser.add_argument("--cache-images", action="store_true")
    parser.add_argument("--rect", action="store_true")
    parser.add_argument("--noautoanchor", action="store_true")
    parser.add_argument("--notest", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    parser.add_argument("--linear-lr", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--lambda-cls-cold", type=float, default=0.0)
    parser.add_argument("--lambda-loc-cold", type=float, default=1.0)
    parser.add_argument("--teacher-det-weight", type=float, default=1.0)
    parser.add_argument("--alpha-non-target", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--cold-loss-mode", choices=("matched", "candidate"), default="matched")
    parser.add_argument("--cold-terms", choices=("tcld", "ncld", "both"), default="both")
    parser.add_argument(
        "--cold-iwm-mode",
        choices=("none", "mean"),
        default="mean",
        help="candidate mode only: no IWM for TCLD/NCLD mechanism sanity, or old layer/image mean IoU scaling.",
    )
    parser.add_argument("--assert-nonnegative-cold", action="store_true")
    parser.add_argument("--candidate-topk", type=int, default=1000)
    parser.add_argument("--candidate-min-conf", type=float, default=0.001)
    parser.add_argument("--candidate-iou-weight-floor", type=float, default=0.0)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--max-batches", type=int, default=-1, help="Debug only: stop each epoch after this many batches.")
    return parser.parse_args()


COLD_STAT_FIELDS = (
    "tcld_loss",
    "ncld_loss",
    "loc_cold",
    "candidate_count",
    "target_candidate_count",
    "nontarget_candidate_count",
    "tcld_terms",
    "ncld_terms",
    "candidate_iou_mean",
    "candidate_iou_positive_count",
    "effective_iwm_weight_mean",
)


def _zero_cold_stats(device: torch.device) -> dict[str, torch.Tensor]:
    return {name: torch.zeros((), device=device) for name in COLD_STAT_FIELDS}


def detach_predictions(pred):
    """Treat teacher predictions as fixed KD targets."""

    if isinstance(pred, torch.Tensor):
        return pred.detach()
    if isinstance(pred, list):
        return [detach_predictions(p) for p in pred]
    if isinstance(pred, tuple):
        return tuple(detach_predictions(p) for p in pred)
    return pred


def yolov5_training_outputs(pred):
    """Return raw YOLOv5 detect-layer outputs for KD/loss code."""

    if isinstance(pred, tuple) and len(pred) == 2 and isinstance(pred[1], list):
        return pred[1]
    return pred


def setup_yolov5_path(root: Path) -> None:
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


class PairedLoadImagesAndLabels:  # thin proxy; inherits at runtime after YOLOv5 imports
    pass


def build_paired_dataset_class(base_cls):
    class _PairedLoadImagesAndLabels(base_cls):
        def __init__(self, *args, teacher_train_path: str, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher_img_files = self._build_teacher_paths(teacher_train_path)

        def _build_teacher_paths(self, teacher_train_path: str) -> list[str]:
            from utils.datasets import img_formats

            teacher_root = Path(teacher_train_path)
            if teacher_root.is_file():
                files = [
                    x.replace("./", str(teacher_root.parent) + os.sep) if x.startswith("./") else x
                    for x in teacher_root.read_text().strip().splitlines()
                ]
            else:
                files = [str(p) for p in teacher_root.rglob("*") if p.suffix[1:].lower() in img_formats]
            files = sorted(str(Path(x)) for x in files if str(x).split(".")[-1].lower() in img_formats)
            by_name = {Path(p).name: p for p in files}
            out = []
            missing = []
            for sar_path in self.img_files:
                name = Path(sar_path).name
                rgb_path = by_name.get(name)
                if rgb_path is None:
                    guess = sar_path.replace(f"{os.sep}sar{os.sep}images{os.sep}", f"{os.sep}rgb{os.sep}images{os.sep}")
                    rgb_path = guess if Path(guess).exists() else None
                if rgb_path is None:
                    missing.append(name)
                    out.append(sar_path)
                else:
                    out.append(rgb_path)
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} paired RGB images, first={missing[:5]}")
            return out

        def __getitem__(self, index):
            py_state = random.getstate()
            np_state = np.random.get_state()
            sar_img, labels, path, shapes = super().__getitem__(index)

            random.setstate(py_state)
            np.random.set_state(np_state)
            old_img_files = self.img_files
            try:
                self.img_files = self.teacher_img_files
                rgb_img, _, rgb_path, _ = super().__getitem__(index)
            finally:
                self.img_files = old_img_files
            return sar_img, rgb_img, labels, path, shapes, rgb_path

        @staticmethod
        def collate_fn(batch):
            img, teacher_img, label, path, shapes, teacher_path = zip(*batch)
            for i, l in enumerate(label):
                l[:, 0] = i
            return torch.stack(img, 0), torch.stack(teacher_img, 0), torch.cat(label, 0), path, shapes, teacher_path

    return _PairedLoadImagesAndLabels


def decoupled_kl_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    alpha_non_target: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    if student_logits.numel() == 0 or student_logits.shape[-1] < 2:
        return student_logits.new_zeros(())
    t = max(float(temperature), eps)
    teacher_probs = F.softmax(teacher_logits / t, dim=-1)
    student_probs = F.softmax(student_logits / t, dim=-1)
    n, k = teacher_probs.shape
    top_idx = teacher_probs.argmax(dim=-1, keepdim=True)
    p_t = teacher_probs.gather(-1, top_idx).squeeze(-1)
    q_t = student_probs.gather(-1, top_idx).squeeze(-1)
    teacher_bin = torch.stack((p_t, 1.0 - p_t), dim=-1).clamp_min(eps)
    student_bin = torch.stack((q_t, 1.0 - q_t), dim=-1).clamp_min(eps)
    tpd = (teacher_bin * (teacher_bin.log() - student_bin.log())).sum(dim=-1)
    mask = F.one_hot(top_idx.squeeze(-1), num_classes=k).bool()
    teacher_rest = teacher_probs.masked_select(~mask).view(n, k - 1).clamp_min(eps)
    student_rest = student_probs.masked_select(~mask).view(n, k - 1).clamp_min(eps)
    teacher_rest = teacher_rest / teacher_rest.sum(dim=-1, keepdim=True).clamp_min(eps)
    student_rest = student_rest / student_rest.sum(dim=-1, keepdim=True).clamp_min(eps)
    npd = (teacher_rest * (teacher_rest.log() - student_rest.log())).sum(dim=-1)
    return (tpd + alpha_non_target * npd).mean() * (t**2)


def _xywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    xy = box[:, :2]
    wh = box[:, 2:4]
    return torch.cat((xy - wh / 2.0, xy + wh / 2.0), dim=1)


def _weighted_distribution_kl(
    teacher_box_xyxy: torch.Tensor,
    student_box_xyxy: torch.Tensor,
    weights: torch.Tensor,
    temperature: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL over CPM box-coordinate distributions.

    CoLD converts box regression outputs into probability distributions with
    Softmax(tau). YOLOv5 v5.0 has point box regression, not DFL bins, so this
    implements the closest native-head analogue: for each coordinate edge,
    form a distribution over the CPM-selected candidate boxes.
    """

    if teacher_box_xyxy.shape[0] < 2:
        return teacher_box_xyxy.new_zeros(())
    t = max(float(temperature), eps)
    teacher_logits = teacher_box_xyxy.detach().T / t
    student_logits = student_box_xyxy.T / t
    teacher_prob = F.softmax(teacher_logits, dim=-1).clamp_min(eps)
    teacher_log_prob = teacher_prob.log()
    student_log_prob = F.log_softmax(student_logits, dim=-1)
    w = weights.detach().clamp_min(0.0)
    if float(w.sum()) <= eps:
        w = torch.ones_like(w)
    w = w / w.sum().clamp_min(eps)
    return (teacher_prob * (teacher_log_prob - student_log_prob) * w.unsqueeze(0)).sum(dim=-1).mean() * (t**2)


def _distribution_kl(teacher_box_xyxy: torch.Tensor, student_box_xyxy: torch.Tensor, temperature: float, eps: float = 1e-8) -> torch.Tensor:
    if teacher_box_xyxy.shape[0] < 2:
        return teacher_box_xyxy.new_zeros(())
    t = max(float(temperature), eps)
    teacher_logit = teacher_box_xyxy.detach().T / t
    student_logit = student_box_xyxy.T / t
    teacher_prob = F.softmax(teacher_logit, dim=-1).clamp_min(eps)
    teacher_log_prob = teacher_prob.log()
    student_log_prob = F.log_softmax(student_logit, dim=-1)
    return (teacher_prob * (teacher_log_prob - student_log_prob)).sum(dim=-1).mean() * (t**2)


def _mean_or_zero(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.mean() if value.numel() else torch.zeros((), device=device)


def _xywhn_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    xy = box[..., :2]
    wh = box[..., 2:4]
    return torch.cat((xy - wh / 2.0, xy + wh / 2.0), dim=-1).clamp(0.0, 1.0)


def _box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if box1.numel() == 0 or box2.numel() == 0:
        return box1.new_zeros((box1.shape[0], box2.shape[0]))
    lt = torch.maximum(box1[:, None, :2], box2[None, :, :2])
    rb = torch.minimum(box1[:, None, 2:], box2[None, :, 2:])
    wh = (rb - lt).clamp_min(0.0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (box1[:, 2] - box1[:, 0]).clamp_min(0.0) * (box1[:, 3] - box1[:, 1]).clamp_min(0.0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp_min(0.0) * (box2[:, 3] - box2[:, 1]).clamp_min(0.0)
    return inter / (area1[:, None] + area2[None, :] - inter + eps)


def _decode_yolov5_layer(pred: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    bs, na, ny, nx, _ = pred.shape
    yv, xv = torch.meshgrid(
        torch.arange(ny, device=pred.device),
        torch.arange(nx, device=pred.device),
        indexing="ij",
    )
    grid = torch.stack((xv, yv), dim=-1).view(1, 1, ny, nx, 2).float()
    gain = pred.new_tensor((nx, ny))
    xy = (pred[..., 0:2].sigmoid() * 2.0 - 0.5 + grid) / gain
    wh = ((pred[..., 2:4].sigmoid() * 2.0) ** 2 * anchors.view(1, na, 1, 1, 2)) / gain
    return _xywhn_to_xyxy(torch.cat((xy, wh), dim=-1).view(bs, -1, 4))


def _candidate_iou_weight(boxes: torch.Tensor, image_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    weights = boxes.new_zeros((boxes.shape[0],))
    if targets.numel() == 0 or boxes.numel() == 0:
        return weights
    max_img_id = int(image_ids.max().item()) if image_ids.numel() else -1
    for img_id in range(max_img_id + 1):
        cand_mask = image_ids == img_id
        gt = targets[targets[:, 0].long() == img_id]
        if gt.numel() == 0:
            continue
        gt_boxes = _xywhn_to_xyxy(gt[:, 2:6])
        cand_indices = cand_mask.nonzero(as_tuple=False).squeeze(1)
        weights[cand_indices] = _box_iou_xyxy(boxes[cand_indices], gt_boxes).max(dim=1).values
    return weights


def cold_candidate_cpm_iwm_loss(
    student_pred,
    teacher_pred,
    targets,
    compute_loss,
    temperature: float,
    alpha_non_target: float,
    cold_terms: str,
    cold_iwm_mode: str,
    candidate_topk: int,
    candidate_min_conf: float,
    candidate_iou_weight_floor: float,
):
    device = targets.device
    tcld_loss_sum = torch.zeros((), device=device)
    ncld_loss_sum = torch.zeros((), device=device)
    stats = _zero_cold_stats(device)
    tcld_terms = 0
    ncld_terms = 0
    iwm_weight_sum = torch.zeros((), device=device)
    iwm_weight_terms = 0
    use_iwm = cold_iwm_mode != "none"

    for layer_idx, (sp, tp) in enumerate(zip(student_pred, teacher_pred)):
        bs = tp.shape[0]
        teacher_box = _decode_yolov5_layer(tp, compute_loss.anchors[layer_idx])
        student_box = _decode_yolov5_layer(sp, compute_loss.anchors[layer_idx])
        obj = tp[..., 4].sigmoid().view(bs, -1)
        if compute_loss.nc > 1:
            cls_prob, hard_cls = tp[..., 5:].sigmoid().view(bs, -1, compute_loss.nc).max(dim=-1)
        else:
            cls_prob = torch.ones_like(obj)
            hard_cls = torch.zeros_like(obj, dtype=torch.long)
        conf = obj * cls_prob

        for b in range(bs):
            valid = conf[b] >= float(candidate_min_conf)
            if not valid.any():
                continue
            valid_idx = valid.nonzero(as_tuple=False).squeeze(1)
            k = min(int(candidate_topk), int(valid_idx.numel()))
            top_local = conf[b, valid_idx].topk(k, sorted=False).indices
            flat_idx = valid_idx[top_local]
            boxes_t = teacher_box[b, flat_idx]
            boxes_s = student_box[b, flat_idx]
            cls_t = hard_cls[b, flat_idx]
            candidate_iou_mean = torch.zeros((), device=device)
            positive_count = 0
            if use_iwm:
                img_ids = torch.full((flat_idx.numel(),), b, dtype=torch.long, device=device)
                iou_weight = _candidate_iou_weight(boxes_t.detach(), img_ids, targets)
                positive_weight = iou_weight[iou_weight > 0]
                positive_count = positive_weight.numel()
                candidate_iou_mean = _mean_or_zero(iou_weight, device)
                group_weight = _mean_or_zero(positive_weight, device)
                if float(group_weight.detach()) <= 0.0:
                    group_weight = candidate_iou_mean
                if candidate_iou_weight_floor > 0:
                    group_weight = group_weight.clamp_min(float(candidate_iou_weight_floor))
            else:
                group_weight = torch.ones((), device=device)

            stats["candidate_count"] += flat_idx.numel()
            stats["candidate_iou_mean"] += candidate_iou_mean.detach()
            stats["candidate_iou_positive_count"] += positive_count
            iwm_weight_sum += group_weight.detach()
            iwm_weight_terms += 1

            for cls_id in range(compute_loss.nc):
                target_mask = cls_t == cls_id
                nontarget_mask = ~target_mask
                target_boxes_t = boxes_t[target_mask]
                target_boxes_s = boxes_s[target_mask]
                nontarget_boxes_t = boxes_t[nontarget_mask]
                nontarget_boxes_s = boxes_s[nontarget_mask]
                target_count = target_boxes_t.shape[0]
                nontarget_count = nontarget_boxes_t.shape[0]
                stats["target_candidate_count"] += target_count
                stats["nontarget_candidate_count"] += nontarget_count
                if cold_terms in ("tcld", "both") and target_count >= 2:
                    term = group_weight.detach() * _distribution_kl(
                        target_boxes_t,
                        target_boxes_s,
                        temperature=temperature,
                    )
                    tcld_loss_sum = tcld_loss_sum + term
                    tcld_terms += 1
                if cold_terms in ("ncld", "both") and nontarget_count >= 2:
                    term = group_weight.detach() * _distribution_kl(
                        nontarget_boxes_t,
                        nontarget_boxes_s,
                        temperature=temperature,
                    )
                    ncld_loss_sum = ncld_loss_sum + term
                    ncld_terms += 1

    enabled_terms = 0
    loc_loss = torch.zeros((), device=device)
    if cold_terms in ("tcld", "both") and tcld_terms:
        tcld_loss = tcld_loss_sum / tcld_terms
        loc_loss = loc_loss + tcld_loss
        enabled_terms += 1
        stats["tcld_loss"] = tcld_loss.detach()
    if cold_terms in ("ncld", "both") and ncld_terms:
        ncld_loss = ncld_loss_sum / ncld_terms
        loc_loss = loc_loss + alpha_non_target * ncld_loss
        enabled_terms += 1
        stats["ncld_loss"] = ncld_loss.detach()
    if enabled_terms:
        loc_loss = loc_loss / enabled_terms
    stats["loc_cold"] = loc_loss.detach()
    stats["tcld_terms"] = torch.as_tensor(float(tcld_terms), device=device)
    stats["ncld_terms"] = torch.as_tensor(float(ncld_terms), device=device)
    if iwm_weight_terms:
        stats["candidate_iou_mean"] = stats["candidate_iou_mean"] / iwm_weight_terms
        stats["effective_iwm_weight_mean"] = iwm_weight_sum / iwm_weight_terms
    return torch.zeros((), device=device), loc_loss, stats["candidate_iou_mean"], stats


def cold_cpm_iwm_loss(
    student_pred,
    teacher_pred,
    targets,
    compute_loss,
    temperature: float,
    alpha_non_target: float,
    cold_terms: str,
):
    from utils.general import bbox_iou

    tcls, tbox, indices, anchors = compute_loss.build_targets(student_pred, targets)
    device = targets.device
    target_loss = torch.zeros((), device=device)
    nontarget_loss = torch.zeros((), device=device)
    iou_mean = torch.zeros((), device=device)
    target_terms = 0
    nontarget_terms = 0
    n_iou_layers = 0
    stats = _zero_cold_stats(device)

    for i, (sp, tp) in enumerate(zip(student_pred, teacher_pred)):
        b, a, gj, gi = indices[i]
        n = b.shape[0]
        if not n:
            continue
        ps = sp[b, a, gj, gi]
        pt = tp[b, a, gj, gi]

        sxy = ps[:, :2].sigmoid() * 2.0 - 0.5
        swh = (ps[:, 2:4].sigmoid() * 2.0) ** 2 * anchors[i]
        txy = pt[:, :2].sigmoid() * 2.0 - 0.5
        twh = (pt[:, 2:4].sigmoid() * 2.0) ** 2 * anchors[i]
        sbox = torch.cat((sxy, swh), 1)
        teacher_box = torch.cat((txy, twh), 1).detach()
        teacher_target_iou = bbox_iou(teacher_box.T, tbox[i], x1y1x2y2=False, CIoU=True).detach().clamp(0.0, 1.0)
        teacher_box_xyxy = _xywh_to_xyxy(teacher_box)
        student_box_xyxy = _xywh_to_xyxy(sbox)

        if compute_loss.nc > 1:
            teacher_hard_cls = pt[:, 5:].detach().argmax(dim=1)
            classes = torch.arange(compute_loss.nc, device=device)
        else:
            teacher_hard_cls = torch.zeros(n, dtype=torch.long, device=device)
            classes = torch.zeros(1, dtype=torch.long, device=device)

        # CPM: partition matched candidate boxes by the teacher classification
        # hard label. For each category, m_t is the target-category set and
        # m_hat_t is the complementary nontarget-category set.
        for cls_id in classes:
            target_mask = teacher_hard_cls == cls_id
            nontarget_mask = ~target_mask
            if int(target_mask.sum().item()) >= 2:
                target_loss = target_loss + _weighted_distribution_kl(
                    teacher_box_xyxy[target_mask],
                    student_box_xyxy[target_mask],
                    teacher_target_iou[target_mask],
                    temperature=temperature,
                )
                target_terms += 1
            if int(nontarget_mask.sum().item()) >= 2:
                nontarget_loss = nontarget_loss + _weighted_distribution_kl(
                    teacher_box_xyxy[nontarget_mask],
                    student_box_xyxy[nontarget_mask],
                    teacher_target_iou[nontarget_mask],
                    temperature=temperature,
                )
                nontarget_terms += 1
        iou_mean = iou_mean + teacher_target_iou.mean()
        n_iou_layers += 1

    enabled_terms = 0
    loc_loss = torch.zeros((), device=device)
    if cold_terms in ("tcld", "both") and target_terms:
        tcld_loss = target_loss / target_terms
        loc_loss = loc_loss + tcld_loss
        enabled_terms += 1
        stats["tcld_loss"] = tcld_loss.detach()
    if cold_terms in ("ncld", "both") and nontarget_terms:
        ncld_loss = nontarget_loss / nontarget_terms
        loc_loss = loc_loss + alpha_non_target * ncld_loss
        enabled_terms += 1
        stats["ncld_loss"] = ncld_loss.detach()
    if enabled_terms:
        loc_loss = loc_loss / enabled_terms
        iou_mean = iou_mean / max(n_iou_layers, 1)
    stats["loc_cold"] = loc_loss.detach()
    stats["candidate_iou_mean"] = iou_mean.detach()
    stats["tcld_terms"] = torch.as_tensor(float(target_terms), device=device)
    stats["ncld_terms"] = torch.as_tensor(float(nontarget_terms), device=device)
    return torch.zeros((), device=device), loc_loss, iou_mean, stats


def load_model(weights: str, cfg: str, nc: int, hyp: dict, device: torch.device):
    from models.experimental import attempt_download
    from models.yolo import Model
    from utils.torch_utils import intersect_dicts

    pretrained = bool(weights) and weights.endswith(".pt")
    if pretrained:
        attempt_download(weights)
        ckpt = torch.load(weights, map_location=device, weights_only=False)
        model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)
        state_dict = ckpt["model"].float().state_dict()
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=["anchor"])
        model.load_state_dict(state_dict, strict=False)
        logging.info("Transferred %g/%g items from %s", len(state_dict), len(model.state_dict()), weights)
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)
    return model


def main() -> None:
    opt = parse_args()
    setup_yolov5_path(opt.yolov5_root)

    import test
    from utils.autoanchor import check_anchors
    from utils.datasets import LoadImagesAndLabels
    from utils.general import (
        check_dataset,
        check_img_size,
        colorstr,
        fitness,
        increment_path,
        init_seeds,
        labels_to_class_weights,
        one_cycle,
        set_logging,
    )
    from utils.loss import ComputeLoss
    from utils.torch_utils import ModelEMA, select_device

    set_logging(-1)
    logger = logging.getLogger(__name__)
    device = select_device(opt.device, batch_size=opt.batch_size)
    cuda = device.type != "cpu"
    init_seeds(2)

    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)
    with open(opt.teacher_data) as f:
        teacher_data_dict = yaml.safe_load(f)
    check_dataset(data_dict)
    check_dataset(teacher_data_dict)
    nc = 1 if opt.single_cls else int(data_dict["nc"])
    names = ["item"] if opt.single_cls and len(data_dict["names"]) != 1 else data_dict["names"]
    assert len(names) == nc

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    last, best = wdir / "last.pt", wdir / "best.pt"
    results_file = save_dir / "results.txt"
    cold_stats_file = save_dir / "cold_stats.csv"
    (save_dir / "hyp.yaml").write_text(yaml.safe_dump(hyp, sort_keys=False))
    opt_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(opt).items()}
    (save_dir / "opt.yaml").write_text(yaml.safe_dump(opt_dict, sort_keys=False))
    cold_stats_file.write_text("epoch," + ",".join(COLD_STAT_FIELDS) + "\n")

    student = load_model(opt.weights, opt.cfg, nc, hyp, device)
    teacher = load_model(opt.teacher_weights, opt.cfg, nc, hyp, device)
    freeze_teacher = opt.teacher_det_weight <= 0.0
    if freeze_teacher:
        for param in teacher.parameters():
            param.requires_grad_(False)
        logger.info("Freezing teacher because teacher_det_weight=%s", opt.teacher_det_weight)

    gs = max(int(student.stride.max()), 32)
    imgsz, imgsz_test = [check_img_size(x, gs) for x in (opt.img_size + opt.img_size[-1:])[:2]]
    nl = student.model[-1].nl
    hyp["box"] *= 3.0 / nl
    hyp["cls"] *= nc / 80.0 * 3.0 / nl
    hyp["obj"] *= (imgsz / 640) ** 2 * 3.0 / nl
    hyp["label_smoothing"] = opt.label_smoothing
    for model in (student, teacher):
        model.nc = nc
        model.hyp = hyp
        model.gr = 1.0
        model.names = names

    train_path = data_dict["train"]
    teacher_train_path = teacher_data_dict["train"]
    PairedDataset = build_paired_dataset_class(LoadImagesAndLabels)
    dataset = PairedDataset(
        train_path,
        imgsz,
        opt.batch_size,
        augment=True,
        hyp=hyp,
        rect=opt.rect,
        cache_images=opt.cache_images,
        single_cls=opt.single_cls,
        stride=gs,
        teacher_train_path=teacher_train_path,
        prefix=colorstr("train: "),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.rect,
        num_workers=opt.workers,
        pin_memory=True,
        collate_fn=PairedDataset.collate_fn,
        drop_last=False,
    )
    nb = len(dataloader)
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc}"

    testloader = DataLoader(
        LoadImagesAndLabels(
            data_dict["val"],
            imgsz_test,
            opt.batch_size * 2,
            hyp=hyp,
            rect=True,
            cache_images=False,
            single_cls=opt.single_cls,
            stride=gs,
            pad=0.5,
            prefix=colorstr("val: "),
        ),
        batch_size=opt.batch_size * 2,
        shuffle=False,
        num_workers=opt.workers,
        pin_memory=True,
        collate_fn=LoadImagesAndLabels.collate_fn,
    )

    if not opt.noautoanchor:
        check_anchors(dataset, model=student, thr=hyp["anchor_t"], imgsz=imgsz)
        student.half().float()
        teacher.half().float()

    student.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    teacher.class_weights = student.class_weights

    pg0, pg1, pg2 = [], [], []
    optim_models = (student,) if freeze_teacher else (student, teacher)
    for model in optim_models:
        for _, m in model.named_modules():
            if hasattr(m, "bias") and isinstance(m.bias, nn.Parameter):
                pg2.append(m.bias)
            if isinstance(m, nn.BatchNorm2d):
                pg0.append(m.weight)
            elif hasattr(m, "weight") and isinstance(m.weight, nn.Parameter):
                pg1.append(m.weight)
    nbs = 64
    total_batch_size = int(opt.effective_batch_size)
    accumulate = max(round(nbs / total_batch_size), 1)
    hyp["weight_decay"] *= total_batch_size * accumulate / nbs
    optimizer = optim.SGD(pg0, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay": hyp["weight_decay"]})
    optimizer.add_param_group({"params": pg2})
    for x in optimizer.param_groups:
        x["initial_lr"] = hyp["lr0"]
    del pg0, pg1, pg2

    lf = (lambda x: (1 - x / (opt.epochs - 1)) * (1.0 - hyp["lrf"]) + hyp["lrf"]) if opt.linear_lr else one_cycle(1, hyp["lrf"], opt.epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    ema = ModelEMA(student)
    compute_student_loss = ComputeLoss(student)
    compute_teacher_loss = ComputeLoss(teacher)
    scaler = amp.GradScaler(enabled=cuda)

    logger.info(
        "Image sizes %s train, %s test; physical batch=%s effective batch=%s; nb=%s",
        imgsz,
        imgsz_test,
        opt.batch_size,
        total_batch_size,
        nb,
    )
    logger.info("Starting native YOLOv5-v5.0 CoLD-style online training for %s epochs", opt.epochs)
    best_fitness = 0.0
    t0 = time.time()
    nw = max(round(hyp["warmup_epochs"] * nb), 1000)
    maps = np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0)

    for epoch in range(opt.epochs):
        student.train()
        teacher.eval() if freeze_teacher else teacher.train()
        mloss = torch.zeros(7, device=device)
        cold_stats_epoch = _zero_cold_stats(device)
        cold_stats_steps = 0
        pbar = tqdm(enumerate(dataloader), total=nb)
        optimizer.zero_grad()
        for i, (imgs, teacher_imgs, targets, paths, _, _) in pbar:
            if opt.max_batches > 0 and i >= opt.max_batches:
                break
            ni = i + nb * epoch
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            teacher_imgs = teacher_imgs.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)

            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, round(np.interp(ni, xi, [1, nbs / total_batch_size])))
                for j, x in enumerate(optimizer.param_groups):
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 2 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            with amp.autocast(enabled=cuda):
                student_pred = student(imgs)
                if freeze_teacher:
                    with torch.no_grad():
                        teacher_pred = teacher(teacher_imgs)
                else:
                    teacher_pred = teacher(teacher_imgs)
                teacher_pred = yolov5_training_outputs(teacher_pred)
                student_det_loss, student_items = compute_student_loss(student_pred, targets)
                if freeze_teacher:
                    teacher_det_loss = torch.zeros((), device=device)
                    teacher_items = torch.zeros_like(student_items)
                else:
                    teacher_det_loss, teacher_items = compute_teacher_loss(teacher_pred, targets)
                teacher_pred_for_kd = detach_predictions(teacher_pred)
                if opt.cold_loss_mode == "candidate":
                    cls_cold, loc_cold, teacher_iou, cold_stats = cold_candidate_cpm_iwm_loss(
                        student_pred,
                        teacher_pred_for_kd,
                        targets,
                        compute_student_loss,
                        temperature=opt.temperature,
                        alpha_non_target=opt.alpha_non_target,
                        cold_terms=opt.cold_terms,
                        cold_iwm_mode=opt.cold_iwm_mode,
                        candidate_topk=opt.candidate_topk,
                        candidate_min_conf=opt.candidate_min_conf,
                        candidate_iou_weight_floor=opt.candidate_iou_weight_floor,
                    )
                else:
                    cls_cold, loc_cold, teacher_iou, cold_stats = cold_cpm_iwm_loss(
                        student_pred,
                        teacher_pred_for_kd,
                        targets,
                        compute_student_loss,
                        temperature=opt.temperature,
                        alpha_non_target=opt.alpha_non_target,
                        cold_terms=opt.cold_terms,
                    )
                check_cold_sign = i % 10 == 0 or i + 1 == nb or (opt.max_batches > 0 and i + 1 >= opt.max_batches)
                if opt.assert_nonnegative_cold and check_cold_sign and float(loc_cold.detach()) < -1e-6:
                    raise RuntimeError(f"CoLD loc loss became negative: {float(loc_cold.detach())}")
                total_loss = (
                    student_det_loss
                    + opt.teacher_det_weight * teacher_det_loss
                    + opt.lambda_cls_cold * cls_cold
                    + opt.lambda_loc_cold * loc_cold
                )

            scaler.scale(total_loss).backward()
            if ni % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(student)

            loss_items = torch.cat(
                (
                    student_items.detach(),
                    torch.stack((teacher_items[-1].detach(), cls_cold.detach(), loc_cold.detach())),
                )
            )
            mloss = (mloss * i + loss_items) / (i + 1)
            for name in COLD_STAT_FIELDS:
                cold_stats_epoch[name] = cold_stats_epoch[name] + cold_stats[name].detach()
            cold_stats_steps += 1
            if check_cold_sign:
                mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)
                pbar.set_description(
                    ("%10s" * 2 + "%10.4g" * 8)
                    % (
                        f"{epoch}/{opt.epochs - 1}",
                        mem,
                        *mloss,
                        teacher_iou.detach(),
                    )
                )

        scheduler.step()
        ema.update_attr(student, include=["yaml", "nc", "hyp", "gr", "names", "stride", "class_weights"])
        final_epoch = epoch + 1 == opt.epochs
        if opt.max_batches > 0:
            logger.info("Debug max-batches=%s reached; skipping validation for epoch %s.", opt.max_batches, epoch)
        elif not opt.notest or final_epoch:
            results, maps, _ = test.test(
                data_dict,
                batch_size=opt.batch_size * 2,
                imgsz=imgsz_test,
                model=ema.ema,
                single_cls=opt.single_cls,
                dataloader=testloader,
                save_dir=save_dir,
                verbose=nc < 50 and final_epoch,
                plots=final_epoch,
                compute_loss=compute_student_loss,
                is_coco=False,
            )

        lr = [x["lr"] for x in optimizer.param_groups]
        with open(results_file, "a") as f:
            f.write(("%10.4g" * len(mloss)) % tuple(mloss.tolist()) + ("%10.4g" * 7) % results + ("%10.4g" * 3) % tuple(lr[:3]) + "\n")
        denom = max(cold_stats_steps, 1)
        with open(cold_stats_file, "a") as f:
            values = [(cold_stats_epoch[name] / denom).item() for name in COLD_STAT_FIELDS]
            f.write(str(epoch) + "," + ",".join(f"{v:.8g}" for v in values) + "\n")
        fi = fitness(np.array(results).reshape(1, -1))
        if fi > best_fitness:
            best_fitness = fi
        if (not opt.nosave) or final_epoch:
            ckpt = {
                "epoch": epoch,
                "best_fitness": best_fitness,
                "training_results": results_file.read_text(),
                "model": deepcopy(student).half(),
                "ema": deepcopy(ema.ema).half(),
                "updates": ema.updates,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            if opt.save_period > 0 and (epoch + 1) % opt.save_period == 0:
                torch.save(ckpt, wdir / f"epoch{epoch + 1}.pt")
            del ckpt

    logger.info("%g epochs completed in %.3f hours.", opt.epochs, (time.time() - t0) / 3600)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
