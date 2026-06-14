#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
YOLOV5_DIR = REPO_ROOT / "external" / "yolov5"
if str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

MODES = (
    "det_only_same_trainer",
    "two_branch_no_kd",
    "raw_proxy_full",
    "paper_atkd_only",
    "paper_ccl_only",
    "paper_full",
    "current_full",
)
PAPER_MODES = {"paper_atkd_only", "paper_ccl_only", "paper_full"}
RAW_PROXY_MODES = {"raw_proxy_full", "current_full"}
DIAG_FIELDS = (
    "epoch",
    "mode",
    "student_box_loss",
    "student_obj_loss",
    "student_cls_loss",
    "teacher_box_loss",
    "teacher_obj_loss",
    "teacher_cls_loss",
    "kd_total_loss",
    "lld_loss",
    "fld_loss",
    "rld_loss",
    "ccl_loss",
    "kd_to_student_det_ratio",
    "atkd_weight",
    "ccl_weight",
    "atkd_loss",
    "weighted_atkd_loss",
    "weighted_ccl_loss",
    "kd_scale",
    "weighted_kd_to_student_det_ratio",
    "cop_valid_candidates",
    "cop_positive_candidates",
    "cop_positive_ratio",
    "cop_class0_count",
    "cop_class1_count",
    "cop_class2_count",
    "neg_candidates_mean",
    "ccl_pos_sim",
    "ccl_neg_sim",
    "ccl_margin",
    "ccl_valid_classes",
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "feature_capture_ok",
    "student_feature_levels",
    "teacher_feature_levels",
    "nan_or_inf_detected",
)

import val as validate  # noqa: E402
from cclkd_yolov5_loss import YoloV5FeatureCapture, cclkd_paper_loss, positive_vectors  # noqa: E402
from models.experimental import attempt_load  # noqa: E402
from models.yolo import Model  # noqa: E402
from utils.autoanchor import check_anchors  # noqa: E402
from utils.callbacks import Callbacks  # noqa: E402
from utils.dataloaders import LoadImagesAndLabels, create_dataloader, seed_worker  # noqa: E402
from utils.downloads import attempt_download  # noqa: E402
from utils.general import (  # noqa: E402
    LOGGER,
    TQDM_BAR_FORMAT,
    check_dataset,
    check_img_size,
    check_suffix,
    colorstr,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    methods,
    one_cycle,
    strip_optimizer,
    yaml_save,
)
from utils.loggers import Loggers  # noqa: E402
from utils.loss import ComputeLoss  # noqa: E402
from utils.metrics import fitness  # noqa: E402
from utils.torch_utils import (  # noqa: E402
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_optimizer,
)


def effective_mode(mode: str) -> str:
    return "raw_proxy_full" if mode == "current_full" else mode


def default_paper_weights(mode: str) -> tuple[float, float]:
    if mode == "paper_atkd_only":
        return 1.0, 0.0
    if mode == "paper_ccl_only":
        return 0.0, 1.0
    if mode == "paper_full":
        return 1.0, 1.0
    return 0.0, 0.0


class PairedYoloV5Dataset(torch.utils.data.Dataset):
    """Return SAR and RGB images with identical YOLOv5 random augmentations."""

    def __init__(
        self,
        sar_path,
        rgb_path,
        imgsz,
        batch_size,
        stride,
        hyp,
        augment=True,
        workers_prefix="train: ",
    ):
        self.sar = LoadImagesAndLabels(
            sar_path,
            imgsz,
            batch_size,
            augment=augment,
            hyp=hyp,
            rect=False,
            cache_images=False,
            single_cls=False,
            stride=int(stride),
            pad=0.0,
            image_weights=False,
            prefix=workers_prefix,
        )
        self.rgb = LoadImagesAndLabels(
            rgb_path,
            imgsz,
            batch_size,
            augment=augment,
            hyp=hyp,
            rect=False,
            cache_images=False,
            single_cls=False,
            stride=int(stride),
            pad=0.0,
            image_weights=False,
            prefix="teacher: ",
        )
        if len(self.sar) != len(self.rgb):
            raise RuntimeError(f"Paired dataset length mismatch: SAR={len(self.sar)} RGB={len(self.rgb)}")
        sar_names = [Path(p).name for p in self.sar.im_files]
        rgb_names = [Path(p).name for p in self.rgb.im_files]
        if sar_names != rgb_names:
            for i, (a, b) in enumerate(zip(sar_names, rgb_names)):
                if a != b:
                    raise RuntimeError(f"Paired filename mismatch at index {i}: SAR={a} RGB={b}")

        self.labels = self.sar.labels
        self.n = len(self.sar)
        self.indices = self.sar.indices

    def __len__(self):
        return len(self.sar)

    def __getitem__(self, index):
        py_state = random.getstate()
        np_state = np.random.get_state()
        sar_img, labels, path, shapes = self.sar[index]
        random.setstate(py_state)
        np.random.set_state(np_state)
        rgb_img, _, _, _ = self.rgb[index]
        return sar_img, rgb_img, labels, path, shapes

    @staticmethod
    def collate_fn(batch):
        sar, rgb, label, path, shapes = zip(*batch)
        for i, lb in enumerate(label):
            lb[:, 0] = i
        return torch.stack(sar, 0), torch.stack(rgb, 0), torch.cat(label, 0), path, shapes


def create_paired_dataloader(sar_path, rgb_path, imgsz, batch_size, stride, hyp, workers, shuffle=True):
    dataset = PairedYoloV5Dataset(sar_path, rgb_path, imgsz, batch_size, stride, hyp=hyp, augment=True)
    nw = min([os.cpu_count() // max(torch.cuda.device_count(), 1), batch_size if batch_size > 1 else 0, workers])
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=True,
        collate_fn=PairedYoloV5Dataset.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader, dataset


def load_yolov5_model(weights: str, cfg: str, nc: int, hyp: dict, device: torch.device):
    check_suffix(weights, ".pt")
    with torch.no_grad():
        weights = attempt_download(weights)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)
    csd = ckpt["model"].float().state_dict()
    csd = intersect_dicts(csd, model.state_dict(), exclude=[])
    model.load_state_dict(csd, strict=False)
    LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")
    return model


def raw_proxy_full_loss(student_preds, teacher_preds, targets, student_loss: ComputeLoss):
    """Legacy raw YOLOv5 head-vector proxy kept only for regression snapshots."""

    with torch.no_grad():
        tcls, _tbox, indices, _anchors = student_loss.build_targets(student_preds, targets)

    s_pos = positive_vectors(student_preds, indices)
    t_pos = positive_vectors(teacher_preds, indices).detach()
    if s_pos.numel() == 0:
        zero = student_preds[0].new_zeros(())
        return zero, torch.stack((zero, zero, zero, zero))

    lld = F.mse_loss(s_pos[:, :4], t_pos[:, :4])
    fld = F.kl_div(
        F.log_softmax(s_pos[:, 4:], dim=-1),
        F.softmax(t_pos[:, 4:], dim=-1),
        reduction="batchmean",
    )

    s_rel = F.normalize(s_pos[:, 4:], dim=-1, eps=1e-6)
    t_rel = F.normalize(t_pos[:, 4:], dim=-1, eps=1e-6)
    if s_rel.shape[0] > 1:
        rld = F.mse_loss(s_rel @ s_rel.T, t_rel @ t_rel.T)
    else:
        rld = s_pos.new_zeros(())

    labels = torch.cat(tcls, 0) if tcls else torch.empty(0, device=s_pos.device, dtype=torch.long)
    ccl = s_pos.new_zeros(())
    used = 0
    if labels.numel() == s_pos.shape[0] and labels.unique().numel() > 1:
        s_norm = F.normalize(s_pos[:, 4:], dim=-1, eps=1e-6)
        t_norm = F.normalize(t_pos[:, 4:], dim=-1, eps=1e-6)
        for cls in labels.unique():
            pos = torch.where(labels == cls)[0]
            neg = torch.where(labels != cls)[0]
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            n = min(pos.numel(), neg.numel(), 256)
            pos = pos[torch.randperm(pos.numel(), device=pos.device)[:n]]
            neg = neg[torch.randperm(neg.numel(), device=neg.device)[:n]]
            pos_sim = (s_norm[pos] * t_norm[pos]).sum(-1) / 0.1
            neg_sim = (s_norm[neg] * t_norm[neg]).sum(-1) / 0.1
            ccl = ccl - F.log_softmax(torch.stack((pos_sim, neg_sim), dim=-1), dim=-1)[:, 0].mean()
            used += 1
        if used:
            ccl = ccl / used

    total = lld + fld + rld + ccl
    return total, torch.stack((lld.detach(), fld.detach(), rld.detach(), ccl.detach()))


def write_diagnostics_csv(path: Path, row: dict):
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DIAG_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in DIAG_FIELDS})


def update_diag_sums(diag_sums: dict[str, float], diag: dict, student_det, kd_loss):
    for key in DIAG_FIELDS:
        if key in {"epoch", "mode"}:
            continue
        if key == "kd_to_student_det_ratio":
            value = float((kd_loss.detach() / student_det.detach().abs().clamp_min(1e-12)).item())
        elif key == "weighted_kd_to_student_det_ratio":
            kd_scale = float(diag.get("kd_scale", 0.0))
            value = float((kd_scale * kd_loss.detach() / student_det.detach().abs().clamp_min(1e-12)).item())
        elif key in diag:
            value = diag[key]
        else:
            continue
        try:
            if key == "nan_or_inf_detected":
                diag_sums[key] = max(diag_sums.get(key, 0.0), float(value))
            else:
                diag_sums[key] = diag_sums.get(key, 0.0) + float(value)
        except (TypeError, ValueError):
            continue


def train(hyp, opt, device, callbacks):
    save_dir = Path(opt.save_dir)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    last, best = weights_dir / "last.pt", weights_dir / "best.pt"
    mode = effective_mode(opt.mode)
    use_teacher = mode != "det_only_same_trainer"
    use_kd = mode in PAPER_MODES or mode in RAW_PROXY_MODES
    use_paper_kd = mode in PAPER_MODES
    if mode == "raw_proxy_full" and not opt.allow_raw_proxy:
        raise RuntimeError(
            "raw_proxy_full/current_full is a legacy regression mode and is blocked by default. "
            "Use --allow-raw-proxy only for historical debugging. Use paper_full for CCLKD reproduction."
        )
    if use_paper_kd and torch.are_deterministic_algorithms_enabled():
        LOGGER.warning(
            "paper_* CCLKD uses grid_sample for FLD; switching deterministic algorithms to warn_only "
            "because CUDA grid_sample backward has no deterministic implementation in this PyTorch build."
        )
        torch.use_deterministic_algorithms(True, warn_only=True)
    default_atkd_weight, default_ccl_weight = default_paper_weights(mode)
    atkd_weight = default_atkd_weight if opt.atkd_weight is None else float(opt.atkd_weight)
    ccl_weight = default_ccl_weight if opt.ccl_weight is None else float(opt.ccl_weight)

    with open(hyp, errors="ignore") as f:
        hyp = yaml.safe_load(f)
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    yaml_save(save_dir / "hyp.yaml", hyp)
    yaml_save(save_dir / "opt.yaml", vars(opt))

    data_dict = check_dataset(opt.data)
    train_path, val_path = data_dict["train"], data_dict["val"]
    teacher_train_path = None
    if use_teacher:
        teacher_dict = check_dataset(opt.teacher_data)
        teacher_train_path = teacher_dict["train"]
    nc = int(data_dict["nc"])
    names = data_dict["names"]

    model = load_yolov5_model(opt.weights, opt.cfg, nc, hyp, device)
    teacher = None
    if use_teacher:
        teacher = load_yolov5_model(opt.teacher_weights, opt.cfg, nc, hyp, device)
        for p in teacher.parameters():
            p.requires_grad_(True)
    amp = bool(opt.amp)

    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)
    if use_teacher:
        train_loader, dataset = create_paired_dataloader(
            train_path, teacher_train_path, imgsz, opt.batch_size, gs, hyp, opt.workers, shuffle=True
        )
        anchor_dataset = dataset.sar
    else:
        train_loader, dataset = create_dataloader(
            train_path,
            imgsz,
            opt.batch_size,
            gs,
            False,
            hyp=hyp,
            augment=True,
            cache=False,
            rect=False,
            rank=-1,
            workers=opt.workers,
            image_weights=False,
            quad=False,
            prefix=colorstr("train: "),
            shuffle=True,
        )
        anchor_dataset = dataset
    labels = np.concatenate(dataset.labels, 0)
    # Match YOLOv5 train.py loss-gain scaling exactly once before ComputeLoss is created.
    nl = de_parallel(model).model[-1].nl
    hyp["box"] *= 3 / nl
    hyp["cls"] *= nc / 80 * 3 / nl
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc = nc
    model.hyp = hyp
    model.names = names
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc
    if teacher is not None:
        teacher.nc = nc
        teacher.hyp = hyp
        teacher.names = names
        teacher.class_weights = model.class_weights

    val_loader = create_dataloader(
        val_path,
        imgsz,
        opt.batch_size * 2,
        gs,
        False,
        hyp=hyp,
        cache=False,
        rect=True,
        rank=-1,
        workers=opt.workers * 2,
        pad=0.5,
        prefix=colorstr("val: "),
    )[0]
    if not opt.noautoanchor:
        check_anchors(anchor_dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)
        model.half().float()
        if teacher is not None:
            teacher.half().float()

    nbs = 64
    accumulate = max(round(nbs / opt.batch_size), 1)
    hyp["weight_decay"] *= opt.batch_size * accumulate / nbs
    optim_model = nn.ModuleList([model, teacher]) if teacher is not None else model
    optimizer = smart_optimizer(optim_model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])
    lf = one_cycle(1, hyp["lrf"], opt.epochs) if opt.cos_lr else lambda x: (1 - x / opt.epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    ema = ModelEMA(model)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    compute_student_loss = ComputeLoss(model)
    compute_teacher_loss = ComputeLoss(teacher) if teacher is not None else None
    student_feature_capture = YoloV5FeatureCapture(model).install() if use_paper_kd else None
    teacher_feature_capture = YoloV5FeatureCapture(teacher).install() if use_paper_kd and teacher is not None else None
    stopper = EarlyStopping(patience=opt.patience)
    best_fitness, maps = 0.0, np.zeros(nc)
    results = (0, 0, 0, 0, 0, 0, 0)
    nb = len(train_loader)
    nw = max(round(hyp["warmup_epochs"] * nb), 100)
    last_opt_step = -1
    t0 = time.time()

    if use_kd and opt.kd_warmup_epochs:
        LOGGER.info(f"Using KD warmup: {opt.kd_warmup_epochs} epochs")
    if opt.mode == "current_full":
        LOGGER.warning("current_full is legacy raw_proxy_full and is not a verified CCLKD implementation.")
    if mode == "raw_proxy_full":
        LOGGER.warning("raw_proxy_full uses the old YOLOv5 head-vector proxy; use paper_full for CCLKD-style audits.")
    if opt.skip_val:
        LOGGER.warning("skip_val=True: validation is disabled for smoke-only execution.")
    LOGGER.info(
        f"YOLOv5x CCLKD audit mode={opt.mode}, effective_mode={mode}, use_teacher={use_teacher}, "
        f"use_kd={use_kd}, batch={opt.batch_size}, imgsz={imgsz}, amp={amp}, "
        f"max_train_batches={opt.max_train_batches}, skip_val={opt.skip_val}, "
        f"atkd_weight={atkd_weight}, ccl_weight={ccl_weight}, kd_weight={opt.kd_weight}"
    )
    LOGGER.info(f"Logging results to {colorstr('bold', save_dir)}")

    for epoch in range(opt.epochs):
        callbacks.run("on_train_epoch_start")
        model.train()
        if teacher is not None:
            teacher.train()
        mloss = torch.zeros(11, device=device)
        epoch_nb = min(nb, opt.max_train_batches) if opt.max_train_batches > 0 else nb
        pbar = tqdm(enumerate(train_loader), total=epoch_nb, bar_format=TQDM_BAR_FORMAT)
        LOGGER.info(
            ("\n" + "%11s" * 13)
            % ("Epoch", "GPU_mem", "s_box", "s_obj", "s_cls", "t_box", "t_obj", "t_cls", "kd", "lld", "fld", "rld", "ccl")
        )
        optimizer.zero_grad()
        diag_sums: dict[str, float] = {}
        diag_count = 0
        for i, batch in pbar:
            if opt.max_train_batches > 0 and i >= opt.max_train_batches:
                break
            ni = i + nb * epoch
            if use_teacher:
                imgs, teacher_imgs, targets, paths, _shapes = batch
                teacher_imgs = teacher_imgs.to(device, non_blocking=True).float() / 255
            else:
                imgs, targets, paths, _shapes = batch
                teacher_imgs = None
            imgs = imgs.to(device, non_blocking=True).float() / 255
            targets = targets.to(device)

            if student_feature_capture is not None:
                student_feature_capture.clear()
            if teacher_feature_capture is not None:
                teacher_feature_capture.clear()

            if ni <= nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / opt.batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            with torch.cuda.amp.autocast(amp):
                student_preds = model(imgs)
                student_det, student_items = compute_student_loss(student_preds, targets)
                teacher_items = torch.zeros(3, device=device)
                kd_loss = student_det.new_zeros(())
                kd_items = torch.zeros(4, device=device)
                kd_diag = {}
                kd_scale = 0.0
                loss = student_det
                if use_teacher:
                    teacher_preds = teacher(teacher_imgs)
                    teacher_det, teacher_items = compute_teacher_loss(teacher_preds, targets)
                    loss = loss + opt.teacher_det_weight * teacher_det
                    if use_kd:
                        if use_paper_kd:
                            kd_loss, kd_items, kd_diag = cclkd_paper_loss(
                                student_preds=student_preds,
                                teacher_preds=teacher_preds,
                                targets=targets,
                                student_loss=compute_student_loss,
                                student_features=student_feature_capture.features,
                                teacher_features=teacher_feature_capture.features,
                                mode=mode,
                                nc=nc,
                                atkd_weight=atkd_weight,
                                ccl_weight=ccl_weight,
                                ccl_source=opt.cclkd_ccl_source,
                                ccl_pair_mode=opt.cclkd_ccl_pair_mode,
                            )
                        else:
                            kd_loss, kd_items = raw_proxy_full_loss(student_preds, teacher_preds, targets, compute_student_loss)
                        kd_scale = opt.kd_weight * min(1.0, epoch / max(float(opt.kd_warmup_epochs), 1.0))
                        kd_diag["kd_scale"] = float(kd_scale)
                        loss = loss + kd_scale * kd_loss

            scaler.scale(loss).backward()
            if ni - last_opt_step >= accumulate:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                if teacher is not None:
                    torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)
                last_opt_step = ni

            display_items = torch.cat((student_items, teacher_items, kd_loss.detach().view(1), kd_items))
            mloss = (mloss * i + display_items) / (i + 1)
            kd_diag["nan_or_inf_detected"] = max(
                float(kd_diag.get("nan_or_inf_detected", 0.0)),
                float(
                    (~torch.isfinite(torch.cat((loss.detach().reshape(-1), kd_loss.detach().reshape(-1)))))
                    .any()
                    .item()
                ),
            )
            kd_diag.setdefault("kd_scale", float(kd_scale))
            update_diag_sums(diag_sums, kd_diag, student_det, kd_loss)
            diag_count += 1
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
            pbar.set_description(("%11s" * 2 + "%11.4g" * 11) % (f"{epoch}/{opt.epochs - 1}", mem, *mloss))

        scheduler.step()
        ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
        final_epoch = epoch + 1 == opt.epochs
        if opt.skip_val:
            results, maps = (0, 0, 0, 0, 0, 0, 0), np.zeros(nc)
        else:
            results, maps, _ = validate.run(
                data_dict,
                batch_size=opt.batch_size * 2,
                imgsz=imgsz,
                half=amp,
                model=ema.ema,
                single_cls=False,
                dataloader=val_loader,
                save_dir=save_dir,
                plots=False,
                callbacks=callbacks,
                compute_loss=compute_student_loss,
            )
        fi = fitness(np.array(results).reshape(1, -1))
        if fi > best_fitness:
            best_fitness = fi
        stop = stopper(epoch=epoch, fitness=fi)
        log_vals = list(mloss[:3]) + list(results) + [g["lr"] for g in optimizer.param_groups]
        callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi)

        avg_diag = {
            k: (v if k == "nan_or_inf_detected" else v / max(diag_count, 1))
            for k, v in diag_sums.items()
        }
        diag_row = {
            "epoch": epoch,
            "mode": mode,
            "student_box_loss": float(mloss[0].item()),
            "student_obj_loss": float(mloss[1].item()),
            "student_cls_loss": float(mloss[2].item()),
            "teacher_box_loss": float(mloss[3].item()),
            "teacher_obj_loss": float(mloss[4].item()),
            "teacher_cls_loss": float(mloss[5].item()),
            "kd_total_loss": float(mloss[6].item()),
            "lld_loss": float(mloss[7].item()),
            "fld_loss": float(mloss[8].item()),
            "rld_loss": float(mloss[9].item()),
            "ccl_loss": float(mloss[10].item()),
        }
        diag_row.update(avg_diag)
        write_diagnostics_csv(save_dir / "cclkd_yolov5_diagnostics.csv", diag_row)
        LOGGER.info(
            "diagnostics: mode=%s cop_positive_ratio=%.4g kd_to_student_det_ratio=%.4g "
            "lld=%.4g fld=%.4g rld=%.4g ccl=%.4g ccl_margin=%.4g",
            mode,
            float(diag_row.get("cop_positive_ratio", 0.0) or 0.0),
            float(diag_row.get("kd_to_student_det_ratio", 0.0) or 0.0),
            float(diag_row.get("lld_loss", 0.0) or 0.0),
            float(diag_row.get("fld_loss", 0.0) or 0.0),
            float(diag_row.get("rld_loss", 0.0) or 0.0),
            float(diag_row.get("ccl_loss", 0.0) or 0.0),
            float(diag_row.get("ccl_margin", 0.0) or 0.0),
        )

        if student_feature_capture is not None:
            student_feature_capture.clear()
        if teacher_feature_capture is not None:
            teacher_feature_capture.clear()
        ckpt = {
            "epoch": epoch,
            "best_fitness": best_fitness,
            "model": deepcopy(de_parallel(model)).half(),
            "ema": deepcopy(ema.ema).half(),
            "updates": ema.updates,
            "optimizer": optimizer.state_dict(),
            "opt": vars(opt),
            "date": datetime.now().isoformat(),
        }
        if teacher is not None:
            ckpt["teacher"] = deepcopy(de_parallel(teacher)).half()
        torch.save(ckpt, last)
        if best_fitness == fi:
            torch.save(ckpt, best)
        if opt.save_period > 0 and epoch % opt.save_period == 0:
            torch.save(ckpt, weights_dir / f"epoch{epoch}.pt")
        del ckpt
        if stop:
            break

    LOGGER.info(f"\n{epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
    for f in (last, best):
        if f.exists():
            try:
                strip_optimizer(f)
            except Exception as exc:
                LOGGER.warning(f"Could not strip optimizer from {f}: {exc}")
    torch.cuda.empty_cache()
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(YOLOV5_DIR / "yolov5x.pt"))
    parser.add_argument("--teacher-weights", default=str(YOLOV5_DIR / "yolov5x.pt"))
    parser.add_argument("--cfg", default="")
    parser.add_argument("--data", required=True)
    parser.add_argument("--teacher-data", required=True)
    parser.add_argument("--hyp", required=True)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--imgsz", "--img", type=int, default=256)
    parser.add_argument("--device", default="1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--noplots", action="store_true")
    parser.add_argument("--evolve", action="store_true")
    parser.add_argument("--resume", default=False)
    parser.add_argument("--upload_dataset", default=False)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--optimizer", default="SGD", choices=("SGD", "Adam", "AdamW"))
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--patience", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--save-period", type=int, default=100)
    parser.add_argument("--noautoanchor", action="store_true")
    parser.add_argument("--teacher-det-weight", type=float, default=1.0)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--kd-warmup-epochs", type=int, default=3)
    parser.add_argument("--atkd-weight", type=float, default=None)
    parser.add_argument("--ccl-weight", type=float, default=None)
    parser.add_argument(
        "--cclkd-ccl-source",
        choices=("box_proxy", "box_class", "roi_feature"),
        default="box_class",
        help="CCL candidate representation. box_proxy reproduces the legacy xywh-only path; "
        "box_class adds objectness/class-j conditioning; roi_feature uses box-sampled Detect-input features.",
    )
    parser.add_argument(
        "--cclkd-ccl-pair-mode",
        choices=("paper_pair", "anchor_teacher_neg"),
        default="anchor_teacher_neg",
        help="paper_pair compares teacher-student positive sets against teacher-student non-target sets; "
        "anchor_teacher_neg pushes student positives away from teacher negatives.",
    )
    parser.add_argument("--mode", choices=MODES, default="paper_full")
    parser.add_argument("--allow-raw-proxy", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=-1)
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--amp", action="store_true", default=True)
    return parser.parse_args()


def main():
    opt = parse_args()
    if effective_mode(opt.mode) == "raw_proxy_full" and not opt.allow_raw_proxy:
        raise RuntimeError(
            "raw_proxy_full/current_full is a legacy regression mode and is blocked by default. "
            "Use --allow-raw-proxy only for historical debugging. Use paper_full for CCLKD reproduction."
        )
    init_seeds(opt.seed + 1, deterministic=True)
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    opt.save_dir = str(save_dir)
    callbacks = Callbacks()
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    if True:
        loggers = Loggers(save_dir, opt.weights, opt, {}, LOGGER)
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))
    device = select_device(opt.device, batch_size=opt.batch_size)
    train(opt.hyp, opt, device, callbacks)


if __name__ == "__main__":
    main()
