#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

MODES = ("det_only_same_trainer", "two_branch_no_kd", "current_full")

import val as validate  # noqa: E402
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


def positive_vectors(preds, indices):
    out = []
    for level, pi in enumerate(preds):
        b, a, gj, gi = indices[level]
        if b.numel():
            out.append(pi[b, a, gj, gi])
    if out:
        return torch.cat(out, 0)
    return preds[0].new_zeros((0, preds[0].shape[-1]))


def cclkd_full_loss(student_preds, teacher_preds, targets, student_loss: ComputeLoss):
    """YOLOv5-adapted CCLKD full loss.

    YOLOv5 has anchor logits rather than YOLO11 DFL logits, so this keeps the
    full signal family but maps it to raw YOLOv5 prediction vectors:
    localization KD, foreground/class KD, relation KD, and class-contrastive KD.
    """

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


def train(hyp, opt, device, callbacks):
    save_dir = Path(opt.save_dir)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    last, best = weights_dir / "last.pt", weights_dir / "best.pt"
    use_teacher = opt.mode in {"two_branch_no_kd", "current_full"}
    use_kd = opt.mode == "current_full"

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
            seed=opt.seed,
        )
        anchor_dataset = dataset
    labels = np.concatenate(dataset.labels, 0)
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
        LOGGER.warning("current_full is under audit and is not a verified CCLKD reproduction.")
    LOGGER.info(
        f"YOLOv5x CCLKD audit mode={opt.mode}, use_teacher={use_teacher}, use_kd={use_kd}, "
        f"batch={opt.batch_size}, imgsz={imgsz}, amp={amp}"
    )
    LOGGER.info(f"Logging results to {colorstr('bold', save_dir)}")

    for epoch in range(opt.epochs):
        callbacks.run("on_train_epoch_start")
        model.train()
        if teacher is not None:
            teacher.train()
        mloss = torch.zeros(10, device=device)
        pbar = tqdm(enumerate(train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
        LOGGER.info(("\n" + "%11s" * 12) % ("Epoch", "GPU_mem", "s_box", "s_obj", "s_cls", "t_box", "t_obj", "t_cls", "kd", "lld", "fld", "ccl"))
        optimizer.zero_grad()
        for i, batch in pbar:
            ni = i + nb * epoch
            if use_teacher:
                imgs, teacher_imgs, targets, paths, _shapes = batch
                teacher_imgs = teacher_imgs.to(device, non_blocking=True).float() / 255
            else:
                imgs, targets, paths, _shapes = batch
                teacher_imgs = None
            imgs = imgs.to(device, non_blocking=True).float() / 255
            targets = targets.to(device)

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
                loss = student_det
                if use_teacher:
                    teacher_preds = teacher(teacher_imgs)
                    teacher_det, teacher_items = compute_teacher_loss(teacher_preds, targets)
                    loss = loss + opt.teacher_det_weight * teacher_det
                    if use_kd:
                        kd_loss, kd_items = cclkd_full_loss(student_preds, teacher_preds, targets, compute_student_loss)
                        kd_scale = opt.kd_weight * min(1.0, epoch / max(float(opt.kd_warmup_epochs), 1.0))
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

            display_items = torch.cat((student_items, teacher_items, kd_loss.detach().view(1), kd_items[[0, 1, 3]]))
            mloss = (mloss * i + display_items) / (i + 1)
            mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
            pbar.set_description(("%11s" * 2 + "%11.4g" * 10) % (f"{epoch}/{opt.epochs - 1}", mem, *mloss))

        scheduler.step()
        ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
        final_epoch = epoch + 1 == opt.epochs
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
            strip_optimizer(f)
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
    parser.add_argument("--save-period", type=int, default=100)
    parser.add_argument("--noautoanchor", action="store_true")
    parser.add_argument("--teacher-det-weight", type=float, default=1.0)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--kd-warmup-epochs", type=int, default=3)
    parser.add_argument("--mode", choices=MODES, default="current_full")
    parser.add_argument("--amp", action="store_true", default=True)
    return parser.parse_args()


def main():
    opt = parse_args()
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
