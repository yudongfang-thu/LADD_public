#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

import train_cold_v5p0_hbb as cold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether CoLD loc loss sends gradients to the teacher.")
    parser.add_argument("--yolov5-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--teacher-data", type=Path, required=True)
    parser.add_argument("--cfg", default="models/yolov5x.yaml")
    parser.add_argument("--weights", default="yolov5x.pt")
    parser.add_argument("--teacher-weights", default="yolov5x.pt")
    parser.add_argument("--hyp", default="data/hyp.cold_paper.yaml")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cold-terms", choices=("tcld", "ncld", "both"), default="both")
    parser.add_argument("--temperature", type=float, default=20.0)
    parser.add_argument("--alpha-non-target", type=float, default=2.0)
    parser.add_argument("--candidate-topk", type=int, default=1000)
    parser.add_argument("--candidate-min-conf", type=float, default=0.001)
    return parser.parse_args()


def grad_norm(model: torch.nn.Module) -> float:
    total = torch.zeros((), device=next(model.parameters()).device)
    for p in model.parameters():
        if p.grad is not None:
            total = total + p.grad.detach().float().pow(2).sum()
    return float(total.sqrt().cpu())


def clear_grads(*models: torch.nn.Module) -> None:
    for model in models:
        for p in model.parameters():
            p.grad = None


def main() -> None:
    args = parse_args()
    os.chdir(args.yolov5_root)
    cold.setup_yolov5_path(args.yolov5_root)

    from utils.datasets import LoadImagesAndLabels
    from utils.general import check_dataset, check_img_size, init_seeds
    from utils.loss import ComputeLoss
    from utils.torch_utils import select_device

    init_seeds(2)
    device = select_device(args.device, batch_size=args.batch_size)
    with open(args.hyp) as f:
        hyp = yaml.safe_load(f)
    with open(args.data) as f:
        data_dict = yaml.safe_load(f)
    with open(args.teacher_data) as f:
        teacher_data_dict = yaml.safe_load(f)
    check_dataset(data_dict)
    nc = int(data_dict["nc"])

    student = cold.load_model(args.weights, args.cfg, nc, hyp, device)
    teacher = cold.load_model(args.teacher_weights, args.cfg, nc, hyp, device)
    student.train()
    teacher.train()
    gs = max(int(student.stride.max()), 32)
    imgsz = check_img_size(args.img_size, gs)
    nl = student.model[-1].nl
    hyp["box"] *= 3.0 / nl
    hyp["cls"] *= nc / 80.0 * 3.0 / nl
    hyp["obj"] *= (imgsz / 640) ** 2 * 3.0 / nl

    paired_cls = cold.build_paired_dataset_class(LoadImagesAndLabels)
    dataset = paired_cls(
        data_dict["train"],
        imgsz,
        args.batch_size,
        augment=True,
        hyp=hyp,
        rect=False,
        cache_images=False,
        single_cls=False,
        stride=gs,
        teacher_train_path=teacher_data_dict["train"],
        prefix="audit: ",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type != "cpu",
        collate_fn=paired_cls.collate_fn,
        drop_last=False,
    )
    imgs, teacher_imgs, targets, *_ = next(iter(loader))
    imgs = imgs.to(device, non_blocking=True).float() / 255.0
    teacher_imgs = teacher_imgs.to(device, non_blocking=True).float() / 255.0
    targets = targets.to(device)
    compute_loss = ComputeLoss(student)

    def run_case(name: str, detach_teacher: bool) -> None:
        clear_grads(student, teacher)
        student_pred = student(imgs)
        teacher_pred = teacher(teacher_imgs)
        if detach_teacher:
            teacher_pred = cold.detach_predictions(teacher_pred)
        _, loc_cold, _, stats = cold.cold_candidate_cpm_iwm_loss(
            student_pred,
            teacher_pred,
            targets,
            compute_loss,
            temperature=args.temperature,
            alpha_non_target=args.alpha_non_target,
            cold_terms=args.cold_terms,
            cold_iwm_mode="none",
            candidate_topk=args.candidate_topk,
            candidate_min_conf=args.candidate_min_conf,
            candidate_iou_weight_floor=0.0,
        )
        loc_cold.backward()
        print(
            f"{name}: loc_cold={float(loc_cold.detach().cpu()):.8f} "
            f"student_grad_norm={grad_norm(student):.8e} "
            f"teacher_grad_norm={grad_norm(teacher):.8e} "
            f"candidate_count={float(stats['candidate_count'].detach().cpu()):.1f} "
            f"tcld_terms={float(stats['tcld_terms'].detach().cpu()):.1f} "
            f"ncld_terms={float(stats['ncld_terms'].detach().cpu()):.1f}"
        )

    run_case("raw_teacher_pred", detach_teacher=False)
    run_case("detached_teacher_pred", detach_teacher=True)


if __name__ == "__main__":
    main()
