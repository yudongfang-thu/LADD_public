#!/usr/bin/env python3
"""
HalluciDet Training Script
Following WACV 2024 paper training protocol
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from copy import copy
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

# Add paths (match other training scripts)
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'shared'))
sys.path.insert(0, str(REPO_ROOT / 'shared' / 'yolo'))

from comparison.hallucidet.hallucidet_model import HallucinationNetwork, HalluciDetModel
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.torch_utils import init_seeds, select_device


class HalluciDetLoss(nn.Module):
    """
    HalluciDet Loss following paper Equation 2:
    L_hall(x, b, θ) = L_cls(f_θ(h_θ(x)), c) + λ·L_reg(f_θ(h_θ(x)), b)

    The loss is computed on the frozen detector's predictions
    on hallucinated images, and gradients flow back to hallucination network only.
    """

    def __init__(self, frozen_detector: nn.Module, lambda_reg: float = 1.0):
        super().__init__()
        self.frozen_detector = frozen_detector
        self.lambda_reg = lambda_reg

        # Use YOLO's detection loss for computing L_cls and L_reg
        self.detection_loss = v8DetectionLoss(frozen_detector)

    def forward_from_preds(
        self,
        preds,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute backprop-able HalluciDet loss from frozen-detector predictions."""
        parsed = self.detection_loss.parse_output(preds)
        if isinstance(parsed, dict) and "one2many" in parsed:
            parsed = parsed["one2many"]
        if not isinstance(parsed, dict) or "boxes" not in parsed or "scores" not in parsed or "feats" not in parsed:
            raise RuntimeError(
                "Frozen detector must return YOLO raw predictions with boxes/scores/feats for HalluciDet loss."
            )
        batch_size = parsed["boxes"].shape[0]
        _, loss_vec, loss_items = self.detection_loss.get_assigned_targets_and_loss(parsed, batch)
        box_loss, cls_loss, dfl_loss = loss_vec[0], loss_vec[1], loss_vec[2]
        total_loss = (cls_loss + self.lambda_reg * (box_loss + dfl_loss)) * batch_size
        loss_dict = {
            'total': total_loss.detach(),
            'box': loss_items[0] if len(loss_items) > 0 else torch.tensor(0.0),
            'cls': loss_items[1] if len(loss_items) > 1 else torch.tensor(0.0),
            'dfl': loss_items[2] if len(loss_items) > 2 else torch.tensor(0.0),
        }
        return total_loss, loss_dict

    def forward(
        self,
        hallucinated_images: torch.Tensor,
        batch: dict[str, torch.Tensor],
        batch_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute detection loss on hallucinated images.

        The detector weights are frozen, but the detector forward is deliberately
        tracked by autograd so detection loss can update the hallucination net.
        """
        preds = self.frozen_detector(hallucinated_images)
        return self.forward_from_preds(preds, batch)


class HalluciDetTrainer:
    """Trainer for HalluciDet following paper's training protocol"""

    PRIMARY_METRIC_KEYS = ("metrics/mAP50-95(B)", "metrics/mAP50-95", "map50_95")

    def __init__(
        self,
        model: HalluciDetModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: Dict[str, Any],
        data: dict[str, Any],
        yolo_args,
        device: str = 'cuda'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.data = data
        self.yolo_args = yolo_args
        self.device = device

        # Loss function
        self.criterion = HalluciDetLoss(
            model.rgb_detector,
            lambda_reg=cfg.get('lambda_reg', 1.0)
        ).to(device)

        # Optimizer - ONLY for hallucination network
        self.optimizer = optim.Adam(
            model.hallucination_net.parameters(),
            lr=cfg['lr'],
            betas=(0.9, 0.999),
            weight_decay=cfg.get('weight_decay', 1e-4)
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg['epochs'],
            eta_min=cfg.get('lr_min', 1e-6)
        )

        # Gradient clipping
        self.grad_clip = cfg.get('grad_clip', 10.0)

        # Training state
        self.epoch = 0
        self.start_epoch = 0
        self.best_metric = float("-inf")  # Higher is better for mAP or -val/loss fallback.
        self.best_metric_key = None
        self._warned_metric_fallback = False
        self.save_dir = Path(self.cfg['save_dir'])
        self.results_file = self.save_dir / "results.csv"

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved = {}
        for k, v in batch.items():
            moved[k] = v.to(self.device, non_blocking=self.device.type == "cuda") if isinstance(v, torch.Tensor) else v
        moved["img"] = moved["img"].float()
        if moved["img"].numel() and moved["img"].max() > 1.5:
            moved["img"] = moved["img"] / 255.0
        return moved

    @staticmethod
    def _to_single_channel(images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] == 1:
            return images
        if images.shape[1] != 3:
            raise RuntimeError(f"HalluciDet expects 1 or 3 input channels, got {images.shape[1]}.")
        return 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]

    def _hallucinate(self, images: torch.Tensor) -> torch.Tensor:
        sar = self._to_single_channel(images)
        hallucinated = self.model.hallucination_net(sar)
        return (hallucinated + 1.0) / 2.0

    def train_one_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        # Keep detector frozen
        self.model.rgb_detector.eval()

        total_loss = 0.0
        cls_loss = 0.0
        box_loss = 0.0
        dfl_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(pbar):
            batch = self._move_batch_to_device(batch)
            hallucinated = self._hallucinate(batch["img"])

            # Compute loss
            loss, loss_dict = self.criterion(hallucinated, batch)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.hallucination_net.parameters(),
                    self.grad_clip
                )

            self.optimizer.step()

            # Accumulate losses
            total_loss += loss_dict['total'].item()
            cls_loss += loss_dict['cls'].item()
            box_loss += loss_dict['box'].item()
            dfl_loss += loss_dict['dfl'].item()
            n_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total'].item():.4f}",
                'cls': f"{loss_dict['cls'].item():.4f}",
                'box': f"{loss_dict['box'].item():.4f}",
            })

        # Average losses
        metrics = {
            'train/loss': total_loss / n_batches,
            'train/cls_loss': cls_loss / n_batches,
            'train/box_loss': box_loss / n_batches,
            'train/dfl_loss': dfl_loss / n_batches,
            'train/lr': self.optimizer.param_groups[0]['lr']
        }

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate loss and mAP on SAR images hallucinated for the frozen RGB detector."""
        self.model.eval()

        total_loss = 0.0
        n_batches = 0
        val_args = copy(self.yolo_args)
        val_args.split = "val"
        val_args.plots = False
        val_args.save_json = False
        val_args.save_txt = False
        val_args.half = False
        validator = DetectionValidator(self.val_loader, save_dir=self.save_dir, args=val_args)
        validator.device = self.device
        validator.data = self.data
        validator.training = False
        validator.stride = getattr(self.model.rgb_detector, "stride", torch.tensor([32], device=self.device))
        validator.init_metrics(self.model)

        for batch in tqdm(self.val_loader, desc="Validating"):
            batch = validator.preprocess(batch)
            hallucinated = self._hallucinate(batch["img"])
            preds = self.model.rgb_detector(hallucinated)
            loss, loss_dict = self.criterion.forward_from_preds(preds, batch)
            total_loss += loss_dict['total'].item()
            n_batches += 1
            infer_preds = preds[0] if isinstance(preds, tuple) else preds
            processed = validator.postprocess(infer_preds)
            validator.update_metrics(processed, batch)

        metrics = {
            'val/loss': total_loss / n_batches
        }
        metrics.update({k: float(v) for k, v in validator.get_stats().items()})

        return metrics

    def _select_primary_metric(self, metrics: Dict[str, Any]) -> tuple[float, str]:
        for key in self.PRIMARY_METRIC_KEYS:
            value = metrics.get(key)
            if value is not None:
                return float(value), key
        if "val/loss" not in metrics:
            raise RuntimeError(
                "Cannot select primary HalluciDet checkpoint metric: no mAP key or val/loss in validation metrics."
            )
        if not self._warned_metric_fallback:
            LOGGER.warning(
                "HalluciDet validation did not report mAP50-95; falling back to -val/loss for best checkpoint."
            )
            self._warned_metric_fallback = True
        return -float(metrics["val/loss"]), "-val/loss"

    def save_checkpoint(self, path: Path, is_best: bool = False, save_epoch: bool = False):
        """Save checkpoint"""
        path.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'epoch': self.epoch,
            'hallucination_net_state': self.model.hallucination_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'best_metric_key': self.best_metric_key,
            'cfg': self.cfg
        }

        torch.save(checkpoint, path / 'last.pt')

        if is_best:
            torch.save(checkpoint, path / 'best.pt')

        if save_epoch:
            torch.save(checkpoint, path / f'epoch_{self.epoch:04d}.pt')

    def load_checkpoint(self, checkpoint_path: str | Path):
        """Resume hallucination-network training from a checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        # PyTorch >=2.6 defaults torch.load(weights_only=True), which rejects
        # the Path objects stored in our own training checkpoint cfg. Resume
        # only loads trusted checkpoints produced by this trainer.
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.hallucination_net.load_state_dict(ckpt['hallucination_net_state'])
        if 'optimizer_state' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scheduler_state' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.best_metric = float(ckpt.get('best_metric', float("-inf")))
        self.best_metric_key = ckpt.get('best_metric_key')
        self.start_epoch = int(ckpt['epoch']) + 1
        LOGGER.info(
            f"Resumed HalluciDet checkpoint {checkpoint_path} at epoch {ckpt['epoch']}; "
            f"continuing from epoch {self.start_epoch}."
        )

    def train(self):
        """Main training loop"""
        save_dir = self.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting training for {self.cfg['epochs']} epochs from epoch {self.start_epoch}...")
        print(f"Save directory: {save_dir}")

        for epoch in range(self.start_epoch, self.cfg['epochs']):
            self.epoch = epoch

            # Train
            train_metrics = self.train_one_epoch()

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step()

            # Log metrics
            all_metrics = {**train_metrics, **val_metrics}
            all_metrics["epoch"] = epoch
            primary_metric, primary_key = self._select_primary_metric(val_metrics)
            all_metrics["primary_metric"] = primary_metric
            all_metrics["primary_metric_key"] = primary_key
            print(f"\nEpoch {epoch}:")
            for k, v in all_metrics.items():
                if isinstance(v, (float, int)):
                    print(f"  {k}: {v:.6f}")
                else:
                    print(f"  {k}: {v}")
            self._append_results(all_metrics)

            # Save checkpoint
            is_best = primary_metric > self.best_metric
            if is_best:
                self.best_metric = primary_metric
                self.best_metric_key = primary_key

            save_period = int(self.cfg.get('save_period', 10))
            save_epoch = save_period > 0 and (epoch + 1) % save_period == 0
            self.save_checkpoint(save_dir, is_best=is_best, save_epoch=save_epoch)

        print(f"\nTraining completed! Best {self.best_metric_key}: {self.best_metric:.6f}")

    def _append_results(self, metrics: Dict[str, Any]) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        fieldnames = ["epoch"] + sorted(k for k in metrics if k != "epoch")
        write_header = not self.results_file.exists()
        if self.results_file.exists():
            with self.results_file.open("r", newline="") as f:
                reader = csv.reader(f)
                existing = next(reader, None)
            if existing:
                fieldnames = list(dict.fromkeys(existing + [k for k in fieldnames if k not in existing]))
        with self.results_file.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({k: metrics.get(k, "") for k in fieldnames})


def build_args(args):
    overrides = {
        "task": "detect",
        "mode": "train",
        "model": args.teacher_weights,
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "rect": False,
        "cache": args.cache,
        "single_cls": False,
        "classes": None,
        "fraction": args.fraction,
        "mosaic": args.mosaic,
        "close_mosaic": 0,
        "multi_scale": 0.0,
        "plots": False,
        "val": True,
        "split": "val",
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "agnostic_nms": False,
        "save_json": False,
        "save_txt": False,
        "save_conf": False,
        "half": False,
        "verbose": False,
    }
    return get_cfg(DEFAULT_CFG, overrides)


def build_hallucidet_dataloaders(args, yolo_args, detector, data):
    gs = max(int(detector.stride.max()), 32)
    train_dataset = build_yolo_dataset(yolo_args, data["train"], args.batch, data, mode="train", rect=False, stride=gs)
    val_dataset = build_yolo_dataset(yolo_args, data["val"], args.batch, data, mode="val", rect=True, stride=gs)
    train_loader = build_dataloader(
        train_dataset,
        batch=args.batch,
        workers=args.workers,
        shuffle=True,
        rank=-1,
        seed=args.seed,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch=args.batch,
        workers=args.workers * 2,
        shuffle=False,
        rank=-1,
        seed=args.seed,
    )
    return train_loader, val_loader


def parse_args():
    parser = argparse.ArgumentParser(description='Train HalluciDet')
    parser.add_argument('--data', type=str, required=True, help='SAR data config')
    parser.add_argument('--teacher-data', type=str, required=True, help='RGB data config')
    parser.add_argument('--teacher-weights', type=str, required=True, help='RGB detector weights')
    parser.add_argument('--imgsz', type=int, default=256, help='Image size')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda-reg', type=float, default=1.0, help='Regression loss weight')
    parser.add_argument('--base-channels', type=int, default=64, help='U-Net base channels')
    parser.add_argument('--project', type=str, default='runs_public/hallucidet', help='Save directory')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    parser.add_argument('--device', type=str, default='0', help='CUDA device')
    parser.add_argument('--workers', type=int, default=8, help='Dataloader workers')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic PyTorch/CUDA behavior')
    parser.add_argument('--cache', default=False, help='Ultralytics dataset cache setting')
    parser.add_argument('--fraction', type=float, default=1.0, help='Training data fraction')
    parser.add_argument('--mosaic', type=float, default=0.0, help='Mosaic augmentation probability')
    parser.add_argument('--save-period', type=int, default=10, help='Checkpoint save period')
    parser.add_argument('--resume', type=str, default='', help='Resume from HalluciDet checkpoint path')
    parser.add_argument('--conf', type=float, default=0.001, help='Validation confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7, help='Validation NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=300, help='Validation max detections per image')

    return parser.parse_args()


def main():
    args = parse_args()

    yolo_args = build_args(args)
    device = select_device(args.device)
    init_seeds(args.seed, deterministic=args.deterministic)
    data = check_det_dataset(args.data)

    # Build model
    print("Building HalluciDet model...")
    hallucination_net = HallucinationNetwork(
        in_channels=1,
        out_channels=3,
        base_channels=args.base_channels,
        use_attention=True
    )

    rgb_detector = YOLO(args.teacher_weights).model
    rgb_detector.args = yolo_args
    rgb_detector.names = data["names"]
    if getattr(rgb_detector.model[-1], "nc", data["nc"]) != data["nc"]:
        raise RuntimeError(
            f"RGB detector nc={getattr(rgb_detector.model[-1], 'nc', None)} does not match dataset nc={data['nc']}."
        )
    model = HalluciDetModel(hallucination_net, rgb_detector)
    model.to(device)

    print(f"Hallucination network parameters: {sum(p.numel() for p in hallucination_net.parameters()):,}")
    print(f"RGB detector frozen: {not any(p.requires_grad for p in rgb_detector.parameters())}")

    # Build dataloaders
    print("Building dataloaders...")
    train_loader, val_loader = build_hallucidet_dataloaders(args, yolo_args, rgb_detector, data)

    # Training config
    cfg = {
        'epochs': args.epochs,
        'lr': args.lr,
        'lambda_reg': args.lambda_reg,
        'weight_decay': 1e-4,
        'grad_clip': 10.0,
        'save_dir': Path(args.project) / args.name,
        'save_period': args.save_period,
    }

    # Create trainer
    trainer = HalluciDetTrainer(model, train_loader, val_loader, cfg, data, yolo_args, device)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()


if __name__ == '__main__':
    main()
