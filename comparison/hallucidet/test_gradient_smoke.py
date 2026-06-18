#!/usr/bin/env python3
"""
HalluciDet gradient smoke tests.

Default mode is fully offline and checks the generic autograd path through a
frozen detector. Pass --teacher-weights to additionally verify the real YOLO
raw-prediction + v8DetectionLoss path used by train_hallucidet.py.
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "shared"))
sys.path.insert(0, str(REPO_ROOT / "shared" / "yolo"))

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG

from comparison.hallucidet.hallucidet_model import HalluciDetModel
from comparison.hallucidet.train_hallucidet import HalluciDetLoss, HalluciDetTrainer


class ToyDetector(nn.Module):
    """Tiny frozen detector surrogate for offline autograd smoke."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 4),
        )

    def forward(self, x):
        return self.net(x)


class ToyHallucinationNet(nn.Module):
    """Small local surrogate for offline gradient tests; not a production method."""

    input_channels = 3
    outputs_unit_range = True

    def __init__(self, channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def _grad_summary(module: nn.Module) -> tuple[int, int]:
    nonzero = 0
    total = 0
    for p in module.parameters():
        if p.grad is not None:
            total += 1
            if p.grad.abs().sum() > 1e-8:
                nonzero += 1
    return nonzero, total


def run_offline_smoke(device: torch.device) -> bool:
    print("[offline] Building toy HalluciDet chain")
    hall = ToyHallucinationNet().to(device)
    model = HalluciDetModel(hall, ToyDetector().to(device), hallucination_input_mode="replicate3").to(device)
    sar = torch.randn(2, 1, 128, 128, device=device)
    preds, hallucinated = model(sar, return_hallucinated=True)
    loss = preds.mean() + hallucinated.mean() * 0.0
    loss.backward()
    hall_nonzero, hall_total = _grad_summary(model.hallucination_net)
    detector_grads = sum(p.grad is not None for p in model.rgb_detector.parameters())
    print(f"[offline] hallucination gradients: {hall_nonzero}/{hall_total}")
    print(f"[offline] detector gradients: {detector_grads}")
    return loss.requires_grad and hall_nonzero > 0 and detector_grads == 0


def _synthetic_yolo_batch(batch_size: int, imgsz: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(batch_size, 3, imgsz, imgsz, device=device),
        "batch_idx": torch.arange(batch_size, device=device, dtype=torch.float32),
        "cls": torch.zeros(batch_size, 1, device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]], device=device).repeat(batch_size, 1),
    }


def run_yolo_loss_smoke(weights: str, device: torch.device, imgsz: int, lambda_reg: float) -> bool:
    print("[yolo] Loading frozen detector:", weights)
    yolo_args = get_cfg(
        DEFAULT_CFG,
        {
            "task": "detect",
            "mode": "train",
            "imgsz": imgsz,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
        },
    )
    detector = YOLO(weights).model.to(device)
    detector.args = yolo_args
    detector.eval()
    hall = ToyHallucinationNet().to(device)
    model = HalluciDetModel(hall, detector, hallucination_input_mode="replicate3").to(device)
    criterion = HalluciDetLoss(model.rgb_detector, lambda_reg=lambda_reg).to(device)
    batch = _synthetic_yolo_batch(batch_size=2, imgsz=imgsz, device=device)
    sar = 0.299 * batch["img"][:, 0:1] + 0.587 * batch["img"][:, 1:2] + 0.114 * batch["img"][:, 2:3]
    hallucinated = model.hallucinate(sar)
    loss, items = criterion(hallucinated, batch)
    print(f"[yolo] loss={float(loss.detach()):.6f} box={float(items['box']):.6f} cls={float(items['cls']):.6f} dfl={float(items['dfl']):.6f}")
    loss.backward()
    hall_nonzero, hall_total = _grad_summary(model.hallucination_net)
    detector_grads = sum(p.grad is not None for p in model.rgb_detector.parameters())
    print(f"[yolo] hallucination gradients: {hall_nonzero}/{hall_total}")
    print(f"[yolo] detector gradients: {detector_grads}")
    return torch.isfinite(loss).item() and loss.requires_grad and hall_nonzero > 0 and detector_grads == 0


class _ResumeSmokeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hallucination_net = ToyHallucinationNet(channels=8)


class _ResumeSmokeTrainer(HalluciDetTrainer):
    """Tiny trainer that reuses HalluciDet checkpoint/result logic without dataset or YOLO dependencies."""

    def __init__(self, save_dir: Path, epochs: int, device: torch.device):
        self.model = _ResumeSmokeModel().to(device)
        self.train_loader = []
        self.val_loader = []
        self.cfg = {
            "epochs": epochs,
            "lr": 1e-3,
            "save_dir": save_dir,
            "save_period": 0,
        }
        self.data = {}
        self.yolo_args = None
        self.device = device
        self.optimizer = torch.optim.SGD(self.model.hallucination_net.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        self.grad_clip = 0.0
        self.epoch = 0
        self.start_epoch = 0
        self.best_metric = float("-inf")
        self.best_metric_key = None
        self._warned_metric_fallback = False
        self.save_dir = Path(save_dir)
        self.results_file = self.save_dir / "results.csv"

    def train_one_epoch(self):
        self.model.train()
        x = torch.rand(1, 3, 32, 32, device=self.device)
        y = self.model.hallucination_net(x)
        loss = y.square().mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {
            "train/loss": float(loss.detach()),
            "train/cls_loss": 0.0,
            "train/box_loss": 0.0,
            "train/dfl_loss": 0.0,
            "train/lr": self.optimizer.param_groups[0]["lr"],
        }

    def validate(self):
        return {
            "val/loss": 1.0 / (self.epoch + 1),
            "metrics/mAP50-95(B)": float(self.epoch),
        }


def run_resume_smoke(device: torch.device, save_dir: str = "") -> bool:
    with tempfile.TemporaryDirectory(prefix="hallucidet_resume_") as tmp:
        root = Path(save_dir) if save_dir else Path(tmp)
        root.mkdir(parents=True, exist_ok=True)
        first = _ResumeSmokeTrainer(root, epochs=1, device=device)
        first.train()
        last = root / "last.pt"
        best = root / "best.pt"
        if not last.exists() or not best.exists():
            print("[resume] missing last.pt or best.pt after first epoch")
            return False
        second = _ResumeSmokeTrainer(root, epochs=2, device=device)
        second.load_checkpoint(last)
        if second.start_epoch != 1:
            print(f"[resume] expected start_epoch=1, got {second.start_epoch}")
            return False
        second.train()
        rows = list(csv.DictReader((root / "results.csv").open()))
        epochs = [int(float(row["epoch"])) for row in rows]
        ckpt = torch.load(root / "last.pt", map_location=device, weights_only=False)
        print(f"[resume] epochs in results.csv: {epochs}")
        print(f"[resume] last.pt epoch: {ckpt['epoch']}")
        return epochs == [0, 1] and int(ckpt["epoch"]) == 1 and best.exists()


def parse_args():
    parser = argparse.ArgumentParser(description="HalluciDet gradient smoke")
    parser.add_argument("--teacher-weights", default="", help="Optional YOLO RGB teacher weights for real loss smoke")
    parser.add_argument("--device", default="cpu", help="cpu, 0, cuda:0, etc.")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--resume-smoke", action="store_true", help="Run lightweight checkpoint resume smoke")
    parser.add_argument("--resume-smoke-dir", default="", help="Optional directory for resume smoke artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    ok = run_offline_smoke(device)
    if args.teacher_weights:
        ok = run_yolo_loss_smoke(args.teacher_weights, device, args.imgsz, args.lambda_reg) and ok
    if args.resume_smoke:
        ok = run_resume_smoke(device, args.resume_smoke_dir) and ok
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
