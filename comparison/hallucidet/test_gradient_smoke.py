#!/usr/bin/env python3
"""
HalluciDet gradient smoke tests.

Default mode is fully offline and checks the generic autograd path through a
frozen detector. Pass --teacher-weights to additionally verify the real YOLO
raw-prediction + v8DetectionLoss path used by train_hallucidet.py.
"""
from __future__ import annotations

import argparse
import sys
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

from comparison.hallucidet.hallucidet_model import HalluciDetModel, HallucinationNetwork
from comparison.hallucidet.train_hallucidet import HalluciDetLoss


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
    hall = HallucinationNetwork(in_channels=1, out_channels=3, base_channels=16, use_attention=True).to(device)
    model = HalluciDetModel(hall, ToyDetector().to(device)).to(device)
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
    hall = HallucinationNetwork(in_channels=1, out_channels=3, base_channels=16, use_attention=True).to(device)
    model = HalluciDetModel(hall, detector).to(device)
    criterion = HalluciDetLoss(model.rgb_detector, lambda_reg=lambda_reg).to(device)
    batch = _synthetic_yolo_batch(batch_size=2, imgsz=imgsz, device=device)
    sar = 0.299 * batch["img"][:, 0:1] + 0.587 * batch["img"][:, 1:2] + 0.114 * batch["img"][:, 2:3]
    hallucinated = (model.hallucination_net(sar) + 1.0) / 2.0
    loss, items = criterion(hallucinated, batch)
    print(f"[yolo] loss={float(loss.detach()):.6f} box={float(items['box']):.6f} cls={float(items['cls']):.6f} dfl={float(items['dfl']):.6f}")
    loss.backward()
    hall_nonzero, hall_total = _grad_summary(model.hallucination_net)
    detector_grads = sum(p.grad is not None for p in model.rgb_detector.parameters())
    print(f"[yolo] hallucination gradients: {hall_nonzero}/{hall_total}")
    print(f"[yolo] detector gradients: {detector_grads}")
    return torch.isfinite(loss).item() and loss.requires_grad and hall_nonzero > 0 and detector_grads == 0


def parse_args():
    parser = argparse.ArgumentParser(description="HalluciDet gradient smoke")
    parser.add_argument("--teacher-weights", default="", help="Optional YOLO RGB teacher weights for real loss smoke")
    parser.add_argument("--device", default="cpu", help="cpu, 0, cuda:0, etc.")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
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
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
