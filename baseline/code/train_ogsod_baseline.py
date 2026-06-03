#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from train_cli_overrides import add_common_detector_train_overrides, collect_common_detector_train_overrides


REPO_ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = REPO_ROOT / "yolo"
SRC_ROOT = REPO_ROOT / "src"
for root in (YOLO_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from d2ad_obb.baseline_trainer import UnifiedAugOBBTrainer  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train clean OGSOD HBB/OBB single-modality baselines.")
    parser.add_argument("--task", choices=("hbb", "obb"), required=True)
    parser.add_argument("--model", required=True, help="YOLO weights or YAML, e.g. yolo11s.pt or yolo11s-obb.pt.")
    parser.add_argument("--data", type=Path, required=True, help="Prepared OGSOD YAML.")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cache", default=False)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs_public" / "ogsod" / "baselines")
    parser.add_argument("--name", required=True)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument(
        "--disable-albumentations",
        action="store_true",
        help="Use an empty Albumentations transform list, disabling YOLO's default Blur/CLAHE/etc. transforms.",
    )
    add_common_detector_train_overrides(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    train_kwargs = dict(
        data=str(args.data.resolve()),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        cache=args.cache,
        patience=args.patience,
        fraction=args.fraction,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
    )
    train_kwargs.update(collect_common_detector_train_overrides(args))
    if args.disable_albumentations:
        train_kwargs["augmentations"] = []
    if args.task == "obb":
        train_kwargs["trainer"] = UnifiedAugOBBTrainer
    else:
        train_kwargs["task"] = "detect"
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
