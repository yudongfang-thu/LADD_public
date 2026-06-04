#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from train_cli_overrides import add_common_detector_train_overrides, collect_common_detector_train_overrides
from train_path_checks import require_existing_file


REPO_ROOT = Path(__file__).resolve().parents[1]
YOLO_ROOT = REPO_ROOT / "yolo"
SRC_ROOT = REPO_ROOT / "src"
for root in (YOLO_ROOT, SRC_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from cold_kd import CoLDOBBTrainer  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO11-OBB adapted CoLD comparison baseline (frozen RGB teacher -> SAR student)."
    )
    parser.add_argument("--model", default="yolo11n-obb.pt", help="Student model weights or YAML.")
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "configs" / "datasets" / "sixiang_sar_obb.yaml",
        help="Student dataset YAML.",
    )
    parser.add_argument(
        "--teacher-data",
        type=Path,
        default=REPO_ROOT / "configs" / "datasets" / "sixiang_rgb_obb.yaml",
        help="Teacher dataset YAML (paired RGB images).",
    )
    parser.add_argument(
        "--teacher-weights",
        default=str(REPO_ROOT / "runs" / "yolo11_obb" / "rgb_yolo11n_obb_clean_e300_b64_gpu4" / "weights" / "best.pt"),
        help="Frozen RGB teacher checkpoint.",
    )
    parser.add_argument("--lambda-kd", type=float, default=1.0, help="Global KD weight.")
    parser.add_argument("--lambda-cls-cold", type=float, default=1.0, help="Class-partition CoLD term weight.")
    parser.add_argument("--lambda-loc-cold", type=float, default=1.0, help="IoU-weighted localization CoLD term weight.")
    parser.add_argument("--alpha-non-target", type=float, default=2.0, help="CoLD non-target weight alpha.")
    parser.add_argument("--temperature", type=float, default=20.0, help="Softmax temperature tau.")
    parser.add_argument("--kd-region", choices=("positive", "all"), default="positive")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--cache", default=False)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs" / "cold_kd")
    parser.add_argument("--name", default="sar_cold_kd_yolo11n_obb")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--fraction", type=float, default=1.0)
    add_common_detector_train_overrides(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher_weights = require_existing_file(args.teacher_weights, "--teacher-weights")
    model = YOLO(args.model)
    train_kwargs = dict(
        trainer=CoLDOBBTrainer,
        data=str(args.data.resolve()),
        teacher_data=str(args.teacher_data.resolve()),
        teacher_weights=teacher_weights,
        lambda_kd=args.lambda_kd,
        lambda_cls_cold=args.lambda_cls_cold,
        lambda_loc_cold=args.lambda_loc_cold,
        alpha_non_target=args.alpha_non_target,
        temperature=args.temperature,
        kd_region=args.kd_region,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        cache=args.cache,
        patience=args.patience,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=args.exist_ok,
        fraction=args.fraction,
    )
    train_kwargs.update(collect_common_detector_train_overrides(args))
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
