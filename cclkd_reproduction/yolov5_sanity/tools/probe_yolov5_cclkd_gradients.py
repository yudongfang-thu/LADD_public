#!/usr/bin/env python3
"""Probe YOLOv5 CCLKD gradient norms and cosines on a few batches.

This is an offline diagnostic helper. It does not modify checkpoints, does not
install hooks into active training jobs, and does not step an optimizer.

Typical use on server 90:

python cclkd_reproduction/yolov5_sanity/tools/probe_yolov5_cclkd_gradients.py \
  --weights <student-or-run-last.pt> \
  --teacher-weights <teacher-or-run-last.pt> \
  --data configs/datasets/ogsod_hbb_sar.yaml \
  --teacher-data configs/datasets/ogsod_hbb_rgb.yaml \
  --hyp cclkd_reproduction/yolov5_sanity/configs/hyp_cold_ogsod.yaml \
  --device 0 --batch-size 8 --max-batches 2 \
  --csv-out /tmp/cclkd_grad_probe_batches.csv \
  --summary-out /tmp/cclkd_grad_probe_summary.csv \
  --md-out /tmp/cclkd_grad_probe_summary.md
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = REPO_ROOT / "cclkd_reproduction" / "yolov5_sanity" / "code"
YOLOV5_DIR = REPO_ROOT / "external" / "yolov5"

COMPONENTS = ("lld", "fld", "rld", "atkd", "ccl", "kd_total")
CSV_FIELDS = [
    "batch",
    "num_targets",
    "loss_student_det",
    "loss_teacher_det",
    "loss_lld",
    "loss_fld",
    "loss_rld",
    "loss_atkd",
    "loss_ccl",
    "loss_kd_total",
    "weighted_atkd_loss",
    "weighted_ccl_loss",
    "grad_norm_det",
    "grad_norm_lld",
    "grad_norm_fld",
    "grad_norm_rld",
    "grad_norm_atkd",
    "grad_norm_ccl",
    "grad_norm_kd_total",
    "cos_det_lld",
    "cos_det_fld",
    "cos_det_rld",
    "cos_det_atkd",
    "cos_det_ccl",
    "cos_det_kd_total",
    "cop_valid_candidates",
    "cop_positive_candidates",
    "cop_positive_ratio",
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "feature_capture_ok",
    "nan_or_inf_detected",
]


def add_import_paths() -> None:
    missing = []
    for path in (CODE_DIR, YOLOV5_DIR):
        if not path.exists():
            missing.append(str(path))
        elif str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if missing:
        raise SystemExit(
            "Missing required local dependency path(s): "
            + ", ".join(missing)
            + ". Run this probe in a clone where external/yolov5 is prepared."
        )


def import_runtime_modules():
    add_import_paths()
    import numpy as np  # noqa: WPS433
    import torch  # noqa: WPS433
    import yaml  # noqa: WPS433
    import cclkd_yolov5_loss as loss_mod  # noqa: WPS433
    import train_yolov5_cclkd_full as train_mod  # noqa: WPS433

    return np, torch, yaml, train_mod, loss_mod


def parse_optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    return float(value)


def default_weights_for_mode(mode: str) -> tuple[float, float]:
    if mode == "paper_atkd_only":
        return 1.0, 0.0
    if mode == "paper_ccl_only":
        return 0.0, 1.0
    if mode == "paper_full":
        return 1.0, 1.0
    raise ValueError(f"Unsupported paper mode for gradient probe: {mode}")


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value_float):
        return "nan"
    return f"{value_float:.10g}"


def mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_float):
            values.append(value_float)
    if not values:
        return None
    return sum(values) / len(values)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in CSV_FIELDS})


def write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["metric", "value"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(summary):
            writer.writerow({"metric": key, "value": fmt(summary[key])})


def write_markdown(path: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# YOLOv5 CCLKD Gradient Probe",
        "",
        "Offline few-batch diagnostic. No optimizer step and no checkpoint write.",
        "",
        "## Inputs",
        "",
        f"- student weights: `{args.weights}`",
        f"- teacher weights: `{args.teacher_weights}`",
        f"- data: `{args.data}`",
        f"- teacher data: `{args.teacher_data}`",
        f"- mode: `{args.mode}`",
        f"- max batches: `{args.max_batches}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key in sorted(summary):
        lines.append(f"| {key} | {fmt(summary[key])} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def checkpoint_source(ckpt: Any, key: str) -> Any:
    if isinstance(ckpt, dict):
        if key in ckpt and ckpt[key] is not None:
            return ckpt[key]
        if "model" in ckpt and ckpt["model"] is not None:
            return ckpt["model"]
    return ckpt


def load_probe_model(train_mod, torch, weights: str, cfg: str, nc: int, hyp: dict, device, ckpt_key: str):
    train_mod.check_suffix(weights, ".pt")
    with torch.no_grad():
        weights = train_mod.attempt_download(weights)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    source = checkpoint_source(ckpt, ckpt_key)
    yaml_cfg = cfg or getattr(source, "yaml", None)
    if yaml_cfg is None:
        raise RuntimeError(f"Could not infer YOLOv5 model yaml from {weights}")
    model = train_mod.Model(yaml_cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)
    if not hasattr(source, "float") or not hasattr(source, "state_dict"):
        raise RuntimeError(f"Checkpoint source {ckpt_key!r} in {weights} is not a YOLOv5 module")
    state = source.float().state_dict()
    state = train_mod.intersect_dicts(state, model.state_dict(), exclude=[])
    model.load_state_dict(state, strict=False)
    train_mod.LOGGER.info(f"Transferred {len(state)}/{len(model.state_dict())} items from {weights} ({ckpt_key})")
    return model


def create_probe_loader(train_mod, torch, sar_path, rgb_path, imgsz, batch_size, stride, hyp, workers, augment):
    dataset = train_mod.PairedYoloV5Dataset(
        sar_path,
        rgb_path,
        imgsz,
        batch_size,
        stride,
        hyp=hyp,
        augment=augment,
        workers_prefix="probe: ",
    )
    workers = min([os.cpu_count() // max(torch.cuda.device_count(), 1), batch_size if batch_size > 1 else 0, workers])
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205)
    loader = train_mod.DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=train_mod.PairedYoloV5Dataset.collate_fn,
        worker_init_fn=train_mod.seed_worker,
        generator=generator,
    )
    return loader, dataset


def prepare_probe(args: argparse.Namespace):
    np, torch, yaml, train_mod, loss_mod = import_runtime_modules()
    train_mod.init_seeds(args.seed + 1, deterministic=True)
    device = train_mod.select_device(args.device, batch_size=args.batch_size)
    with open(args.hyp, errors="ignore") as handle:
        hyp_for_data = yaml.safe_load(handle)
    data_dict = train_mod.check_dataset(args.data)
    teacher_dict = train_mod.check_dataset(args.teacher_data)
    nc = int(data_dict["nc"])
    names = data_dict["names"]

    model = load_probe_model(train_mod, torch, args.weights, args.cfg, nc, hyp_for_data, device, args.student_ckpt_key)
    teacher = load_probe_model(
        train_mod,
        torch,
        args.teacher_weights,
        args.cfg,
        nc,
        hyp_for_data,
        device,
        args.teacher_ckpt_key,
    )
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    stride = max(int(model.stride.max()), 32)
    imgsz = train_mod.check_img_size(args.imgsz, stride, floor=stride * 2)
    loader, dataset = create_probe_loader(
        train_mod,
        torch,
        data_dict["train"],
        teacher_dict["train"],
        imgsz,
        args.batch_size,
        stride,
        hyp_for_data,
        args.workers,
        augment=not args.no_augment,
    )

    hyp = dict(hyp_for_data)
    labels = np.concatenate(dataset.labels, 0)
    nl = train_mod.de_parallel(model).model[-1].nl
    hyp["box"] *= 3 / nl
    hyp["cls"] *= nc / 80 * 3 / nl
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl
    hyp["label_smoothing"] = args.label_smoothing
    model.nc = nc
    model.hyp = hyp
    model.names = names
    model.class_weights = train_mod.labels_to_class_weights(dataset.labels, nc).to(device) * nc
    teacher.nc = nc
    teacher.hyp = hyp
    teacher.names = names
    teacher.class_weights = model.class_weights

    student_loss = train_mod.ComputeLoss(model)
    teacher_loss = train_mod.ComputeLoss(teacher)
    student_capture = loss_mod.YoloV5FeatureCapture(model).install()
    teacher_capture = loss_mod.YoloV5FeatureCapture(teacher).install()
    return torch, train_mod, loss_mod, device, nc, model, teacher, loader, student_loss, teacher_loss, student_capture, teacher_capture


def component_losses(loss_mod, student_preds, teacher_preds, targets, student_loss, student_features, teacher_features, mode, nc, args):
    candidates = loss_mod.collect_yolov5_positive_candidates(student_preds, targets, student_loss)
    teacher_vectors = loss_mod.positive_vectors(teacher_preds, candidates.indices).detach()
    student_feature_levels = len(student_features)
    teacher_feature_levels = len(teacher_features)
    zero = student_preds[0].new_zeros(())
    if candidates.vectors.numel() == 0:
        return {
            "lld": zero,
            "fld": zero,
            "rld": zero,
            "atkd": zero,
            "ccl": zero,
            "kd_total": zero,
        }, {
            "cop_valid_candidates": 0.0,
            "cop_positive_candidates": 0.0,
            "cop_positive_ratio": 0.0,
            "temperature_mean": 0.0,
            "temperature_min": 0.0,
            "temperature_max": 0.0,
            "feature_capture_ok": 0.0,
            "nan_or_inf_detected": 0.0,
        }

    cop = loss_mod.build_teacher_cop(teacher_vectors, candidates.labels, nc)
    valid_mask = cop["valid_mask"]
    positive_mask = cop["positive_mask"]
    student_box_probs = candidates.vectors[:, :4].sigmoid()
    teacher_box_probs = teacher_vectors[:, :4].sigmoid()
    detached_teacher_features = [feature.detach() for feature in teacher_features]
    student_sampled, student_features_ok = loss_mod.sample_box_features(
        student_features,
        candidates.levels,
        candidates.batch_indices,
        candidates.grid_x,
        candidates.grid_y,
        roi_grid_size=args.roi_grid_size,
    )
    teacher_sampled, teacher_features_ok = loss_mod.sample_box_features(
        detached_teacher_features,
        candidates.levels,
        candidates.batch_indices,
        candidates.grid_x,
        candidates.grid_y,
        roi_grid_size=args.roi_grid_size,
    )

    lld = candidates.vectors.new_zeros(())
    fld = candidates.vectors.new_zeros(())
    rld = candidates.vectors.new_zeros(())
    ccl = candidates.vectors.new_zeros(())
    temperatures = []
    ccl_terms = []
    class_weights = []
    safe_labels = candidates.labels.clamp(min=0, max=max(nc - 1, 0))

    for cls in range(nc):
        class_pos = positive_mask & (safe_labels == cls)
        if class_pos.sum() == 0:
            continue
        class_neg = valid_mask & ~class_pos
        temperature = loss_mod.adaptive_temperature_from_teacher_scores(
            cop["teacher_scores"],
            class_pos,
            t_min=args.t_min,
            t_max=args.t_max,
            entropy_scale=args.entropy_scale,
        )
        temperatures.append(temperature.detach())
        lld = lld + loss_mod.spatial_distribution_kl(student_box_probs, teacher_box_probs, class_pos, temperature)
        lld = lld + loss_mod.spatial_distribution_kl(student_box_probs, teacher_box_probs, class_neg, temperature)
        fld = fld + loss_mod.feature_kl(student_sampled, teacher_sampled, class_pos, temperature)
        rld = rld + loss_mod.relationship_loss(student_sampled, teacher_sampled, class_pos, temperature)
        if class_neg.sum() > 0:
            ccl_terms.append(
                loss_mod.contrastive_loss(
                    student_box_probs,
                    teacher_box_probs,
                    class_pos,
                    class_neg,
                    temperature,
                    tau=args.contrastive_temperature,
                )
            )
            class_weights.append(1.0 / class_pos.sum().detach().float().clamp_min(1.0))

    class_count = max(len(temperatures), 1)
    lld = lld / class_count
    fld = fld / class_count
    rld = rld / class_count
    if ccl_terms:
        weights = loss_mod.torch.stack(class_weights)
        weights = weights / weights.sum().clamp_min(1e-6)
        ccl = loss_mod.torch.stack(ccl_terms).mul(weights.to(ccl_terms[0].device)).sum()

    atkd = lld + fld + rld
    weighted_atkd = float(args.atkd_weight) * atkd
    weighted_ccl = float(args.ccl_weight) * ccl
    kd_total = float(args.kd_scale) * (weighted_atkd + weighted_ccl)
    temp_tensor = loss_mod.torch.stack(temperatures) if temperatures else candidates.vectors.new_zeros(1)
    finite_tensors = loss_mod.torch.stack((lld.detach(), fld.detach(), rld.detach(), ccl.detach(), kd_total.detach()))
    nan_or_inf = float((~loss_mod.torch.isfinite(finite_tensors)).any().item())
    valid_count = float(valid_mask.sum().detach().item())
    positive_count = float(positive_mask.sum().detach().item())
    diagnostics = {
        "cop_valid_candidates": valid_count,
        "cop_positive_candidates": positive_count,
        "cop_positive_ratio": positive_count / max(valid_count, 1.0),
        "temperature_mean": float(temp_tensor.detach().mean().item()),
        "temperature_min": float(temp_tensor.detach().min().item()),
        "temperature_max": float(temp_tensor.detach().max().item()),
        "feature_capture_ok": float(bool(student_features_ok and teacher_features_ok and student_feature_levels and teacher_feature_levels)),
        "nan_or_inf_detected": nan_or_inf,
    }
    return {
        "lld": lld,
        "fld": fld,
        "rld": rld,
        "atkd": atkd,
        "ccl": ccl,
        "kd_total": kd_total,
        "weighted_atkd": weighted_atkd,
        "weighted_ccl": weighted_ccl,
    }, diagnostics


def grad_tuple(torch, loss, params):
    if loss is None or not getattr(loss, "requires_grad", False):
        return [None for _ in params]
    return torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)


def grad_norm_and_cos(torch, grads, reference=None) -> tuple[float, float | None]:
    norm_sq = 0.0
    ref_sq = 0.0
    dot = 0.0
    for grad, ref in zip(grads, reference or [None for _ in grads]):
        if grad is not None:
            norm_sq += float(grad.detach().float().pow(2).sum().item())
        if reference is not None and ref is not None:
            ref_sq += float(ref.detach().float().pow(2).sum().item())
        if reference is not None and grad is not None and ref is not None:
            dot += float((grad.detach().float() * ref.detach().float()).sum().item())
    norm = math.sqrt(max(norm_sq, 0.0))
    if reference is None:
        return norm, None
    ref_norm = math.sqrt(max(ref_sq, 0.0))
    if norm == 0.0 or ref_norm == 0.0:
        return norm, None
    return norm, dot / (norm * ref_norm)


def scalar(loss) -> float:
    try:
        return float(loss.detach().item())
    except Exception:
        return math.nan


def run_probe(args: argparse.Namespace) -> list[dict[str, Any]]:
    torch, _train_mod, loss_mod, device, nc, model, teacher, loader, student_loss, teacher_loss, student_capture, teacher_capture = prepare_probe(args)
    model.train()
    teacher.train()
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    rows: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        imgs, teacher_imgs, targets, _paths, _shapes = batch
        imgs = imgs.to(device, non_blocking=True).float() / 255
        teacher_imgs = teacher_imgs.to(device, non_blocking=True).float() / 255
        targets = targets.to(device)
        student_capture.clear()
        teacher_capture.clear()
        model.zero_grad(set_to_none=True)
        teacher.zero_grad(set_to_none=True)

        student_preds = model(imgs)
        student_det, _student_items = student_loss(student_preds, targets)
        teacher_preds = teacher(teacher_imgs)
        losses, diagnostics = component_losses(
            loss_mod,
            student_preds,
            teacher_preds,
            targets,
            student_loss,
            student_capture.features,
            teacher_capture.features,
            args.mode,
            nc,
            args,
        )
        teacher_det, _teacher_items = teacher_loss(teacher_preds, targets)
        det_grads = grad_tuple(torch, student_det, params)
        det_norm, _ = grad_norm_and_cos(torch, det_grads)

        row: dict[str, Any] = {
            "batch": batch_index,
            "num_targets": int(targets.shape[0]),
            "loss_student_det": scalar(student_det),
            "loss_teacher_det": scalar(teacher_det),
            "loss_lld": scalar(losses["lld"]),
            "loss_fld": scalar(losses["fld"]),
            "loss_rld": scalar(losses["rld"]),
            "loss_atkd": scalar(losses["atkd"]),
            "loss_ccl": scalar(losses["ccl"]),
            "loss_kd_total": scalar(losses["kd_total"]),
            "weighted_atkd_loss": scalar(losses["weighted_atkd"]),
            "weighted_ccl_loss": scalar(losses["weighted_ccl"]),
            "grad_norm_det": det_norm,
        }
        for component in COMPONENTS:
            grads = grad_tuple(torch, losses[component], params)
            norm, cosine = grad_norm_and_cos(torch, grads, det_grads)
            row[f"grad_norm_{component}"] = norm
            row[f"cos_det_{component}"] = cosine
            del grads
        row.update(diagnostics)
        rows.append(row)
        del det_grads, student_preds, teacher_preds, losses
        torch.cuda.empty_cache()
    return rows


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "batches": len(rows),
        "mode": args.mode,
        "atkd_weight": args.atkd_weight,
        "ccl_weight": args.ccl_weight,
        "kd_scale": args.kd_scale,
    }
    for field in CSV_FIELDS:
        if field in {"batch"}:
            continue
        value = mean_numeric(rows, field)
        if value is not None:
            summary[f"mean_{field}"] = value
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Student checkpoint or YOLOv5 weights.")
    parser.add_argument("--teacher-weights", required=True, help="Teacher checkpoint or YOLOv5 weights.")
    parser.add_argument("--student-ckpt-key", default="model", help="Checkpoint key for student model.")
    parser.add_argument("--teacher-ckpt-key", default="teacher", help="Checkpoint key for teacher model; falls back to model.")
    parser.add_argument("--cfg", default="", help="Optional YOLOv5 model yaml override.")
    parser.add_argument("--data", required=True, help="SAR dataset yaml.")
    parser.add_argument("--teacher-data", required=True, help="RGB dataset yaml.")
    parser.add_argument("--hyp", required=True, help="YOLOv5 hyp yaml.")
    parser.add_argument("--imgsz", "--img", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--mode", choices=("paper_atkd_only", "paper_ccl_only", "paper_full"), default="paper_full")
    parser.add_argument("--atkd-weight", type=parse_optional_float, default=None)
    parser.add_argument("--ccl-weight", type=parse_optional_float, default=None)
    parser.add_argument("--kd-scale", type=float, default=1.0)
    parser.add_argument("--t-min", type=float, default=0.5)
    parser.add_argument("--t-max", type=float, default=5.0)
    parser.add_argument("--entropy-scale", type=float, default=5.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.1)
    parser.add_argument("--roi-grid-size", type=int, default=3)
    parser.add_argument("--no-augment", action="store_true", help="Disable train augmentations for deterministic probing.")
    parser.add_argument("--csv-out", required=True, type=Path, help="Per-batch CSV output.")
    parser.add_argument("--summary-out", required=True, type=Path, help="Summary CSV output.")
    parser.add_argument("--md-out", required=True, type=Path, help="Summary Markdown output.")
    args = parser.parse_args()
    default_atkd, default_ccl = default_weights_for_mode(args.mode)
    if args.atkd_weight is None:
        args.atkd_weight = default_atkd
    if args.ccl_weight is None:
        args.ccl_weight = default_ccl
    if args.max_batches < 1:
        parser.error("--max-batches must be >= 1")
    return args


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    rows = run_probe(args)
    summary = summarize(rows, args)
    write_csv(args.csv_out, rows)
    write_summary_csv(args.summary_out, summary)
    write_markdown(args.md_out, summary, args)
    print(args.csv_out)
    print(args.summary_out)
    print(args.md_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
