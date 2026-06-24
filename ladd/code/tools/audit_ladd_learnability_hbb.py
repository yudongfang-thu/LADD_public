#!/usr/bin/env python3
"""Offline learnability audit for LADD HBB checkpoints.

The audit asks whether SAR student features are closer to / can linearly predict
teacher z_t better than u_t. It does not modify the checkpoint and it does not add
anything to the inference graph.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
SHARED_ROOT = REPO_ROOT / "shared"
YOLO_ROOT = SHARED_ROOT / "yolo"
SRC_ROOT = REPO_ROOT / "ladd" / "code" / "src"
for path in (str(SRC_ROOT), str(SHARED_ROOT), str(YOLO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import DEFAULT_CFG, get_cfg  # noqa: E402
from ultralytics.data import build_dataloader  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.utils.tal import make_anchors  # noqa: E402

from d2ad_obb.aug_policy import apply_unified_paired_aug_policy  # noqa: E402
from d2ad_obb.paired_dataset import PairedOBBDataset  # noqa: E402
from teacher_student_decomposition_kd_hbb.loss import (  # noqa: E402
    TeacherStudentDecompositionKDNRRLTeacherUAuxLossHBB,
    _unwrap_teacher_preds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path, help="LADD/dynamic checkpoint.")
    parser.add_argument("--model", type=str, default="", help="Optional model yaml, retained for provenance.")
    parser.add_argument("--data", required=True, type=Path, help="SAR dataset yaml.")
    parser.add_argument("--teacher-data", required=True, type=Path, help="RGB teacher dataset yaml.")
    parser.add_argument("--teacher-weights", required=True, type=Path, help="Frozen RGB teacher checkpoint.")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-batches", type=int, default=20)
    parser.add_argument("--max-tokens-per-level", type=int, default=4096)
    parser.add_argument("--fg-only", action="store_true")
    parser.add_argument("--include-bg-sample", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-teacher-pairs", action="store_true")
    parser.add_argument("--save-features", action="store_true")
    parser.add_argument(
        "--gradient-audit",
        action="store_true",
        help="Reserved hook for distillability gradient audit; first version records a not-run note.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(args: argparse.Namespace, student_model: torch.nn.Module) -> tuple[Any, Any]:
    data = check_det_dataset(str(args.data))
    teacher_data = check_det_dataset(str(args.teacher_data))
    cfg = get_cfg(DEFAULT_CFG)
    cfg.imgsz = args.imgsz
    cfg.cache = False
    cfg.single_cls = False
    cfg.task = "detect"
    cfg.classes = None
    cfg.fraction = 1.0
    cfg.rect = args.split == "val"
    cfg.mosaic = 0.0
    cfg.mixup = 0.0
    cfg.copy_paste = 0.0
    cfg.close_mosaic = 0
    apply_unified_paired_aug_policy(cfg)
    stride = max(int(getattr(student_model, "stride", torch.tensor([32])).max()), 32)
    dataset = PairedOBBDataset(
        img_path=data[args.split],
        teacher_img_path=teacher_data[args.split],
        imgsz=args.imgsz,
        batch_size=args.batch,
        augment=False,
        hyp=cfg,
        rect=args.split == "val",
        cache=None,
        single_cls=False,
        stride=stride,
        pad=0.0,
        prefix=f"{args.split}: ",
        task="detect",
        classes=None,
        data=data,
        fraction=1.0,
    )
    loader = build_dataloader(dataset, batch=args.batch, workers=args.workers, shuffle=False, rank=-1, seed=args.seed)
    return data, loader


def move_batch(batch: dict[str, Any], device: torch.device, shuffle_teacher_pairs: bool) -> dict[str, Any]:
    out = dict(batch)
    out["img"] = out["img"].to(device, non_blocking=True).float() / 255.0
    out["teacher_img"] = out["teacher_img"].to(device, non_blocking=True).float() / 255.0
    if shuffle_teacher_pairs and out["teacher_img"].shape[0] > 1:
        out["teacher_img"] = out["teacher_img"][torch.randperm(out["teacher_img"].shape[0], device=device)]
    for key in ("batch_idx", "cls", "bboxes"):
        if key in out and isinstance(out[key], torch.Tensor):
            out[key] = out[key].to(device)
    return out


def unwrap_preds(outputs: Any) -> dict[str, torch.Tensor]:
    return _unwrap_teacher_preds(outputs)


def assign_tokens(
    criterion: TeacherStudentDecompositionKDNRRLTeacherUAuxLossHBB,
    preds: dict[str, torch.Tensor],
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
    pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
    anchor_points, stride_tensor = make_anchors(preds["feats"], criterion.stride, 0.5)
    batch_size = pred_scores.shape[0]
    dtype = pred_scores.dtype
    imgsz = torch.tensor(preds["feats"][0].shape[2:], device=criterion.device, dtype=dtype) * criterion.stride[0]
    batch_idx = batch["batch_idx"].view(-1, 1)
    targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = criterion.preprocess(targets.to(criterion.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
    pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
    bboxes_for_assigner = pred_bboxes.clone().detach() * stride_tensor
    _, _, target_scores, fg_mask, _ = criterion.assigner(
        pred_scores.detach().sigmoid(),
        bboxes_for_assigner.type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )
    return fg_mask.bool(), target_scores


def squared_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (F.normalize(a, dim=-1, eps=1e-6) - F.normalize(b, dim=-1, eps=1e-6)).pow(2).sum(dim=-1)


def summarize(values: torch.Tensor, prefix: str) -> dict[str, float]:
    values = values.detach().float().reshape(-1).cpu()
    if values.numel() == 0:
        return {f"{prefix}_{name}": float("nan") for name in ("mean", "median", "q25", "q75")}
    return {
        f"{prefix}_mean": float(values.mean().item()),
        f"{prefix}_median": float(torch.quantile(values, 0.50).item()),
        f"{prefix}_q25": float(torch.quantile(values, 0.25).item()),
        f"{prefix}_q75": float(torch.quantile(values, 0.75).item()),
    }


def fit_ridge_probe(x: torch.Tensor, y: torch.Tensor, seed: int, ridge: float = 1e-3) -> dict[str, float]:
    x = x.float().cpu()
    y = y.float().cpu()
    if x.shape[0] < 16:
        return {"mse": float("nan"), "r2": float("nan"), "cos": float("nan")}
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=generator)
    split = max(1, int(0.8 * x.shape[0]))
    train_idx, test_idx = perm[:split], perm[split:]
    if test_idx.numel() == 0:
        test_idx = train_idx
    x_train = torch.cat((x[train_idx], torch.ones(train_idx.numel(), 1)), dim=1)
    x_test = torch.cat((x[test_idx], torch.ones(test_idx.numel(), 1)), dim=1)
    y_train = y[train_idx]
    y_test = y[test_idx]
    eye = torch.eye(x_train.shape[1])
    eye[-1, -1] = 0.0
    lhs = x_train.T @ x_train + ridge * eye
    rhs = x_train.T @ y_train
    try:
        weight = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        weight = torch.linalg.lstsq(lhs, rhs).solution
    pred = x_test @ weight
    mse = F.mse_loss(pred, y_test).item()
    var = y_test.var(unbiased=False).clamp_min(1e-12).item()
    cos = F.cosine_similarity(pred, y_test, dim=-1).mean().item()
    return {"mse": float(mse), "r2": float(1.0 - mse / var), "cos": float(cos)}


def optional_task_auc(features: torch.Tensor, labels: torch.Tensor, seed: int) -> tuple[float, float]:
    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    if len(np.unique(labels_np)) < 2:
        return float("nan"), float("nan")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss, roc_auc_score
    except Exception:
        return float("nan"), float("nan")
    x_np = features.detach().float().cpu().numpy()
    if x_np.shape[0] > 10000:
        rng = np.random.default_rng(seed)
        idx = rng.choice(x_np.shape[0], 10000, replace=False)
        x_np = x_np[idx]
        labels_np = labels_np[idx]
    clf = LogisticRegression(max_iter=200, random_state=seed, n_jobs=1)
    clf.fit(x_np, labels_np)
    prob = clf.predict_proba(x_np)[:, 1]
    return float(roc_auc_score(labels_np, prob)), float(log_loss(labels_np, prob, labels=[0, 1]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    student = YOLO(str(args.weights)).model.to(device).eval()
    teacher = YOLO(str(args.teacher_weights)).model.to(device).eval()
    for model in (student, teacher):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    criterion = TeacherStudentDecompositionKDNRRLTeacherUAuxLossHBB(student, teacher_model=teacher)
    _, loader = make_dataset(args, student)

    samples: dict[str, dict[str, list[torch.Tensor]]] = {}
    direct_rows: list[dict[str, Any]] = []
    level_counts: dict[str, int] = {}

    with torch.no_grad():
        for batch_id, raw_batch in enumerate(loader):
            if batch_id >= args.max_batches:
                break
            batch = move_batch(raw_batch, device, args.shuffle_teacher_pairs)
            preds = unwrap_preds(student(batch["img"]))
            teacher_preds = unwrap_preds(teacher(batch["teacher_img"]))
            if not preds or "feats" not in preds or "student_base_feats" not in preds:
                raise RuntimeError("Student forward did not expose LADD HBB feature dictionaries.")
            if not teacher_preds or "feats" not in teacher_preds:
                raise RuntimeError("Teacher forward did not expose feature dictionaries.")
            fg_mask, target_scores = assign_tokens(criterion, preds, batch)

            offset = 0
            for level_idx, (teacher_feat, student_raw, z_s_map) in enumerate(
                zip(teacher_preds["feats"], preds["student_base_feats"], preds["z_s_feats"])
            ):
                level_name = f"P{level_idx + 3}"
                z_t_map, u_t_map, _, _ = student.teacher_decomposition[level_idx](teacher_feat)
                if hasattr(student, "student_reachability"):
                    q_s_map = student.student_reachability[level_idx](student_raw)
                else:
                    q_s_map = student_raw
                q_s = q_s_map.permute(0, 2, 3, 1).reshape(-1, q_s_map.shape[1])
                z_t = z_t_map.permute(0, 2, 3, 1).reshape(-1, z_t_map.shape[1])
                u_t = u_t_map.permute(0, 2, 3, 1).reshape(-1, u_t_map.shape[1])
                z_s = z_s_map.permute(0, 2, 3, 1).reshape(-1, z_s_map.shape[1])
                n_tokens = z_t_map.shape[2] * z_t_map.shape[3]
                fg_level = fg_mask[:, offset : offset + n_tokens].reshape(-1)
                score_level = target_scores[:, offset : offset + n_tokens].reshape(-1, target_scores.shape[-1])
                offset += n_tokens

                if args.fg_only:
                    candidate = torch.where(fg_level)[0]
                elif args.include_bg_sample:
                    fg_idx = torch.where(fg_level)[0]
                    bg_idx = torch.where(~fg_level)[0]
                    max_each = max(1, args.max_tokens_per_level // 2)
                    fg_idx = fg_idx[torch.randperm(fg_idx.numel(), device=device)[:max_each]] if fg_idx.numel() > max_each else fg_idx
                    bg_idx = bg_idx[torch.randperm(bg_idx.numel(), device=device)[:max_each]] if bg_idx.numel() > max_each else bg_idx
                    candidate = torch.cat((fg_idx, bg_idx), dim=0)
                else:
                    candidate = torch.arange(q_s.shape[0], device=device)
                remaining = max(0, args.max_tokens_per_level - level_counts.get(level_name, 0))
                if remaining <= 0 or candidate.numel() == 0:
                    continue
                if candidate.numel() > remaining:
                    candidate = candidate[torch.randperm(candidate.numel(), device=device)[:remaining]]
                level_counts[level_name] = level_counts.get(level_name, 0) + int(candidate.numel())

                d_s_z = squared_l2(q_s[candidate], z_t[candidate])
                d_s_u = squared_l2(q_s[candidate], u_t[candidate])
                gap = d_s_u - d_s_z
                row = {
                    "batch": batch_id,
                    "level": level_name,
                    "tokens": int(candidate.numel()),
                    "fg_ratio": float(fg_level[candidate].float().mean().item()),
                    "learnability_positive_ratio": float((gap > 0).float().mean().item()),
                }
                row.update(summarize(gap, "learnability_gap_direct"))
                row.update(summarize(d_s_z, "d_s_z"))
                row.update(summarize(d_s_u, "d_s_u"))
                direct_rows.append(row)

                bucket = samples.setdefault(level_name, {"q": [], "raw": [], "zs": [], "z": [], "u": [], "fg": [], "score": []})
                bucket["q"].append(q_s[candidate].detach().cpu())
                bucket["raw"].append(student_raw.permute(0, 2, 3, 1).reshape(-1, student_raw.shape[1])[candidate].detach().cpu())
                bucket["zs"].append(z_s[candidate].detach().cpu())
                bucket["z"].append(z_t[candidate].detach().cpu())
                bucket["u"].append(u_t[candidate].detach().cpu())
                bucket["fg"].append(fg_level[candidate].detach().cpu().long())
                bucket["score"].append(score_level[candidate].detach().cpu())

    if not direct_rows:
        raise RuntimeError("No audit tokens were collected. Check split, labels, and max-token settings.")

    write_csv(args.output_dir / "learnability_audit_per_batch.csv", direct_rows)

    per_level_rows: list[dict[str, Any]] = []
    global_gap_parts: list[torch.Tensor] = []
    global_fg_parts: list[torch.Tensor] = []
    for level_name, bucket in samples.items():
        merged = {key: torch.cat(values, dim=0) for key, values in bucket.items() if values}
        q, z, u, fg = merged["q"], merged["z"], merged["u"], merged["fg"]
        gap = squared_l2(q, u) - squared_l2(q, z)
        global_gap_parts.append(gap.detach().cpu())
        global_fg_parts.append(fg.detach().cpu())
        probe_z = fit_ridge_probe(q, z, args.seed)
        probe_u = fit_ridge_probe(q, u, args.seed)
        auc_z, ce_z = optional_task_auc(z, fg, args.seed)
        auc_u, ce_u = optional_task_auc(u, fg, args.seed)
        row = {
            "level": level_name,
            "tokens": int(q.shape[0]),
            "fg_ratio": float(fg.float().mean().item()),
            "learnability_positive_ratio": float((gap > 0).float().mean().item()),
            "mse_probe_z": probe_z["mse"],
            "mse_probe_u": probe_u["mse"],
            "r2_probe_z": probe_z["r2"],
            "r2_probe_u": probe_u["r2"],
            "cos_probe_z": probe_z["cos"],
            "cos_probe_u": probe_u["cos"],
            "learnability_gap_probe": probe_z["r2"] - probe_u["r2"],
            "task_auc_z": auc_z,
            "task_auc_u": auc_u,
            "task_ce_z": ce_z,
            "task_ce_u": ce_u,
        }
        row.update(summarize(gap, "learnability_gap_direct"))
        per_level_rows.append(row)
    write_csv(args.output_dir / "learnability_audit_per_level.csv", per_level_rows)

    gap = torch.cat(global_gap_parts, dim=0)
    fg = torch.cat(global_fg_parts, dim=0)
    total_tokens = int(sum(row["tokens"] for row in per_level_rows))

    def weighted_level_metric(name: str) -> float:
        numerator = 0.0
        denominator = 0.0
        for row in per_level_rows:
            value = row.get(name)
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(value):
                continue
            weight = float(row.get("tokens", 0))
            numerator += value * weight
            denominator += weight
        return numerator / denominator if denominator > 0 else float("nan")

    summary = {
        "weights": str(args.weights),
        "model": args.model,
        "data": str(args.data),
        "teacher_data": str(args.teacher_data),
        "teacher_weights": str(args.teacher_weights),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "max_batches": args.max_batches,
        "max_tokens_per_level": args.max_tokens_per_level,
        "fg_only": int(args.fg_only),
        "include_bg_sample": int(args.include_bg_sample),
        "shuffle_teacher_pairs": int(args.shuffle_teacher_pairs),
        "summary_probe_scope": "per_level_token_weighted",
        "tokens": total_tokens,
        "fg_ratio": float(fg.float().mean().item()),
        "learnability_positive_ratio": float((gap > 0).float().mean().item()),
        "mse_probe_z": weighted_level_metric("mse_probe_z"),
        "mse_probe_u": weighted_level_metric("mse_probe_u"),
        "r2_probe_z": weighted_level_metric("r2_probe_z"),
        "r2_probe_u": weighted_level_metric("r2_probe_u"),
        "cos_probe_z": weighted_level_metric("cos_probe_z"),
        "cos_probe_u": weighted_level_metric("cos_probe_u"),
        "learnability_gap_probe": weighted_level_metric("learnability_gap_probe"),
        "task_auc_z": weighted_level_metric("task_auc_z"),
        "task_auc_u": weighted_level_metric("task_auc_u"),
        "task_ce_z": weighted_level_metric("task_ce_z"),
        "task_ce_u": weighted_level_metric("task_ce_u"),
        "gradient_audit_status": "not_implemented_reserved" if args.gradient_audit else "not_requested",
    }
    summary.update(summarize(gap, "learnability_gap_direct"))
    write_csv(args.output_dir / "learnability_audit_summary.csv", [summary])

    config = vars(args).copy()
    config["output_dir"] = str(args.output_dir)
    (args.output_dir / "learnability_audit_config.yaml").write_text(
        "\n".join(f"{key}: {json.dumps(str(value) if isinstance(value, Path) else value, ensure_ascii=False)}" for key, value in sorted(config.items())) + "\n",
        encoding="utf-8",
    )
    if args.save_features:
        np.savez_compressed(
            args.output_dir / "features_sample.npz",
            gap=gap.numpy(),
            fg=fg.numpy(),
        )
    notes = [
        "# LADD HBB Learnability Audit Notes",
        "",
        "- This audit is eval/no_grad and does not modify the checkpoint.",
        "- Direct gap is d(q_s,u_t)-d(q_s,z_t); positive means z_t is closer to SAR q_s.",
        "- Probe gap is R2(q_s->z_t)-R2(q_s->u_t); positive supports z_t being more SAR-learnable.",
        "- High task_auc_u is not automatically a failure; u_t may be RGB-private but task-useful.",
        "- Paired-vs-shuffled should be compared by running this tool twice with and without --shuffle-teacher-pairs.",
        f"- Gradient audit status: {summary['gradient_audit_status']}.",
    ]
    (args.output_dir / "learnability_audit_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
