#!/usr/bin/env python3
"""Summarize BatchNorm running statistics from a YOLO/LADD checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, help="Path to best.pt/last.pt/checkpoint.pt")
    parser.add_argument("--repo-root", type=Path, default=None, help="Optional public repo root for import paths")
    parser.add_argument("--prefer", choices=("ema", "model"), default="ema", help="Checkpoint model source")
    parser.add_argument("--top", type=int, default=20, help="Number of layers to print")
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path")
    return parser.parse_args()


def add_repo_paths(repo_root: Path | None) -> None:
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root,
        repo_root / "ladd" / "code",
        repo_root / "ladd" / "code_versions" / "current_hbb",
        repo_root / "shared" / "yolo",
        repo_root / "shared" / "yolo" / "ultralytics",
    ]
    for path in candidates:
        if path.exists():
            sys.path.insert(0, str(path))


def load_state_dict(weights: Path, prefer: str) -> dict[str, Any]:
    import torch

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    obj: Any = ckpt
    if isinstance(ckpt, dict):
        if prefer == "ema" and ckpt.get("ema") is not None:
            obj = ckpt["ema"]
        elif ckpt.get("model") is not None:
            obj = ckpt["model"]
        elif ckpt.get("state_dict") is not None:
            obj = ckpt["state_dict"]
        elif ckpt.get("model_state_dict") is not None:
            obj = ckpt["model_state_dict"]

    if hasattr(obj, "float"):
        obj = obj.float()
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported checkpoint object type: {type(obj)!r}")


def tensor_stats(value: Any) -> dict[str, float | bool]:
    import torch

    tensor = value.detach().float().cpu()
    finite = torch.isfinite(tensor)
    clean = tensor[finite]
    if clean.numel() == 0:
        return {
            "finite": False,
            "mean": math.nan,
            "max": math.nan,
            "min": math.nan,
            "mean_abs": math.nan,
            "max_abs": math.nan,
        }
    return {
        "finite": bool(finite.all().item()),
        "mean": float(clean.mean().item()),
        "max": float(clean.max().item()),
        "min": float(clean.min().item()),
        "mean_abs": float(clean.abs().mean().item()),
        "max_abs": float(clean.abs().max().item()),
    }


def collect_bn_rows(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for key, value in state.items():
        if not key.endswith(("running_mean", "running_var")):
            continue
        layer, field = key.rsplit(".", 1)
        rows.setdefault(layer, {"layer": layer})
        stats = tensor_stats(value)
        prefix = "mean" if field == "running_mean" else "var"
        for stat_key, stat_value in stats.items():
            rows[layer][f"{prefix}_{stat_key}"] = stat_value
    return sorted(rows.values(), key=lambda row: float(row.get("var_max", -1.0)), reverse=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "layer",
        "mean_finite",
        "mean_mean",
        "mean_max",
        "mean_min",
        "mean_mean_abs",
        "mean_max_abs",
        "var_finite",
        "var_mean",
        "var_max",
        "var_min",
        "var_mean_abs",
        "var_max_abs",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    add_repo_paths(args.repo_root)
    state = load_state_dict(args.weights, args.prefer)
    rows = collect_bn_rows(state)
    if not rows:
        raise SystemExit("No BatchNorm running_mean/running_var keys found.")

    print(f"checkpoint: {args.weights}")
    print(f"bn_layers: {len(rows)}")
    print("top_by_running_var:")
    for row in rows[: args.top]:
        print(
            f"{row['layer']}: "
            f"var_max={row.get('var_max')} var_mean={row.get('var_mean')} "
            f"mean_max_abs={row.get('mean_max_abs')} finite={row.get('var_finite')}"
        )

    if args.csv:
        write_csv(args.csv, rows)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
