#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

FIELDS = [
    "paper_result_id",
    "dataset",
    "protocol",
    "task",
    "method",
    "method_label",
    "model_size",
    "seed",
    "student_modality",
    "teacher_modality",
    "inference_modality",
    "imgsz",
    "epochs",
    "batch",
    "mosaic",
    "close_mosaic",
    "phase_chain",
    "student_init",
    "teacher_init",
    "run_tag",
    "project_dir",
    "run_dir",
    "results_csv",
    "args_yaml",
    "manifest",
    "code_commit",
    "git_dirty",
    "best_epoch",
    "best_AP50_95",
    "best_AP50",
    "final_epoch",
    "final_AP50_95",
    "final_AP50",
    "status",
    "usable_for_main_table",
    "invalid_reason",
    "notes",
]

MAP_KEYS = (
    "metrics/mAP50-95(B)",
    "metrics/mAP50-95",
    "metrics/mAP_0.5:0.95",
    "metrics/mAP50-95(M)",
    "map50_95",
)
MAP50_KEYS = (
    "metrics/mAP50(B)",
    "metrics/mAP50",
    "metrics/mAP_0.5",
    "metrics/mAP50(M)",
    "map50",
)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path | None) -> str:
    if path and path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def parse_kv_text(text: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip("'\"")
        meta[key.strip()] = value
    return meta


def parse_yaml_simple(path: Path | None) -> dict[str, Any]:
    if not path or not path.is_file():
        return {}
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        data: dict[str, Any] = {}
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in raw or raw.startswith(" "):
                continue
            key, value = raw.split(":", 1)
            data[key.strip()] = value.strip().strip("'\"")
        return data


def find_sidecar(run_dir: Path, names: tuple[str, ...], max_up: int = 5) -> Path | None:
    current = run_dir
    for _ in range(max_up + 1):
        for name in names:
            candidate = current / name
            if candidate.is_file():
                return candidate
        current = current.parent
    return None


def load_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def first_float(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return None


def first_value(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return ""


def infer_model_seed(text: str, meta: dict[str, Any]) -> tuple[str, str]:
    model = first_value(meta.get("model_size"), meta.get("size"))
    seed = first_value(meta.get("seed"), meta.get("SEED"))
    if not model:
        m = re.search(r"yolo11([nslmx])|ogsod11([nslmx])", text)
        if m:
            model = m.group(1) or m.group(2)
    if not seed:
        m = re.search(r"(?:seed|_s)(0|42|123)(?:\D|$)", text)
        if m:
            seed = m.group(1)
    return model, seed


def infer_method(text: str, meta: dict[str, Any]) -> tuple[str, str, str]:
    method = first_value(meta.get("method"), meta.get("paper_method"), meta.get("comparison_kd_profile"))
    lower = f"{text} {method}".lower()
    if "clean_a1b_dynprobe" in lower or "dynamic_probe" in lower or "ladd_probea" in lower:
        return "ladd_probea", "LADD Probe-A / LADD-clean A1B, ours", "A1->B"
    if "clean_a1b_dyn" in lower:
        return "ladd_dynamic_ablation", "LADD Dynamic ablation", "A1->B"
    if "clean_a1b" in lower:
        return "ladd_static_ablation", "LADD Static ablation", "A1->B"
    if "cmdistill" in lower:
        return "cmdistill", "CMDistill-style / paper-aligned adaptation", "B-only"
    if re.search(r"(^|[/_\-])fgd([/_\-]|$)", lower):
        return "fgd", "FGD-style / FGD-YOLO adaptation", "B-only"
    if re.search(r"(^|[/_\-])ld([/_\-]|$)", lower):
        return "ld", "LD", "B-only"
    if "hallucidet" in lower:
        return "hallucidet_yolo", "HalluciDet-YOLO adaptation", "standalone"
    if "cclkd" in lower and "online" in lower:
        return "cclkd_online", "CCLKD online comparison", "online"
    if "rgb" in lower and "baseline" in lower:
        return "rgb_teacher", "RGB teacher", "baseline"
    if "sar" in lower and "baseline" in lower:
        return "sar_baseline", "SAR baseline", "baseline"
    return first_value(method, "unknown"), first_value(meta.get("method_label"), "unknown"), first_value(meta.get("phase_chain"), "")


def as_float_text(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def gate_row(row: dict[str, str], meta: dict[str, Any]) -> tuple[str, str, str]:
    reasons: list[str] = []
    if row["protocol"] != "mosaic100":
        reasons.append("protocol_not_mosaic100")
    if row["imgsz"] != "256":
        reasons.append("imgsz_not_256")
    if row["epochs"] != "800":
        reasons.append("epochs_not_800")
    if as_float_text(row["mosaic"]) not in {"1", "1.0"}:
        reasons.append("mosaic_not_1.0")
    if row["close_mosaic"] != "700":
        reasons.append("close_mosaic_not_700")
    method = row["method"]
    text = f"{row['run_tag']} {row['run_dir']} {meta}"
    if method == "ladd_probea":
        if row["phase_chain"] != "A1->B":
            reasons.append("ladd_not_a1b")
        if "a2" in text.lower() and "ladd_b_a2_core" not in text.lower():
            reasons.append("ladd_contains_a2")
        if first_value(meta.get("ladd_a1b_mode"), meta.get("LADD_A1B_MODE")) not in {"dynamic_probe", ""}:
            reasons.append("ladd_not_dynamic_probe")
    if method == "cmdistill":
        if first_value(meta.get("kd_calibration_mode"), meta.get("KD_CALIBRATION_MODE")) not in {"affine", ""}:
            reasons.append("cmdistill_not_affine")
    if method == "cclkd_online" and "frozen" in text.lower():
        reasons.append("cclkd_not_online")
    if not row["results_csv"]:
        reasons.append("missing_results_csv")
    if not row["args_yaml"]:
        reasons.append("missing_args_yaml")
    if not row["manifest"]:
        reasons.append("missing_manifest")
    if not row["code_commit"]:
        reasons.append("missing_code_commit")
    if reasons:
        return "invalid", "no", ";".join(reasons)
    return "verified", "yes", ""


def collect_one(results_csv: Path) -> dict[str, str]:
    run_dir = results_csv.parent
    args_yaml = run_dir / "args.yaml"
    if not args_yaml.is_file():
        args_yaml = find_sidecar(run_dir, ("args.yaml",)) or args_yaml
    manifest = find_sidecar(
        run_dir,
        ("paper_run_meta.env", "run_meta_clean_a1b.env", "manifest.txt"),
        max_up=6,
    )
    args_data = parse_yaml_simple(args_yaml if args_yaml.is_file() else None)
    meta = {**args_data, **parse_kv_text(read_text(manifest))}
    rows = load_results(results_csv)
    if not rows:
        raise ValueError(f"No rows in {results_csv}")
    final_row = rows[-1]
    map_values = [first_float(r, MAP_KEYS) for r in rows]
    best_idx = max(range(len(rows)), key=lambda i: map_values[i] if map_values[i] is not None else float("-inf"))
    best_row = rows[best_idx]
    best_map = first_float(best_row, MAP_KEYS)
    final_map = first_float(final_row, MAP_KEYS)
    best_map50 = first_float(best_row, MAP50_KEYS)
    final_map50 = first_float(final_row, MAP50_KEYS)

    text = f"{run_dir} {results_csv} {meta}"
    model_size, seed = infer_model_seed(text, meta)
    method, method_label, phase_chain = infer_method(text, meta)
    protocol = first_value(meta.get("paper_protocol"), meta.get("protocol"), meta.get("PROTOCOL"))
    if not protocol:
        mosaic = first_value(meta.get("mosaic"), meta.get("MOSAIC"), args_data.get("mosaic"))
        close = first_value(meta.get("close_mosaic"), meta.get("CLOSE_MOSAIC"), args_data.get("close_mosaic"))
        protocol = "mosaic100" if as_float_text(mosaic) in {"1", "1.0"} and str(close) == "700" else "unknown"
    row = {
        "paper_result_id": first_value(meta.get("paper_result_id"), f"{protocol}_{method}_yolo11{model_size}_s{seed}_{run_dir.name}"),
        "dataset": first_value(meta.get("dataset"), "OGSOD-1.0"),
        "protocol": protocol,
        "task": first_value(meta.get("task"), "hbb"),
        "method": method,
        "method_label": first_value(meta.get("method_label"), method_label),
        "model_size": model_size,
        "seed": seed,
        "student_modality": first_value(meta.get("student_modality"), "SAR" if method != "rgb_teacher" else "RGB"),
        "teacher_modality": first_value(meta.get("teacher_modality"), "RGB" if method not in {"sar_baseline", "rgb_teacher"} else "none"),
        "inference_modality": first_value(meta.get("inference_modality"), "SAR" if method != "rgb_teacher" else "RGB"),
        "imgsz": first_value(meta.get("imgsz"), meta.get("IMGSZ"), args_data.get("imgsz")),
        "epochs": first_value(meta.get("epochs"), meta.get("epochs_b"), meta.get("EPOCHS"), args_data.get("epochs")),
        "batch": first_value(meta.get("batch"), meta.get("batch_size"), meta.get("BATCH_SIZE"), args_data.get("batch")),
        "mosaic": first_value(meta.get("mosaic"), meta.get("b_mosaic"), meta.get("MOSAIC"), args_data.get("mosaic")),
        "close_mosaic": first_value(meta.get("close_mosaic"), meta.get("b_close_mosaic"), meta.get("CLOSE_MOSAIC"), args_data.get("close_mosaic")),
        "phase_chain": first_value(meta.get("phase_chain"), phase_chain),
        "student_init": first_value(meta.get("sar_baseline"), meta.get("student_init"), meta.get("model"), args_data.get("model")),
        "teacher_init": first_value(meta.get("rgb_teacher"), meta.get("teacher_init"), meta.get("teacher_weights"), args_data.get("teacher_weights")),
        "run_tag": first_value(meta.get("run_tag"), run_dir.name),
        "project_dir": first_value(meta.get("project_dir"), args_data.get("project")),
        "run_dir": rel(run_dir),
        "results_csv": rel(results_csv),
        "args_yaml": rel(args_yaml) if args_yaml.is_file() else "",
        "manifest": rel(manifest) if manifest else "",
        "code_commit": first_value(meta.get("code_commit"), meta.get("git_commit")),
        "git_dirty": first_value(meta.get("git_dirty"), "unknown"),
        "best_epoch": first_value(best_row.get("epoch"), str(best_idx)),
        "best_AP50_95": as_float_text(best_map),
        "best_AP50": as_float_text(best_map50),
        "final_epoch": first_value(final_row.get("epoch"), str(len(rows) - 1)),
        "final_AP50_95": as_float_text(final_map),
        "final_AP50": as_float_text(final_map50),
        "status": "",
        "usable_for_main_table": "",
        "invalid_reason": "",
        "notes": "",
    }
    status, usable, reason = gate_row(row, meta)
    row["status"] = status
    row["usable_for_main_table"] = usable
    row["invalid_reason"] = reason
    return row


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            candidate = Path(pattern)
            if candidate.exists():
                matches = [str(candidate)]
        for match in matches:
            path = Path(match)
            if path.is_dir():
                path = path / "results.csv"
            if path.name == "results.csv" and path.is_file():
                paths.append(path)
    return sorted(set(paths))


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect paper-facing result rows into canonical CSV.")
    parser.add_argument("--glob", action="append", default=[], help="Glob for results.csv or run directories. Can be repeated.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    results_paths = expand_inputs(args.glob)
    rows = [collect_one(path) for path in results_paths]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    print(f"Wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
