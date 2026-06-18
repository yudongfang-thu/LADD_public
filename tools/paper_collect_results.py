#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAPER_PROTOCOL_ID = "ogsod_hbb_nomosaic_clean_a1b_probea_20260619"
INVALID_GIT_COMMITS = {"unknown", "dirty", "none", "null"}
RECOGNIZED_METHODS = {
    "ladd_probea",
    "cmdistill",
    "fgd",
    "ld",
    "hallucidet_yolo",
    "cclkd_online",
    "rgb_teacher",
    "sar_baseline",
}
FORBIDDEN_PATTERNS = (
    ("smoke", re.compile(r"(^|[_\-/\s])smoke($|[_\-/\s])", re.IGNORECASE)),
    ("partial", re.compile(r"(^|[_\-/\s])partial($|[_\-/\s])", re.IGNORECASE)),
    ("snapshot", re.compile(r"(^|[_\-/\s])snapshot($|[_\-/\s])", re.IGNORECASE)),
    ("diagnostic", re.compile(r"(^|[_\-/\s])diagnostic($|[_\-/\s])", re.IGNORECASE)),
    ("archive", re.compile(r"(^|[_\-/\s])archive($|[_\-/\s])", re.IGNORECASE)),
    ("old", re.compile(r"(^|[_\-/\s])old($|[_\-/\s])", re.IGNORECASE)),
    ("legacy", re.compile(r"(^|[_\-/\s])legacy($|[_\-/\s])", re.IGNORECASE)),
    ("bn-freeze", re.compile(r"bn[-_]freeze", re.IGNORECASE)),
    ("a1-a2-b", re.compile(r"a1[-_]a2[-_]b", re.IGNORECASE)),
    ("probe_only", re.compile(r"(^|[_\-/\s])probe[-_]only($|[_\-/\s])", re.IGNORECASE)),
    ("probe_run", re.compile(r"(^|[_\-/\s])probe[-_]run($|[_\-/\s])", re.IGNORECASE)),
    ("diagnostic_probe", re.compile(r"(^|[_\-/\s])diagnostic[-_]probe($|[_\-/\s])", re.IGNORECASE)),
)


@dataclass(frozen=True)
class InputSpec:
    target: str
    base_dir: Path
    args_yaml: str = ""
    manifest: str = ""
    data_yaml: str = ""
    teacher_data_yaml: str = ""
    student_data_yaml: str = ""

FIELDS = [
    "dataset",
    "task",
    "protocol_id",
    "method",
    "method_display",
    "model_size",
    "seed",
    "init_type",
    "student_modality",
    "teacher_modality",
    "inference_modality",
    "imgsz",
    "epochs",
    "batch",
    "mosaic",
    "close_mosaic",
    "phase_chain",
    "ladd_mode",
    "run_tag",
    "project_dir",
    "results_csv",
    "args_yaml",
    "manifest",
    "data_yaml",
    "teacher_data_yaml",
    "student_data_yaml",
    "git_commit",
    "best_ap50_95",
    "best_ap50",
    "final_ap50_95",
    "final_ap50",
    "best_epoch",
    "status",
    "claim_usable",
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


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    base_candidate = base_dir / path
    if base_candidate.exists():
        return base_candidate
    root_candidate = ROOT / path
    if root_candidate.exists():
        return root_candidate
    return base_candidate


def parse_kv_text(text: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        meta[key.strip()] = value.strip().strip("'\"")
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


def find_sidecar(run_dir: Path, names: tuple[str, ...], max_up: int = 6) -> Path | None:
    current = run_dir
    for _ in range(max_up + 1):
        for name in names:
            candidate = current / name
            if candidate.is_file():
                return candidate
        current = current.parent
    return None


def first_value(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return ""


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


def as_float_text(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def is_sha_like(value: str) -> bool:
    lowered = str(value or "").strip().lower()
    return lowered not in INVALID_GIT_COMMITS and re.fullmatch(r"[0-9a-f]{7,40}", lowered) is not None


def is_truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def forbidden_labels(text: str) -> list[str]:
    return [label for label, pattern in FORBIDDEN_PATTERNS if pattern.search(text)]


def infer_model_seed(text: str, meta: dict[str, Any]) -> tuple[str, str]:
    model = first_value(meta.get("model_size"), meta.get("size"), meta.get("SIZE"))
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


def infer_method(text: str, meta: dict[str, Any]) -> tuple[str, str, str, str, str]:
    method = first_value(meta.get("method"), meta.get("paper_method"), meta.get("comparison_kd_profile"))
    lower = f"{text} {method}".lower()
    if "clean_a1b_dynprobe" in lower or "dynamic_probe" in lower or "ladd_probea" in lower:
        return "ladd_probea", "LADD Probe-A / LADD-clean A1B", "A1->B", "dynamic_probe", "sar_baseline"
    if "cmdistill" in lower:
        return "cmdistill", "CMDistill-style / paper-aligned adaptation", "B-only", "", "transferred_kd"
    if re.search(r"(^|[/_\-])fgd([/_\-]|$)", lower):
        return "fgd", "FGD-style / FGD-YOLO adaptation", "B-only", "", "transferred_kd"
    if re.search(r"(^|[/_\-])ld([/_\-]|$)", lower):
        return "ld", "LD", "B-only", "", "transferred_kd"
    if "hallucidet" in lower:
        return "hallucidet_yolo", "HalluciDet-YOLO adaptation", "standalone", "", "standalone"
    if "cclkd" in lower and "online" in lower:
        return "cclkd_online", "CCLKD online comparison", "online", "", "from_yolo_pretrain"
    if "rgb" in lower and "baseline" in lower:
        return "rgb_teacher", "RGB teacher", "baseline", "", "yolo_pretrain"
    if "sar" in lower and "baseline" in lower:
        return "sar_baseline", "SAR baseline", "baseline", "", "yolo_pretrain"
    return first_value(method, "unknown"), first_value(meta.get("method_label"), "unknown"), first_value(meta.get("phase_chain"), ""), "", ""


def load_results(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def gate_row(row: dict[str, str]) -> tuple[str, str, str]:
    reasons: list[str] = []
    if row["protocol_id"] != PAPER_PROTOCOL_ID:
        reasons.append("protocol_id_mismatch")
    if row["imgsz"] != "256":
        reasons.append("imgsz_not_256")
    if row["epochs"] != "800":
        reasons.append("epochs_not_800")
    if as_float_text(row["mosaic"]) not in {"0", "0.0"}:
        reasons.append("mosaic_not_0.0")
    if row["close_mosaic"] != "0":
        reasons.append("close_mosaic_not_0")
    if row["seed"] not in {"0", "42", "123"}:
        reasons.append("nonpaper_seed")
    if not row["results_csv"]:
        reasons.append("missing_results_csv")
    if not row["args_yaml"]:
        reasons.append("missing_args_yaml")
    if not row["manifest"]:
        reasons.append("missing_manifest")
    if not row.get("_paper_run") or not is_truthy(row.get("_paper_run")):
        reasons.append("not_paper_run")
    if not row["git_commit"]:
        reasons.append("missing_git_commit")
    elif not is_sha_like(row["git_commit"]):
        reasons.append("invalid_git_commit")
    if row["method"] not in RECOGNIZED_METHODS:
        reasons.append("method_unrecognized")
    if row["status"] not in {"complete", "verified", "main_table"}:
        reasons.append("status_not_verified")
    text = " ".join(row.values()).lower()
    for label in forbidden_labels(text):
        reasons.append(f"forbidden_{label}")
    if row["method"] == "ladd_probea":
        if row["ladd_mode"] != "dynamic_probe":
            reasons.append("ladd_not_dynamic_probe")
        if "clean_a1b_dynprobe" not in row["run_tag"]:
            reasons.append("ladd_tag_missing_dynprobe")
        if row["phase_chain"] != "A1->B":
            reasons.append("ladd_not_a1b")
    if row["method"] == "cmdistill" and "vedai" in text:
        reasons.append("cmdistill_vedai_not_ogsod_main")
    if row["method"] == "cclkd_online" and ("online" not in text or "frozen" in text):
        reasons.append("cclkd_not_online_main")
    if reasons:
        return "collected_unverified", "no", ";".join(reasons)
    return row["status"], "yes", ""


def collect_one(spec: InputSpec) -> dict[str, str]:
    results_csv = resolve_path(spec.target, spec.base_dir)
    run_dir = results_csv.parent
    args_yaml = resolve_path(spec.args_yaml, spec.base_dir) if spec.args_yaml else run_dir / "args.yaml"
    if not spec.args_yaml and not args_yaml.is_file():
        args_yaml = find_sidecar(run_dir, ("args.yaml",)) or args_yaml
    manifest = resolve_path(spec.manifest, spec.base_dir) if spec.manifest else find_sidecar(run_dir, ("paper_run_meta.env", "run_meta_clean_a1b.env", "manifest.txt"))
    data_yaml = resolve_path(spec.data_yaml, spec.base_dir) if spec.data_yaml else None
    teacher_data_yaml = resolve_path(spec.teacher_data_yaml, spec.base_dir) if spec.teacher_data_yaml else None
    student_data_yaml = resolve_path(spec.student_data_yaml, spec.base_dir) if spec.student_data_yaml else None
    args_data = parse_yaml_simple(args_yaml if args_yaml.is_file() else None)
    meta = {**args_data, **parse_kv_text(read_text(manifest))}
    if not data_yaml:
        meta_data_yaml = first_value(meta.get("data_yaml"), meta.get("data_cfg"), meta.get("DATA_CFG"))
        data_yaml = resolve_path(meta_data_yaml, spec.base_dir) if meta_data_yaml else None
    if not teacher_data_yaml:
        meta_teacher_yaml = first_value(meta.get("teacher_data_yaml"), meta.get("teacher_data_cfg"), meta.get("TEACHER_DATA_CFG"))
        teacher_data_yaml = resolve_path(meta_teacher_yaml, spec.base_dir) if meta_teacher_yaml else None
    if not student_data_yaml:
        meta_student_yaml = first_value(meta.get("student_data_yaml"), meta.get("student_data_cfg"), meta.get("STUDENT_DATA_CFG"))
        student_data_yaml = resolve_path(meta_student_yaml, spec.base_dir) if meta_student_yaml else None

    rows = load_results(results_csv)
    if not rows:
        raise ValueError(f"No rows in {results_csv}")
    final_row = rows[-1]
    map_values = [first_float(r, MAP_KEYS) for r in rows]
    best_idx = max(range(len(rows)), key=lambda i: map_values[i] if map_values[i] is not None else float("-inf"))
    best_row = rows[best_idx]

    text = f"{run_dir} {results_csv} {meta}"
    model_size, seed = infer_model_seed(text, meta)
    method, method_display, phase_chain, ladd_mode, init_type = infer_method(text, meta)
    git_commit = first_value(meta.get("git_commit"), meta.get("code_commit"))
    if not is_sha_like(git_commit):
        git_commit = ""
    row = {
        "dataset": first_value(meta.get("dataset"), "OGSOD-1.0"),
        "task": first_value(meta.get("task"), "hbb"),
        "protocol_id": first_value(meta.get("protocol_id"), meta.get("paper_protocol_id")),
        "method": method,
        "method_display": first_value(meta.get("method_display"), meta.get("method_label"), method_display),
        "model_size": model_size,
        "seed": seed,
        "init_type": first_value(meta.get("init_type"), init_type),
        "student_modality": first_value(meta.get("student_modality"), "SAR" if method != "rgb_teacher" else "RGB"),
        "teacher_modality": first_value(meta.get("teacher_modality"), "RGB" if method not in {"sar_baseline", "rgb_teacher"} else "none"),
        "inference_modality": first_value(meta.get("inference_modality"), "SAR" if method != "rgb_teacher" else "RGB"),
        "imgsz": first_value(meta.get("imgsz"), meta.get("IMGSZ"), args_data.get("imgsz")),
        "epochs": first_value(meta.get("epochs"), meta.get("epochs_b"), meta.get("EPOCHS"), args_data.get("epochs")),
        "batch": first_value(meta.get("batch"), meta.get("batch_size"), meta.get("BATCH_SIZE"), args_data.get("batch")),
        "mosaic": first_value(meta.get("mosaic"), meta.get("b_mosaic"), meta.get("MOSAIC"), args_data.get("mosaic")),
        "close_mosaic": first_value(meta.get("close_mosaic"), meta.get("b_close_mosaic"), meta.get("CLOSE_MOSAIC"), args_data.get("close_mosaic")),
        "phase_chain": first_value(meta.get("phase_chain"), phase_chain),
        "ladd_mode": first_value(meta.get("ladd_mode"), meta.get("ladd_a1b_mode"), meta.get("LADD_A1B_MODE"), ladd_mode),
        "run_tag": first_value(meta.get("run_tag"), run_dir.name),
        "project_dir": first_value(meta.get("project_dir"), args_data.get("project"), rel(run_dir.parent)),
        "results_csv": rel(results_csv),
        "args_yaml": rel(args_yaml) if args_yaml.is_file() else "",
        "manifest": rel(manifest) if manifest and manifest.is_file() else "",
        "data_yaml": rel(data_yaml) if data_yaml and data_yaml.is_file() else first_value(meta.get("data_yaml"), meta.get("data_cfg"), meta.get("DATA_CFG")),
        "teacher_data_yaml": rel(teacher_data_yaml) if teacher_data_yaml and teacher_data_yaml.is_file() else first_value(meta.get("teacher_data_yaml"), meta.get("teacher_data_cfg"), meta.get("TEACHER_DATA_CFG")),
        "student_data_yaml": rel(student_data_yaml) if student_data_yaml and student_data_yaml.is_file() else first_value(meta.get("student_data_yaml"), meta.get("student_data_cfg"), meta.get("STUDENT_DATA_CFG")),
        "git_commit": git_commit,
        "best_ap50_95": as_float_text(first_float(best_row, MAP_KEYS)),
        "best_ap50": as_float_text(first_float(best_row, MAP50_KEYS)),
        "final_ap50_95": as_float_text(first_float(final_row, MAP_KEYS)),
        "final_ap50": as_float_text(first_float(final_row, MAP50_KEYS)),
        "best_epoch": first_value(best_row.get("epoch"), str(best_idx)),
        "status": first_value(meta.get("status"), meta.get("paper_status"), meta.get("claim_status")),
        "claim_usable": "",
        "notes": "",
        "_paper_run": first_value(meta.get("paper_run"), meta.get("PAPER_RUN")),
    }
    status, usable, notes = gate_row(row)
    row["status"] = status
    row["claim_usable"] = usable
    row["notes"] = notes
    return row


def expand_specs(specs: list[InputSpec]) -> list[InputSpec]:
    results: list[InputSpec] = []
    for spec in specs:
        matches = glob.glob(str(resolve_path(spec.target, spec.base_dir)), recursive=True) or [str(resolve_path(spec.target, spec.base_dir))]
        for match in matches:
            path = Path(match)
            if path.is_dir():
                path = path / "results.csv"
            if path.name == "results.csv" and path.is_file():
                results.append(
                    InputSpec(
                        target=str(path),
                        base_dir=spec.base_dir,
                        args_yaml=spec.args_yaml,
                        manifest=spec.manifest,
                        data_yaml=spec.data_yaml,
                        teacher_data_yaml=spec.teacher_data_yaml,
                        student_data_yaml=spec.student_data_yaml,
                    )
                )
    seen: set[tuple[str, str, str]] = set()
    unique: list[InputSpec] = []
    for spec in sorted(results, key=lambda item: item.target):
        key = (str(resolve_path(spec.target, spec.base_dir)), spec.args_yaml, spec.manifest)
        if key in seen:
            continue
        seen.add(key)
        unique.append(spec)
    return unique


def registry_specs(path: Path) -> list[InputSpec]:
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    out: list[InputSpec] = []
    for row in rows:
        candidate = first_value(row.get("results_csv"), row.get("canonical_path"), row.get("path"))
        if candidate:
            out.append(
                InputSpec(
                    target=candidate,
                    base_dir=path.resolve().parent,
                    args_yaml=first_value(row.get("args_yaml")),
                    manifest=first_value(row.get("manifest"), row.get("paper_run_meta"), row.get("paper_run_meta_env"), row.get("meta_path")),
                    data_yaml=first_value(row.get("data_yaml"), row.get("data_cfg"), row.get("DATA_CFG")),
                    teacher_data_yaml=first_value(row.get("teacher_data_yaml"), row.get("teacher_data_cfg"), row.get("TEACHER_DATA_CFG")),
                    student_data_yaml=first_value(row.get("student_data_yaml"), row.get("student_data_cfg"), row.get("STUDENT_DATA_CFG")),
                )
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect paper-facing result rows into canonical CSV.")
    parser.add_argument("--input", nargs="*", default=[], help="Registry CSVs, run directories, or results.csv files.")
    parser.add_argument("--runs", nargs="*", default=[], help="Run directories or results.csv files.")
    parser.add_argument("--glob", action="append", default=[], help="Glob for run directories or results.csv files.")
    parser.add_argument("--registry", type=Path, help="Registry CSV with results paths.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    specs = [InputSpec(target=item, base_dir=Path.cwd()) for item in list(args.runs) + list(args.glob)]
    for item in args.input:
        path = Path(item)
        if path.suffix == ".csv" and path.name != "results.csv" and path.is_file():
            specs.extend(registry_specs(path))
        else:
            specs.append(InputSpec(target=item, base_dir=Path.cwd()))
    if args.registry:
        specs.extend(registry_specs(args.registry))
    result_specs = expand_specs(specs)
    rows = [collect_one(spec) for spec in result_specs]

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
