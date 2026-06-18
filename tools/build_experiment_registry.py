#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


AP_KEYS = (
    "metrics/mAP50-95(B)",
    "metrics/mAP50-95",
    "map50_95",
    "AP",
    "ap",
)
AP50_KEYS = (
    "metrics/mAP50(B)",
    "metrics/mAP50",
    "map50",
    "AP50",
    "ap50",
)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_simple_kv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            continue
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in out:
            out[key] = value
    return out


def find_nearest_manifest(run_dir: Path, root: Path) -> Path | None:
    for parent in [run_dir, *run_dir.parents]:
        if parent == root.parent:
            break
        manifest = parent / "manifest.txt"
        if manifest.exists():
            return manifest
    return None


def infer_server(path: Path, args_text: str, manifest_text: str) -> str:
    s = str(path)
    t = args_text + "\n" + manifest_text
    lower = (s + "\n" + t).lower()
    if "server_artifacts/server90" in s or "ladd90_formal_baselines" in s or "/mnt/datay/" in lower:
        return "ladd90"
    if "server_artifacts/dual4090" in s:
        return "dual4090_old"
    if "ladd4090" in s or "main_4090" in s or "/root/shared-nvme/" in lower:
        return "ladd4090"
    if "autodl" in lower or "4090d" in lower or "/root/autodl" in lower:
        return "autodl"
    if "117" in lower or "5880" in lower:
        return "server117"
    return "unknown"


def infer_family(path: Path) -> str:
    s = str(path).lower()
    name = path.parent.name.lower()
    if "cclkd_reproduction/yolov5_sanity" in s:
        return "cclkd_yolov5x"
    if "cclkd_reproduction" in s or "online_cclkd" in s or "cclkd" in name:
        return "cclkd_yolo11_or_diag"
    if "baselines" in s or "/baseline" in s or name.startswith(("sar_", "rgb_")):
        return "baseline"
    if "comparison" in s or any(k in name for k in ("fgd", "ld_", "hallucidet")):
        return "comparison"
    if "ladd" in s or name.startswith("ladd_"):
        return "ladd"
    return "other"


def infer_method(path: Path, family: str) -> str:
    s = str(path).lower()
    name = path.parent.name.lower()
    if family == "baseline":
        if "rgb" in s or name.startswith("rgb_"):
            return "rgb_baseline"
        if "sar" in s or name.startswith("sar_"):
            return "sar_baseline"
        return "baseline"
    for method in ("hallucidet", "cclkd", "fgd", "ld"):
        if method in s or method in name:
            return method
    if family == "ladd":
        return "ladd"
    return family


def infer_validity(path: Path, family: str) -> str:
    s = str(path).lower()
    if "archive_legacy_ladd_20260618" in s:
        return "diagnostic"
    if "invalid" in s or "nc5" in s or "protocol_gap/server_artifacts/dual4090" in s:
        return "invalid_or_diagnostic"
    if family == "ladd" and "clean_a1b" in s:
        return "candidate_or_unknown"
    if any(k in s for k in ("smoke", "probe", "snapshot", "partial", "old", "diagnostic", "diag_")):
        return "diagnostic"
    if family in {"ladd", "comparison", "baseline", "cclkd_yolov5x"}:
        return "candidate_or_unknown"
    return "unknown"


def infer_model_seed_phase(run_dir: Path, meta: dict[str, str]) -> tuple[str, str, str]:
    text = " ".join([run_dir.name, str(run_dir), " ".join(f"{k}={v}" for k, v in meta.items())])
    model = ""
    m = re.search(r"yolo(?:v5)?(?:11)?([nslmx])", text, flags=re.IGNORECASE)
    if m:
        model = m.group(1).lower()
    seed = meta.get("seed", "")
    if not seed:
        sm = re.search(r"(?:^|[_-])s(\d+)(?:[_-]|$)", run_dir.name)
        if sm:
            seed = sm.group(1)
    phase = meta.get("phase", "")
    if not phase:
        pm = re.search(r"(?:^|_)(a1|a2|b1|b2|b)(?:_|$)", run_dir.name.lower())
        if pm:
            phase = pm.group(1)
    return model, seed, phase


def parse_floatish(value: str) -> float | None:
    try:
        return float(str(value).strip().strip("'\""))
    except (TypeError, ValueError):
        return None


def parse_intish(value: str) -> int | None:
    try:
        return int(float(str(value).strip().strip("'\"")))
    except (TypeError, ValueError):
        return None


def infer_mosaic_value(path: Path, meta: dict[str, str]) -> str:
    for key in ("mosaic", "manifest_mosaic", "MOSAIC"):
        if key in meta and str(meta[key]).strip() != "":
            parsed = parse_floatish(meta[key])
            return f"{parsed:g}" if parsed is not None else str(meta[key]).strip()
    s = str(path).lower()
    if "nomosaic" in s or "no-mosaic" in s:
        return "0"
    if "mosaic1" in s or "mosaic_1" in s or "mosaic-1" in s:
        return "1"
    return ""


def infer_epochs_planned(run_dir: Path, meta: dict[str, str]) -> str:
    for key in (
        "epochs",
        "manifest_epochs",
        "EPOCHS",
        "epochs_b",
        "manifest_epochs_b",
        "EPOCHS_B",
        "epochs_a2",
        "manifest_epochs_a2",
        "EPOCHS_A2",
    ):
        if key in meta:
            parsed = parse_intish(meta[key])
            if parsed is not None:
                return str(parsed)
    text = f"{run_dir.name} {run_dir}"
    patterns = (
        r"_e(\d+)(?:_|$)",
        r"(?:^|[_-])b(\d+)(?:sched|[_-]|$)",
        r"(\d+)ep",
        r"(?:^|[_-])B(\d+)(?:[_-]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def infer_protocol_fields(path: Path, family: str, validity: str, meta: dict[str, str]) -> dict[str, str]:
    s = str(path).lower()
    run_name = path.parent.name.lower()
    is_ladd_archive = family == "ladd" and "archive_legacy_ladd_20260618" in s
    is_clean_a1b = family == "ladd" and "clean_a1b" in s
    is_clean_probea = is_clean_a1b and ("dynprobe" in s or "dynamic_probe" in s or "probea" in s)
    is_clean_dynamic = is_clean_a1b and ("dynamic" in s or "_dyn" in s) and not is_clean_probea
    is_clean_static = is_clean_a1b and not is_clean_probea and not is_clean_dynamic
    mosaic = infer_mosaic_value(path, meta)
    epochs = infer_epochs_planned(path.parent, meta)
    epochs_i = parse_intish(epochs) if epochs else None

    probe_like = "probe" in s and not is_clean_probea
    smoke_like = any(k in s for k in ("smoke", "dryrun", "partial")) or probe_like or (
        epochs_i is not None and epochs_i <= 20
    )
    mosaic_f = parse_floatish(mosaic) if mosaic else None

    if is_clean_a1b and (mosaic_f == 0.0 or "nomosaic" in s):
        protocol_id = "nomosaic_clean_a1b"
    elif is_clean_a1b and (mosaic_f == 1.0 or "mosaic100" in s or "mosaic_first100" in s):
        protocol_id = "mosaic100_clean_a1b"
    elif smoke_like:
        protocol_id = "smoke_probe_partial"
    elif ("formal_nomosaic" in s or "nomosaic" in s or mosaic_f == 0.0) and epochs_i is not None and epochs_i >= 700:
        protocol_id = "formal_nomosaic_800"
    elif "formal_nomosaic" in s or "nomosaic" in s or mosaic_f == 0.0:
        protocol_id = "formal_nomosaic_compressed"
    elif "cclkd" in s and (mosaic_f == 1.0 or "mosaic1" in s):
        protocol_id = "cclkd_paper_mosaic400"
    elif mosaic_f == 1.0 or "mosaic90" in s or "mosaic" in s:
        protocol_id = "historical_mosaic_mainline"
    else:
        protocol_id = "unknown_protocol"

    if smoke_like:
        schedule_class = "smoke_probe_partial"
    elif protocol_id == "cclkd_paper_mosaic400":
        schedule_class = "paper_400"
    elif epochs_i is not None and epochs_i >= 700:
        schedule_class = "full_800"
    elif epochs_i is not None:
        schedule_class = "compressed"
    else:
        schedule_class = "unknown"

    if family.startswith("cclkd"):
        experiment_line = "cclkd_reproduction"
    elif family == "comparison":
        experiment_line = "comparison_methods"
    elif family == "ladd":
        if is_ladd_archive:
            experiment_line = "ladd_legacy_archive"
        elif is_clean_probea:
            experiment_line = "ladd_clean_a1b_mainline"
        elif is_clean_a1b:
            experiment_line = "ladd_clean_a1b_ablation"
        else:
            experiment_line = "ladd_legacy_or_unreviewed"
    elif family == "baseline":
        experiment_line = "baseline_reference"
    else:
        experiment_line = "other"

    if validity == "invalid_or_diagnostic":
        role = "archive_or_invalid"
        claim_usable = "no"
    elif is_ladd_archive:
        role = "diagnostic"
        claim_usable = "no"
    elif validity == "diagnostic" or smoke_like:
        role = "diagnostic"
        claim_usable = "partial"
    elif family == "baseline":
        role = "baseline"
        claim_usable = "yes" if protocol_id == "formal_nomosaic_800" else "partial"
    elif family == "comparison":
        role = "comparison"
        claim_usable = "yes" if protocol_id == "formal_nomosaic_800" else "partial"
    elif family == "ladd":
        if is_clean_probea and protocol_id == "mosaic100_clean_a1b":
            role = "mainline_candidate"
            claim_usable = "yes"
        elif is_clean_static or is_clean_dynamic:
            role = "ablation"
            claim_usable = "partial"
        elif is_clean_probea:
            role = "robustness_or_appendix"
            claim_usable = "partial"
        else:
            role = "diagnostic"
            claim_usable = "no"
    elif family.startswith("cclkd"):
        role = "cclkd_reproduction"
        claim_usable = "partial"
    else:
        role = "unknown"
        claim_usable = "partial"

    if any(k in run_name for k in ("old", "deprecated", "invalid")):
        role = "archive_or_invalid"
        claim_usable = "no"

    return {
        "experiment_line": experiment_line,
        "protocol_id": protocol_id,
        "mosaic": mosaic,
        "epochs_planned": epochs,
        "schedule_class": schedule_class,
        "role": role,
        "claim_usable": claim_usable,
    }


def summarize_csv(path: Path) -> dict[str, str]:
    try:
        rows = list(csv.DictReader(path.open(newline="", encoding="utf-8", errors="ignore")))
    except Exception as exc:
        return {
            "rows": "0",
            "metric_key": "",
            "best_ap": "",
            "best_epoch": "",
            "last_ap": "",
            "last_epoch": "",
            "best_ap50": "",
            "last_ap50": "",
            "csv_error": str(exc),
        }
    if not rows:
        return {
            "rows": "0",
            "metric_key": "",
            "best_ap": "",
            "best_epoch": "",
            "last_ap": "",
            "last_epoch": "",
            "best_ap50": "",
            "last_ap50": "",
            "csv_error": "",
        }
    keys = rows[0].keys()

    def values_for(candidates: tuple[str, ...]) -> tuple[str, list[tuple[str, float]]]:
        key = next((k for k in candidates if k in keys), "")
        vals: list[tuple[str, float]] = []
        if key:
            for idx, row in enumerate(rows):
                try:
                    value = float(str(row.get(key, "")).strip())
                except ValueError:
                    continue
                if math.isfinite(value):
                    epoch = row.get("epoch") or row.get("Epoch") or str(idx)
                    vals.append((str(epoch), value))
        return key, vals

    metric_key, ap_vals = values_for(AP_KEYS)
    _, ap50_vals = values_for(AP50_KEYS)
    if ap_vals:
        best_epoch, best_ap = max(ap_vals, key=lambda item: item[1])
        last_epoch, last_ap = ap_vals[-1]
    else:
        best_epoch = best_ap = last_epoch = last_ap = ""
    if ap50_vals:
        _, best_ap50 = max(ap50_vals, key=lambda item: item[1])
        _, last_ap50 = ap50_vals[-1]
    else:
        best_ap50 = last_ap50 = ""
    return {
        "rows": str(len(rows)),
        "metric_key": metric_key,
        "best_ap": f"{best_ap:.6g}" if isinstance(best_ap, float) else "",
        "best_epoch": str(best_epoch),
        "last_ap": f"{last_ap:.6g}" if isinstance(last_ap, float) else "",
        "last_epoch": str(last_epoch),
        "best_ap50": f"{best_ap50:.6g}" if isinstance(best_ap50, float) else "",
        "last_ap50": f"{last_ap50:.6g}" if isinstance(last_ap50, float) else "",
        "csv_error": "",
    }


def build_registry(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    results = sorted(p for p in root.rglob("results.csv") if ".git" not in p.parts)
    records: list[dict[str, str]] = []
    for results_path in results:
        run_dir = results_path.parent
        args_path = run_dir / "args.yaml"
        diag_path = run_dir / "ladd_diagnostics.csv"
        manifest_path = find_nearest_manifest(run_dir, root)
        args_text = read_text(args_path)
        manifest_text = read_text(manifest_path) if manifest_path else ""
        meta = parse_simple_kv(args_text)
        meta.update({f"manifest_{k}": v for k, v in parse_simple_kv(manifest_text).items()})
        family = infer_family(results_path)
        validity = infer_validity(results_path, family)
        model, seed, phase = infer_model_seed_phase(run_dir, meta)
        protocol_fields = infer_protocol_fields(results_path, family, validity, meta)
        record = {
            "run_id": run_dir.name,
            "family": family,
            "method": infer_method(results_path, family),
            "source_server": infer_server(results_path, args_text, manifest_text),
            "validity": validity,
            **protocol_fields,
            "model_size": model,
            "seed": seed,
            "phase": phase,
            "results_hash": sha1_file(results_path),
            "canonical_path": str(run_dir.relative_to(root)),
            "results_path": str(results_path.relative_to(root)),
            "args_path": str(args_path.relative_to(root)) if args_path.exists() else "",
            "diagnostics_path": str(diag_path.relative_to(root)) if diag_path.exists() else "",
            "manifest_path": str(manifest_path.relative_to(root)) if manifest_path else "",
            "weights_present": str((run_dir / "weights").exists()).lower(),
            "git_commit": meta.get("manifest_git_commit", meta.get("git_commit", "")),
            "server_tag": meta.get("manifest_server_tag", meta.get("server_tag", "")),
            "project": meta.get("project", meta.get("project_dir", "")),
            "name": meta.get("name", meta.get("run_name", "")),
        }
        record.update(summarize_csv(results_path))
        records.append(record)

    by_hash: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_hash[record["results_hash"]].append(record)

    duplicate_rows: list[dict[str, str]] = []
    for result_hash, group in sorted(by_hash.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(group) <= 1:
            continue
        canonical = sorted(group, key=lambda r: (r["validity"] != "candidate_or_unknown", len(r["canonical_path"])))[0]
        for record in group:
            duplicate_rows.append(
                {
                    "results_hash": result_hash,
                    "canonical_run_id": canonical["run_id"],
                    "canonical_path": canonical["canonical_path"],
                    "alias_run_id": record["run_id"],
                    "alias_path": record["canonical_path"],
                    "source_server": record["source_server"],
                    "family": record["family"],
                }
            )

    summary = {
        "total_results": len(records),
        "by_source_server": Counter(record["source_server"] for record in records),
        "by_family": Counter(record["family"] for record in records),
        "by_validity": Counter(record["validity"] for record in records),
        "duplicate_hash_groups": sum(1 for group in by_hash.values() if len(group) > 1),
        "files_in_duplicate_hash_groups": sum(len(group) for group in by_hash.values() if len(group) > 1),
    }
    summary = {
        key: dict(value) if isinstance(value, Counter) else value
        for key, value in summary.items()
    }
    return records, duplicate_rows, summary


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_protocol_summary(records: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        key = (
            record["experiment_line"],
            record["protocol_id"],
            record["schedule_class"],
            record["role"],
            record["claim_usable"],
        )
        groups[key].append(record)

    def format_counter(counter: Counter[str]) -> str:
        return ";".join(f"{key}:{value}" for key, value in counter.most_common())

    rows: list[dict[str, str]] = []
    for (line, protocol, schedule, role, claim_usable), group in sorted(groups.items()):
        rows.append(
            {
                "experiment_line": line,
                "protocol_id": protocol,
                "schedule_class": schedule,
                "role": role,
                "claim_usable": claim_usable,
                "num_runs": str(len(group)),
                "families": format_counter(Counter(record["family"] for record in group)),
                "methods": format_counter(Counter(record["method"] for record in group)),
                "servers": format_counter(Counter(record["source_server"] for record in group)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight local registry for LADD experiment results.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=Path("docs/experiments/registry"))
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    records, duplicates, summary = build_registry(root)
    write_csv(out_dir / "experiment_registry_20260614.csv", records)
    write_csv(out_dir / "duplicate_results_20260614.csv", duplicates)
    write_csv(out_dir / "protocol_summary_20260614.csv", build_protocol_summary(records))
    (out_dir / "experiment_registry_summary_20260614.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
