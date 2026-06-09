#!/usr/bin/env python3
"""Sanity-check OGSOD YOLOv5 HBB dataset yaml files."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - runtime fallback for minimal envs
    yaml = None

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_NC = 3
EXPECTED_COLD_NAMES = ["oil_tank", "bridge", "harbor"]
PAPER_TRAIN = 14665
PAPER_TEST = 3666
KNOWN_ALT_SPLIT = (14664, 3667)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="SAR YOLO dataset yaml")
    parser.add_argument("--teacher-data", default=None, help="Optional RGB YOLO dataset yaml")
    parser.add_argument(
        "--output-dir",
        default="cclkd_reproduction/yolov5_sanity/results",
        help="Directory for dataset_sanity_<modality>.json/md",
    )
    parser.add_argument("--sample-labels", type=int, default=20)
    parser.add_argument("--sample-images", type=int, default=20)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read dataset yaml files.")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def names_to_list(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, list):
        return [str(x) for x in names]
    return []


def infer_modality(path: Path, yaml_data: dict[str, Any], fallback: str) -> str:
    text = f"{path} {yaml_data.get('path', '')}".lower()
    if "rgb" in text:
        return "rgb"
    if "sar" in text:
        return "sar"
    return fallback


def split_entries(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def resolve_entry(root: Path, entry: str) -> Path:
    path = Path(entry)
    if path.is_absolute():
        return path
    return root / path


def read_image_list(root: Path, entry: str) -> tuple[list[Path], list[str]]:
    warnings: list[str] = []
    resolved = resolve_entry(root, entry)
    if resolved.is_file() and resolved.suffix == ".txt":
        images: list[Path] = []
        for line in resolved.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            img = Path(line)
            if not img.is_absolute():
                img = resolved.parent / img
            images.append(img)
        return images, warnings
    if resolved.is_dir():
        return sorted(p for p in resolved.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES), warnings
    warnings.append(f"Image split path does not exist: {resolved}")
    return [], warnings


def label_path_for_image(image: Path) -> Path:
    parts = list(image.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image.parent.parent / "labels" / image.parent.name / f"{image.stem}.txt"


def label_roots_for_entry(root: Path, entry: str) -> list[Path]:
    resolved = resolve_entry(root, entry)
    candidates: list[Path] = []
    if resolved.is_file() and resolved.suffix == ".txt":
        images, _ = read_image_list(root, entry)
        return [label_path_for_image(img) for img in images]
    parts = list(resolved.parts)
    for idx, part in enumerate(parts):
        if part == "images":
            parts[idx] = "labels"
            candidates.append(Path(*parts))
            break
    if not candidates:
        candidates.append(resolved.parent.parent / "labels" / resolved.name)
    return candidates


def collect_label_files(root: Path, entries: list[str]) -> list[Path]:
    files: list[Path] = []
    for entry in entries:
        for label_root in label_roots_for_entry(root, entry):
            if label_root.is_file():
                files.append(label_root)
            elif label_root.is_dir():
                files.extend(sorted(label_root.rglob("*.txt")))
    return sorted(set(files))


def inspect_labels(label_files: list[Path], sample_n: int) -> tuple[Counter[int], list[str], list[str]]:
    counts: Counter[int] = Counter()
    warnings: list[str] = []
    errors: list[str] = []
    sample = random.Random(0).sample(label_files, min(sample_n, len(label_files))) if label_files else []

    for label in label_files:
        try:
            lines = label.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = label.read_text(errors="replace").splitlines()
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            try:
                cls = int(float(parts[0]))
            except Exception:
                errors.append(f"Invalid class id in {label}: {line}")
                continue
            counts[cls] += 1

    for label in sample:
        try:
            lines = label.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = label.read_text(errors="replace").splitlines()
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                errors.append(f"Label row has fewer than 5 columns in {label}: {line}")
                continue
            try:
                cls = int(float(parts[0]))
                x, y, w, h = map(float, parts[1:5])
            except Exception:
                errors.append(f"Could not parse label row in {label}: {line}")
                continue
            if not (0 <= cls < EXPECTED_NC):
                errors.append(f"Class id out of range [0, 2] in {label}: {line}")
            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                errors.append(f"xywh not normalized to [0, 1] in {label}: {line}")
            if w <= 0 or h <= 0:
                errors.append(f"bbox width/height must be > 0 in {label}: {line}")

    if not label_files:
        warnings.append("No label files found; dataset root may be unavailable on this machine.")
    return counts, warnings, errors


def inspect_image_sizes(images: list[Path], sample_n: int) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if not images:
        return [], ["No images available for dimension sampling."]
    try:
        from PIL import Image
    except Exception as exc:
        return [], [f"PIL is unavailable; skipped image size sampling: {exc}"]

    sample = random.Random(0).sample(images, min(sample_n, len(images)))
    sizes: list[dict[str, Any]] = []
    for image in sample:
        try:
            with Image.open(image) as im:
                sizes.append({"path": str(image), "width": im.width, "height": im.height})
        except Exception as exc:
            warnings.append(f"Could not read image size for {image}: {exc}")
    return sizes, warnings


def compare_split_counts(train_count: int | None, test_count: int | None) -> list[str]:
    warnings: list[str] = []
    if train_count is None or test_count is None:
        return warnings
    if (train_count, test_count) == (PAPER_TRAIN, PAPER_TEST):
        return warnings
    if (train_count, test_count) == KNOWN_ALT_SPLIT:
        warnings.append(
            "Split count is train 14664 / test 3667 rather than paper train 14665 / test 3666; record this in result interpretation."
        )
    else:
        warnings.append(
            f"Split count differs from CoLD/CCLKD paper: train {train_count}, test {test_count}; expected {PAPER_TRAIN}/{PAPER_TEST}."
        )
    return warnings


def check_one(yaml_path: Path, output_dir: Path, fallback_modality: str) -> tuple[dict[str, Any], bool]:
    errors: list[str] = []
    warnings: list[str] = []

    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset yaml does not exist: {yaml_path}")

    data = load_yaml(yaml_path)
    modality = infer_modality(yaml_path, data, fallback_modality)
    names = names_to_list(data.get("names"))
    nc = data.get("nc")
    if nc != EXPECTED_NC:
        errors.append(f"nc must be 3, got {nc}")
    if len(names) != EXPECTED_NC:
        warnings.append(f"names should contain 3 classes, got {names}")
    if names != EXPECTED_COLD_NAMES:
        warnings.append(
            "Class order differs from CoLD table order Oil Tank / Bridge / Harbor. "
            f"YAML order is: {names}. Use mapping Oil Tank -> storage_tank, Bridge -> bridge, Harbor -> harbor when reporting per-class AP."
        )

    root_value = data.get("path", yaml_path.parent)
    root = Path(str(root_value))
    if not root.is_absolute():
        root = yaml_path.parent / root

    split_report: dict[str, Any] = {}
    split_images: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        entries = split_entries(data.get(split))
        images: list[Path] = []
        split_warnings: list[str] = []
        for entry in entries:
            got, got_warnings = read_image_list(root, entry)
            images.extend(got)
            split_warnings.extend(got_warnings)
        split_images[split] = sorted(set(images))
        split_report[split] = {
            "entries": entries,
            "image_count": len(split_images[split]),
            "warnings": split_warnings,
        }
        warnings.extend(split_warnings)

    test_images = split_images["test"] or split_images["val"]
    warnings.extend(compare_split_counts(len(split_images["train"]), len(test_images)))

    train_label_files = collect_label_files(root, split_entries(data.get("train")))
    eval_entries = split_entries(data.get("test")) or split_entries(data.get("val"))
    eval_label_files = collect_label_files(root, eval_entries)
    label_files = sorted(set(train_label_files + eval_label_files))
    instance_counts, label_warnings, label_errors = inspect_labels(label_files, sample_n=20)
    warnings.extend(label_warnings)
    errors.extend(label_errors)

    sampled_images = split_images["train"] + test_images
    image_sizes, image_warnings = inspect_image_sizes(sampled_images, sample_n=20)
    warnings.extend(image_warnings)

    report = {
        "yaml": str(yaml_path),
        "modality": modality,
        "root": str(root),
        "nc": nc,
        "names": names,
        "class_mapping_for_cold": {
            "Oil Tank": "storage_tank",
            "Bridge": "bridge",
            "Harbor": "harbor",
        },
        "splits": split_report,
        "label_file_count": len(label_files),
        "train_label_file_count": len(train_label_files),
        "eval_label_file_count": len(eval_label_files),
        "instance_counts_by_class_id": {str(k): v for k, v in sorted(instance_counts.items())},
        "instance_counts_by_name": {
            names[k] if 0 <= k < len(names) else str(k): v for k, v in sorted(instance_counts.items())
        },
        "sampled_image_sizes": image_sizes,
        "paper_split_reference": {"train": PAPER_TRAIN, "test": PAPER_TEST},
        "warnings": warnings,
        "errors": errors,
        "status": "fail" if errors else "warning" if warnings else "ok",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"dataset_sanity_{modality}.json"
    md_path = output_dir / f"dataset_sanity_{modality}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return report, bool(errors)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Dataset sanity: {report['modality']}",
        "",
        f"- yaml: `{report['yaml']}`",
        f"- root: `{report['root']}`",
        f"- status: **{report['status']}**",
        f"- nc: {report['nc']}",
        f"- names: {', '.join(report['names'])}",
        f"- label files: {report['label_file_count']}",
        "",
        "## Split Counts",
        "",
        "| split | images | entries |",
        "|---|---:|---|",
    ]
    for split, info in report["splits"].items():
        lines.append(f"| {split} | {info['image_count']} | `{info['entries']}` |")
    lines.extend(["", "## Instance Counts", "", "| class id/name | instances |", "|---|---:|"])
    for name, count in report["instance_counts_by_name"].items():
        lines.append(f"| {name} | {count} |")
    lines.extend(["", "## CoLD Class Mapping", ""])
    for paper_name, yaml_name in report["class_mapping_for_cold"].items():
        lines.append(f"- {paper_name} -> `{yaml_name}`")
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in report["warnings"])
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {e}" for e in report["errors"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    targets = [(Path(args.data), "sar")]
    if args.teacher_data:
        targets.append((Path(args.teacher_data), "rgb"))

    any_fail = False
    for yaml_path, fallback in targets:
        report, failed = check_one(yaml_path, output_dir, fallback)
        any_fail = any_fail or failed
        print(f"{report['modality']}: {report['status']} -> {output_dir / ('dataset_sanity_' + report['modality'] + '.json')}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
