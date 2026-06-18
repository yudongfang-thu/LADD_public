#!/usr/bin/env python3
"""Prepare VEDAI as paired RGB/IR YOLO HBB datasets for CMDistill evidence runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image


CLASS_ID_TO_NAME = {
    1: "car",
    11: "pickup",
    5: "camper",
    2: "truck",
    10: "other",
    4: "tractor",
    23: "boat",
    9: "van",
}
CLASS_NAMES = list(CLASS_ID_TO_NAME.values())
CLASS_ID_TO_INDEX = {class_id: idx for idx, class_id in enumerate(CLASS_ID_TO_NAME)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", default="512", choices=["512", "1024"])
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Defaults to CMDISTILL_NATIVE_DATA_ROOT or native_reproduction/data.",
    )
    parser.add_argument(
        "--split",
        default="paper80_seed0",
        choices=["paper80_seed0", "official_fold01"],
        help="paper80_seed0 follows the CMDistill paper's approximate 8:2 split; official_fold01 uses VEDAI fold01.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="Use symlinks by default to avoid duplicating the image archive.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--check-images", type=int, default=8)
    return parser.parse_args()


def default_data_root() -> Path:
    script = Path(__file__).resolve()
    repro_dir = script.parents[1]
    return Path(os.environ.get("CMDISTILL_NATIVE_DATA_ROOT", repro_dir / "data")).resolve()


def read_ids_from_file(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if item:
                ids.append(item.zfill(8))
    return ids


def strict_image_ids(image_dir: Path, suffix: str) -> set[str]:
    ids: set[str] = set()
    for path in image_dir.glob(f"*_{suffix}.png"):
        stem = path.name[:8]
        if len(stem) == 8 and stem.isdigit() and path.name == f"{stem}_{suffix}.png":
            ids.add(stem)
    return ids


def load_annotations(annotation_file: Path, image_size: int) -> tuple[dict[str, list[str]], Counter, Counter]:
    labels: dict[str, list[str]] = defaultdict(list)
    kept_counter: Counter = Counter()
    ignored_counter: Counter = Counter()

    with annotation_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 15:
                raise ValueError(f"{annotation_file}:{line_no}: expected 15 fields, got {len(parts)}")

            image_id = parts[0].zfill(8)
            xs = [float(v) for v in parts[4:8]]
            ys = [float(v) for v in parts[8:12]]
            class_id = int(float(parts[12]))

            if class_id not in CLASS_ID_TO_INDEX:
                ignored_counter[class_id] += 1
                continue

            x_min = max(0.0, min(xs))
            y_min = max(0.0, min(ys))
            x_max = min(float(image_size), max(xs))
            y_max = min(float(image_size), max(ys))
            width = max(0.0, x_max - x_min)
            height = max(0.0, y_max - y_min)
            if width <= 0 or height <= 0:
                ignored_counter[class_id] += 1
                continue

            x_center = (x_min + x_max) / 2.0 / image_size
            y_center = (y_min + y_max) / 2.0 / image_size
            norm_width = width / image_size
            norm_height = height / image_size
            class_index = CLASS_ID_TO_INDEX[class_id]
            labels[image_id].append(
                f"{class_index} {x_center:.8f} {y_center:.8f} {norm_width:.8f} {norm_height:.8f}"
            )
            kept_counter[class_id] += 1

    return labels, kept_counter, ignored_counter


def make_split(
    ids: list[str],
    annotations_dir: Path,
    split: str,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    ids = sorted(ids)
    if split == "official_fold01":
        train = [item for item in read_ids_from_file(annotations_dir / "fold01.txt") if item in ids]
        val = [item for item in read_ids_from_file(annotations_dir / "fold01test.txt") if item in ids]
        return train, val

    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    val_count = round(len(shuffled) * val_ratio)
    val = sorted(shuffled[:val_count])
    train = sorted(shuffled[val_count:])
    return train, val


def reset_output(root: Path, overwrite: bool) -> None:
    if not root.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{root} exists; pass --overwrite to regenerate")
    for child in ["images", "labels", "configs", "metadata", "splits"]:
        target = root / child
        if target.exists():
            shutil.rmtree(target)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        rel = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(rel)


def write_yaml(path: Path, dataset_root: Path, modality: str) -> None:
    names = ", ".join(f"'{name}'" for name in CLASS_NAMES)
    content = (
        f"path: {dataset_root}\n"
        f"train: images/{modality}/train\n"
        f"val: images/{modality}/val\n"
        f"test: images/{modality}/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: [{names}]\n"
    )
    path.write_text(content, encoding="utf-8")


def sample_check_images(paths: Iterable[Path], limit: int) -> list[dict[str, object]]:
    checked = []
    for path in list(paths)[:limit]:
        with Image.open(path) as im:
            checked.append({"path": str(path), "size": list(im.size), "mode": im.mode})
    return checked


def main() -> None:
    args = parse_args()
    data_root = (args.data_root or default_data_root()).resolve()
    resolution = int(args.resolution)
    extracted_root = data_root / "interim" / "VEDAI" / args.resolution
    annotations_dir = extracted_root / f"Annotations{args.resolution}"
    images_dir = extracted_root / f"Vehicules{args.resolution}"
    annotation_file = annotations_dir / f"annotation{args.resolution}.txt"

    if not annotation_file.exists() or not images_dir.exists():
        raise FileNotFoundError(
            f"Missing extracted VEDAI files under {extracted_root}. "
            f"Run scripts/extract_vedai.sh {args.resolution} first."
        )

    rgb_ids = strict_image_ids(images_dir, "co")
    ir_ids = strict_image_ids(images_dir, "ir")
    paired_ids = sorted(rgb_ids & ir_ids)
    if not paired_ids:
        raise RuntimeError(f"No strict RGB/IR pairs found in {images_dir}")

    labels_by_id, kept_counter, ignored_counter = load_annotations(annotation_file, resolution)
    sample_ids = paired_ids
    images_with_kept_labels = sorted([image_id for image_id in paired_ids if image_id in labels_by_id])
    train_ids, val_ids = make_split(sample_ids, annotations_dir, args.split, args.val_ratio, args.seed)

    output_root = data_root / "processed" / f"VEDAI{args.resolution}_paper8_hbb_{args.split}"
    reset_output(output_root, args.overwrite)
    output_root.mkdir(parents=True, exist_ok=True)

    split_ids = {"train": train_ids, "val": val_ids}
    per_split_counts: dict[str, Counter] = {}
    paired_rows = []

    for split_name, ids in split_ids.items():
        split_counter: Counter = Counter()
        (output_root / "splits" / f"{split_name}.txt").parent.mkdir(parents=True, exist_ok=True)
        (output_root / "splits" / f"{split_name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

        for image_id in ids:
            label_lines = labels_by_id.get(image_id, [])
            for line in label_lines:
                split_counter[int(line.split()[0])] += 1

            for modality, suffix in [("rgb", "co"), ("ir", "ir")]:
                src_image = images_dir / f"{image_id}_{suffix}.png"
                dst_image = output_root / "images" / modality / split_name / f"{image_id}.png"
                link_or_copy(src_image, dst_image, args.link_mode)

                dst_label = output_root / "labels" / modality / split_name / f"{image_id}.txt"
                dst_label.parent.mkdir(parents=True, exist_ok=True)
                dst_label.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

            paired_rows.append(
                {
                    "split": split_name,
                    "image_id": image_id,
                    "rgb_image": f"images/rgb/{split_name}/{image_id}.png",
                    "ir_image": f"images/ir/{split_name}/{image_id}.png",
                    "label": f"labels/rgb/{split_name}/{image_id}.txt",
                    "num_labels": len(label_lines),
                }
            )

        per_split_counts[split_name] = split_counter

    configs_dir = output_root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    write_yaml(configs_dir / f"vedai{args.resolution}_rgb_hbb.yaml", output_root, "rgb")
    write_yaml(configs_dir / f"vedai{args.resolution}_ir_hbb.yaml", output_root, "ir")

    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (metadata_dir / "paired_samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "image_id", "rgb_image", "ir_image", "label", "num_labels"],
        )
        writer.writeheader()
        writer.writerows(paired_rows)

    with (metadata_dir / "class_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class_index", "class_name", "count"])
        for split_name in ["train", "val"]:
            for idx, name in enumerate(CLASS_NAMES):
                writer.writerow([split_name, idx, name, per_split_counts[split_name].get(idx, 0)])

    check_paths = [output_root / row["rgb_image"] for row in paired_rows] + [
        output_root / row["ir_image"] for row in paired_rows
    ]
    metadata = {
        "dataset": "VEDAI",
        "resolution": resolution,
        "source_extracted_root": str(extracted_root),
        "output_root": str(output_root),
        "split": args.split,
        "seed": args.seed,
        "val_ratio": args.val_ratio if args.split == "paper80_seed0" else None,
        "link_mode": args.link_mode,
        "class_id_to_index": {str(k): v for k, v in CLASS_ID_TO_INDEX.items()},
        "class_names": CLASS_NAMES,
        "strict_rgb_images": len(rgb_ids),
        "strict_ir_images": len(ir_ids),
        "paired_images": len(paired_ids),
        "sample_images": len(sample_ids),
        "images_with_kept_labels": len(images_with_kept_labels),
        "train_images": len(train_ids),
        "val_images": len(val_ids),
        "kept_original_class_counts": {str(k): v for k, v in sorted(kept_counter.items())},
        "ignored_original_class_counts": {str(k): v for k, v in sorted(ignored_counter.items())},
        "checked_images": sample_check_images(check_paths, args.check_images),
        "bbox_policy": "oriented quadrilateral min/max converted to axis-aligned HBB, clamped to image bounds",
    }
    (metadata_dir / "prepare_manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
