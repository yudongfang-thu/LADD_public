#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/paper/prepare_paper_dataset_yamls.sh /path/to/OGSOD-1.0

Generates:
  configs/paper/datasets/ogsod_hbb_sar.yaml
  configs/paper/datasets/ogsod_hbb_rgb.yaml

Expected layout:
  <root>/sar/images/train
  <root>/sar/images/test
  <root>/rgb/images/train
  <root>/rgb/images/test
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DATA_ROOT="${1:-}"
[[ -n "$DATA_ROOT" ]] || { usage >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

python3 - "$DATA_ROOT" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
if not root.is_dir():
    raise SystemExit(f"Dataset root does not exist: {root}")

classes = ["bridge", "harbor", "storage_tank"]
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def count_images(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in image_exts)


counts: dict[tuple[str, str], int] = {}
for modality in ("sar", "rgb"):
    for split in ("train", "test"):
        image_dir = root / modality / "images" / split
        if not image_dir.is_dir():
            raise SystemExit(f"Missing image directory: {image_dir}")
        counts[(modality, split)] = count_images(image_dir)
        if counts[(modality, split)] <= 0:
            raise SystemExit(f"No images found under {image_dir}")

for split in ("train", "test"):
    if counts[("sar", split)] != counts[("rgb", split)]:
        raise SystemExit(
            f"SAR/RGB {split} image count mismatch: "
            f"{counts[('sar', split)]} vs {counts[('rgb', split)]}"
        )

out_dir = Path("configs/paper/datasets")
out_dir.mkdir(parents=True, exist_ok=True)

for modality in ("sar", "rgb"):
    path = out_dir / f"ogsod_hbb_{modality}.yaml"
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(classes))
    path.write_text(
        f"path: {root / modality}\n"
        "train: images/train\n"
        "val: images/test\n"
        "test: images/test\n"
        "\n"
        "nc: 3\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )
    print(f"Wrote {path}")

print("OK: SAR/RGB train/test image counts match and class order is fixed.")
PY
