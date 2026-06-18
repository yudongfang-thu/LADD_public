#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${CMDISTILL_NATIVE_DATA_ROOT:-${REPRO_DIR}/data}"
DV_DIR="${DATA_ROOT}/raw/DroneVehicle"

echo "DroneVehicle root: ${DV_DIR}"

if [[ ! -d "${DV_DIR}" ]]; then
  cat >&2 <<EOF
[missing] ${DV_DIR}

Please download the official BaiduYun splits and place them under:
  ${DV_DIR}/train
  ${DV_DIR}/val
  ${DV_DIR}/test

Official links:
  Train      https://pan.baidu.com/s/1ptZCJ1mKYqFnMnsgqEyoGg  code: ngar
  Validation https://pan.baidu.com/s/1e6e9mESZecpME4IEdU8t3Q  code: jnj6
  Test       https://pan.baidu.com/s/1JlXO4jEUQgkR1Vco1hfKhg  code: tqwc
EOF
  exit 1
fi

for split in train val validation test; do
  if [[ -d "${DV_DIR}/${split}" ]]; then
    count="$(find "${DV_DIR}/${split}" -type f | wc -l | tr -d ' ')"
    echo "[found] ${split}: ${count} files"
  fi
done

image_count="$(find "${DV_DIR}" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) | wc -l | tr -d ' ')"
label_count="$(find "${DV_DIR}" -type f \( -iname '*.txt' -o -iname '*.xml' -o -iname '*.json' \) | wc -l | tr -d ' ')"

echo "Images found: ${image_count}"
echo "Label-like files found: ${label_count}"

if [[ "${image_count}" -eq 0 ]]; then
  echo "[warn] No images found yet. The archives may still need extraction." >&2
fi

if [[ "${label_count}" -eq 0 ]]; then
  echo "[warn] No label-like files found yet. Check whether labels are packaged separately." >&2
fi
