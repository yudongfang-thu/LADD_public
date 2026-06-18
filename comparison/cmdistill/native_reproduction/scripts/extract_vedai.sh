#!/usr/bin/env bash
set -euo pipefail

RESOLUTION="${1:-512}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${CMDISTILL_NATIVE_DATA_ROOT:-${REPRO_DIR}/data}"
RAW_DIR="${DATA_ROOT}/raw/VEDAI/${RESOLUTION}"
EXTRACT_DIR="${DATA_ROOT}/interim/VEDAI/${RESOLUTION}"

case "${RESOLUTION}" in
  512|1024) ;;
  *)
    echo "Usage: $0 [512|1024]" >&2
    exit 2
    ;;
esac

ann_tar="${RAW_DIR}/Annotations${RESOLUTION}.tar"
if [[ ! -f "${ann_tar}" ]]; then
  echo "[missing] ${ann_tar}" >&2
  echo "Run scripts/download_vedai.sh ${RESOLUTION} first." >&2
  exit 1
fi

shopt -s nullglob
vehicle_parts=("${RAW_DIR}/Vehicules${RESOLUTION}.tar."*)
if [[ "${#vehicle_parts[@]}" -eq 0 ]]; then
  echo "[missing] ${RAW_DIR}/Vehicules${RESOLUTION}.tar.*" >&2
  exit 1
fi

mkdir -p "${EXTRACT_DIR}"

if [[ -d "${EXTRACT_DIR}/Annotations${RESOLUTION}" ]]; then
  echo "[ok] annotations already extracted"
else
  echo "[extract] Annotations${RESOLUTION}.tar"
  tar -xf "${ann_tar}" -C "${EXTRACT_DIR}"
fi

if [[ -d "${EXTRACT_DIR}/Vehicules${RESOLUTION}" ]]; then
  echo "[ok] images already extracted"
else
  echo "[extract] Vehicules${RESOLUTION}.tar.*"
  cat "${vehicle_parts[@]}" | tar -xf - -C "${EXTRACT_DIR}"
fi

if [[ -f "${RAW_DIR}/DevKit.tar" && ! -d "${EXTRACT_DIR}/DevKit" ]]; then
  echo "[extract] DevKit.tar"
  tar -xf "${RAW_DIR}/DevKit.tar" -C "${EXTRACT_DIR}"
fi

echo "VEDAI ${RESOLUTION} extracted: ${EXTRACT_DIR}"
