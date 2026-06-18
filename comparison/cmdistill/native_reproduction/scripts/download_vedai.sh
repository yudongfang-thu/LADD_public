#!/usr/bin/env bash
set -euo pipefail

RESOLUTION="${1:-512}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_ROOT="${CMDISTILL_NATIVE_DATA_ROOT:-${REPRO_DIR}/data}"
BASE_URL="https://downloads.greyc.fr/vedai"
OUT_DIR="${DATA_ROOT}/raw/VEDAI/${RESOLUTION}"

case "${RESOLUTION}" in
  512)
    FILES=(
      "Annotations512.tar:1753088"
      "Vehicules512.tar.001:699400192"
      "Vehicules512.tar.002:593733632"
      "DevKit.tar:543232"
      "TermsandConditionsofUseVeDAI2014.pdf:53320"
    )
    ;;
  1024)
    FILES=(
      "Annotations1024.tar:1768960"
      "Vehicules1024.tar.001:699400192"
      "Vehicules1024.tar.002:699400192"
      "Vehicules1024.tar.003:699400192"
      "Vehicules1024.tar.004:699400192"
      "Vehicules1024.tar.005:93268992"
      "DevKit.tar:543232"
      "TermsandConditionsofUseVeDAI2014.pdf:53320"
    )
    ;;
  *)
    echo "Usage: $0 [512|1024]" >&2
    exit 2
    ;;
esac

mkdir -p "${OUT_DIR}"

file_size() {
  wc -c < "$1" | tr -d ' '
}

download_one() {
  local item="$1"
  local file="${item%%:*}"
  local expected="${item##*:}"
  local target="${OUT_DIR}/${file}"

  if [[ -f "${target}" ]]; then
    local actual
    actual="$(file_size "${target}")"
    if [[ "${actual}" == "${expected}" ]]; then
      echo "[ok] ${file} (${actual} bytes)"
      return
    fi
    echo "[warn] ${file} exists but size=${actual}, expected=${expected}; resuming"
  fi

  echo "[download] ${file}"
  curl -fL --retry 5 --retry-delay 5 -C - \
    -o "${target}" \
    "${BASE_URL}/${file}"

  local actual
  actual="$(file_size "${target}")"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "[error] ${file} size=${actual}, expected=${expected}" >&2
    exit 1
  fi
}

for item in "${FILES[@]}"; do
  download_one "${item}"
done

cat > "${OUT_DIR}/DOWNLOAD_MANIFEST.txt" <<EOF
Dataset: VEDAI
Resolution release: ${RESOLUTION}
Source: ${BASE_URL}/
Downloaded at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
Data root: ${DATA_ROOT}
Files:
$(printf '  %s\n' "${FILES[@]}")
EOF

echo
echo "VEDAI ${RESOLUTION} download complete: ${OUT_DIR}"

if [[ "${EXTRACT:-0}" == "1" ]]; then
  EXTRACT_DIR="${DATA_ROOT}/raw/VEDAI/${RESOLUTION}_extracted"
  mkdir -p "${EXTRACT_DIR}"
  echo "[extract] ${EXTRACT_DIR}"
  cat "${OUT_DIR}/Vehicules${RESOLUTION}.tar."* > "${OUT_DIR}/Vehicules${RESOLUTION}.tar"
  tar -xf "${OUT_DIR}/Vehicules${RESOLUTION}.tar" -C "${EXTRACT_DIR}"
  tar -xf "${OUT_DIR}/Annotations${RESOLUTION}.tar" -C "${EXTRACT_DIR}"
  tar -xf "${OUT_DIR}/DevKit.tar" -C "${EXTRACT_DIR}"
  echo "Extraction complete: ${EXTRACT_DIR}"
fi
