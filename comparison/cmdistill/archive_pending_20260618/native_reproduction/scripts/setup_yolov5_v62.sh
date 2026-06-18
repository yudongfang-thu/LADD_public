#!/usr/bin/env bash
set -euo pipefail

YOLOV5_DIR="${YOLOV5_DIR:-/root/autodl-tmp/yolov5-v6.2}"
PYTHON_BIN="${PYTHON:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif [[ -x /root/miniconda3/bin/python ]]; then
    PYTHON_BIN="/root/miniconda3/bin/python"
  else
    echo "No Python interpreter found. Set PYTHON=/path/to/python." >&2
    exit 1
  fi
fi

if [[ ! -d "${YOLOV5_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${YOLOV5_DIR}")"
  git clone --depth 1 --branch v6.2 https://github.com/ultralytics/yolov5.git "${YOLOV5_DIR}"
else
  echo "[ok] YOLOv5 repo exists: ${YOLOV5_DIR}"
fi

mapfile -t numpy_alias_files < <(grep -RIlE 'np\.(int|float)\b' \
  "${YOLOV5_DIR}/data" "${YOLOV5_DIR}/models" "${YOLOV5_DIR}/utils" 2>/dev/null || true)
if [[ "${#numpy_alias_files[@]}" -gt 0 ]]; then
  perl -0pi -e 's/np\.int\b/int/g; s/np\.float\b/float/g' "${numpy_alias_files[@]}"
  echo "[ok] patched YOLOv5 NumPy legacy aliases"
fi

"${PYTHON_BIN}" - <<'PY'
import importlib.util
import subprocess
import sys

required = {
    "yaml": "PyYAML",
    "cv2": "opencv-python",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "Pillow",
    "psutil": "psutil",
    "scipy": "scipy",
    "seaborn": "seaborn",
    "thop": "thop",
    "torch": "torch",
    "torchvision": "torchvision",
    "tqdm": "tqdm",
}
missing = [pkg for mod, pkg in required.items() if importlib.util.find_spec(mod) is None]
if missing:
    install = [pkg for pkg in missing if pkg not in {"torch", "torchvision"}]
    if any(pkg in {"torch", "torchvision"} for pkg in missing):
        raise SystemExit(f"Missing torch packages: {missing}; install them manually to match the CUDA image.")
    print("[pip install]", " ".join(install))
    subprocess.check_call([sys.executable, "-m", "pip", "install", *install])
else:
    print("[ok] Python requirements available")
PY

if [[ ! -f "${YOLOV5_DIR}/yolov5s.pt" ]]; then
  curl -fL --retry 5 --retry-delay 5 \
    -o "${YOLOV5_DIR}/yolov5s.pt" \
    "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt"
else
  echo "[ok] yolov5s.pt exists"
fi

cd "${YOLOV5_DIR}"
"${PYTHON_BIN}" train.py --help >/dev/null
echo "YOLOv5 v6.2 ready: ${YOLOV5_DIR}"
