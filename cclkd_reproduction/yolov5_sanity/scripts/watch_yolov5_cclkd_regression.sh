#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash cclkd_reproduction/yolov5_sanity/scripts/watch_yolov5_cclkd_regression.sh \
    --candidate-results path/to/candidate/results.csv \
    --baseline-results path/to/baseline/results.csv \
    [--epoch 50] [--min-ap-gap -0.05] [--kill-pid PID]

Default rule:
  epoch >= 50 and candidate AP is more than 0.05 below baseline -> STOP.
  epoch >= 100 and candidate AP is more than 0.10 below baseline -> STRONG STOP.

This script does not kill anything unless --kill-pid is provided.
EOF
}

CANDIDATE_RESULTS=""
BASELINE_RESULTS=""
EPOCH="50"
MIN_AP_GAP="-0.05"
KILL_PID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --candidate-results)
      CANDIDATE_RESULTS="${2:?missing candidate results path}"
      shift 2
      ;;
    --baseline-results)
      BASELINE_RESULTS="${2:?missing baseline results path}"
      shift 2
      ;;
    --epoch)
      EPOCH="${2:?missing epoch}"
      shift 2
      ;;
    --min-ap-gap)
      MIN_AP_GAP="${2:?missing min AP gap}"
      shift 2
      ;;
    --kill-pid)
      KILL_PID="${2:?missing PID}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CANDIDATE_RESULTS" || -z "$BASELINE_RESULTS" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -f "$CANDIDATE_RESULTS" ]]; then
  echo "Missing candidate results: $CANDIDATE_RESULTS" >&2
  exit 2
fi
if [[ ! -f "$BASELINE_RESULTS" ]]; then
  echo "Missing baseline results: $BASELINE_RESULTS" >&2
  exit 2
fi

python3 - "$CANDIDATE_RESULTS" "$BASELINE_RESULTS" "$EPOCH" "$MIN_AP_GAP" "${KILL_PID:-}" <<'PY'
import csv
import os
import signal
import sys

candidate_path, baseline_path, target_epoch_s, min_gap_s, kill_pid_s = sys.argv[1:6]
target_epoch = int(float(target_epoch_s))
min_gap = float(min_gap_s)


def read_results(path):
    rows = {}
    with open(path, newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            clean = {k.strip(): v.strip() for k, v in row.items()}

            def get_float(*keys):
                for key in keys:
                    if key in clean and clean[key] != "":
                        return float(clean[key])
                raise KeyError(keys)

            epoch = int(float(clean.get("epoch") or clean.get("Epoch") or idx))
            rows[epoch] = {
                "AP": get_float("metrics/mAP_0.5:0.95", "metrics/mAP50-95(B)"),
                "AP50": get_float("metrics/mAP_0.5", "metrics/mAP50(B)"),
            }
    return rows


candidate = read_results(candidate_path)
baseline = read_results(baseline_path)
common = sorted(set(candidate) & set(baseline))
if not common:
    print("No common epochs found between candidate and baseline.", file=sys.stderr)
    sys.exit(2)

if target_epoch in common:
    epoch = target_epoch
else:
    epoch = min(common, key=lambda e: (abs(e - target_epoch), -e))

c = candidate[epoch]
b = baseline[epoch]
ap_gap = c["AP"] - b["AP"]
ap50_gap = c["AP50"] - b["AP50"]

print(
    "epoch={epoch} candidate_AP={c_ap:.6f} baseline_AP={b_ap:.6f} "
    "gap_AP={gap:.6f} candidate_AP50={c_ap50:.6f} baseline_AP50={b_ap50:.6f} gap_AP50={gap50:.6f}".format(
        epoch=epoch,
        c_ap=c["AP"],
        b_ap=b["AP"],
        gap=ap_gap,
        c_ap50=c["AP50"],
        b_ap50=b["AP50"],
        gap50=ap50_gap,
    )
)

stop = False
strong = False
if epoch >= 100 and ap_gap < -0.10:
    stop = True
    strong = True
elif epoch >= 50 and ap_gap < min_gap:
    stop = True

if not stop:
    print("OK: candidate is not significantly below baseline by the configured rule.")
    sys.exit(0)

if strong:
    print("STRONG STOP: candidate is significantly below baseline.")
else:
    print("STOP: candidate is significantly below baseline.")

if kill_pid_s:
    pid = int(kill_pid_s)
    print(f"Killing PID {pid} because --kill-pid was provided.")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"PID {pid} is not running.")

sys.exit(10)
PY
