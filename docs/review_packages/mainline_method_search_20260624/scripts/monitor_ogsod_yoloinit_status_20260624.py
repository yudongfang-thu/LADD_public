#!/usr/bin/env python3
"""Summarize OGSOD YOLO-init mainline search runs.

Run from the LADD_public repo root on ladd4090-zw1 or ladd3090-zw1.
"""

from __future__ import annotations

import argparse
import csv
import glob
import shlex
import subprocess
import time
from pathlib import Path


RUNS_4090 = {
    "det-only": "runs_public/dronevehicle_method_search/ogsod_nomosaic_autodl_condition/yolo_detonly/ogsod_nomix_yolo_detonly_existingcache_yolo11n_e800_b64_img256_s0_20260624_110207_b/results.csv",
    "ProbeA": "runs_public/dronevehicle_method_search/ogsod_nomosaic_autodl_condition/yolo_probeA/ogsod_nomix_yolo_probeA_existingcache_yolo11n_e800_b64_img256_s0_20260624_113044_b/results.csv",
    "dynamic": "runs_public/dronevehicle_method_search/ogsod_nomosaic_autodl_condition/yolo_dynamic/ogsod_nomix_yolo_dynamic_existingcache_yolo11n_e800_b64_img256_s0_20260624_125211_b/results.csv",
    "oldcommit_ProbeA": "runs_public/dronevehicle_method_search/ogsod_nomosaic_autodl_condition/yolo_probeA_oldcommit6f663c6b/ogsod_nomix_yolo_probeA_oldcommit6f663c6b_existingcache_yolo11n_e700_b64_img256_s0_20260624_154303_b/results.csv",
    "dynamic_kd0p5": "runs_public/ogsod/hbb/yoloinit_dynamic_sweep_20260624/dynamic_kd0p5_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_kd0p5_yoloinit_yolo11n_e800_b64_img256_s0_20260624_210844_gpu0/results.csv",
    "dynamic_reach0p5": "runs_public/ogsod/hbb/yoloinit_dynamic_sweep_20260624/dynamic_reach0p5_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_reach0p5_yoloinit_yolo11n_e800_b64_img256_s0_20260624_210844_gpu0/results.csv",
    "dynamic_srec0p05": "runs_public/ogsod/hbb/yoloinit_dynamic_sweep_20260624/dynamic_srec0p05_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_srec0p05_yoloinit_yolo11n_e800_b64_img256_s0_20260624_210844_gpu1/results.csv",
    "dynamic_teacher_projectedraw": "runs_public/ogsod/hbb/yoloinit_dynamic_sweep_20260624/dynamic_teacher_projectedraw_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_teacher_projectedraw_yoloinit_yolo11n_e800_b64_img256_s0_20260624_210844_gpu1/results.csv",
}

RUNS_3090 = {
    "detonly_control": "runs_public/ogsod/hbb/yoloinit_mainline_search_20260624/detonly_control_yoloinit/yolo11n/seed0/ogsod_yoloinit_detonly_control_yoloinit_yolo11n_e800_b64_img256_s0_20260624_161654_gpu0/results.csv",
    "singleproj": "runs_public/ogsod/hbb/yoloinit_mainline_search_20260624/dynamic_singleproj_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_singleproj_yoloinit_yolo11n_e800_b64_img256_s0_20260624_1605_gpu0/results.csv",
    "wo_s_rec": "runs_public/ogsod/hbb/yoloinit_mainline_search_20260624/dynamic_wo_s_rec_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_wo_s_rec_yoloinit_yolo11n_e800_b64_img256_s0_20260624_1608_gpu1/results.csv",
    "wo_reach": "runs_public/ogsod/hbb/yoloinit_mainline_search_20260624/dynamic_wo_reach_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_wo_reach_yoloinit_yolo11n_e800_b64_img256_s0_20260624_161654_gpu1/results.csv",
    "dynamic_plain": "runs_public/ogsod/hbb/yoloinit_mainline_search_20260624/dynamic_plain_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_plain_yoloinit_yolo11n_e800_b64_img256_s0_20260624_201305_gpu1/results.csv",
    "dynamic_kd2p0": "runs_public/ogsod/hbb/yoloinit_dynamic_sweep_20260624/dynamic_kd2p0_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_kd2p0_yoloinit_yolo11n_e800_b64_img256_s0_20260624_210845_gpu0/results.csv",
    "dynamic_corewarm60": "runs_public/ogsod/hbb/yoloinit_dynamic_sweep_20260624/dynamic_corewarm60_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_corewarm60_yoloinit_yolo11n_e800_b64_img256_s0_20260624_210845_gpu0/results.csv",
    "dynamic_kd0p25": "runs_public/ogsod/hbb/yoloinit_dynamic_sweep_20260624/dynamic_kd0p25_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_kd0p25_yoloinit_yolo11n_e800_b64_img256_s0_20260624_210845_gpu1/results.csv",
    "dynamic_reach_rawinput": "runs_public/ogsod/hbb/yoloinit_dynamic_sweep_20260624/dynamic_reach_rawinput_yoloinit/yolo11n/seed0/ogsod_yoloinit_dynamic_reach_rawinput_yoloinit_yolo11n_e800_b64_img256_s0_20260624_210845_gpu1/results.csv",
}

LOG_GLOBS_4090 = [
    "logs/dronevehicle_method_search/sub2k_seed0_fullval/autodl_condition_followups_20260624/**/*.log",
    "logs/dronevehicle_method_search/sub2k_seed0_fullval/autodl_condition_followups_20260624/**/*.out",
    "logs/dronevehicle_method_search/sub2k_seed0_fullval/autodl_condition_followups_20260624/**/*.err",
    "logs/ogsod_yoloinit_dynamic_sweep_20260624/**/*.log",
    "logs/ogsod_yoloinit_dynamic_sweep_20260624/**/*.out",
    "logs/ogsod_yoloinit_dynamic_sweep_20260624/**/*.err",
]

LOG_GLOBS_3090 = [
    "logs/ogsod_yoloinit_mainline_search_20260624/**/*.log",
    "logs/ogsod_yoloinit_mainline_search_20260624/**/*.out",
    "logs/ogsod_yoloinit_mainline_search_20260624/**/*.err",
    "logs/ogsod_yoloinit_dynamic_sweep_20260624/**/*.log",
    "logs/ogsod_yoloinit_dynamic_sweep_20260624/**/*.out",
    "logs/ogsod_yoloinit_dynamic_sweep_20260624/**/*.err",
]

BAD_PATTERNS = (
    "Traceback",
    "CUDA out of memory",
    "CUDA OOM",
    "RuntimeError",
    "NaN",
    "batch fallback",
)


def shell(cmd: str) -> str:
    return subprocess.getoutput(cmd)


def process_summary() -> str:
    raw = shell(
        "ps -eo pid,ppid,stat,etime,%cpu,%mem,args | "
        "grep -E 'train_ladd|train_ogsod|python3|python' | grep -v grep"
    )
    groups: dict[tuple[str, str, str], dict[str, object]] = {}
    other = []
    for line in raw.splitlines():
        parts = line.split(None, 6)
        if len(parts) < 7:
            other.append(line[:220])
            continue
        pid, ppid, stat, elapsed, cpu, mem, args = parts
        try:
            argv = shlex.split(args)
        except ValueError:
            argv = args.split()
        name = Path(argv[1]).name if len(argv) > 1 and argv[1].endswith(".py") else Path(argv[0]).name
        device = ""
        run_name = ""
        for idx, token in enumerate(argv):
            if token == "--device" and idx + 1 < len(argv):
                device = argv[idx + 1]
            elif token == "--name" and idx + 1 < len(argv):
                run_name = argv[idx + 1]
        if not run_name:
            other.append(f"{pid} {stat} etime={elapsed} cpu={cpu} mem={mem} {name}".strip())
            continue
        key = (name, device, run_name)
        group = groups.setdefault(
            key,
            {
                "count": 0,
                "pids": [],
                "elapsed": elapsed,
                "cpu_total": 0.0,
                "mem_total": 0.0,
                "states": {},
            },
        )
        group["count"] = int(group["count"]) + 1
        pids = group["pids"]
        assert isinstance(pids, list)
        if len(pids) < 4:
            pids.append(pid)
        try:
            group["cpu_total"] = float(group["cpu_total"]) + float(cpu)
            group["mem_total"] = float(group["mem_total"]) + float(mem)
        except ValueError:
            pass
        states = group["states"]
        assert isinstance(states, dict)
        states[stat] = int(states.get(stat, 0)) + 1

    lines = []
    if other:
        lines.append("other: " + " ; ".join(other[:5]))
    for (name, device, run_name), group in sorted(groups.items(), key=lambda item: item[0][2]):
        states = group["states"]
        assert isinstance(states, dict)
        state_text = ",".join(f"{state}:{count}" for state, count in sorted(states.items()))
        pids = ",".join(group["pids"])  # type: ignore[arg-type]
        lines.append(
            f"{run_name} | script={name} | device={device or '?'} | procs={group['count']} "
            f"| pids={pids} | etime~{group['elapsed']} | cpu_total={float(group['cpu_total']):.1f} "
            f"| mem_total={float(group['mem_total']):.1f} | states={state_text}"
        )
    return "\n".join(lines)


def metric_value(row: dict[str, str], *names: str) -> float:
    for name in names:
        if row.get(name):
            return float(row[name])
    stripped = {key.strip(): value for key, value in row.items()}
    for name in names:
        if stripped.get(name):
            return float(stripped[name])
    raise KeyError(names)


def read_results(path: str) -> list[dict[str, float]] | None:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    for idx, row in enumerate(rows, 1):
        parsed.append(
            {
                "idx": float(idx),
                "ap50": metric_value(row, "metrics/mAP50(B)", "metrics/mAP50"),
                "ap5095": metric_value(row, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
            }
        )
    return parsed


def mean_last(rows: list[dict[str, float]], count: int) -> float:
    values = [row["ap5095"] for row in rows[-min(count, len(rows)) :]]
    return sum(values) / len(values)


def delta_late(candidate: list[dict[str, float]], control: list[dict[str, float]], matched: int, count: int) -> float:
    start = max(0, matched - count)
    values = [candidate[i]["ap5095"] - control[i]["ap5095"] for i in range(start, matched)]
    return sum(values) / len(values)


def status_for(matched: int, latest_delta: float, late20_delta: float) -> str:
    if matched < 100:
        return "pre100"
    if late20_delta >= 0.020 and latest_delta > 0:
        return "STRONG_EARLY"
    if late20_delta >= 0.010 and latest_delta > 0:
        return "PROMISING_EARLY"
    if matched >= 120 and late20_delta <= 0:
        return "LOW_PRIORITY"
    return "WATCH"


def scan_logs(patterns: list[str]) -> tuple[int, list[tuple[str, list[str]]]]:
    files: set[str] = set()
    for pattern in patterns:
        files.update(glob.glob(pattern, recursive=True))
    bad = []
    for file_name in sorted(files):
        try:
            text = Path(file_name).read_text(errors="ignore")[-30000:]
        except OSError:
            continue
        hits = [pattern for pattern in BAD_PATTERNS if pattern in text]
        if hits:
            bad.append((file_name, hits))
    return len(files), bad


def summarize(server: str) -> None:
    if server == "4090":
        runs = RUNS_4090
        control_name = "det-only"
        log_globs = LOG_GLOBS_4090
        label = "ladd4090-zw1"
    else:
        runs = RUNS_3090
        control_name = "detonly_control"
        log_globs = LOG_GLOBS_3090
        label = "ladd3090-zw1"

    data = {name: read_results(path) for name, path in runs.items()}
    control = data.get(control_name)

    print(f"SERVER {label}")
    print("TIME", time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    print("GPU")
    print(shell("nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"))
    print("PROCS")
    print(process_summary())
    print("RESULTS")
    for name, rows in data.items():
        if rows is None:
            print(f"{name} | MISSING")
            continue
        latest = rows[-1]
        best = max(rows, key=lambda row: row["ap5095"])
        late = {count: mean_last(rows, count) for count in (5, 10, 20, 50)}
        fields = [
            name,
            f"rows={len(rows)}",
            f"latest={latest['ap5095']:.5f}/{latest['ap50']:.5f}",
            f"best={best['ap5095']:.5f}@{int(best['idx'])}",
            "late=" + "/".join(f"{late[count]:.5f}" for count in (5, 10, 20, 50)),
        ]
        if name != control_name and control:
            matched = min(len(rows), len(control))
            latest_delta = rows[matched - 1]["ap5095"] - control[matched - 1]["ap5095"]
            late20_delta = delta_late(rows, control, matched, 20)
            epoch100_delta = rows[99]["ap5095"] - control[99]["ap5095"] if matched >= 100 else None
            positive = sum(1 for i in range(matched) if rows[i]["ap5095"] > control[i]["ap5095"])
            fields.extend(
                [
                    f"matched={matched}",
                    f"d_latest={latest_delta:+.5f}",
                    f"d_late20={late20_delta:+.5f}",
                    "d_epoch100=" + (f"{epoch100_delta:+.5f}" if epoch100_delta is not None else "NA"),
                    f"pos={positive}/{matched}",
                    f"status={status_for(matched, latest_delta, late20_delta)}",
                ]
            )
        print(" | ".join(fields))
    print("LOGSCAN")
    log_count, bad = scan_logs(log_globs)
    print(f"log_count={log_count}")
    print(f"bad={bad[:20]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", choices=("4090", "3090"), required=True)
    args = parser.parse_args()
    summarize(args.server)


if __name__ == "__main__":
    main()
