#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.is_file():
        return rows
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float] = {}
            for key, value in row.items():
                try:
                    parsed[key.strip()] = float(value)
                except (TypeError, ValueError):
                    pass
            if parsed:
                rows.append(parsed)
    return rows


def log_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%F %T")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {text}\n")
    print(f"[{stamp}] {text}", flush=True)


def screen_exists(screen_name: str) -> bool:
    out = subprocess.run(["screen", "-ls"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return screen_name in out.stdout


def stop_target(screen_name: str, target_pattern: str, log_path: Path) -> None:
    log_line(log_path, f"STOP requested for screen={screen_name} pattern={target_pattern}")
    subprocess.run(["screen", "-S", screen_name, "-X", "quit"], check=False)
    time.sleep(3)
    if target_pattern:
        current_pid = os.getpid()
        out = subprocess.run(["pgrep", "-f", target_pattern],
                             text=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL,
                             check=False)
        pids = [int(line) for line in out.stdout.splitlines() if line.strip().isdigit()]
        for pid in pids:
            if pid != current_pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        time.sleep(2)
        for pid in pids:
            if pid != current_pid:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def decide(rows: list[dict[str, float]], args: argparse.Namespace) -> tuple[str, str]:
    if not rows:
        return "WAIT", "no results.csv rows yet"
    if "metrics/mAP_0.5" not in rows[-1] or "epoch" not in rows[-1]:
        return "WAIT", "required mAP50/epoch columns missing"

    last_epoch = int(rows[-1]["epoch"])
    latest = rows[-1]["metrics/mAP_0.5"]
    best = max(row["metrics/mAP_0.5"] for row in rows if "metrics/mAP_0.5" in row)
    best_epoch = int(max(rows, key=lambda row: row.get("metrics/mAP_0.5", -1.0))["epoch"])

    window = min(args.window, len(rows))
    if len(rows) > window:
        prev_best = max(row.get("metrics/mAP_0.5", -1.0) for row in rows[:-window])
    else:
        prev_best = 0.0
    recent_best = max(row.get("metrics/mAP_0.5", -1.0) for row in rows[-window:])
    improvement = best - prev_best

    reason = (
        f"epoch={last_epoch} latest={latest:.4f} best={best:.4f}@{best_epoch} "
        f"recent_best={recent_best:.4f} prev_best={prev_best:.4f} improve={improvement:.4f}"
    )

    if last_epoch >= args.min_epoch and len(rows) > window and improvement < args.min_improvement:
        if best < args.min_best_map50:
            return "STOP", reason + " below plateau threshold"
        return "STOP", reason + " plateau after reaching target threshold"
    return "CONTINUE", reason


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--target-screen", required=True)
    parser.add_argument("--target-pattern", required=True)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=600)
    parser.add_argument("--min-epoch", type=int, default=120)
    parser.add_argument("--window", type=int, default=35)
    parser.add_argument("--min-best-map50", type=float, default=0.62)
    parser.add_argument("--min-improvement", type=float, default=0.008)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_line(args.log, f"watch start results={args.results} target_screen={args.target_screen}")
    while True:
        rows = read_rows(args.results)
        decision, reason = decide(rows, args)
        log_line(args.log, f"{decision}: {reason}")
        if decision == "STOP":
            stop_target(args.target_screen, args.target_pattern, args.log)
            return
        if rows and int(rows[-1].get("epoch", -1)) >= 299:
            log_line(args.log, "training reached final epoch; watcher exiting")
            return
        if not screen_exists(args.target_screen):
            log_line(args.log, "target screen no longer exists; watcher exiting")
            return
        time.sleep(max(args.poll_seconds, 30))


if __name__ == "__main__":
    main()
