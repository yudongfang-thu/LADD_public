#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


CORE_COLUMNS = {
    "effective_alpha_kd": "base_alpha_kd",
    "effective_alpha_s_rec": "base_alpha_s_rec",
    "effective_alpha_sep": "base_alpha_sep",
    "effective_lambda_residual_aux": "base_lambda_residual_aux",
}

REQUIRED_COLUMNS = (
    "epoch",
    "effective_alpha_kd",
    "effective_alpha_s_rec",
    "effective_alpha_sep",
    "effective_lambda_residual_aux",
    "b_loss_warmup_multiplier",
    "b_loss_warmup_active",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check LADD warmup diagnostics from ladd_diagnostics.csv.")
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--mode", choices=("core", "kd"), default="core")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--base-alpha-kd", type=float, default=1.0)
    parser.add_argument("--base-alpha-s-rec", type=float, default=0.1)
    parser.add_argument("--base-alpha-sep", type=float, default=0.05)
    parser.add_argument("--base-lambda-residual-aux", type=float, default=0.25)
    parser.add_argument("--tol", type=float, default=1e-4)
    return parser.parse_args()


def expected_multiplier(epoch: int, start: int, end: int) -> float:
    if end <= start:
        return 1.0 if epoch >= start else 0.0
    if epoch <= start:
        return 0.0
    if epoch >= end:
        return 1.0
    return (epoch - start) / float(end - start)


def as_float(value: str, *, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise AssertionError(f"{name}: cannot parse float from {value!r}") from exc
    if not math.isfinite(out):
        raise AssertionError(f"{name}: non-finite value {out}")
    return out


def assert_close(name: str, got: float, expected: float, tol: float) -> None:
    if abs(got - expected) > tol:
        raise AssertionError(f"{name}: got {got}, expected {expected}, tol={tol}")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Missing diagnostics file: {path}")
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"Empty diagnostics file: {path}")
        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise SystemExit(f"Missing required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise SystemExit(f"No diagnostics rows found: {path}")
    return rows


def main() -> None:
    args = parse_args()
    rows = read_rows(args.diagnostics)

    bases = {
        "base_alpha_kd": args.base_alpha_kd,
        "base_alpha_s_rec": args.base_alpha_s_rec,
        "base_alpha_sep": args.base_alpha_sep,
        "base_lambda_residual_aux": args.base_lambda_residual_aux,
    }

    for row in rows:
        epoch = int(as_float(row["epoch"], name="epoch"))
        mult = expected_multiplier(epoch, args.start, args.end)

        if args.mode == "core":
            assert_close(
                f"epoch={epoch} b_loss_warmup_multiplier",
                as_float(row["b_loss_warmup_multiplier"], name="b_loss_warmup_multiplier"),
                mult,
                args.tol,
            )
            expected_active = 1.0 if mult < 1.0 else 0.0
            assert_close(
                f"epoch={epoch} b_loss_warmup_active",
                as_float(row["b_loss_warmup_active"], name="b_loss_warmup_active"),
                expected_active,
                args.tol,
            )
            for col, base_name in CORE_COLUMNS.items():
                assert_close(
                    f"epoch={epoch} {col}",
                    as_float(row[col], name=col),
                    bases[base_name] * mult,
                    args.tol,
                )
        else:
            assert_close(
                f"epoch={epoch} effective_alpha_kd",
                as_float(row["effective_alpha_kd"], name="effective_alpha_kd"),
                args.base_alpha_kd * mult,
                args.tol,
            )

    print(f"PASS warmup smoke: {args.diagnostics}")


if __name__ == "__main__":
    main()
