from __future__ import annotations

import argparse


_FLOAT_OVERRIDE_ARGS = (
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
    "degrees",
    "perspective",
    "translate",
    "scale",
    "mosaic",
    "mixup",
    "cutmix",
    "fliplr",
    "flipud",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "erasing",
)

_STR_OVERRIDE_ARGS = (
    "optimizer",
)

_INT_OVERRIDE_ARGS = (
    "close_mosaic",
    "save_period",
    "seed",
)

_BOOL_STORE_TRUE_ARGS = (
    "cos_lr",
)


def add_common_detector_train_overrides(parser: argparse.ArgumentParser) -> None:
    """Expose a small set of Ultralytics train args for controlled ablations."""
    for name in _FLOAT_OVERRIDE_ARGS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=None, dest=name)
    for name in _STR_OVERRIDE_ARGS:
        parser.add_argument(f"--{name.replace('_', '-')}", default=None, dest=name)
    for name in _INT_OVERRIDE_ARGS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=None, dest=name)
    for name in _BOOL_STORE_TRUE_ARGS:
        parser.add_argument(f"--{name.replace('_', '-')}", action="store_true", default=None, dest=name)
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_const",
        const=True,
        default=None,
        help="Force Ultralytics deterministic=True for reproducible runs.",
    )
    parser.add_argument(
        "--non-deterministic",
        dest="deterministic",
        action="store_const",
        const=False,
        help="Force Ultralytics deterministic=False. Useful when collecting natural run variance.",
    )


def collect_common_detector_train_overrides(args: argparse.Namespace) -> dict[str, float | int | bool]:
    overrides: dict[str, float | int | bool] = {}
    for name in _FLOAT_OVERRIDE_ARGS + _STR_OVERRIDE_ARGS + _INT_OVERRIDE_ARGS:
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    for name in _BOOL_STORE_TRUE_ARGS:
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    deterministic = getattr(args, "deterministic", None)
    if deterministic is not None:
        overrides["deterministic"] = deterministic
    return overrides
