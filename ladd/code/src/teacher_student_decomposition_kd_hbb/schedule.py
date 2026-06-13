from __future__ import annotations

import math
from typing import Mapping


NON_DETECTION_PHASE_LOSS_SCALES = (
    "rec",
    "teacher_sep",
    "match",
    "unmatch",
    "task",
    "kd",
    "student_rec",
    "student_sep",
    "residual_aux",
    "teacher_private_aux",
    "mask",
    "recon_task",
    "rs_comp",
    "r_obb",
    "s_repel",
    "path_b",
    "r_sar",
    "dkd",
    "proto_cls",
)


TRACKED_LADD_WEIGHT_KEYS = (
    "alpha_kd",
    "alpha_s_rec",
    "alpha_sep",
    "lambda_residual_aux",
    "lambda_reach",
    "lambda_match_inner",
    "lambda_rank_inner",
)


def clamp_unit(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def compute_kd_multiplier(
    *,
    phase: str,
    epoch_1based: int,
    decay_mode: str = "none",
    decay_start_epoch: int = -1,
    decay_end_epoch: int = -1,
    final_mult: float = 1.0,
    stop_after_epoch: int = -1,
) -> float:
    """Return the B-phase KD multiplier for a 1-based phase epoch."""
    if str(phase).lower() != "b":
        return 1.0

    epoch = int(epoch_1based)
    stop_after = int(stop_after_epoch)
    if stop_after >= 0 and epoch >= stop_after:
        return 0.0

    mode = str(decay_mode or "none").lower()
    start = int(decay_start_epoch)
    end = int(decay_end_epoch)
    final_mult = clamp_unit(final_mult)

    if mode in {"warmup", "warmup_linear", "linear_warmup", "ramp_linear"}:
        if start < 0:
            start = 0
        if end <= start:
            return final_mult if epoch >= start else 0.0
        if epoch <= start:
            return 0.0
        if epoch >= end:
            return final_mult
        progress = (epoch - start) / float(end - start)
        return progress * final_mult

    if mode == "none" or start < 0:
        return 1.0
    if mode == "step":
        return final_mult if epoch >= start else 1.0
    if end <= start:
        return final_mult if epoch >= start else 1.0
    if epoch <= start:
        return 1.0
    if epoch >= end:
        return final_mult

    progress = (epoch - start) / float(end - start)
    if mode == "linear":
        return 1.0 + progress * (final_mult - 1.0)
    if mode == "cosine":
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_mult + (1.0 - final_mult) * cosine

    raise ValueError(f"Unsupported ladd_kd_decay_mode={decay_mode}")


def is_kd_warmup_active(
    *,
    phase: str,
    epoch_1based: int,
    decay_mode: str = "none",
    decay_end_epoch: int = -1,
) -> bool:
    """Return whether the configured B-phase KD warmup has not reached full strength yet."""
    if str(phase).lower() != "b":
        return False
    mode = str(decay_mode or "none").lower()
    if mode not in {"warmup", "warmup_linear", "linear_warmup", "ramp_linear"}:
        return False
    end = int(decay_end_epoch)
    if end < 0:
        return False
    return int(epoch_1based) < end


def is_det_only_phase(*, phase: str, ladd_b_det_only: bool = False, ladd_a2_det_only: bool = False) -> bool:
    phase = str(phase).lower()
    return (phase == "b" and bool(ladd_b_det_only)) or (phase == "a2" and bool(ladd_a2_det_only))


def apply_det_only_phase_scales(
    scales: Mapping[str, float],
    *,
    phase: str,
    ladd_b_det_only: bool = False,
    ladd_a2_det_only: bool = False,
) -> dict[str, float]:
    """Return phase loss scales with non-detection terms disabled when requested."""
    updated = dict(scales)
    if is_det_only_phase(phase=phase, ladd_b_det_only=ladd_b_det_only, ladd_a2_det_only=ladd_a2_det_only):
        for key in NON_DETECTION_PHASE_LOSS_SCALES:
            updated[key] = 0.0
    return updated


def compute_effective_ladd_weights(
    *,
    phase: str,
    epoch_1based: int,
    base_weights: Mapping[str, float],
    decay_mode: str = "none",
    decay_start_epoch: int = -1,
    decay_end_epoch: int = -1,
    final_mult: float = 1.0,
    stop_after_epoch: int = -1,
    ladd_b_det_only: bool = False,
    ladd_a2_det_only: bool = False,
) -> dict[str, float]:
    """Compute the tracked effective LADD weights for diagnostics and runtime application."""
    weights = {key: float(base_weights.get(key, 0.0)) for key in TRACKED_LADD_WEIGHT_KEYS}
    base_alpha_kd = float(base_weights.get("alpha_kd", 0.0))
    kd_multiplier = compute_kd_multiplier(
        phase=phase,
        epoch_1based=epoch_1based,
        decay_mode=decay_mode,
        decay_start_epoch=decay_start_epoch,
        decay_end_epoch=decay_end_epoch,
        final_mult=final_mult,
        stop_after_epoch=stop_after_epoch,
    )
    weights["alpha_kd"] *= kd_multiplier
    weights["base_alpha_kd"] = base_alpha_kd
    if is_det_only_phase(phase=phase, ladd_b_det_only=ladd_b_det_only, ladd_a2_det_only=ladd_a2_det_only):
        for key in weights:
            if key != "base_alpha_kd":
                weights[key] = 0.0
    weights["kd_multiplier"] = kd_multiplier
    weights["kd_warmup_active"] = float(
        is_kd_warmup_active(
            phase=phase,
            epoch_1based=epoch_1based,
            decay_mode=decay_mode,
            decay_end_epoch=decay_end_epoch,
        )
    )
    return weights
