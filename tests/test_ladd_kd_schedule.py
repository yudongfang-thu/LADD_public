from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "ladd" / "code" / "src"
SHARED = ROOT / "shared"
YOLO = SHARED / "yolo"
for path in (SHARED, YOLO, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from teacher_student_decomposition_kd_hbb.schedule import (  # noqa: E402
    apply_det_only_phase_scales,
    compute_effective_ladd_weights,
    compute_kd_multiplier,
)


BASE_WEIGHTS = {
    "alpha_kd": 1.0,
    "alpha_s_rec": 0.1,
    "alpha_sep": 0.05,
    "lambda_residual_aux": 0.25,
    "lambda_reach": 1.0,
    "lambda_match_inner": 1.0,
    "lambda_rank_inner": 1.0,
}


def test_kd_multiplier_none_is_constant():
    assert compute_kd_multiplier(phase="b", epoch_1based=1) == 1.0
    assert compute_kd_multiplier(phase="b", epoch_1based=300, decay_mode="none") == 1.0
    assert compute_kd_multiplier(phase="a2", epoch_1based=300, decay_mode="linear", decay_start_epoch=1) == 1.0


def test_kd_multiplier_linear_schedule():
    kwargs = dict(phase="b", decay_mode="linear", decay_start_epoch=200, decay_end_epoch=300, final_mult=0.0)
    assert compute_kd_multiplier(epoch_1based=199, **kwargs) == 1.0
    assert compute_kd_multiplier(epoch_1based=200, **kwargs) == 1.0
    assert abs(compute_kd_multiplier(epoch_1based=250, **kwargs) - 0.5) < 1e-9
    assert compute_kd_multiplier(epoch_1based=300, **kwargs) == 0.0
    assert compute_kd_multiplier(epoch_1based=301, **kwargs) == 0.0


def test_kd_multiplier_step_schedule():
    kwargs = dict(phase="b", decay_mode="step", decay_start_epoch=200, final_mult=0.5)
    assert compute_kd_multiplier(epoch_1based=199, **kwargs) == 1.0
    assert compute_kd_multiplier(epoch_1based=200, **kwargs) == 0.5
    assert compute_kd_multiplier(epoch_1based=400, **kwargs) == 0.5


def test_kd_stop_after_takes_priority():
    kwargs = dict(
        phase="b",
        decay_mode="step",
        decay_start_epoch=200,
        final_mult=0.5,
        stop_after_epoch=220,
    )
    assert compute_kd_multiplier(epoch_1based=219, **kwargs) == 0.5
    assert compute_kd_multiplier(epoch_1based=220, **kwargs) == 0.0
    assert compute_kd_multiplier(epoch_1based=400, **kwargs) == 0.0


def test_b_det_only_zeroes_non_detection_weights_and_scales():
    weights = compute_effective_ladd_weights(
        phase="b",
        epoch_1based=250,
        base_weights=BASE_WEIGHTS,
        decay_mode="linear",
        decay_start_epoch=200,
        decay_end_epoch=300,
        final_mult=0.0,
        ladd_b_det_only=True,
    )
    assert weights["kd_multiplier"] == 0.5
    for key, value in weights.items():
        if key != "kd_multiplier":
            assert value == 0.0

    scales = apply_det_only_phase_scales(
        {"det": 1.0, "kd": 1.0, "student_rec": 1.0, "residual_aux": 1.0},
        phase="b",
        ladd_b_det_only=True,
    )
    assert scales["det"] == 1.0
    assert scales["kd"] == 0.0
    assert scales["student_rec"] == 0.0
    assert scales["residual_aux"] == 0.0


def test_a2_det_only_zeroes_non_detection_weights_and_scales():
    weights = compute_effective_ladd_weights(
        phase="a2",
        epoch_1based=10,
        base_weights=BASE_WEIGHTS,
        ladd_a2_det_only=True,
    )
    assert weights["kd_multiplier"] == 1.0
    for key, value in weights.items():
        if key != "kd_multiplier":
            assert value == 0.0

    scales = apply_det_only_phase_scales(
        {"det": 1.0, "rec": 1.0, "match": 1.0, "unmatch": 1.0, "task": 1.0},
        phase="a2",
        ladd_a2_det_only=True,
    )
    assert scales["det"] == 1.0
    assert scales["rec"] == 0.0
    assert scales["match"] == 0.0
    assert scales["unmatch"] == 0.0
    assert scales["task"] == 0.0


def test_model_train_refreshes_effective_weights_before_phase_assert():
    paths = [
        ROOT / "ladd" / "code" / "src" / "teacher_student_decomposition_kd_hbb" / "trainer.py",
        ROOT
        / "ladd"
        / "code_versions"
        / "current_hbb"
        / "src"
        / "teacher_student_decomposition_kd_hbb"
        / "trainer.py",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        start = text.find("    def _model_train(self):")
        end = text.find("    def optimizer_step(self):", start)
        assert start >= 0, path
        assert end >= 0, path
        model_train = text[start:end]
        refresh_idx = model_train.find("self._refresh_effective_ladd_weights()")
        assert_idx = model_train.find(
            'self._assert_b_phase_frozen_modules(model, context="after_model_train_and_bn_freeze")'
        )
        assert refresh_idx >= 0, path
        assert assert_idx >= 0, path
        assert refresh_idx < assert_idx, path
