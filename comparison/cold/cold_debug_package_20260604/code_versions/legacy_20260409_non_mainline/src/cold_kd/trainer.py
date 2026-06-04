from __future__ import annotations

from copy import copy

from ultralytics.models import yolo
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import unwrap_model

from vanilla_kd.trainer import VanillaKDOBBTrainer

from .loss import CoLDOBBLoss


class CoLDOBBTrainer(VanillaKDOBBTrainer):
    """YOLO11-OBB adapted CoLD trainer under the repo's frozen-teacher comparison setup."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        overrides = {} if overrides is None else dict(overrides)
        self.cold_cfg = {
            "alpha_non_target": float(overrides.pop("alpha_non_target", 2.0)),
            "temperature": float(overrides.pop("temperature", 20.0)),
            "lambda_cls_cold": float(overrides.pop("lambda_cls_cold", 1.0)),
            "lambda_loc_cold": float(overrides.pop("lambda_loc_cold", 1.0)),
            "kd_region": overrides.pop("kd_region", "positive"),
        }
        super().__init__(cfg, overrides, _callbacks)

    def get_validator(self):
        self.loss_names = (
            "box_loss",
            "cls_loss",
            "dfl_loss",
            "angle_loss",
            "cls_cold_loss",
            "loc_cold_loss",
            "mean_iou_weight",
            "mean_teacher_top_conf",
        )
        return yolo.obb.OBBValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def _setup_train(self):
        super(VanillaKDOBBTrainer, self)._setup_train()
        self.teacher_model = self._load_teacher_model()
        student_model = unwrap_model(self.model)
        student_model.criterion = CoLDOBBLoss(
            student_model,
            teacher_model=self.teacher_model,
            lambda_kd=self.kd_cfg["lambda_kd"],
            lambda_cls_cold=self.cold_cfg["lambda_cls_cold"],
            lambda_loc_cold=self.cold_cfg["lambda_loc_cold"],
            alpha_non_target=self.cold_cfg["alpha_non_target"],
            temperature=self.cold_cfg["temperature"],
            kd_region=self.cold_cfg["kd_region"],
        )
        if self.ema:
            self.ema.ema.criterion = CoLDOBBLoss(
                self.ema.ema,
                teacher_model=None,
                lambda_kd=self.kd_cfg["lambda_kd"],
                lambda_cls_cold=self.cold_cfg["lambda_cls_cold"],
                lambda_loc_cold=self.cold_cfg["lambda_loc_cold"],
                alpha_non_target=self.cold_cfg["alpha_non_target"],
                temperature=self.cold_cfg["temperature"],
                kd_region=self.cold_cfg["kd_region"],
            )
