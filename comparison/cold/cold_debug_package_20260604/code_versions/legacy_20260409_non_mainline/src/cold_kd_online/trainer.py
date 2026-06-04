from __future__ import annotations

from copy import copy

import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.models import yolo
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model

from cold_kd.trainer import CoLDOBBTrainer

from .loss import OnlineCoLDOBBLoss


class OnlineCoLDOBBTrainer(CoLDOBBTrainer):
    """Online CoLD trainer: RGB teacher and SAR student are jointly updated."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        overrides = {} if overrides is None else dict(overrides)
        self.online_cfg = {
            "teacher_det_weight": float(overrides.pop("teacher_det_weight", 0.1)),
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
            "teacher_det_loss",
            "mean_iou_weight",
            "mean_teacher_top_conf",
        )
        return yolo.obb.OBBValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def _make_criterion(self, model, teacher_model):
        return OnlineCoLDOBBLoss(
            model,
            teacher_model=teacher_model,
            lambda_kd=self.kd_cfg["lambda_kd"],
            lambda_cls_cold=self.cold_cfg["lambda_cls_cold"],
            lambda_loc_cold=self.cold_cfg["lambda_loc_cold"],
            alpha_non_target=self.cold_cfg["alpha_non_target"],
            temperature=self.cold_cfg["temperature"],
            kd_region=self.cold_cfg["kd_region"],
            teacher_det_weight=self.online_cfg["teacher_det_weight"],
        )

    def _load_teacher_model(self):
        teacher = YOLO(self.kd_cfg["teacher_weights"]).model
        teacher = teacher.to(self.device)
        teacher.float()
        teacher.nc = self.data["nc"]
        teacher.names = self.data["names"]
        teacher.args = self.args
        teacher.train()
        for p in teacher.parameters():
            p.requires_grad_(True)
        return teacher

    @staticmethod
    def _group_trainable_params(module: nn.Module) -> dict[str, list[torch.nn.Parameter]]:
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        groups = {"weight": [], "bn": [], "bias": []}
        for module_name, submodule in module.named_modules():
            for param_name, param in submodule.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:
                    groups["bias"].append(param)
                elif isinstance(submodule, bn) or "logit_scale" in fullname:
                    groups["bn"].append(param)
                else:
                    groups["weight"].append(param)
        return groups

    def _attach_teacher_to_optimizer(self) -> None:
        existing = {group.get("param_group"): group for group in self.optimizer.param_groups}
        teacher_groups = self._group_trainable_params(self.teacher_model)
        added = {}
        for group_name, params in teacher_groups.items():
            if not params:
                continue
            template = existing.get(group_name, self.optimizer.param_groups[0])
            new_group = {k: v for k, v in template.items() if k != "params"}
            new_group["params"] = params
            new_group["param_group"] = group_name
            new_group["online_teacher"] = True
            self.optimizer.add_param_group(new_group)
            added[group_name] = len(params)

        prev_last_epoch = getattr(self.scheduler, "last_epoch", -1)
        self._setup_scheduler()
        self.scheduler.last_epoch = prev_last_epoch

        if RANK in {-1, 0}:
            LOGGER.info(
                "Attached online CoLD teacher params to optimizer: "
                + ", ".join(f"{k}={v}" for k, v in sorted(added.items()))
            )

    def _setup_train(self):
        super()._setup_train()
        student_model = unwrap_model(self.model)
        student_model.criterion = self._make_criterion(student_model, self.teacher_model)
        if self.ema:
            self.ema.ema.criterion = self._make_criterion(self.ema.ema, None)
        self._attach_teacher_to_optimizer()

    def _model_train(self):
        super()._model_train()
        if self.teacher_model is not None:
            self.teacher_model.train()

    def optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.teacher_model is not None:
            params.extend(p for p in self.teacher_model.parameters() if p.requires_grad)
        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
