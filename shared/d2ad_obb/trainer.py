from __future__ import annotations

from copy import copy
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models import yolo
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.torch_utils import unwrap_model

from .aug_policy import apply_unified_paired_aug_policy
from .loss import D2ADOBBLoss
from .model import D2ADOBBModel
from .paired_dataset import PairedOBBDataset


class D2ADOBBTrainer(OBBTrainer):
    """YOLO11-OBB trainer with paired teacher images and D2AD-R losses."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        overrides = {} if overrides is None else dict(overrides)
        self.d2ad_cfg = {
            "teacher_data": overrides.pop("teacher_data", None),
            "teacher_weights": overrides.pop("teacher_weights", "yolo11n-obb.pt"),
            "proj_dim": int(overrides.pop("proj_dim", 64)),
            "lambda_inv": float(overrides.pop("lambda_inv", 1.0)),
            "lambda_orth": float(overrides.pop("lambda_orth", 0.05)),
            "lambda_rec": float(overrides.pop("lambda_rec", 0.0)),
            "teacher_target_mode": overrides.pop("teacher_target_mode", "raw"),
        }
        if self.d2ad_cfg["teacher_data"] is None:
            raise ValueError("D2ADOBBTrainer requires 'teacher_data'.")
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)
        self.teacher_data = check_det_dataset(str(self.d2ad_cfg["teacher_data"]))
        self.teacher_model = None

        apply_unified_paired_aug_policy(self.args)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        if mode != "train":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
        teacher_path = self.teacher_data["train"]
        return PairedOBBDataset(
            img_path=img_path,
            teacher_img_path=teacher_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=(mode == "train"),
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=gs,
            pad=0.0,
            prefix=f"{mode}: ",
            task=self.args.task,
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction,
        )

    def get_model(self, cfg: str | dict | None = None, weights: str | Path | None = None, verbose: bool = True):
        model = D2ADOBBModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            d2ad_proj_dim=self.d2ad_cfg["proj_dim"],
            teacher_target_mode=self.d2ad_cfg["teacher_target_mode"],
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = (
            "box_loss",
            "cls_loss",
            "dfl_loss",
            "angle_loss",
            "inv_loss",
            "orth_loss",
            "rec_loss",
        )
        return yolo.obb.OBBValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        teacher_img = batch.get("teacher_img")
        if teacher_img is not None:
            batch["teacher_img"] = teacher_img.to(self.device, non_blocking=self.device.type == "cuda").float() / 255
        return batch

    def _setup_train(self):
        super()._setup_train()
        self.teacher_model = self._load_teacher_model()
        student_model = unwrap_model(self.model)
        student_model.criterion = D2ADOBBLoss(
            student_model,
            teacher_model=self.teacher_model,
            lambda_inv=self.d2ad_cfg["lambda_inv"],
            lambda_orth=self.d2ad_cfg["lambda_orth"],
            lambda_rec=self.d2ad_cfg["lambda_rec"],
        )
        if self.ema:
            self.ema.ema.criterion = D2ADOBBLoss(
                self.ema.ema,
                teacher_model=None,
                lambda_inv=self.d2ad_cfg["lambda_inv"],
                lambda_orth=self.d2ad_cfg["lambda_orth"],
                lambda_rec=self.d2ad_cfg["lambda_rec"],
            )

    def _load_teacher_model(self):
        teacher = YOLO(self.d2ad_cfg["teacher_weights"]).model
        teacher = teacher.to(self.device)
        teacher.float()
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher
