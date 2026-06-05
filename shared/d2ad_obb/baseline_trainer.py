from __future__ import annotations

from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import unwrap_model

from .aug_policy import apply_unified_paired_aug_policy
from .paired_dataset import PairedOBBDataset


class UnifiedAugOBBTrainer(OBBTrainer):
    """Single-modality OBB training with the same ``paired_v8_transforms`` stack as D2AD-R (no teacher)."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        overrides = {} if overrides is None else dict(overrides)
        overrides["task"] = "obb"
        super().__init__(cfg, overrides, _callbacks)
        apply_unified_paired_aug_policy(self.args)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        if mode != "train":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
        return PairedOBBDataset(
            img_path=img_path,
            teacher_img_path=None,
            pair_teacher=False,
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
