from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ultralytics.data.augment import Compose
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.patches import imread

from .paired_augment import PairedFormat, PairedLetterBox, paired_v8_transforms


class PairedOBBDataset(YOLODataset):
    """YOLO OBB dataset that adds a paired teacher image for train-time distillation.

    When ``augment=True``, geometric transforms are synchronized so that both
    student and teacher images receive identical spatial transformations.

    Set ``pair_teacher=False`` to use the same ``paired_v8_transforms`` stack without a
    teacher (single-modality baselines aligned with D2AD-R augmentation).
    """

    def __init__(
        self,
        *args,
        teacher_img_path: str | Path | None = None,
        pair_teacher: bool = True,
        data: dict | None = None,
        task: str = "obb",
        **kwargs,
    ):
        self.pair_teacher = pair_teacher
        if pair_teacher:
            if teacher_img_path is None:
                raise ValueError("teacher_img_path is required when pair_teacher=True")
            self.teacher_root = Path(teacher_img_path).resolve()
        else:
            self.teacher_root = None
        super().__init__(*args, data=data, task=task, **kwargs)
        # Keep symlink names intact for datasets such as VEDAI where paired
        # RGB/IR links share a basename but point to *_co.png / *_ir.png files.
        self.student_root = Path(self.im_files[0]).expanduser().absolute().parent

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def build_transforms(self, hyp: dict | None = None) -> Compose:
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = paired_v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([PairedLetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            PairedFormat(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                return_obb=self.use_obb,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,
            )
        )
        return transforms

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        label = super().get_image_and_label(index)
        if not self.pair_teacher:
            return label
        teacher_file = self._resolve_teacher_file(Path(label["im_file"]).expanduser().absolute())
        teacher_img = self._load_teacher_image(teacher_file, target_shape=label["resized_shape"])
        label["teacher_img"] = teacher_img  # numpy HWC — flows through paired transforms
        label["teacher_im_file"] = str(teacher_file)
        return label

    def _resolve_teacher_file(self, student_file: Path) -> Path:
        try:
            rel = student_file.relative_to(self.student_root)
        except ValueError:
            rel = Path(student_file.name)
        teacher_file = self.teacher_root / rel
        if not teacher_file.exists():
            raise FileNotFoundError(f"Paired teacher image not found: {teacher_file}")
        return teacher_file

    def _load_teacher_image(self, teacher_file: Path, target_shape: tuple[int, int] | None = None) -> np.ndarray:
        """Load teacher image and resize to match student's resized_shape."""
        img = imread(str(teacher_file), flags=self.cv2_flag)
        if img is None:
            raise FileNotFoundError(f"Teacher image could not be read: {teacher_file}")
        if img.ndim == 2:
            img = img[..., None]
        if target_shape is not None:
            h, w = target_shape
            if (img.shape[0], img.shape[1]) != (h, w):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
                if img.ndim == 2:
                    img = img[..., None]
        return img

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        import torch

        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "teacher_img", "text_feats", "sem_masks"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            elif k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            elif k in {"teacher_im_file", "im_file", "ori_shape", "resized_shape", "ratio_pad"}:
                value = list(value)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch
