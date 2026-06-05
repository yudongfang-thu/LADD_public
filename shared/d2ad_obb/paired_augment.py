"""Paired augmentation transforms for synchronized teacher/student geometric augmentation.

Each Paired* class wraps the corresponding Ultralytics transform to also process
``labels["teacher_img"]`` with **identical** geometric parameters.  Colour-only
transforms (HSV, Albumentations) remain student-only — they do not affect spatial
alignment.

Usage::

    from d2ad_obb.paired_augment import paired_v8_transforms, PairedFormat
    transforms = paired_v8_transforms(dataset, imgsz, hyp)
    transforms.append(PairedFormat(...))
"""
from __future__ import annotations

import random
from typing import Any

import cv2
import numpy as np

from ultralytics.data.augment import (
    Albumentations,
    Compose,
    Format,
    LetterBox,
    MixUp,
    Mosaic,
    RandomFlip,
    RandomHSV,
    RandomPerspective,
)
from ultralytics.utils.instance import Instances


# ---------------------------------------------------------------------------
# PairedLetterBox
# ---------------------------------------------------------------------------

class PairedLetterBox(LetterBox):
    """LetterBox that also pads ``labels["teacher_img"]`` identically."""

    def __call__(self, labels: dict[str, Any] | None = None, image: np.ndarray = None):
        teacher_img = labels.get("teacher_img") if labels else None
        labels = super().__call__(labels=labels, image=image)
        if teacher_img is not None and isinstance(labels, dict):
            # teacher_img has the same spatial dims as student → same padding.
            labels["teacher_img"] = super().__call__(labels=None, image=teacher_img)
        return labels


# ---------------------------------------------------------------------------
# PairedMosaic
# ---------------------------------------------------------------------------

class PairedMosaic(Mosaic):
    """Mosaic that composes teacher images on an identical canvas."""

    def _mosaic3(self, labels: dict[str, Any]) -> dict[str, Any]:
        mosaic_labels = []
        s = self.imgsz
        has_teacher = "teacher_img" in labels
        teacher_img3 = None

        for i in range(3):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            if i == 0:
                img3 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)
                if has_teacher:
                    t = labels_patch["teacher_img"]
                    teacher_img3 = np.full((s * 3, s * 3, t.shape[2]), 114, dtype=np.uint8)
                h0, w0 = h, w
                c = s, s, s + w, s + h
            elif i == 1:
                c = s + w0, s, s + w0 + w, s + h
            elif i == 2:
                c = s - w, s + h0 - h, s, s + h0

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)

            img3[y1:y2, x1:x2] = img[y1 - padh:, x1 - padw:]
            if has_teacher:
                t = labels_patch.get("teacher_img")
                if t is not None:
                    teacher_img3[y1:y2, x1:x2] = t[y1 - padh:, x1 - padw:]

            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)

        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img3[-self.border[0]: self.border[0], -self.border[1]: self.border[1]]
        if has_teacher and teacher_img3 is not None:
            final_labels["teacher_img"] = teacher_img3[
                -self.border[0]: self.border[0], -self.border[1]: self.border[1]
            ]
        return final_labels

    def _mosaic4(self, labels: dict[str, Any]) -> dict[str, Any]:
        mosaic_labels = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)
        has_teacher = "teacher_img" in labels
        teacher_img4 = None

        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                if has_teacher:
                    t = labels_patch["teacher_img"]
                    teacher_img4 = np.full((s * 2, s * 2, t.shape[2]), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            if has_teacher:
                t = labels_patch.get("teacher_img")
                if t is not None:
                    teacher_img4[y1a:y2a, x1a:x2a] = t[y1b:y2b, x1b:x2b]

            padw = x1a - x1b
            padh = y1a - y1b
            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)

        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img4
        if has_teacher and teacher_img4 is not None:
            final_labels["teacher_img"] = teacher_img4
        return final_labels

    def _mosaic9(self, labels: dict[str, Any]) -> dict[str, Any]:
        mosaic_labels = []
        s = self.imgsz
        hp, wp = -1, -1
        has_teacher = "teacher_img" in labels
        teacher_img9 = None

        for i in range(9):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            if i == 0:
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)
                if has_teacher:
                    t = labels_patch["teacher_img"]
                    teacher_img9 = np.full((s * 3, s * 3, t.shape[2]), 114, dtype=np.uint8)
                h0, w0 = h, w
                c = s, s, s + w, s + h
            elif i == 1:
                c = s, s - h, s + w, s
            elif i == 2:
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)

            img9[y1:y2, x1:x2] = img[y1 - padh:, x1 - padw:]
            if has_teacher:
                t = labels_patch.get("teacher_img")
                if t is not None:
                    teacher_img9[y1:y2, x1:x2] = t[y1 - padh:, x1 - padw:]
            hp, wp = h, w

            labels_patch = self._update_labels(labels_patch, padw + self.border[0], padh + self.border[1])
            mosaic_labels.append(labels_patch)

        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img9[-self.border[0]: self.border[0], -self.border[1]: self.border[1]]
        if has_teacher and teacher_img9 is not None:
            final_labels["teacher_img"] = teacher_img9[
                -self.border[0]: self.border[0], -self.border[1]: self.border[1]
            ]
        return final_labels


# ---------------------------------------------------------------------------
# PairedRandomPerspective
# ---------------------------------------------------------------------------

class PairedRandomPerspective(RandomPerspective):
    """RandomPerspective that warps teacher_img with the same affine matrix M."""

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        labels.pop("ratio_pad", None)

        img = labels["img"]
        teacher_img = labels.get("teacher_img")
        cls = labels["cls"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop("mosaic_border", self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2

        img, M, scale = self.affine_transform(img, border)

        # Apply the same M to teacher image.
        if teacher_img is not None:
            if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
                if self.perspective:
                    teacher_img = cv2.warpPerspective(
                        teacher_img, M, dsize=self.size, borderValue=(114, 114, 114)
                    )
                else:
                    teacher_img = cv2.warpAffine(
                        teacher_img, M[:2], dsize=self.size, borderValue=(114, 114, 114)
                    )
                if teacher_img.ndim == 2:
                    teacher_img = teacher_img[..., None]
            labels["teacher_img"] = teacher_img

        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)
        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)

        new_instances = Instances(bboxes, segments, keypoints, bbox_format="xyxy", normalized=False)
        new_instances.clip(*self.size)

        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        i = self.box_candidates(
            box1=instances.bboxes.T,
            box2=new_instances.bboxes.T,
            area_thr=0.01 if len(segments) else 0.10,
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels


# ---------------------------------------------------------------------------
# PairedRandomFlip
# ---------------------------------------------------------------------------

class PairedRandomFlip(RandomFlip):
    """RandomFlip that also flips teacher_img."""

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        img = labels["img"]
        teacher_img = labels.get("teacher_img")
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w

        if self.direction == "vertical" and random.random() < self.p:
            img = np.flipud(img)
            if teacher_img is not None:
                teacher_img = np.flipud(teacher_img)
            instances.flipud(h)
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])
        if self.direction == "horizontal" and random.random() < self.p:
            img = np.fliplr(img)
            if teacher_img is not None:
                teacher_img = np.fliplr(teacher_img)
            instances.fliplr(w)
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])

        labels["img"] = np.ascontiguousarray(img)
        if teacher_img is not None:
            labels["teacher_img"] = np.ascontiguousarray(teacher_img)
        labels["instances"] = instances
        return labels


# ---------------------------------------------------------------------------
# PairedMixUp
# ---------------------------------------------------------------------------

class PairedMixUp(MixUp):
    """MixUp that blends teacher images with the same ratio."""

    def _mix_transform(self, labels: dict[str, Any]) -> dict[str, Any]:
        r = np.random.beta(32.0, 32.0)
        labels2 = labels["mix_labels"][0]

        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)

        t1 = labels.get("teacher_img")
        t2 = labels2.get("teacher_img")
        if t1 is not None and t2 is not None:
            labels["teacher_img"] = (t1 * r + t2 * (1 - r)).astype(np.uint8)

        labels["instances"] = Instances.concatenate(
            [labels["instances"], labels2["instances"]], axis=0
        )
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        return labels


# ---------------------------------------------------------------------------
# PairedFormat
# ---------------------------------------------------------------------------

class PairedFormat(Format):
    """Format that also converts teacher_img to a CHW tensor."""

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        teacher_img = labels.pop("teacher_img", None)
        labels = super().__call__(labels)
        if teacher_img is not None:
            labels["teacher_img"] = self._format_img(teacher_img)
        return labels


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def paired_v8_transforms(dataset, imgsz: int, hyp) -> Compose:
    """Build paired augmentation pipeline (Ultralytics v8 transforms + teacher sync).

    CopyPaste and CutMix are disabled (p=0) because paired implementations are
    too complex.  Albumentations and RandomHSV apply to the student only.
    """
    mosaic = PairedMosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = PairedRandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=None if getattr(hyp, "stretch", False) else PairedLetterBox(new_shape=(imgsz, imgsz)),
    )
    pre_transform = Compose([mosaic, affine])

    return Compose(
        [
            pre_transform,
            PairedMixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            Albumentations(p=1.0, transforms=getattr(hyp, "augmentations", None)),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            PairedRandomFlip(direction="vertical", p=hyp.flipud),
            PairedRandomFlip(direction="horizontal", p=hyp.fliplr),
        ]
    )
